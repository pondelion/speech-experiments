"""固定shapeのORT sessionと固定I/OでCUDA Graphを安全に再生する。"""

import gc
import types
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def install(torch, pipe, t2s, paths, profile=False, bucket=0):
    import numpy as np
    import onnx
    import onnxruntime as ort
    from transformers.modeling_outputs import BaseModelOutput

    sessions = t2s._ort_sessions
    sessions.clear()
    stats = {
        "bucket": bucket,
        "sessions_created": 0,
        "cache_hits": 0,
        "runs": 0,
        "max_profiles_per_graph": 2,
    }

    class Captured(torch.nn.Module):
        def __init__(self, name, path):
            super().__init__()
            self.name = name
            self.path = path
            self.cache = OrderedDict()

        def state(self, values, shape):
            key = tuple(tuple(x.shape) for x in values)
            if key in self.cache:
                stats["cache_hits"] += 1
                self.cache.move_to_end(key)
                return self.cache[key]
            if len(self.cache) == 2:
                old_key, old_state = self.cache.popitem(last=False)
                sessions.pop((self.name, old_key), None)
                del old_state
                gc.collect()
            directory = ROOT / "runs" / "static" / self.path.parent.name
            directory.mkdir(parents=True, exist_ok=True)
            suffix = "_".join("x".join(map(str, s)) for s in key)
            path = directory / f"{self.name}_{suffix}.onnx"
            if (
                not path.exists()
                or path.stat().st_mtime_ns < self.path.stat().st_mtime_ns
            ):
                model = onnx.load(str(self.path))
                for spec, value in zip(model.graph.input, values):
                    dims = spec.type.tensor_type.shape.dim
                    for dim, size in zip(dims, value.shape):
                        dim.ClearField("dim_param")
                        dim.dim_value = size
                for dim, size in zip(
                    model.graph.output[0].type.tensor_type.shape.dim, shape
                ):
                    dim.ClearField("dim_param")
                    dim.dim_value = size
                del model.graph.value_info[:]
                onnx.save(model, str(path))
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            if profile:
                options.enable_profiling = True
                options.profile_file_prefix = str(directory / "profile")
            options.add_session_config_entry(
                "session.disable_cpu_ep_fallback", "1"
            )
            provider = {
                "device_id": 0,
                "use_tf32": 0,
                "enable_cuda_graph": 1,
                "user_compute_stream": str(
                    torch.cuda.current_stream().cuda_stream
                ),
            }
            session = ort.InferenceSession(
                str(path),
                options,
                providers=[("CUDAExecutionProvider", provider)],
            )
            session.disable_fallback()
            inputs = [torch.empty_like(x.contiguous()) for x in values]
            output = torch.empty(shape, device="cuda", dtype=torch.float32)
            binding = session.io_binding()
            dtypes = {
                torch.float32: np.float32,
                torch.int64: np.int64,
                torch.bool: np.bool_,
            }
            for spec, value in zip(session.get_inputs(), inputs):
                binding.bind_input(
                    spec.name,
                    "cuda",
                    0,
                    dtypes[value.dtype],
                    tuple(value.shape),
                    value.data_ptr(),
                )
            binding.bind_output(
                "y",
                "cuda",
                0,
                np.float32,
                tuple(output.shape),
                output.data_ptr(),
            )
            state = session, binding, inputs, output
            stats["sessions_created"] += 1
            self.cache[key] = state
            sessions[(self.name, key)] = session
            return state

        def run(self, values, shape):
            length = shape[1]
            if length > 32768:
                raise ValueError("Sequence exceeds the RoPE cache limit")
            if bucket:
                target = ((length + bucket - 1) // bucket) * bucket
                sequence_inputs = {
                    "encoder": {0, 1},
                    "blocks": {0, 2},
                    "velocity": {0, 1, 3},
                }[self.name]
                padded = []
                for index, value in enumerate(values):
                    if index not in sequence_inputs or target == length:
                        padded.append(value)
                        continue
                    dims = list(value.shape)
                    dims[1] = target
                    fill = value.dtype == torch.bool
                    buffer = torch.full(
                        dims, fill, device=value.device, dtype=value.dtype
                    )
                    buffer[:, :length].copy_(value)
                    padded.append(buffer)
                values = padded
                shape = (shape[0], target, shape[2])
            session, binding, inputs, output = self.state(values, shape)
            for target, source in zip(inputs, values):
                target.copy_(source)
            session.run_with_iobinding(binding)
            stats["runs"] += 1
            return output[:, :length].clone()

        def forward(self, x, t, pad):
            return self.run([x, t, pad], tuple(x.shape))

    class Encoder(Captured):
        def forward(self, inputs_embeds, attention_mask):
            return BaseModelOutput(
                last_hidden_state=self.run(
                    [inputs_embeds, attention_mask], tuple(inputs_embeds.shape)
                )
            )

    t2s.encoder = Encoder("encoder", paths[0])
    t2s.blocks = torch.nn.ModuleList([Captured("blocks", paths[1])])
    velocity = Captured("velocity", paths[2])

    def run_velocity(self, x, cond, t, pad, text=None, text_pad=None):
        return velocity.run([x, cond, t, pad], tuple(x.shape))

    pipe.s2m.velocity = types.MethodType(run_velocity, pipe.s2m)
    gc.collect()
    torch.cuda.empty_cache()
    return stats
