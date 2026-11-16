"""学習済みの数式を保ち、固定幅の分割とRoPEを直接exportする。"""

import types
from pathlib import Path


def flatten_gemm():
    """rank-3 Linearを2D Gemmへ変換し、CPUのbias融合を可能にする。"""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    root = Path(__file__).resolve().parent
    report = {}
    for name in ["encoder", "blocks", "velocity"]:
        source = root / "runs" / "optimized_fp32" / f"{name}_fp32.onnx"
        model = onnx.shape_inference.infer_shapes(onnx.load(str(source)))
        constants = {
            x.name: numpy_helper.to_array(x) for x in model.graph.initializer
        }
        rank = {
            x.name: len(x.type.tensor_type.shape.dim)
            for x in list(model.graph.input) + list(model.graph.value_info)
        }
        consumers = {}
        for node in model.graph.node:
            for value in node.input:
                consumers.setdefault(value, []).append(node)
        removed = set()
        nodes = []
        count = 0
        for node in model.graph.node:
            if node.name in removed:
                continue
            weight = (
                constants.get(node.input[1])
                if (node.op_type == "MatMul" and len(node.input) == 2)
                else None
            )
            users = consumers.get(node.output[0], [])
            if (
                weight is None
                or weight.ndim != 2
                or rank.get(node.input[0]) != 3
                or len(users) != 1
                or users[0].op_type != "Add"
            ):
                nodes.append(node)
                continue
            add = users[0]
            bias = next((v for v in add.input if v != node.output[0]), "")
            if bias not in constants or constants[bias].shape != (
                weight.shape[1],
            ):
                nodes.append(node)
                continue
            prefix = node.output[0] + "_gemm"
            arrays = {
                "flat": [-1, weight.shape[0]],
                "start": [0],
                "end": [2],
                "axis": [0],
                "width": [weight.shape[1]],
            }
            for key, value in arrays.items():
                model.graph.initializer.append(
                    numpy_helper.from_array(
                        np.array(value, dtype=np.int64), prefix + key
                    )
                )
            nodes += [
                helper.make_node(
                    "Reshape",
                    [node.input[0], prefix + "flat"],
                    [prefix + "input"],
                ),
                helper.make_node(
                    "Gemm",
                    [prefix + "input", node.input[1], bias],
                    [prefix + "out"],
                    name=node.name + "/flat_gemm",
                ),
                helper.make_node("Shape", [node.input[0]], [prefix + "shape"]),
                helper.make_node(
                    "Slice",
                    [
                        prefix + "shape",
                        prefix + "start",
                        prefix + "end",
                        prefix + "axis",
                    ],
                    [prefix + "leading"],
                ),
                helper.make_node(
                    "Concat",
                    [prefix + "leading", prefix + "width"],
                    [prefix + "output_shape"],
                    axis=0,
                ),
                helper.make_node(
                    "Reshape",
                    [prefix + "out", prefix + "output_shape"],
                    list(add.output),
                    name=node.name + "/unflatten",
                ),
            ]
            removed.add(add.name)
            count += 1
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        onnx.checker.check_model(model)
        output = root / "runs" / "gemm_fp32" / source.name
        output.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(output))
        report[name] = count
    import json

    path = root / "results" / "gemm_export.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(report)


def quantized():
    """CPU整数MatMulの粒度と範囲を切り分ける派生graphを作る。"""
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    root = Path(__file__).resolve().parent
    for mode, per_channel, reduce_range in [
        ("opt8", True, False),
        ("pt8", False, False),
        ("rr8", True, True),
    ]:
        for name in ["encoder", "blocks", "velocity"]:
            source = root / "runs" / "optimized_fp32" / f"{name}_fp32.onnx"
            dest = root / "runs" / f"{mode}_int8" / f"{name}_int8.onnx"
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            quantize_dynamic(
                str(source),
                str(dest),
                per_channel=per_channel,
                reduce_range=reduce_range,
                weight_type=QuantType.QInt8,
                op_types_to_quantize=["MatMul"],
                extra_options={"MatMulConstBOnly": True},
            )
            onnx.checker.check_model(onnx.load(str(dest)))
            print("QUANTIZED", mode, name, flush=True)


def install(torch, pipe, t2s):
    import torch.nn.functional as F

    class Rotary(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, cos, sin, position):
            ids = torch.arange(x.shape[2], device=x.device) + position[0]
            c, s = cos[ids], sin[ids]
            real, imag = x[..., 0::2], x[..., 1::2]
            return torch.stack(
                [real * c - imag * s, real * s + imag * c], dim=-1
            ).flatten(-2)

        @staticmethod
        def symbolic(g, x, cos, sin, position):
            return g.op(
                "com.microsoft::RotaryEmbedding",
                x,
                position,
                cos,
                sin,
                interleaved_i=1,
            ).setType(x.type())

    def forward(self, x, t_emb, pad, text=None, text_pad=None):
        if text is not None:
            raise ValueError("Cross attention needs a separate export path")
        width = self.qkv.in_features
        batch, length = x.shape[:2]
        values = self.ada(t_emb).unsqueeze(1).split(width, dim=-1)
        shift1, scale1, gate1, shift2, scale2, gate2 = values
        h = self.norm1(x) * (1 + scale1) + shift1
        qkv = self.qkv(h).reshape(
            batch, length, 3, self.heads, width // self.heads
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q = Rotary.apply(q, self.rope_cos, self.rope_sin, self.rope_start)
        k = Rotary.apply(k, self.rope_cos, self.rope_sin, self.rope_start)
        attention = F.scaled_dot_product_attention(
            q, k, v, attn_mask=(~pad)[:, None, None, :]
        )
        attention = attention.transpose(1, 2).reshape(batch, length, width)
        x = x + gate1 * self.out(attention)
        h = self.norm2(x) * (1 + scale2) + shift2
        return x + gate2 * self.mlp(h)

    for block in list(t2s.blocks) + list(pipe.s2m.blocks):
        head_dim = block.qkv.in_features // block.heads
        freq = 1.0 / (
            10000 ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        angle = torch.outer(torch.arange(32768).float(), freq)
        block.register_buffer("rope_cos", angle.cos())
        block.register_buffer("rope_sin", angle.sin())
        block.register_buffer("rope_start", torch.tensor([0]))
        block.forward = types.MethodType(forward, block)


if __name__ == "__main__":
    import sys

    if "--quantize" in sys.argv:
        quantized()
    else:
        flatten_gemm()
