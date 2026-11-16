"""exp3の接続を共用し、session・graph・計算精度を独立に比較する。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def prepare(torch, pipe, t2s, device, config, profile=False):
    sys.path.insert(0, str(ROOT.parent / "exp0003"))
    import onnx_models as base

    if config.endswith("_async"):
        raise RuntimeError(
            "Rejected: asynchronous TTS screening produced nonfinite output"
        )
    if config == "torch" or config.startswith("torch_"):
        threads = (
            int(config.rsplit("_", 1)[-1])
            if config.rsplit("_", 1)[-1].isdigit()
            else 8
        )
        torch.set_num_threads(threads)
        if config == "torch_bf16":
            if device != "cuda":
                raise ValueError("The BF16 reference is CUDA-only")
            t2s.to(dtype=torch.bfloat16)
            pipe.s2m.to(dtype=torch.bfloat16)
        if "int8" in config:
            from quantize import dynamic_int8

            pipe.s2m = dynamic_int8(torch, pipe.s2m)
            t2s = dynamic_int8(torch, t2s, exclude={"head"})
        return {
            "backend": "pytorch_dynamic_int8"
            if "int8" in config
            else "pytorch_eager",
            "threads": threads,
        }, t2s
    import onnxruntime as ort

    threads = 8
    quantized_opt = config.startswith(
        ("opt8_", "pt8_", "rr8_", "native8_", "native7_")
    )
    if config.startswith(("native8_", "native7_")):
        from quantize import dynamic_int8

        pipe.s2m = dynamic_int8(torch, pipe.s2m)
        t2s = dynamic_int8(torch, t2s, exclude={"head"})
    optimized = config.startswith(("opt_", "gemm_")) or quantized_opt
    fused = config.startswith(("attn_", "attn8_", "attn16_", "opt_", "gemm_"))
    fused = fused or quantized_opt
    precision = (
        "int8"
        if config.startswith("attn8_") or quantized_opt
        else ("fp16" if config.startswith("attn16_") else "fp32")
    )
    pool_config = config.split("_", 1)[1] if fused else config
    shared = pool_config.startswith("shared")
    spin = pool_config == "legacy"
    if len(pool_config.split("_")) > 1 and pool_config.split("_")[1].isdigit():
        threads = int(pool_config.split("_")[1])
    frontend_threads = 1 if "_front1" in config else 8
    torch.set_num_threads(frontend_threads)
    if shared:
        ort.set_global_thread_pool_sizes(
            intra_op_num_threads=threads, inter_op_num_threads=1
        )

    def session(path, options, providers):
        if profile:
            directory = ROOT / "runs" / "profiles"
            directory.mkdir(parents=True, exist_ok=True)
            options.enable_profiling = True
            options.profile_file_prefix = str(
                directory / f"{device}_{config}_{path.stem}"
            )
        options.intra_op_num_threads = threads
        options.add_session_config_entry(
            "session.intra_op.allow_spinning", "1" if spin else "0"
        )
        options.add_session_config_entry(
            "session.inter_op.allow_spinning", "0"
        )
        if shared:
            options.use_per_session_threads = False
        result = ort.InferenceSession(
            str(path), sess_options=options, providers=providers
        )
        return result

    paths = (
        [
            ROOT
            / "runs"
            / (
                f"optimized_{precision}"
                if optimized
                else f"attention_{precision}"
            )
            / f"{n}_{precision}.onnx"
            for n in ["encoder", "blocks", "velocity"]
        ]
        if fused
        else None
    )
    if config.startswith("gemm_"):
        paths = [
            ROOT / "runs" / "gemm_fp32" / f"{n}_fp32.onnx"
            for n in ["encoder", "blocks", "velocity"]
        ]
    if quantized_opt:
        prefix = config.split("_", 1)[0]
        if prefix == "native8":
            prefix = "pt8"
        paths = [
            ROOT / "runs" / f"{prefix}_int8" / f"{n}_int8.onnx"
            for n in ["encoder", "blocks", "velocity"]
        ]
    metadata, t2s = base.prepare(
        torch,
        pipe,
        t2s,
        device,
        precision,
        session_factory=session,
        model_paths=paths,
    )
    if config.endswith(("_cg", "_cg128")):
        if device != "cuda":
            raise ValueError("CUDA Graph is CUDA-only")
        from capture import install

        metadata["capture"] = install(
            torch,
            pipe,
            t2s,
            paths,
            profile=profile,
            bucket=128 if config.endswith("_cg128") else 0,
        )
        metadata["cuda_graph"] = True
    metadata.update(
        config=config,
        threads_per_pool=threads,
        shared_thread_pool=shared,
        session_spinning_request=spin,
        global_pool_spinning="ORT environment default" if shared else None,
        frontend_threads=frontend_threads,
        fused_attention=fused,
        async_execution=config.endswith("_async"),
    )
    return metadata, t2s
