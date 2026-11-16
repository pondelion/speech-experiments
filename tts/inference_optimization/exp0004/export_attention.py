"""SDPAを標準Attention(opset 23)としてexportし、細かな演算への分解を避ける。"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def derived():
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.onnx_model import ONNXModel
    from onnxruntime.transformers.float16 import convert_float_to_float16

    for name in ["encoder", "blocks", "velocity"]:
        src = ROOT / "runs" / "attention_fp32" / f"{name}_fp32.onnx"
        for precision in ["int8", "fp16"]:
            directory = ROOT / "runs" / f"attention_{precision}"
            directory.mkdir(parents=True, exist_ok=True)
            dest = directory / f"{name}_{precision}.onnx"
            if dest.exists():
                continue
            if precision == "int8":
                quantize_dynamic(
                    str(src),
                    str(dest),
                    per_channel=True,
                    weight_type=QuantType.QInt8,
                    op_types_to_quantize=["MatMul"],
                    extra_options={"MatMulConstBOnly": True},
                )
            else:
                model = convert_float_to_float16(
                    onnx.load(str(src)), keep_io_types=True
                )
                ONNXModel(model).topological_sort()
                onnx.checker.check_model(model)
                onnx.save(model, str(dest))
            onnx.checker.check_model(onnx.load(str(dest)))
            print("DERIVED", name, precision, flush=True)


def main():
    sys.path.insert(0, str(ROOT.parent / "exp0001"))
    sys.path.insert(0, str(ROOT.parent / "exp0003"))
    from benchmark import build_models
    from onnx_models import export_models

    torch, pipe, t2s, *_ = build_models("cpu")
    torch.set_num_threads(8)
    optimized = "--optimized" in sys.argv
    if optimized:
        from graph_rewrite import install

        install(torch, pipe, t2s)
    import torch.nn.functional as F

    original = F.scaled_dot_product_attention

    class Attention(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, mask, scale):
            return original(q, k, v, attn_mask=mask, scale=scale)

        @staticmethod
        @torch.onnx.symbolic_helper.parse_args("v", "v", "v", "v", "f")
        def symbolic(g, q, k, v, mask, scale):
            return g.op(
                "Attention", q, k, v, mask, is_causal_i=0, scale_f=scale
            ).setType(q.type())

    def sdpa(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ):
        if dropout_p or is_causal or enable_gqa:
            raise RuntimeError(
                "このexportは非causal・dropoutなし・GQAなしのSDPAを対象とする"
            )
        scale = 1 / math.sqrt(q.shape[-1]) if scale is None else scale
        if attn_mask is None:
            attn_mask = torch.ones(
                (q.shape[0], 1, q.shape[-2], k.shape[-2]),
                dtype=torch.bool,
                device=q.device,
            )
        # ORT CPU Attention currently requires an explicit query-length mask.
        mask = attn_mask.expand(
            q.shape[0], attn_mask.shape[1], q.shape[-2], k.shape[-2]
        )
        return Attention.apply(q, k, v, mask, scale)

    try:
        F.scaled_dot_product_attention = sdpa
        export_models(
            torch,
            pipe,
            t2s,
            output_dir=ROOT
            / "runs"
            / ("optimized_fp32" if optimized else "attention_fp32"),
            report_dir=ROOT
            / "results"
            / ("optimized_export" if optimized else "attention"),
            opset=23,
        )
    finally:
        F.scaled_dot_product_attention = original


if __name__ == "__main__":
    if "--derived-only" in sys.argv:
        derived()
    else:
        main()
