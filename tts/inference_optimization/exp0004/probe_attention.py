"""候補runtimeの可変長・padding付き出力と実行providerを検査する。"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from runtime import prepare

ROOT = Path(__file__).resolve().parent


def main():
    sys.path.insert(0, str(ROOT.parent / "exp0001"))
    from benchmark import build_models

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--config", default="attn_nospin_8")
    parser.add_argument("--reference-int8", action="store_true")
    args = parser.parse_args()
    torch, pipe, t2s, *_ = build_models(args.device)
    torch.set_num_threads(8)
    if args.reference_int8:
        sys.path.insert(0, str(ROOT.parent / "exp0003"))
        from quantize import dynamic_int8

        pipe.s2m = dynamic_int8(torch, pipe.s2m)
        t2s = dynamic_int8(torch, t2s, exclude={"head"})
    cases = []
    for batch, length in [(1, 8), (2, 32), (4, 65)]:
        torch.manual_seed(0)
        x = torch.randn(batch, length, 512, device=args.device)
        pad = torch.zeros(batch, length, dtype=torch.bool, device=args.device)
        pad[:, -3:] = True
        with torch.inference_mode():
            inputs = [x, (~pad).long()]
            ref = t2s.encoder(
                inputs_embeds=inputs[0], attention_mask=inputs[1]
            ).last_hidden_state
            cases.append(("encoder", inputs, ref.cpu()))
            inputs = [x, torch.randn(batch, 512, device=args.device), pad]
            ref = x
            for block in t2s.blocks:
                ref = block(ref, inputs[1], pad)
            cases.append(("blocks", inputs, ref.cpu()))
            inputs = [
                torch.randn(batch, length, 80, device=args.device),
                x,
                torch.rand(batch, device=args.device),
                pad,
            ]
            cases.append(
                ("velocity", inputs, pipe.s2m.velocity(*inputs).cpu())
            )
    meta, t2s = prepare(
        torch, pipe, t2s, args.device, args.config, profile=True
    )
    errors = []
    for name, inputs, ref in cases:
        with torch.inference_mode():
            if name == "encoder":
                out = t2s.encoder(
                    inputs_embeds=inputs[0], attention_mask=inputs[1]
                ).last_hidden_state
            elif name == "blocks":
                out = inputs[0]
                for block in t2s.blocks:
                    out = block(out, inputs[1], inputs[2])
            else:
                out = pipe.s2m.velocity(*inputs)
            assert torch.isfinite(out).all()
            err = float((out.cpu() - ref).norm() / ref.norm())
        tolerance = (
            0.15
            if args.config.startswith(
                (
                    "attn8_",
                    "opt8_",
                    "pt8_",
                    "rr8_",
                    "native8_",
                    "native7_",
                    "torch_int8",
                )
            )
            else (0.02 if args.config.startswith("attn16_") else 0.001)
        )
        if err > tolerance:
            raise RuntimeError(
                f"{name} relative error {err} exceeds {tolerance}"
            )
        errors.append(
            {"graph": name, "shape": list(out.shape), "relative_l2": err}
        )
    placement = {}
    for name, session in getattr(t2s, "_ort_sessions", {}).items():
        raw = json.loads(Path(session.end_profiling()).read_text())
        counts = Counter(
            (e["args"].get("op_name"), e["args"].get("provider"))
            for e in raw
            if e.get("name", "").endswith("_kernel_time")
            and "provider" in e.get("args", {})
        )
        heavy = {
            "MatMul",
            "Gemm",
            "Attention",
            "MultiHeadAttention",
            "MatMulInteger",
            "MatMulNBits",
            "FusedMatMul",
            "LayerNormalization",
            "SkipLayerNormalization",
            "Gelu",
            "FastGelu",
        }
        if args.device == "cuda" and any(
            provider == "CPUExecutionProvider" and op in heavy
            for op, provider in counts
        ):
            raise RuntimeError(f"heavy CPU fallback in {name}: {dict(counts)}")
        placement[str(name)] = {
            f"{op}:{provider}": count
            for (op, provider), count in counts.items()
        }
    result = {
        "device": args.device,
        "config": args.config,
        "runtime": meta,
        "errors": errors,
        "provider_nodes": placement,
    }
    out = ROOT / "results" / "probes"
    out.mkdir(parents=True, exist_ok=True)
    suffix = "_reference_int8" if args.reference_int8 else ""
    (out / f"{args.device}_{args.config}{suffix}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        "PASS",
        args.device,
        args.config,
        "max relative_l2",
        max(e["relative_l2"] for e in errors),
        flush=True,
    )


if __name__ == "__main__":
    main()
