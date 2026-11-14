"""torchaoの重み量子化backendを小さなLinearで検証する。"""
import argparse
import json
from pathlib import Path


def main():
    import torch
    import torchao
    from torchao.quantization import quantize_, Int8WeightOnlyConfig, Int4WeightOnlyConfig
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--tiled", action="store_true", help="INT4をPyTorchのtile-packed経路で試す")
    args = parser.parse_args()
    torch.set_num_threads(8)
    result = {"device": args.device, "torch": torch.__version__, "torchao": torchao.__version__, "cases": {}}
    for bits in [8, 4]:
        for dtype in [torch.float32, torch.bfloat16]:
            key = f"int{bits}_{str(dtype).split('.')[-1]}"
            try:
                torch.manual_seed(0)
                model = torch.nn.Sequential(torch.nn.Linear(512, 512)).to(device=args.device, dtype=dtype).eval()
                x = torch.randn(1, 128, 512, device=args.device, dtype=dtype)
                with torch.inference_mode():
                    ref = model(x).float()
                    config = Int8WeightOnlyConfig(set_inductor_config=False) if bits == 8 else Int4WeightOnlyConfig(group_size=128, set_inductor_config=False)
                    if bits == 4 and args.tiled:
                        config.int4_packing_format = type(config.int4_packing_format).TILE_PACKED_TO_4D
                    quantize_(model, config)
                    out = model(x).float()
                result["cases"][key] = {"ok": True, "weight_type": type(model[0].weight).__name__,
                    "finite": bool(torch.isfinite(out).all()),
                    "relative_l2": ((out-ref).norm()/ref.norm()).item()}
            except Exception as exc:
                # 公開結果にローカルpathを残さない。
                message = str(exc).replace(str(Path.home()), "<local>")
                result["cases"][key] = {"ok": False, "error_type": type(exc).__name__, "message": message[:500]}
    root = Path(__file__).resolve().parent / "results" / "weight_only"
    root.mkdir(parents=True, exist_ok=True)
    suffix = "_tiled" if args.tiled else ""
    result["int4_packing"] = "tile_packed_to_4d" if args.tiled else "plain"
    (root / f"probe_{args.device}{suffix}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
