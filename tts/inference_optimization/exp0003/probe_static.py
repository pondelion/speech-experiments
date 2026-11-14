"""Dで使う既存SmoothQuant/static INT8 APIの単体確認。"""
import argparse
import json
from pathlib import Path


def main():
    import torch
    from torchao.quantization import quantize_, Int8StaticActivationInt8WeightConfig
    from torchao.quantization.granularity import PerTensor, PerRow
    from torchao.prototype.smoothquant import SmoothQuantConfig
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    output = {}
    for alpha in [None, 0.5, 0.75]:
        try:
            torch.manual_seed(0)
            model = torch.nn.Sequential(torch.nn.Linear(512, 512)).to(args.device).eval()
            x = torch.randn(1, 64, 512, device=args.device)
            base = Int8StaticActivationInt8WeightConfig(granularity=[PerTensor(), PerRow()], set_inductor_config=False)
            # prototype observerはdictを要求するが、0.18のconfigはdataclassを返す。
            kwargs = base.get_act_quant_kwargs()
            base.get_act_quant_kwargs = lambda kwargs=kwargs: {**vars(kwargs), "is_symmetric": True}
            with torch.no_grad():
                ref = model(x)
                quantize_(model, SmoothQuantConfig(base, step="prepare", alpha=alpha, use_running_absmax=True))
                model(x)
                quantize_(model, SmoothQuantConfig(base, step="prepare_for_smoothquant_smoothing_factor", alpha=alpha, use_running_absmax=True))
                model(x)
                quantize_(model, SmoothQuantConfig(base, step="convert", alpha=alpha, use_running_absmax=True))
                for layer in model.modules():
                    if isinstance(layer, torch.nn.Linear):
                        layer.weight.act_quant_scale = layer.weight.act_quant_scale.reshape(*([1]*x.ndim))
                        if layer.weight.act_quant_zero_point is not None:
                            layer.weight.act_quant_zero_point = layer.weight.act_quant_zero_point.reshape(*([1]*x.ndim))
                if args.compile:
                    model=torch.compile(model,fullgraph=True,dynamic=True)
                out = model(x)
            output[str(alpha)] = {"ok": True, "finite": bool(torch.isfinite(out).all()),
                "relative_l2": ((ref-out).norm()/ref.norm()).item()}
        except Exception as exc:
            import traceback
            output[str(alpha)] = {"ok": False, "error_type": type(exc).__name__, "message": str(exc)[:300],
                "trace": [{"function":f.name,"line":f.line} for f in traceback.extract_tb(exc.__traceback__)[-4:]]}
    path = Path(__file__).resolve().parent / "results" / "static"
    path.mkdir(parents=True, exist_ok=True)
    suffix='_compiled' if args.compile else ''
    (path/f"probe_{args.device}{suffix}.json").write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
