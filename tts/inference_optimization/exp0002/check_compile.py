"""固定テンソルによるeager/compileの数値比較。音声ファイルを使わない。"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "exp0001"))
from benchmark import CHECKPOINTS, import_models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    args = parser.parse_args()
    torch, _, load_s2m, _, _, _ = import_models()
    torch.set_num_threads(8)
    torch.manual_seed(0)
    model = load_s2m(CHECKPOINTS["s2m"], args.device)
    eager = model.velocity
    compiled = torch.compile(eager, fullgraph=True, dynamic=False)
    # 中程度の長さ、CFG相当のbatch=2、末尾8 frameをpaddingにする。
    frames = 508
    x = torch.randn(2, frames, 80, device=args.device)
    cond = torch.randn(2, frames, model.dim, device=args.device)
    cond[1].zero_()
    pad = torch.zeros(2, frames, dtype=torch.bool, device=args.device)
    pad[:, -8:] = True

    def compare(a, b):
        diff = (a - b).float()
        return {
            "max_abs": diff.abs().max().item(),
            "rmse": diff.square().mean().sqrt().item(),
            "relative_l2": (diff.norm() / a.float().norm().clamp_min(1e-12)).item(),
            "allclose": torch.allclose(a, b, atol=1e-4, rtol=1e-4),
            "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
        }

    result = {"device": args.device, "torch": torch.__version__, "seed": 0,
              "shape": list(x.shape), "atol": 1e-4, "rtol": 1e-4, "single_step": {}}
    with torch.inference_mode():
        for value in [0.0, 0.5, 0.96]:
            t = torch.full((2,), value, device=args.device)
            result["single_step"][str(value)] = compare(eager(x, cond, t, pad), compiled(x, cond, t, pad))
        a, b = x.clone(), x.clone()
        for step in range(25):
            t = torch.full((2,), step / 25, device=args.device)
            a = a + eager(a, cond, t, pad) / 25
            b = b + compiled(b, cond, t, pad) / 25
        result["synthetic_25_step_rollout"] = compare(a, b)
    output = Path(__file__).resolve().parent / "results" / f"compile_numerics_{args.device}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if not all(row["finite"] and row["allclose"] for row in
               [*result["single_step"].values(), result["synthetic_25_step_rollout"]]):
        raise SystemExit("numerical check failed")


if __name__ == "__main__":
    main()
