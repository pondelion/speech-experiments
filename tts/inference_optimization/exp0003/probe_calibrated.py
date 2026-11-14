"""インストール済みprototype APIの動作確認。"""
import json
from pathlib import Path


def main():
    import torch
    from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig, quantize_
    from torchao.prototype.awq import AWQConfig
    from torchao.prototype.gptq import GPTQConfig
    result = {}
    for method, bits in [("awq", 4), ("gptq", 8), ("gptq", 4)]:
        try:
            dtype = torch.bfloat16 if bits == 4 else torch.float32
            torch.manual_seed(0)
            model = torch.nn.Sequential(torch.nn.Linear(512, 512)).cuda().to(dtype).eval()
            x = torch.randn(1, 64, 512, device="cuda", dtype=dtype)
            base = Int8WeightOnlyConfig(set_inductor_config=False) if bits == 8 else Int4WeightOnlyConfig(set_inductor_config=False)
            if bits == 4:
                base.int4_packing_format = type(base.int4_packing_format).TILE_PACKED_TO_4D
            with torch.inference_mode():
                ref = model(x).float()
                prepare = AWQConfig(base, step="prepare") if method == "awq" else GPTQConfig(step="observe", base_config=base)
                quantize_(model, prepare)
                model(x)
                convert = AWQConfig(base, step="convert") if method == "awq" else GPTQConfig(step="convert", base_config=base)
                quantize_(model, convert)
                repack_check = None
                if method == "gptq" and bits == 4:
                    from repack import repack_plain_int4
                    # packing変換はinference tensorを作らないcontextで行う。
                    with torch.inference_mode(False), torch.no_grad():
                        w, repack_check = repack_plain_int4(torch, model[0].weight)
                        model[0].weight = torch.nn.Parameter(w, requires_grad=False)
                out = model(x).float()
            result[f"{method}{bits}"] = {"ok": True, "relative_l2": ((ref-out).norm()/ref.norm()).item(),
                                       "weight_type": type(model[0].weight).__name__, "repack_check": repack_check}
        except Exception as exc:
            result[f"{method}{bits}"] = {"ok": False, "error_type": type(exc).__name__, "message": str(exc)[:300]}
    path = Path(__file__).resolve().parent / "results" / "weight_only" / "probe_calibrated.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
