"""CPU動的INT8の初期比較と、小容量MP3の生成。"""
import argparse
import io
import json
from pathlib import Path
import sys
import statistics
import time
import warnings

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "exp0001"))
from benchmark import build_models, environment, load_inputs, one_run, summarize
from quantize import dynamic_int8, weight_only


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--target", choices=["none", "t2s", "s2m", "both"], default="none")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--no-samples", action="store_true")
    parser.add_argument("--wo-bits", type=int, choices=[4, 8])
    parser.add_argument("--group-size", type=int, choices=[32, 64, 128, 256], default=128)
    parser.add_argument("--calibration", choices=["awq", "gptq"])
    parser.add_argument("--packing", choices=["tile_packed_to_4d", "plain"], default="tile_packed_to_4d")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeatsは1以上")
    target = "s2m" if args.quantize else args.target
    if target != "none" and args.device != "cpu" and not args.wo_bits:
        parser.error("この動的INT8 backendはCPU専用")
    if args.wo_bits and target == "none":
        parser.error("wo-bitsにはtargetが必要")
    if args.dtype == "bfloat16" and args.device != "cuda":
        parser.error("このrunnerのBF16条件はCUDAのみ")
    condition = f"{args.device}_" + (f"{target}_int8" if target != "none" else "fp32")
    if args.wo_bits:
        condition = f"{args.device}_{target}_wo{args.wo_bits}_{args.dtype}"
        if args.wo_bits == 4 and args.group_size != 128:
            condition += f"_g{args.group_size}"
        if args.wo_bits == 4 and args.packing == "plain":
            condition += "_plain"
        if args.calibration:
            condition += f"_{args.calibration}"
    elif args.dtype == "bfloat16":
        condition = f"{args.device}_bf16"
    torch, pipe, t2s, vocab, seed_everything = build_models(args.device)
    torch.set_num_threads(8)
    original_bytes = io.BytesIO()
    torch.save(pipe.s2m.state_dict(), original_bytes)
    t2s_original = io.BytesIO()
    torch.save(t2s.state_dict(), t2s_original)
    if args.dtype == "bfloat16":
        t2s = t2s.to(dtype=torch.bfloat16)
        pipe.s2m = pipe.s2m.to(dtype=torch.bfloat16)
    start = time.perf_counter()
    wo_metadata = {}
    if args.calibration:
        if target != "both" or not args.wo_bits:
            parser.error("calibrationはtarget bothとwo-bitsを指定")
        from calibrate import calibrated_models
        wo_metadata["calibration"] = calibrated_models(torch, pipe, t2s, vocab, seed_everything,
            args.calibration, args.wo_bits, args.group_size, ROOT, condition)
    elif target in ["s2m", "both"]:
        if args.wo_bits:
            wo_metadata["s2m"] = weight_only(torch, pipe.s2m, args.wo_bits, group_size=args.group_size, packing=args.packing)
        else:
            pipe.s2m = dynamic_int8(torch, pipe.s2m)
    if target in ["t2s", "both"] and not args.calibration:
        # device propertyがhead.weightを直接参照するため、headはFP32に残す。
        if args.wo_bits:
            wo_metadata["t2s"] = weight_only(torch, t2s, args.wo_bits, exclude={"head"}, group_size=args.group_size, packing=args.packing)
        else:
            t2s = dynamic_int8(torch, t2s, exclude={"head"})
    conversion_s = time.perf_counter() - start
    packed = io.BytesIO()
    torch.save(pipe.s2m.state_dict(), packed)
    t2s_packed = io.BytesIO()
    torch.save(t2s.state_dict(), t2s_packed)
    rows, samples = {}, []
    for item in load_inputs("all"):
        one_run(torch, pipe, t2s, vocab, seed_everything, item["text"], args.device, 0, args.dtype)
        times, shapes = [], []
        for repeat in range(args.repeats):
            path = ROOT / "results" / "samples" / "eval_texts" / condition / f"{item['id']}.mp3"
            stages, shape = one_run(torch, pipe, t2s, vocab, seed_everything, item["text"], args.device,
                                    1 + repeat, args.dtype, path if repeat == args.repeats - 1 and not args.no_samples else None)
            times.append(stages)
            shapes.append(shape)
        rows[item["id"]] = {
            "stages": {key: summarize([r[key] for r in times]) for key in times[0]},
            "rtf_median": statistics.median(s["rtf"] for s in shapes),
            "finite": all(s["mel_all_finite"] and s["wave_all_finite"] for s in shapes),
            "sample_seed": args.repeats if not args.no_samples else None,
        }
        samples.append({"group": "eval_texts", "condition": condition, "id": item["id"],
                        "text": item["text"], "category": item["id"],
                        "file": str(path.relative_to(ROOT / "results" / "samples"))})
        print(f"finished {condition}/{item['id']}", flush=True)
    quantized_names = [name for name, module in pipe.s2m.named_modules()
                       if isinstance(module, torch.ao.nn.quantized.dynamic.Linear)]
    result = {"condition": condition, "threads": 8, "warmup": 1, "repeats": args.repeats,
              "dtype": args.dtype, "weight_only": wo_metadata,
              "quantized_linear_names": quantized_names,
              "t2s_quantized_linear_names": [name for name, m in t2s.named_modules()
                  if isinstance(m, torch.ao.nn.quantized.dynamic.Linear)],
              "excluded_t2s_layers": ["head"] if target in ["t2s", "both"] else [],
              "t2s_state_bytes_fp32": t2s_original.tell(), "t2s_state_bytes": t2s_packed.tell(),
              "s2m_state_bytes_fp32": original_bytes.tell(), "s2m_state_bytes": packed.tell(),
              "conversion_s": conversion_s, "cases": rows, "environment": environment(torch, args)}
    results = args.output_dir
    results.mkdir(parents=True, exist_ok=True)
    (results / f"{condition}.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.no_samples:
        return
    index_path = ROOT / "results" / "samples" / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else []
    index = [r for r in index if r["condition"] != condition] + samples
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
