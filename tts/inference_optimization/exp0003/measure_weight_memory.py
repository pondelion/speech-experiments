"""Cの選定条件を独立プロセスで測る。size計測用bufferは作らない。"""
import argparse
import gc
import json
from pathlib import Path
import resource
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "exp0001"))
from benchmark import build_models, load_inputs, one_run, environment
from quantize import weight_only


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--bits", type=int, choices=[0, 4, 8], default=0)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--packing", default="tile_packed_to_4d")
    parser.add_argument("--calibration", choices=["awq", "gptq"])
    args = parser.parse_args()
    condition = f"{args.device}_both_wo{args.bits}_{args.dtype}" if args.bits else f"{args.device}_" + ("bf16" if args.dtype=="bfloat16" else "fp32")
    if args.bits == 4 and args.group_size != 128:
        condition += f"_g{args.group_size}"
    if args.bits == 4 and args.packing == "plain":
        condition += "_plain"
    if args.calibration:
        condition += f"_{args.calibration}"
    torch, pipe, t2s, vocab, seed = build_models(args.device)
    torch.set_num_threads(8)
    if args.dtype == "bfloat16":
        t2s = t2s.to(dtype=torch.bfloat16)
        pipe.s2m = pipe.s2m.to(dtype=torch.bfloat16)
    if args.calibration:
        cache = ROOT / "runs" / "calibration" / f"{condition}.pt"
        if not cache.exists():
            raise RuntimeError("メモリ測定前にcalibration checkpointを作成してください")
        from calibrate import calibrated_models
        calibrated_models(torch, pipe, t2s, vocab, seed, args.calibration, args.bits,
                          args.group_size, ROOT, condition)
    elif args.bits:
        weight_only(torch, t2s, args.bits, exclude={"head"}, group_size=args.group_size, packing=args.packing)
        weight_only(torch, pipe.s2m, args.bits, group_size=args.group_size, packing=args.packing)
    gc.collect()
    result = {"condition": condition, "environment": environment(torch, args),
              "load_strategy": "source_checkpoint_then_cached_quantized" if args.calibration else
                  ("source_checkpoint_then_runtime_quantization" if args.bits else "source_checkpoint")}
    if args.device == "cuda":
        torch.cuda.synchronize()
        result["load_convert_peak_vram_bytes"] = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    one_run(torch, pipe, t2s, vocab, seed, load_inputs("long")[0]["text"], args.device, 0, args.dtype)
    gc.collect()
    result["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if args.device == "cuda":
        result["long_inference_peak_vram_bytes"] = torch.cuda.max_memory_allocated()
        result["long_inference_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    else:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                result["rss_after_long_inference_bytes"] = int(line.split()[1]) * 1024
    output = ROOT / "results" / "weight_only" / f"memory_{condition}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"memory {condition} done", flush=True)


if __name__ == "__main__":
    main()
