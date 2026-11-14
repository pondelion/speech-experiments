"""serialize用bufferを作らず、独立processでCPU RAMを測る。"""
import argparse
import gc
import json
from pathlib import Path
import resource
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "exp0001"))
from benchmark import build_models, load_inputs, one_run
from quantize import dynamic_int8


def rss():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()
    torch, pipe, t2s, vocab, seed = build_models("cpu")
    torch.set_num_threads(8)
    if args.quantize:
        t2s = dynamic_int8(torch, t2s, exclude={"head"})
        pipe.s2m = dynamic_int8(torch, pipe.s2m)
    gc.collect()
    result = {"condition": "cpu_both_int8" if args.quantize else "cpu_fp32", "threads": 8,
              "rss_after_load_convert_bytes": rss()}
    item = load_inputs("long")[0]
    one_run(torch, pipe, t2s, vocab, seed, item["text"], "cpu", 0, "float32")
    gc.collect()
    result.update(rss_after_long_inference_bytes=rss(),
                  process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    path = ROOT / "results" / "validation" / f"memory_{result['condition']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
