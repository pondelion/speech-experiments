"""CPU/GPUの遅延を中文1文で切り分ける。正式比較・CERとは別の探索。"""

import argparse
import fcntl
import json
import resource
import statistics
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

from runtime import prepare

ROOT = Path(__file__).resolve().parent


def main():
    sys.path.insert(0, str(ROOT.parent / "exp0001"))
    from benchmark import build_models, environment, load_inputs, one_run

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--kernel-profile", action="store_true")
    args = parser.parse_args()
    out = ROOT / "results" / "screen"
    out.mkdir(parents=True, exist_ok=True)
    suffix = "_profile" if args.profile else ""
    if args.kernel_profile:
        suffix += "_kernels"
    target = out / f"{args.device}_{args.config}{suffix}.json"
    if target.exists():
        print("SKIP", target.name, flush=True)
        return
    lock_dir = ROOT / "runs" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock = (lock_dir / "phase.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    torch, pipe, t2s, vocab, seed = build_models(args.device)
    torch.set_num_threads(8)
    metadata, t2s = prepare(
        torch, pipe, t2s, args.device, args.config, profile=args.profile
    )
    dtype = "bfloat16" if args.config == "torch_bf16" else "float32"
    text = load_inputs("medium")[0]["text"]
    one_run(torch, pipe, t2s, vocab, seed, text, args.device, 0, dtype)
    rows = []
    before = resource.getrusage(resource.RUSAGE_SELF)
    profiler = (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        )
        if args.kernel_profile
        else nullcontext()
    )
    with profiler as prof:
        for index in range(args.repeats):
            stages, shape = one_run(
                torch,
                pipe,
                t2s,
                vocab,
                seed,
                text,
                args.device,
                index + 1,
                dtype,
            )
            rows.append(stages)
            assert shape["mel_all_finite"] and shape["wave_all_finite"]
    after = resource.getrusage(resource.RUSAGE_SELF)
    cpu_s = (after.ru_utime + after.ru_stime) - (
        before.ru_utime + before.ru_stime
    )
    result = {
        "device": args.device,
        "config": args.config,
        "runtime": metadata,
        "repeats": args.repeats,
        "environment": environment(torch, args),
        "process_cpu_s": cpu_s,
        "mean_active_cpu_cores": cpu_s / sum(r["end_to_end_s"] for r in rows),
        "stages_median_s": {
            k: statistics.median(r[k] for r in rows) for k in rows[0]
        },
        "runs": rows,
    }
    if args.kernel_profile:
        kernels = defaultdict(float)
        counts = defaultdict(int)
        launch_us = 0.0
        for event in prof.events():
            if event.device_type == torch.autograd.DeviceType.CUDA:
                kernels[event.name] += event.self_device_time_total
                counts[event.name] += 1
            elif event.name.startswith("cudaLaunch"):
                launch_us += event.self_cpu_time_total
        result["kernel_profile"] = {
            "cuda_event_ms": sum(kernels.values()) / 1000,
            "cuda_launch_api_ms": launch_us / 1000,
            "event_count": sum(counts.values()),
            "top_events_ms": dict(
                sorted(
                    ((k, v / 1000) for k, v in kernels.items()),
                    key=lambda x: -x[1],
                )[:15]
            ),
            "top_event_counts": dict(
                sorted(counts.items(), key=lambda x: -x[1])[:15]
            ),
        }
    if args.profile:
        result["profiling_includes_warmup"] = True
        result["profiles"] = {}
        for name, session in t2s._ort_sessions.items():
            events = json.loads(Path(session.end_profiling()).read_text())
            ops = defaultdict(float)
            nodes = defaultdict(float)
            for event in events:
                if event.get("name", "").endswith("_kernel_time"):
                    args_event = event.get("args", {})
                    key = (
                        args_event.get("op_name", "unknown")
                        + ":"
                        + args_event.get("provider", "unknown")
                    )
                    ops[key] += event.get("dur", 0) / 1000
                    nodes[event["name"]] += event.get("dur", 0) / 1000
            result["profiles"][str(name)] = {
                "operator_ms": dict(sorted(ops.items(), key=lambda x: -x[1])),
                "top_nodes_ms": dict(
                    sorted(nodes.items(), key=lambda x: -x[1])[:20]
                ),
            }
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(args.config, result["stages_median_s"], flush=True)


if __name__ == "__main__":
    main()
