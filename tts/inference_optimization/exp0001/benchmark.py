"""exp0001: hubert_ja_k256/ttsの区間別ベンチマーク。

--check-onlyはcheckpointと入力だけを検査し、torchのimportもモデルのloadも行わない。
通常実行はGPUが空いているときに行う。
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parent
TTS_DIR = EXP_DIR.parents[1]
REPO_DIR = EXP_DIR.parents[2]
BASE_EXP = TTS_DIR / "indextts_bases" / "exp0003"
CHECKPOINTS = {
    "codec": BASE_EXP / "runs" / "codec_hubert_ja_k256" / "codec.pt",
    "s2m": BASE_EXP / "runs" / "s2m_hubert_ja_k256" / "s2m.pt",
    "t2s": BASE_EXP / "runs" / "t2s_hubert_ja_k256" / "t2s.pt",
}
SR = 24_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="all", help="inputs.jsonのid、またはall")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=0, help="CPU thread数。0はPyTorchの既定値")
    parser.add_argument("--compile-s2m", action="store_true", help="S2Mのテンソル演算部をtorch.compileする")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32",
                        help="float16/bfloat16はCUDA autocastで測る")
    parser.add_argument("--check-only", action="store_true", help="モデルをloadせず、入力とpathだけ確認する")
    parser.add_argument("--output", type=Path, default=EXP_DIR / "results" / "metrics.json")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_inputs(case: str) -> list[dict]:
    items = json.loads((EXP_DIR / "inputs.json").read_text())
    selected = items if case == "all" else [x for x in items if x["id"] == case]
    if not selected:
        raise SystemExit(f"unknown case: {case}")
    if any(not x.get("text") for x in selected):
        raise SystemExit("inputs.jsonに空のtextがあります")
    return selected


def check_files() -> dict:
    missing = [str(path) for path in CHECKPOINTS.values() if not path.is_file()]
    if missing:
        raise SystemExit("checkpointがありません:\n" + "\n".join(missing))
    return {name: {"path": str(path.relative_to(REPO_DIR)), "bytes": path.stat().st_size}
            for name, path in CHECKPOINTS.items()}


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] if lo == hi else ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def summarize(values: list[float]) -> dict:
    return {
        "median_s": statistics.median(values),
        "p90_s": percentile(values, 0.9),
        "min_s": min(values),
        "max_s": max(values),
        "runs_s": values,
    }


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_DIR, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sync(torch, device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def timed(torch, device: str, fn):
    sync(torch, device)
    start = time.perf_counter()
    value = fn()
    sync(torch, device)
    return value, time.perf_counter() - start


def import_models():
    # 既存expと同じimport名(common、exp0001、exp0002)を解決する。
    sys.path[:0] = [str(TTS_DIR / "indextts_bases" / "exp0003"), str(TTS_DIR), str(TTS_DIR / "indextts_bases")]
    import torch
    from exp0001.common.pipeline import TTSPipeline, load_s2m
    from exp0002.common.utils import load_t2s
    from common.utils import load_ssl_codec
    from lib_common.utils import seed_everything
    return torch, TTSPipeline, load_s2m, load_t2s, load_ssl_codec, seed_everything


def build_models(device: str):
    torch, TTSPipeline, load_s2m, load_t2s, load_ssl_codec, seed_everything = import_models()
    seed_everything(0)
    codec, normalizer, _, _ = load_ssl_codec(CHECKPOINTS["codec"], device)
    s2m = load_s2m(CHECKPOINTS["s2m"], device)
    t2s, vocab, method = load_t2s(CHECKPOINTS["t2s"], device)
    if method != "hybrid":
        raise RuntimeError(f"expected hybrid T2S, got {method}")
    pipe = TTSPipeline(codec, normalizer, s2m, device=device)
    return torch, pipe, t2s, vocab, seed_everything


def one_run(torch, pipe, t2s, vocab, seed_everything, text: str, device: str, seed: int,
            dtype: str, preview_path=None) -> tuple[dict, dict]:
    seed_everything(seed)
    stages = {}
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(dtype)
    amp = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype and device.startswith("cuda") else nullcontext()
    with torch.inference_mode():
        text_ids, stages["text_preprocess_s"] = timed(
            torch, device, lambda: [torch.tensor(vocab.encode(text))]
        )
        with amp:
            durations, stages["duration_s"] = timed(
                torch, device, lambda: t2s.predict_durations(text_ids)
            )
            codes, stages["t2s_tokens_s"] = timed(
                torch, device, lambda: t2s.generate(text_ids, durations=durations)
            )
            mels, stages["s2m_s"] = timed(
                torch, device, lambda: pipe.codes_to_mel(codes, n_steps=25, cfg=0.7)
            )
        # Griffin-Limの複素数演算はBF16未対応。モデル精度とは分け、ボコーダは常にFP32で測る。
        mels = [mel.float() for mel in mels]
        waves, stages["vocoder_s"] = timed(
            torch, device, lambda: [pipe.mel_to_wave(mel) for mel in mels]
        )
    stages["end_to_end_s"] = sum(stages.values())
    audio_s = len(waves[0]) / SR
    shape = {
        "text_chars": len(text),
        "text_tokens": len(text_ids[0]),
        "audio_tokens": len(codes[0]),
        "mel_frames": len(mels[0]),
        "audio_s": audio_s,
        "rtf": stages["end_to_end_s"] / audio_s,
        "x_realtime": audio_s / stages["end_to_end_s"],
        "codes_sha256": hashlib.sha256(codes[0].numpy().tobytes()).hexdigest(),
        "mel_all_finite": bool(torch.isfinite(mels[0]).all()),
        "wave_all_finite": bool(torch.isfinite(waves[0]).all()),
    }
    if preview_path is not None:
        import soundfile as sf
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(preview_path, waves[0].numpy(), SR, format="MP3",
                 bitrate_mode="CONSTANT", compression_level=0.75)
    return stages, shape


def environment(torch, args: argparse.Namespace) -> dict:
    cuda_device = None
    if args.device.startswith("cuda"):
        cuda_device = {
            "name": torch.cuda.get_device_name(args.device),
            "capability": list(torch.cuda.get_device_capability(args.device)),
            "total_memory_bytes": torch.cuda.get_device_properties(args.device).total_memory,
        }
    return {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": args.device,
        "cuda_device": cuda_device,
        "pid": os.getpid(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.repeats < 1:
        raise SystemExit("warmupは0以上、repeatsは1以上にしてください")
    inputs, checkpoints = load_inputs(args.case), check_files()
    if args.check_only:
        print(json.dumps({"status": "ok", "inputs": inputs, "checkpoints": checkpoints}, indent=2, ensure_ascii=False))
        return
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"output already exists: {args.output} (--overwriteで置換)")

    load_start = time.perf_counter()
    torch, pipe, t2s, vocab, seed_everything = build_models(args.device)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    sync(torch, args.device)
    cold_load_s = time.perf_counter() - load_start
    if args.compile_s2m:
        pipe.s2m.velocity = torch.compile(pipe.s2m.velocity, fullgraph=True, dynamic=False)

    results = {}
    for item in inputs:
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(args.device)
        warmup_start = time.perf_counter()
        for i in range(args.warmup):
            one_run(torch, pipe, t2s, vocab, seed_everything, item["text"], args.device, args.seed + i, args.dtype)
        warmup_elapsed_s = time.perf_counter() - warmup_start
        runs, shapes = [], []
        for i in range(args.repeats):
            stage, shape = one_run(
                torch, pipe, t2s, vocab, seed_everything, item["text"], args.device,
                args.seed + args.warmup + i, args.dtype
            )
            runs.append(stage)
            shapes.append(shape)
        result = {
            "text": item["text"],
            "warmup_elapsed_s": warmup_elapsed_s,
            "stages": {name: summarize([run[name] for run in runs]) for name in runs[0]},
            "shape": {
                "text_chars": shapes[-1]["text_chars"],
                "text_tokens": shapes[-1]["text_tokens"],
                "audio_tokens": shapes[-1]["audio_tokens"],
                "mel_frames": shapes[-1]["mel_frames"],
                "codes_sha256_last_run": shapes[-1]["codes_sha256"],
                "mel_all_finite": all(shape["mel_all_finite"] for shape in shapes),
                "wave_all_finite": all(shape["wave_all_finite"] for shape in shapes),
                "audio_s": summarize([shape["audio_s"] for shape in shapes]),
            },
            "rtf": summarize([shape["rtf"] for shape in shapes]),
            "x_realtime": summarize([shape["x_realtime"] for shape in shapes]),
        }
        if args.device.startswith("cuda"):
            result["peak_vram_bytes"] = torch.cuda.max_memory_allocated(args.device)
        # Linuxのru_maxrssはKiB。モデルを含むprocess全体のpeakで、case間ではresetされない。
        result["peak_process_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        results[item["id"]] = result

    payload = {
        "experiment": "inference_optimization/exp0001",
        "condition": "hubert_ja_k256/tts",
        "settings": {
            "seed": args.seed,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "batch": 1,
            "threads_requested": args.threads,
            "compile_s2m": args.compile_s2m,
            "dtype": args.dtype,
            "t2s_steps": 50,
            "t2s_cfg": 2.5,
            "s2m_steps": 25,
            "s2m_cfg": 0.7,
            "griffin_lim_iterations": 64,
        },
        "checkpoints": checkpoints,
        "cold_load_s": cold_load_s,
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    env_name = "environment_gpu.json" if args.device.startswith("cuda") else "environment_cpu.json"
    env_path = args.output.parent / env_name
    env_path.write_text(json.dumps(environment(torch, args), indent=2, ensure_ascii=False) + "\n")
    print(f"saved {args.output.name}")


if __name__ == "__main__":
    main()
