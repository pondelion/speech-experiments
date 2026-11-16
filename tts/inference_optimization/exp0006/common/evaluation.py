"""部品別・TTS全体の共通速度測定と既存品質結果の出典。"""

import json
import statistics
import warnings

import torch

from .paths import ROOT, benchmark_module, relative


def dynamic_int8(model, exclude=()):
    """既存最速CPUと同じLinearの動的INT8。"""
    names = {
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name not in exclude
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return torch.ao.quantization.quantize_dynamic(
            model, names, dtype=torch.qint8, inplace=False
        )


def measure(teacher, vocab, device, repeats, pipe=None):
    bench = benchmark_module()
    if pipe is None:
        _, pipe, _, _, seed = bench.build_models(device)
    else:
        seed = bench.import_models()[-1]
    model = teacher
    if device == "cpu":
        model = dynamic_int8(teacher, exclude={"head"})
        pipe.s2m = dynamic_int8(pipe.s2m)
    results = {}
    for item in bench.load_inputs("all"):
        bench.one_run(
            torch,
            pipe,
            model,
            vocab,
            seed,
            item["text"],
            device,
            0,
            "float32",
        )
        runs = [
            bench.one_run(
                torch,
                pipe,
                model,
                vocab,
                seed,
                item["text"],
                device,
                i + 1,
                "float32",
            )
            for i in range(repeats)
        ]
        assert all(
            shape["wave_all_finite"] and shape["mel_all_finite"]
            for _, shape in runs
        )
        results[item["id"]] = {
            "median_s": {
                key: statistics.median(stages[key] for stages, _ in runs)
                for key in runs[0][0]
            },
            "runs": [
                {"stages_s": stages, "shape": shape} for stages, shape in runs
            ],
        }
        print(f"BENCH {device}/{item['id']} done", flush=True)
    return {
        "runtime": "pytorch",
        "dtype": "float32",
        "linear_quantization": "dynamic_int8" if device == "cpu" else None,
        "t2s_excluded_quantization_layers": ["head"]
        if device == "cpu"
        else [],
        "batch_size": 1,
        "threads": torch.get_num_threads(),
        "warmup": 1,
        "repeats": repeats,
        "cases": results,
    }


def reference_quality(device):
    """同一重み・生成条件の既存CER。新規測定とは区別する。"""
    path = ROOT.parent / "exp0004" / "results" / "summary.json"
    rows = json.loads(path.read_text())
    config = "torch_int8" if device == "cpu" else "torch"
    selected = [
        r for r in rows if r["device"] == device and r["config"] == config
    ]
    if not selected:
        raise ValueError("Missing reference quality")
    row = selected[-1]
    return {
        "reused": True,
        "source": relative(path),
        "original_source": row["quality_source"],
        "condition": row["condition"],
        "test_cer_pct": row["test_cer_pct"],
        "practical_cer_pct": row["eval_texts_cer_pct"],
        "note": "Teacher unchanged; not a new CER measurement",
    }


def evaluate_condition(args):
    """学習済みstudentのCPU/GPU速度・155文CER・代表MP3を保存。"""
    import gc
    import resource

    import soundfile as soundfile

    from .models import reload_student
    from .paths import RESULTS, RUNS, digest, write_json

    condition = f"t2s_l{args.layers}_{args.mode}_s{args.seed}"
    target = RESULTS / "evaluation" / f"{args.device}_{condition}.json"
    if target.exists():
        raise FileExistsError(f"Completed evaluation exists: {condition}")
    checkpoint = RUNS / condition / "t2s.pt"
    model = reload_student(checkpoint, args.device)
    bench = benchmark_module()
    _, pipe, original, vocab, seed = bench.build_models(args.device)
    del original
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    torch.set_num_threads(8)
    benchmark = measure(model, vocab, args.device, args.repeats, pipe=pipe)
    benchmark_memory = {
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated() / 2**20
            if args.device == "cuda"
            else None
        ),
        "rss_high_water_mib": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        ),
        "scope": "model load and batch-1 benchmark; before CER and ASR",
    }
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    if args.device == "cpu":
        model = dynamic_int8(model, exclude={"head"})
        pipe.s2m = dynamic_int8(pipe.s2m)
    from lib_common.asr_eval import ASR, cer_scores, utterance_cer
    from lib_common.audio import SR
    from lib_common.data import list_jsut, split
    from lib_common.eval_sets import load_eval_texts

    label = f"{args.device}_{condition}"
    previews = []
    for item in bench.load_inputs("all"):
        path = RESULTS / "samples" / label / f"{item['id']}.mp3"
        _, shape = bench.one_run(
            torch,
            pipe,
            model,
            vocab,
            seed,
            item["text"],
            args.device,
            0,
            "float32",
            preview_path=path,
        )
        previews.append(
            {
                "condition": label,
                "id": item["id"],
                "text": item["text"],
                "seed": 0,
                "path": relative(path),
                "bytes": path.stat().st_size,
                "audio_s": shape["audio_s"],
            }
        )
    groups = {
        "test": split(list_jsut(), "test"),
        "practical": load_eval_texts(["practical"]),
    }
    audio, references = {}, {}
    for group, items in groups.items():
        seed(0)
        codes = []
        with torch.inference_mode():
            for start in range(0, len(items), 16):
                ids = [
                    torch.tensor(vocab.encode(u.text))
                    for u in items[start : start + 16]
                ]
                codes.extend(
                    model.generate(ids, durations=model.predict_durations(ids))
                )
            seed(0)
            mels = []
            for start in range(0, len(codes), 4):
                mels.extend(
                    pipe.codes_to_mel(
                        codes[start : start + 4], n_steps=25, cfg=0.7, batch=4
                    )
                )
            paths = []
            for item, mel in zip(items, mels):
                wave = pipe.mel_to_wave(mel.float())
                if not (
                    torch.isfinite(mel).all() and torch.isfinite(wave).all()
                ):
                    raise FloatingPointError(f"Nonfinite audio: {item.id}")
                path = RUNS / "quality" / label / group / f"{item.id}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                soundfile.write(path, wave.numpy(), SR)
                paths.append(str(path))
        audio[group] = paths
        references[group] = items
        print(f"SYNTHESIZED {label}/{group} {len(paths)}", flush=True)
    peak_cuda = (
        torch.cuda.max_memory_allocated() / 2**20
        if args.device == "cuda"
        else None
    )
    rss_before_asr = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    del model, pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    asr = ASR("cuda")
    quality = {}
    for group, paths in audio.items():
        hypotheses = asr.transcribe(paths, batch_size=16)
        items = references[group]
        scores = cer_scores([u.text for u in items], hypotheses)
        quality[group] = {
            "utterances": len(items),
            "scores": scores,
            "reading_cer_pct": 100 * scores["reading"],
            "items": [
                {
                    "id": item.id,
                    "reference": item.text,
                    "hypothesis": hyp,
                    "reading_cer": utterance_cer(item.text, hyp),
                }
                for item, hyp in zip(items, hypotheses)
            ],
        }
    result = {
        "complete": True,
        "condition": condition,
        "device": args.device,
        "checkpoint": relative(checkpoint),
        "checkpoint_sha256": digest(checkpoint),
        "benchmark": benchmark,
        "benchmark_memory": benchmark_memory,
        "quality": quality,
        "previews": previews,
        "peak_cuda_allocated_mib_before_asr": peak_cuda,
        "rss_high_water_mib_before_asr": rss_before_asr,
        "rss_high_water_mib_with_asr": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        ),
        "quality_seed": 0,
        "t2s_batch": 16,
        "s2m_batch": 4,
    }
    write_json(target, result)
    print(f"EVALUATION COMPLETE {label}", flush=True)
