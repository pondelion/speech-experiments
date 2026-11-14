"""FP32/FP16/BF16の読みCERを、既存expと同じ評価文・ASRで測る。"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json

import soundfile as sf

from benchmark import EXP_DIR, build_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="動作確認用。各groupの先頭N文だけ")
    return parser.parse_args()


def amp_context(torch, dtype: str, device: str):
    value = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(dtype)
    return torch.autocast(device_type="cuda", dtype=value) if value and device.startswith("cuda") else nullcontext()


def synthesize(torch, pipe, t2s, vocab, texts: list[str], dtype: str, device: str, batch: int):
    codes = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch):
            chunk = texts[start:start + batch]
            ids = [torch.tensor(vocab.encode(text)) for text in chunk]
            with amp_context(torch, dtype, device):
                durations = t2s.predict_durations(ids)
                codes += t2s.generate(ids, durations=durations)
        with amp_context(torch, dtype, device):
            mels = pipe.codes_to_mel(codes, n_steps=25, cfg=0.7, batch=batch)
        # Griffin-Limは複素数BF16が未対応なので、全条件FP32に固定する。
        waves = [pipe.mel_to_wave(mel.float()) for mel in mels]
    return codes, mels, waves


def main() -> None:
    args = parse_args()
    torch, pipe, t2s, vocab, seed_everything = build_models(args.device)
    # build_models()が既存expのimport pathを設定した後に共通評価をimportする。
    from lib_common.asr_eval import ASR, cer_scores
    from lib_common.audio import SR
    from lib_common.data import list_jsut, split
    from lib_common.eval_sets import load_eval_texts

    groups = {
        "test": split(list_jsut(), "test"),
        "eval_texts": load_eval_texts(["practical"]),
    }
    if args.limit:
        groups = {name: items[:args.limit] for name, items in groups.items()}

    out_dir = EXP_DIR / "runs" / "precision" / args.dtype
    audio_paths = {}
    metadata = {}
    seed_everything(args.seed)
    for group, items in groups.items():
        codes, mels, waves = synthesize(
            torch, pipe, t2s, vocab, [item.text for item in items], args.dtype, args.device, args.batch
        )
        group_dir = out_dir / group
        group_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for item, wave in zip(items, waves):
            path = group_dir / f"{item.id}.wav"
            sf.write(path, wave.numpy(), SR)
            paths.append(str(path))
        audio_paths[group] = paths
        metadata[group] = {
            "utterances": len(items),
            "audio_tokens": sum(len(code) for code in codes),
            "mel_all_finite": all(bool(torch.isfinite(mel).all()) for mel in mels),
            "wave_all_finite": all(bool(torch.isfinite(wave).all()) for wave in waves),
        }
        print(f"synthesized {group}: {len(items)} utterances")

    asr = ASR(args.device)
    result = {}
    for group, items in groups.items():
        hypotheses = asr.transcribe(audio_paths[group], batch_size=args.batch)
        result[group] = {
            **cer_scores([item.text for item in items], hypotheses),
            **metadata[group],
        }
        print(f"{group}: reading CER {result[group]['reading']:.4f}")

    if args.limit:
        print("--limitのためresultsには保存しません")
        return
    result_path = EXP_DIR / "results" / "precision_quality.json"
    results = json.loads(result_path.read_text()) if result_path.exists() else {}
    results[args.dtype] = result
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"saved {result_path.relative_to(EXP_DIR)}")


if __name__ == "__main__":
    main()
