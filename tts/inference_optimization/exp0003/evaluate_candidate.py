"""FP32/両方INT8を追加seedで評価する。既存MP3は増やさない。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "exp0001"))
from benchmark import build_models
from quantize import dynamic_int8


def main():
    import soundfile as sf
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    torch, pipe, t2s, vocab, seed = build_models("cpu")
    torch.set_num_threads(8)
    from lib_common.data import list_jsut, split
    from lib_common.eval_sets import load_eval_texts
    from lib_common.asr_eval import ASR, cer_scores
    from lib_common.audio import SR
    groups = {"test": split(list_jsut(), "test"), "eval_texts": load_eval_texts(["practical"])}
    result = {"seed": args.seed, "threads": 8, "t2s_batch": 16, "s2m_batch": 4, "conditions": {}}
    audio = {}
    for condition in ["cpu_fp32", "cpu_both_int8"]:
        if condition == "cpu_both_int8":
            t2s = dynamic_int8(torch, t2s, exclude={"head"})
            pipe.s2m = dynamic_int8(torch, pipe.s2m)
        audio[condition] = {}
        result["conditions"][condition] = {}
        for group, items in groups.items():
            seed(args.seed)
            codes = []
            with torch.inference_mode():
                for start in range(0, len(items), 16):
                    ids = [torch.tensor(vocab.encode(u.text)) for u in items[start:start + 16]]
                    codes += t2s.generate(ids, durations=t2s.predict_durations(ids))
                    print(f"T2S {condition}/{group} {min(start + 16, len(items))}/{len(items)}", flush=True)
                seed(args.seed)
                mels = []
                for start in range(0, len(codes), 4):
                    mels += pipe.codes_to_mel(codes[start:start + 4], n_steps=25, cfg=0.7, batch=4)
                    print(f"S2M {condition}/{group} {min(start + 4, len(codes))}/{len(codes)}", flush=True)
            paths = []
            finite = True
            for item, mel in zip(items, mels):
                wave = pipe.mel_to_wave(mel.float())
                finite = finite and bool(torch.isfinite(mel).all() and torch.isfinite(wave).all())
                path = ROOT / "runs" / f"seed{args.seed}" / condition / group / f"{item.id}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, wave.numpy(), SR)
                paths.append(str(path))
            audio[condition][group] = paths
            result["conditions"][condition][group] = {"utterances": len(items), "finite": finite}
    asr = ASR("cuda")
    for condition, gr in audio.items():
        for group, paths in gr.items():
            scores = cer_scores([u.text for u in groups[group]], asr.transcribe(paths, batch_size=16))
            result["conditions"][condition][group]["cer"] = scores
            print(f"CER {condition}/{group}: {scores['reading']:.6f}", flush=True)
    output = ROOT / "results" / "validation" / f"quality_seed{args.seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
