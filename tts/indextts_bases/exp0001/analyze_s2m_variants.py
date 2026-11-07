"""分析: S2M の学習条件ごとに、同じトークン列を入力したときの読みの正確さ(CER)を比べる。

トークン列は次の 3 種類を 1 回だけ作り、全ての S2M で使い回す(違いが S2M だけになるように):
    gt:        正解の音声から作ったトークン
    tf_sample: T2S の teacher forcing での予測(推論と同じ設定でサンプル)。正解と同じ長さ
    free:      T2S の自由な生成(推論と同じ)
テキストの隠れ状態を条件にする S2M には、読み上げる文章も渡す。

使い方:
    python analyze_s2m_variants.py --t2s t2s_ph_drop03 --s2m s2m s2m_f0 s2m_f1 s2m_f2 s2m_f3
出力: runs/eval/s2m_compare/{S2M の名前}/{トークン列}/(音声)、results/s2m_compare.json
"""
import argparse
import json

import soundfile as sf
import torch

import common  # noqa: F401  tts/lib_common を import できるようにする
from common.pipeline import TTSPipeline, load_s2m, load_t2s
from common.utils import EXP_DIR, RUNS_DIR, encode_all, load_codec
from lib_common.asr_eval import ASR, cer_scores
from lib_common.audio import SR
from lib_common.data import load_dataset, split
from lib_common.utils import Logger, seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t2s", default="t2s_ph_drop03", help="トークン列を作る T2S の runs/ の名前")
    parser.add_argument("--s2m", nargs="+", required=True, help="比べる S2M の runs/ の名前")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    torch.set_num_threads(4)
    log = Logger(RUNS_DIR / "eval" / "log_s2m_compare.txt")
    test = split(load_dataset(None, log), "test")
    texts = [u.text for u in test]
    pipe = TTSPipeline.from_checkpoints(RUNS_DIR / "codec" / "codec.pt", RUNS_DIR / "s2m" / "s2m.pt",
                                        RUNS_DIR / args.t2s / "t2s.pt", device=args.device)
    gt = encode_all(pipe.codec, [pipe.normalizer.normalize(u.mel) for u in test], args.device)  # 各 [U_i]
    ids = [torch.tensor(pipe.vocab.encode(t)) for t in texts]
    tokens = {"gt": gt, "tf_sample": pipe.t2s.predict_teacher_forced(ids, gt, sample=True),
              "free": pipe.text_to_codes(texts)}
    asr = ASR(args.device)
    out = {"t2s": args.t2s}
    for name in args.s2m:
        pipe.s2m = load_s2m(RUNS_DIR / name / "s2m.pt", args.device)
        if pipe.s2m.text_t2s:  # テキストの隠れ状態を作る T2S(S2M の学習時と同じもの)
            pipe.s2m.text_encoder = load_t2s(RUNS_DIR / pipe.s2m.text_t2s / "t2s.pt", args.device)
        out[name] = {}
        for kind, cs in tokens.items():
            d = RUNS_DIR / "eval" / "s2m_compare" / name / kind
            d.mkdir(parents=True, exist_ok=True)
            for u, m in zip(test, pipe.codes_to_mel(cs, texts=texts)):
                sf.write(d / f"{u.id}.wav", pipe.mel_to_wave(m).numpy(), SR)
            out[name][kind] = cer_scores(texts, asr.transcribe([str(d / f"{u.id}.wav") for u in test]))
            log(f"{name:10s} {kind:10s} reading CER {out[name][kind]['reading']:.4f}")
        (EXP_DIR / "results" / "s2m_compare.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
