"""F0 の条件を足した S2M(train_s2m.py)を評価する(common/evaluator.py。乱数の種 0・1・2 の平均)。

条件(--cond で 1 つずつ。1 つのプロセスで続けてモデルを替えると GPU のエラーで落ちることがあるため):
    gt_tokens          正解のトークン → 元の S2M(上限の比較用)
    gt_tokens_gt_f0    正解のトークン + 正解の F0 → F0 の条件つきの S2M(F0 の条件の上限)
    tts                テキスト → T2S(F0 の補助の損失、生成の CFG 0)→ トークン + T2S が予測した F0 → F0 の条件つきの S2M
(T2S のトークン → 元の S2M は、evaluate_t2s.py の `t2s_f0aux@cfg0`)

使い方:
    python evaluate_s2m.py --cond tts --s2m s2m_f0cond
出力: results/evaluate_s2m.json に追記
"""
import argparse
import json

import numpy as np
import soundfile as sf
import torch

import common  # noqa: F401
from lib_common.asr_eval import cer_scores
from lib_common.audio import SR
from lib_common.eval_sets import load_eval_texts
from common.evaluator import Evaluator
from common.s2m_f0 import F0Condition, codes_to_mel_f0
from exp0001.common.pipeline import TTSPipeline
from exp0002.common.utils import load_t2s
from lib_common.data import load_dataset, split
from lib_common.utils import Logger, seed_everything
from sweep_inference import DEFAULT_S2M, DEFAULT_T2S, E3, E4, EXP
from train_t2s import f0_targets


def practical_scores(ev: Evaluator, items: list, mels: list, out_dir) -> dict:
    """実用文の CER と UTMOS(Evaluator の Vocos・ASR・UTMOS を使う)。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for t, m in zip(items, mels):
        sf.write(out_dir / f"{t.id}.wav", ev.wave(m), SR)
        paths.append(str(out_dir / f"{t.id}.wav"))
    return {"cer": cer_scores([t.text for t in items], ev.asr.transcribe(paths))["reading"],
            "utmos": sum(ev.mos.score(paths)) / len(paths)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cond", choices=["gt_tokens", "gt_tokens_gt_f0", "tts"], required=True)
    parser.add_argument("--s2m", default="s2m_f0cond")
    parser.add_argument("--t2s", default="t2s_f0aux", help="tts の条件で使う T2S(runs/ の名前)。既定以外なら結果の名前に付く")
    parser.add_argument("--practical", action="store_true", help="tts の条件で、実用文 55 文の CER・UTMOS も測る")
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(4)
    log = Logger(EXP / "runs" / "log_evaluate_s2m.txt")
    test = split(load_dataset(None, log), "test")
    pipe = TTSPipeline.from_checkpoints(E3 / "codec_hubert_ja_k256" / "codec.pt", E3 / "s2m_hubert_ja_k256" / "s2m.pt")
    ev = Evaluator(test, E4 / "vocos" / "vocoder.pt", EXP / "runs" / "eval" / "s2m")
    gt_codes = [torch.load(E3 / "codec_hubert_ja_k256" / "tokens.pt")[u.id].long() for u in test]
    if args.cond != "gt_tokens":  # F0 の条件つきの S2M に差し替える
        ck = torch.load(EXP / "runs" / args.s2m / "s2m.pt", map_location="cpu")
        pipe.s2m.load_state_dict(ck["s2m"])
        f0cond = F0Condition(pipe.codec.hidden_dim).to(pipe.device).eval()
        f0cond.load_state_dict(ck["f0cond"])
    if args.cond == "tts":
        t2s, vocab, _ = load_t2s(EXP / "runs" / args.t2s / "t2s.pt", "cuda")
    res = []
    for seed in range(args.seeds):
        seed_everything(seed)
        if args.cond == "tts":
            mels, durs = [], []
            for s in range(0, len(test), 16):
                ids = [torch.tensor(vocab.encode(u.text)) for u in test[s: s + 16]]
                d = t2s.predict_durations(ids)
                codes = t2s.generate(ids, durations=d, **{**DEFAULT_T2S, "cfg": 0.0})
                f0s = t2s.predict_f0(ids, [c.cuda() for c in codes], d)
                mels += codes_to_mel_f0(pipe, f0cond, codes, f0s, **DEFAULT_S2M)
                durs += [x.cpu().numpy() for x in d]
        else:
            durs = ev.gt_durs  # 正解のトークンの長さは録音と同じなので、モーラの区間は jsut-lab の時刻
            if args.cond == "gt_tokens":
                mels = pipe.codes_to_mel(gt_codes, **DEFAULT_S2M)
            else:
                stats = torch.load(EXP / "runs" / args.s2m / "s2m.pt", map_location="cpu")["f0_stats"]
                f0s = f0_targets(test, gt_codes, torch.load(EXP / "runs" / "f0.pt"), stats)[0]
                mels = codes_to_mel_f0(pipe, f0cond, gt_codes, f0s, **DEFAULT_S2M)
        res.append(ev(f"{args.cond}_seed{seed}", mels, durs))
        log(f"{args.cond:16s} seed {seed} " + " ".join(f"{k} {v:.3f}" for k, v in res[-1].items() if not k.startswith("n_")))
    name = args.cond if args.t2s == "t2s_f0aux" or args.cond != "tts" else f"tts_{args.t2s}"
    practical = None
    if args.practical and args.cond == "tts":  # 実用文(正解の録音が無いので CER・UTMOS だけ)。種 0 の 1 回
        seed_everything(0)
        ets = load_eval_texts(["practical"])
        mels = []
        for s in range(0, len(ets), 16):
            ids = [torch.tensor(vocab.encode(t.text)) for t in ets[s: s + 16]]
            d = t2s.predict_durations(ids)
            codes = t2s.generate(ids, durations=d, **{**DEFAULT_T2S, "cfg": 0.0})
            mels += codes_to_mel_f0(pipe, f0cond, codes, t2s.predict_f0(ids, [c.cuda() for c in codes], d), **DEFAULT_S2M)
        practical = practical_scores(ev, ets, mels, EXP / "runs" / "eval" / "practical" / name)
        log(f"{name} practical {practical}")
    path = EXP / "results" / "evaluate_s2m.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    if practical:
        out.setdefault("practical", {})[name] = practical
    out[name] = {"s2m": args.s2m if args.cond != "gt_tokens" else "s2m_hubert_ja_k256", "seeds": res,
                      "mean": {k: float(np.mean([r[k] for r in res])) for k in res[0]},
                      "std": {k: float(np.std([r[k] for r in res])) for k in res[0]}}
    path.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
