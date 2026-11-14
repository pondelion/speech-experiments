"""テスト 100 文(と実用文 55 文)で評価する(common/evaluator.py の指標。乱数の種 0・1・2 の平均)。

条件(--cond で 1 つずつ。1 つのプロセスで続けてモデルを替えると GPU のエラーで落ちることがあるため):
    gt_mel     正解の Mel → ボコーダ(ボコーダの上限)
    gt_tokens  正解のトークン → codec の decoder → S2M → ボコーダ(S2M までの上限)
    tts        テキスト → T2S → codec の decoder → S2M → ボコーダ
生成の設定は公式の推論の既定: T2S はサンプリング(温度 0.8、top_p 0.8、top_k 30)+ ビームサーチ 3 + 繰り返しのペナルティ 10、
S2M は Euler 25 step・CFG 0.7、Mel の長さ = codec の特徴のフレーム数 × 1.72。

使い方:
    python evaluate.py --cond tts --codec codec --s2m s2m --t2s t2s --vocoder bigvgan_v2_22k [--practical]
出力: results/evaluate.json に追記(名前は --cond、--tag があれば {cond}_{tag})。音声は runs/eval/(git の対象外)
"""
import argparse
import json

import numpy as np
import torch

import common
from common.evaluator import Evaluator
from common.s2m import MEL_PER_CODEC_FRAME
from common.text import frontend
from lib_common.data import load_dataset, split
from lib_common.eval_sets import load_eval_texts
from lib_common.utils import Logger, seed_everything
from train_codec import load_codec
from train_s2m import load_s2m
from train_t2s import load_t2s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cond", choices=["gt_mel", "gt_tokens", "tts"], required=True)
    parser.add_argument("--codec", default="codec")
    parser.add_argument("--s2m", default="s2m")
    parser.add_argument("--t2s", default="t2s")
    parser.add_argument("--vocoder", default=None, help="runs/{vocoder}/vocoder.pt。無ければ Griffin-Lim")
    parser.add_argument("--practical", action="store_true", help="tts のとき、実用文 55 文の CER・UTMOS も測る(種 0)")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--limit", type=int, default=None, help="動作確認用にテスト文を先頭から絞る")
    args = parser.parse_args()
    torch.set_num_threads(4)
    name = args.cond if args.tag is None else f"{args.cond}_{args.tag}"
    log = Logger(common.RUNS_DIR / "log_evaluate.txt")
    log(f"args: {vars(args)}")
    test = split(load_dataset(None, log), "test")[: args.limit]
    ev = Evaluator(test, common.RUNS_DIR / args.vocoder / "vocoder.pt" if args.vocoder else None)
    if args.cond != "gt_mel":
        codec, _, _ = load_codec(common.RUNS_DIR / args.codec / "codec.pt", "cuda")
        s2m = load_s2m(common.RUNS_DIR / args.s2m / "s2m.pt", "cuda")
        tokens = torch.load(common.RUNS_DIR / args.codec / "tokens.pt")
    if args.cond == "tts":
        t2s, vocab = load_t2s(common.RUNS_DIR / args.t2s / "t2s.pt", "cuda")

    @torch.no_grad()
    def synth(codes: torch.Tensor) -> torch.Tensor:
        feat = codec.decode(codes.long().cuda()[None])                                # [1, 2U, 1024]
        n = torch.tensor([int(feat.shape[1] * MEL_PER_CODEC_FRAME)], device="cuda")
        return s2m.generate(feat, n)[0].cpu()                                         # [T, F]

    def tts(text: str) -> torch.Tensor:
        return synth(t2s.generate(torch.tensor(vocab.encode(frontend(text)))))

    res = []
    for seed in range(args.seeds if args.cond != "gt_mel" else 1):
        seed_everything(seed)
        if args.cond == "gt_mel":
            mels = [u.mel for u in test]
        elif args.cond == "gt_tokens":
            mels = [synth(tokens[u.id]) for u in test]
        else:
            mels = [tts(u.text) for u in test]
        res.append(ev(mels, common.RUNS_DIR / "eval" / name / f"seed{seed}"))
        log(f"{name} seed {seed} " + " ".join(f"{k} {v:.3f}" for k, v in res[-1].items() if not k.startswith("n_")))
    out = {"args": vars(args), "seeds": res, "mean": {k: float(np.mean([r[k] for r in res])) for k in res[0]},
           "std": {k: float(np.std([r[k] for r in res])) for k in res[0]}}
    if args.practical and args.cond == "tts":
        seed_everything(0)
        items = load_eval_texts(["practical"])
        out["practical"] = ev.practical(items, [tts(t.text) for t in items], common.RUNS_DIR / "eval" / name / "practical")
        log(f"{name} practical {out['practical']}")
    path = common.EXP_DIR / "results" / "evaluate.json"
    path.parent.mkdir(exist_ok=True)
    allres = json.loads(path.read_text()) if path.exists() else {}
    allres[name] = out
    path.write_text(json.dumps(allres, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
