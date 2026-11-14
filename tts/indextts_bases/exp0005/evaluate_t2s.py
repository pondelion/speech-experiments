"""exp0005 で学習した T2S を、基準(exp0003 の `t2s_hubert_ja_k256`)と同じ S2M・Vocos・評価(common/evaluator.py)で比べる。

生成の設定は既定(sweep_inference.py の DEFAULT_T2S・DEFAULT_S2M)。乱数の種 0・1・2 の 3 回測り、平均と標準偏差を出す。

使い方:
    python evaluate_t2s.py --runs t2s_f0aux
出力: results/evaluate_t2s.json(T2S ごと・種ごとの指標と、平均・標準偏差)
"""
import argparse
import json

import numpy as np
import torch

import common  # noqa: F401
from common.evaluator import Evaluator
from exp0001.common.pipeline import TTSPipeline
from exp0002.common.utils import load_t2s
from lib_common.data import load_dataset, split
from lib_common.utils import Logger, seed_everything
from sweep_inference import DEFAULT_S2M, DEFAULT_T2S, E3, E4, EXP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="runs/ の T2S の名前(基準は自動で入れる)")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--t2s_cfg", type=float, default=None, help="T2S の CFG の強さ(既定は DEFAULT_T2S の 2.5)。結果の名前に @cfg が付く")
    args = parser.parse_args()
    torch.set_num_threads(4)
    log = Logger(EXP / "runs" / "log_evaluate_t2s.txt")
    test = split(load_dataset(None, log), "test")
    pipe = TTSPipeline.from_checkpoints(E3 / "codec_hubert_ja_k256" / "codec.pt", E3 / "s2m_hubert_ja_k256" / "s2m.pt")
    ev = Evaluator(test, E4 / "vocos" / "vocoder.pt", EXP / "runs" / "eval" / "t2s")
    path = EXP / "results" / "evaluate_t2s.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    # 基準は、結果がまだ無いときだけ測る。1 つのプロセスで続けてモデルを替えると GPU のエラーで落ちることがあるので、
    # T2S は 1 回に 1 つずつ指定する方が安全
    t2s_kw = {**DEFAULT_T2S, **({"cfg": args.t2s_cfg} if args.t2s_cfg is not None else {})}
    suffix = f"@cfg{args.t2s_cfg:g}" if args.t2s_cfg is not None else ""
    for run in ([] if "baseline" in out or suffix else ["baseline"]) + args.runs:
        name = run + suffix
        t2s, vocab, _ = load_t2s(E3 / "t2s_hubert_ja_k256" / "t2s.pt" if run == "baseline" else EXP / "runs" / run / "t2s.pt", "cuda")
        res = []
        for seed in range(args.seeds):
            seed_everything(seed)
            mels, durs = [], []
            for s in range(0, len(test), 16):
                ids = [torch.tensor(vocab.encode(u.text)) for u in test[s: s + 16]]
                d = t2s.predict_durations(ids)
                mels += pipe.codes_to_mel(t2s.generate(ids, durations=d, **t2s_kw), **DEFAULT_S2M)
                durs += [x.cpu().numpy() for x in d]
            res.append(ev(f"{name}_seed{seed}", mels, durs))
            log(f"{name:16s} seed {seed} " + " ".join(f"{k} {v:.3f}" for k, v in res[-1].items() if not k.startswith("n_")))
        out[name] = {"seeds": res, "mean": {k: float(np.mean([r[k] for r in res])) for k in res[0]},
                     "std": {k: float(np.std([r[k] for r in res])) for k in res[0]}}
        path.write_text(json.dumps(out, indent=2))
        del t2s
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
