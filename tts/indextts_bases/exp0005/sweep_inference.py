"""生成の設定(学習し直し不要)で抑揚がどう変わるか: T2S の温度・候補の数・CFG、S2M の CFG を 1 つずつ変える。

基準は exp0004 の構成(日本語 HuBERT + ハイブリッド型 + Vocos)の既定の設定
(T2S: 50 回・温度 0.9・確率の上位 2% からサンプル・CFG 2.5、S2M: Euler 25 step・CFG 0.7)。
各設定は乱数の種 0 で 1 回ずつ。既定の設定だけは種 0・1・2 の 3 回測り、ばらつきの目安にする。

使い方:
    python sweep_inference.py
出力: results/sweep_inference.json(設定ごとの指標)。音声は runs/eval/sweep/(git の対象外)
"""
import json
from pathlib import Path

import torch

import common  # noqa: F401
from common.evaluator import Evaluator
from exp0001.common.pipeline import TTSPipeline
from exp0002.common.utils import load_t2s
from lib_common.data import load_dataset, split
from lib_common.utils import Logger, seed_everything

EXP = Path(__file__).resolve().parent
E3, E4 = EXP.parent / "exp0003" / "runs", EXP.parent / "exp0004" / "runs"
DEFAULT_T2S = {"n_steps": 50, "temperature": 0.9, "filter_thres": 0.98, "cfg": 2.5}
DEFAULT_S2M = {"cfg": 0.7}
SETTINGS = [  # (名前, T2S の設定の変更, S2M の設定の変更)
    ("t2s_temp0.5", {"temperature": 0.5}, {}), ("t2s_temp1.5", {"temperature": 1.5}, {}),
    ("t2s_top10pct", {"filter_thres": 0.9}, {}),
    ("t2s_cfg0", {"cfg": 0.0}, {}), ("t2s_cfg1", {"cfg": 1.0}, {}), ("t2s_cfg4", {"cfg": 4.0}, {}),
    ("s2m_cfg0", {}, {"cfg": 0.0}), ("s2m_cfg1.5", {}, {"cfg": 1.5}),
]


def main():
    torch.set_num_threads(4)
    log = Logger(EXP / "runs" / "log_sweep.txt")
    test = split(load_dataset(None, log), "test")
    pipe = TTSPipeline.from_checkpoints(E3 / "codec_hubert_ja_k256" / "codec.pt", E3 / "s2m_hubert_ja_k256" / "s2m.pt")
    t2s, vocab, _ = load_t2s(E3 / "t2s_hubert_ja_k256" / "t2s.pt", "cuda")
    ev = Evaluator(test, E4 / "vocos" / "vocoder.pt", EXP / "runs" / "eval" / "sweep")

    def run(name: str, t2s_kw: dict, s2m_kw: dict, seed: int) -> dict:
        seed_everything(seed)
        mels, durs = [], []
        for s in range(0, len(test), 16):
            ids = [torch.tensor(vocab.encode(u.text)) for u in test[s: s + 16]]
            d = t2s.predict_durations(ids)
            mels += pipe.codes_to_mel(t2s.generate(ids, durations=d, **{**DEFAULT_T2S, **t2s_kw}), **{**DEFAULT_S2M, **s2m_kw})
            durs += [x.cpu().numpy() for x in d]
        res = ev(name, mels, durs)
        log(f"{name:16s} " + " ".join(f"{k} {v:.3f}" for k, v in res.items() if not k.startswith("n_")))
        return res

    out = {f"default_seed{s}": run(f"default_seed{s}", {}, {}, s) for s in range(3)}
    for name, t2s_kw, s2m_kw in SETTINGS:
        out[name] = run(name, t2s_kw, s2m_kw, 0)
        (EXP / "results" / "sweep_inference.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
