"""比べる相手: exp0005 の一番良い構成を、exp0006 と同じ測り方(common/evaluator.py。モーラの区間は DTW で写す)で測り直す。

exp0005 の構成: ハイブリッド型の T2S(F0 の補助の損失、重み 2.0、`exp0005/runs/t2s_f0aux_w2`)、生成の CFG 0、
T2S が予測した F0 を条件にする S2M(`exp0005/runs/s2m_f0cond`)、日本語 HuBERT のトークン、Vocos(24kHz、exp0004)。
音声の設定は exp0005 と同じ 24kHz(jsut24k)にする。乱数の種 0・1・2 と、実用文 55 文(種 0)。

使い方:
    python evaluate_exp0005.py
出力: results/evaluate.json の exp0005_best
"""
import os

os.environ["TTS_AUDIO_PROFILE"] = "jsut24k"  # common を import する前に決める(exp0005 は 24kHz)

import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import common  # noqa: E402
from common.evaluator import Evaluator  # noqa: E402
from exp0001.common.pipeline import TTSPipeline  # noqa: E402
from exp0002.common.utils import load_t2s  # noqa: E402
from exp0005.common.s2m_f0 import F0Condition, codes_to_mel_f0  # noqa: E402
from lib_common.data import load_dataset, split  # noqa: E402
from lib_common.eval_sets import load_eval_texts  # noqa: E402
from lib_common.utils import Logger, seed_everything  # noqa: E402

BASES = common.EXP_DIR.parent
E3, E4, E5 = BASES / "exp0003" / "runs", BASES / "exp0004" / "runs", BASES / "exp0005" / "runs"
T2S_KW = {"n_steps": 50, "temperature": 0.9, "filter_thres": 0.98, "cfg": 0.0}  # exp0005 の既定(CFG だけ 0)
S2M_KW = {"cfg": 0.7}


def main():
    torch.set_num_threads(4)
    log = Logger(common.RUNS_DIR / "log_evaluate.txt")
    test = split(load_dataset(None, log), "test")
    pipe = TTSPipeline.from_checkpoints(E3 / "codec_hubert_ja_k256" / "codec.pt", E3 / "s2m_hubert_ja_k256" / "s2m.pt")
    ck = torch.load(E5 / "s2m_f0cond" / "s2m.pt", map_location="cpu")
    pipe.s2m.load_state_dict(ck["s2m"])
    f0cond = F0Condition(pipe.codec.hidden_dim).to(pipe.device).eval()
    f0cond.load_state_dict(ck["f0cond"])
    t2s, vocab, _ = load_t2s(E5 / "t2s_f0aux_w2" / "t2s.pt", "cuda")
    ev = Evaluator(test, E4 / "vocos" / "vocoder.pt")

    def tts(texts: list[str]) -> list[torch.Tensor]:
        mels = []
        for s in range(0, len(texts), 16):
            ids = [torch.tensor(vocab.encode(t)) for t in texts[s: s + 16]]
            d = t2s.predict_durations(ids)
            codes = t2s.generate(ids, durations=d, **T2S_KW)
            mels += codes_to_mel_f0(pipe, f0cond, codes, t2s.predict_f0(ids, [c.cuda() for c in codes], d), **S2M_KW)
        return mels

    res = []
    for seed in range(3):
        seed_everything(seed)
        res.append(ev(tts([u.text for u in test]), common.RUNS_DIR / "eval" / "exp0005_best" / f"seed{seed}"))
        log(f"exp0005_best seed {seed} " + " ".join(f"{k} {v:.3f}" for k, v in res[-1].items() if not k.startswith("n_")))
    out = {"seeds": res, "mean": {k: float(np.mean([r[k] for r in res])) for k in res[0]},
           "std": {k: float(np.std([r[k] for r in res])) for k in res[0]}}
    seed_everything(0)
    items = load_eval_texts(["practical"])
    out["practical"] = ev.practical(items, tts([t.text for t in items]), common.RUNS_DIR / "eval" / "exp0005_best" / "practical")
    log(f"exp0005_best practical {out['practical']}")
    path = common.EXP_DIR / "results" / "evaluate.json"
    path.parent.mkdir(exist_ok=True)
    allres = json.loads(path.read_text()) if path.exists() else {}
    allres["exp0005_best"] = out
    path.write_text(json.dumps(allres, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
