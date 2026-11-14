"""試聴用の音声を作る: 基準と、一番良い構成(F0 の補助の損失 + 生成の CFG 0)を、同じ文で聞き比べる(乱数の種 0、ボコーダは Vocos)。

テスト文の先頭 3 文と、実用文の各カテゴリの先頭 1 文。results/samples/ に MP3 を保存し、index.json に追記する。

使い方:
    python make_samples.py
"""
import argparse
import json

import soundfile as sf
import torch

import common  # noqa: F401
from common.s2m_f0 import F0Condition, codes_to_mel_f0
from exp0001.common.pipeline import TTSPipeline
from exp0002.common.utils import load_t2s
from lib_common.audio import SR, save_preview
from lib_common.data import load_dataset, split
from lib_common.eval_sets import load_eval_texts
from lib_common.neural_vocoder import load_vocoder
from lib_common.utils import seed_everything
from sweep_inference import DEFAULT_S2M, DEFAULT_T2S, E3, E4, EXP

CONDITIONS = {  # 条件の名前 → (T2S の checkpoint, T2S の生成の設定の変更, F0 の条件つきの S2M の名前(None なら元の S2M))
    "baseline": (E3 / "t2s_hubert_ja_k256" / "t2s.pt", {}, None),
    "f0aux_cfg0": (EXP / "runs" / "t2s_f0aux" / "t2s.pt", {"cfg": 0.0}, None),
    "f0aux_cfg0_s2mf0": (EXP / "runs" / "t2s_f0aux" / "t2s.pt", {"cfg": 0.0}, "s2m_f0cond"),
    "f0aux_w2_cfg0_s2mf0": (EXP / "runs" / "t2s_f0aux_w2" / "t2s.pt", {"cfg": 0.0}, "s2m_f0cond"),  # F0 の補助の損失の重み 2.0
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None, help="作る条件(既定は全て)")
    args = parser.parse_args()
    test = split(load_dataset(None, print), "test")[:3]
    firsts = {}
    for t in load_eval_texts(["practical"]):
        firsts.setdefault(t.category, t)
    groups = [("test", test), ("eval_texts", list(firsts.values()))]
    pipe = TTSPipeline.from_checkpoints(E3 / "codec_hubert_ja_k256" / "codec.pt", E3 / "s2m_hubert_ja_k256" / "s2m.pt")
    voc = load_vocoder(E4 / "vocos" / "vocoder.pt", "cuda")
    ipath = EXP / "results" / "samples" / "index.json"
    index = json.loads(ipath.read_text()) if ipath.exists() else []
    for cond, (path, kw, s2m_f0) in CONDITIONS.items():
        if args.only and cond not in args.only:
            continue
        t2s, vocab, _ = load_t2s(path, "cuda")
        if s2m_f0:  # F0 の条件つきの S2M に差し替える
            ck = torch.load(EXP / "runs" / s2m_f0 / "s2m.pt", map_location="cpu")
            pipe.s2m.load_state_dict(ck["s2m"])
            f0cond = F0Condition(pipe.codec.hidden_dim).to(pipe.device).eval()
            f0cond.load_state_dict(ck["f0cond"])
        for group, items in groups:
            seed_everything(0)
            ids = [torch.tensor(vocab.encode(x.text)) for x in items]
            d = t2s.predict_durations(ids)
            codes = t2s.generate(ids, durations=d, **{**DEFAULT_T2S, **kw})
            if s2m_f0:
                mels = codes_to_mel_f0(pipe, f0cond, codes, t2s.predict_f0(ids, [c.cuda() for c in codes], d), **DEFAULT_S2M)
            else:
                mels = pipe.codes_to_mel(codes, **DEFAULT_S2M)
            d, sd = EXP / "runs" / "eval" / "samples" / group / cond, EXP / "results" / "samples" / group / cond
            d.mkdir(parents=True, exist_ok=True)
            sd.mkdir(parents=True, exist_ok=True)
            index = [r for r in index if not (r["group"] == group and r["condition"] == cond)]
            for x, m in zip(items, mels):
                sf.write(d / f"{x.id}.wav", voc.generate(m.cuda()[None])[0].cpu().numpy(), SR)
                save_preview(d / f"{x.id}.wav", sd / f"{x.id}.mp3")
                index.append({"group": group, "condition": cond, "id": x.id, "text": x.text,
                              "category": getattr(x, "category", "jsut_test"), "file": f"{group}/{cond}/{x.id}.mp3"})
    ipath.write_text(json.dumps(index, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
