"""外来語のアクセントの確認: F0 のずれが大きかった 5 文(分析 5)で、pyopenjtalk の辞書のアクセントと元の録音の高さの動きを比べる。

GPU は使わない。元の録音の F0 の動きが辞書の高・低と合っている割合(hl_agreement、モーラ単位)を、
5 文と、それ以外のテスト文とで比べる。辞書のアクセントが外れていれば、5 文の方が元の録音でも低くなる。
句ごとに、モーラの音素・辞書の高低(H/L)・元の録音の隣のモーラからの動き(↑/↓)も保存する。

使い方:
    python analyze_loanword_accent.py
出力: results/loanword_accent.json
"""
import json

import numpy as np

import common  # noqa: F401
from common.evaluator import FPT
from exp0002.common import utils
from exp0002.common.utils import phoneme_durations
from lib_common.audio import load_wave
from lib_common.data import load_dataset, split
from lib_common.prosody import MORA_END, _mora_f0, f0, mora_accent_scores, mora_phrases
from lib_common.text import phoneme_tokens
from sweep_inference import EXP

SAME = utils._same
WORST = ["VOICEACTRESS100_042", "VOICEACTRESS100_055", "VOICEACTRESS100_085", "VOICEACTRESS100_037", "VOICEACTRESS100_018"]


def mora_texts(tokens: list[str]) -> list[list[str]]:
    """mora_phrases と同じ区切り方で、句ごとのモーラの音素の文字列を作る。"""
    phrases, cur, buf = [], [], ""
    for t in tokens:
        if t in {"#", "_", "^", "$", "?"}:
            if cur:
                phrases.append(cur)
            cur, buf = [], ""
        elif t not in {"[", "]"}:
            buf += t
            if t in MORA_END:
                cur.append(buf)
                buf = ""
    if cur:
        phrases.append(cur)
    return phrases


def main():
    test = split(load_dataset(None, print), "test")
    groups = {"worst5": [], "others": []}
    detail = []
    for u in test:
        tk = phoneme_tokens(u.text)
        d = phoneme_durations(u, tk, len(u.mel) // FPT)
        if d is None and u.id in WORST:  # 5 文のうち 4 文は、外来語の読みの 1 か所(ヴ: b と v など)が録音のラベルと違うので、
            utils._same = lambda *a: True  # 音素の数が同じなら、音素の文字の一致は見ずにラベルの時刻を使う
            d = phoneme_durations(u, tk, len(u.mel) // FPT)
            utils._same = SAME
        if d is None:
            continue
        ref = f0(load_wave(u.path))
        ph = mora_phrases(tk, d, FPT)
        groups["worst5" if u.id in WORST else "others"].append((ref, ph))
        if u.id in WORST:
            rows = []
            for p, txt in zip(ph, mora_texts(tk)):
                x = _mora_f0(*ref, p)
                move = ["-"] + ["?" if np.isnan(a) or np.isnan(b) else ("↑" if b > a else "↓") for a, b in zip(x[:-1], x[1:])]
                rows.append({"moras": " ".join(txt), "dict": "".join("H" if h else "L" for _, _, h in p), "real": "".join(move)})
            detail.append({"id": u.id, "text": u.text, **mora_accent_scores([ref], [ph]), "phrases": rows})
    out = {k: mora_accent_scores([r for r, _ in v], [p for _, p in v]) | {"n_utts": len(v)} for k, v in groups.items()}
    out["detail"] = detail
    for k in ["worst5", "others"]:
        print(k, out[k])
    for x in detail:
        print(x["id"], f"{x['hl_agreement']:.3f}", x["n_pairs"])
    (EXP / "results" / "loanword_accent.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
