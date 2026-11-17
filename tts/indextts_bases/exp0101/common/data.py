"""JSUT + JVS の発話の一覧(README の「データ」と results/split.json のとおりに分ける)と、音素ごとの長さの正解。

分け方:
    train:       JSUT の学習データ(lib_common.data の train)+ JSUT の voiceactress100 のうち学習に使う 75 文
                 + 学習に使う 90 人の parallel100 の 75 文と nonpara30
    val:         JSUT の検証データ(lib_common.data の val、100 文)+ 学習に使う 90 人の nonpara30 から 1 人 1 発話(ID の順で先頭)
    test_unseen: 学習に使わない 10 人の parallel100 のテスト文 25 文(ゼロショット)
    test_seen:   学習に使った話者から選んだ 10 人の parallel100 のテスト文 25 文(声は学習で見た、文は見ていない)
    prompt:      参照音声の候補。学習に使わない 10 人の nonpara30(学習には入っていない)
学習に使わない 10 人の音声は、test_unseen と prompt にしか入らない。テスト文は、どの話者の分も train・val に入らない。
JVS のささやき声(whisper10)・裏声(falset10)は使わない。

Utterance の id は、JSUT は今までと同じ(例 BASIC5000_0001)、JVS は「話者/発話 ID」(例 jvs001/VOICEACTRESS100_001)。
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch

from exp0002.common.utils import PROSODY_MARKS, TOKEN_SEC, _same
from lib_common.data import JSUT_LAB_DIR, list_jsut

from . import DATASETS, EXP_DIR

JVS = DATASETS / "jvs" / "jvs_ver1"


@dataclass
class Utt:
    id: str
    speaker: str          # "jsut" か "jvs001" など
    path: str
    text: str
    lab: str              # 音素の時刻のラベルのパス
    lab_unit: float       # ラベルの時刻の単位(秒)。jsut-lab は 1e-7、JVS は 1
    split: str = ""
    mel: torch.Tensor | None = field(default=None, repr=False)


def load_split() -> dict:
    return json.loads((EXP_DIR / "results" / "split.json").read_text())


def _jvs(speaker: str, subset: str) -> list[Utt]:
    """JVS の 1 人の 1 つのセットの発話(wav と書き起こしの両方があるものだけ)。"""
    d = JVS / speaker / subset
    out = []
    for line in (d / "transcripts_utf8.txt").read_text(encoding="utf-8").splitlines():
        uid, text = line.split(":", 1)
        wav = d / "wav24kHz16bit" / f"{uid}.wav"
        if wav.exists():
            out.append(Utt(f"{speaker}/{uid}", speaker, str(wav), text, str(d / "lab" / "mon" / f"{uid}.lab"), 1.0))
    return out


def list_all() -> list[Utt]:
    sp = load_split()
    test_ids, train_ids = set(sp["test_sentences"]), set(sp["train_sentences"])
    out = []
    for u in list_jsut():
        utt = Utt(u.id, "jsut", u.path, u.text, str(JSUT_LAB_DIR / u.subset / "lab" / f"{u.id}.lab"), 1e-7)
        if u.subset == "repeat500" and f"VOICEACTRESS100_{u.id[-3:]}" in test_ids:
            continue  # repeat500 は voiceactress100 の 100 文を 5 回ずつ読んだもの(番号が対応する)。テスト文に当たる分は使わない
        if u.split in ("train", "val"):
            utt.split = u.split
        elif u.id in train_ids:  # voiceactress100 のうち学習に使う 75 文
            utt.split = "train"
        else:
            continue             # テスト文 25 文の JSUT の分は使わない(比べる相手のフェーズ A のモデルは、テキストだけで測る)
        out.append(utt)
    for s in sp["train_speakers"]:
        out += [_set(u, "val" if i == 0 else "train") for i, u in enumerate(_jvs(s, "nonpara30"))]
        for u in _jvs(s, "parallel100"):
            uid = u.id.split("/")[1]
            if uid in train_ids:
                out.append(_set(u, "train"))
            elif uid in test_ids and s in sp["seen_eval_speakers"]:
                out.append(_set(u, "test_seen"))
    for s in sp["unseen_speakers"]:
        out += [_set(u, "test_unseen") for u in _jvs(s, "parallel100") if u.id.split("/")[1] in test_ids]
        out += [_set(u, "prompt") for u in _jvs(s, "nonpara30")]
    # 念のため、句読点・空白を除いた文面がテスト文と同じ発話を、学習・検証から除く(ほかのセットから同じ文が入るのを防ぐ)
    test_norm = {_norm(t) for t in sp["test_texts"].values()}
    return [u for u in out if not (u.split in ("train", "val") and _norm(u.text) in test_norm)]


def _norm(text: str) -> str:
    return re.sub(r"[、。,.!?！？「」『』()（）\s]", "", text)


def _set(u: Utt, split: str) -> Utt:
    u.split = split
    return u


def load_alignment(u: Utt) -> list[tuple[float, float, str]]:
    """音素の時刻のラベル → (開始の秒, 終了の秒, 音素) のリスト。jsut-lab はフルコンテキストのラベル、JVS は mon(音素だけ)。"""
    rows = []
    for line in open(u.lab, encoding="utf-8").read().splitlines():
        start, end, label = line.split(maxsplit=2)
        ph = label.split("-", 1)[1].split("+", 1)[0] if "-" in label else label
        rows.append((float(start) * u.lab_unit, float(end) * u.lab_unit, ph))
    return rows


def phoneme_durations(u: Utt, tokens: list[str], n_codes: int, token_sec: float = TOKEN_SEC) -> list[int] | None:
    """exp0002 の phoneme_durations と同じ計算(ラベルの読み込みだけ JVS にも対応させたもの)。
    並びが合わないか、ラベルのファイルが無ければ None(JVS には、ラベルの無い発話がある)。"""
    if not Path(u.lab).exists():
        return None
    align = load_alignment(u)
    sounding = [t for t in tokens if t not in PROSODY_MARKS]
    if len(sounding) != len(align) or not all(_same(t, n, p) for t, (n, (_, _, p)) in zip(sounding, enumerate(align))):
        return None
    ends, prev = [], 0
    for _, e, _ in align:
        prev = max(prev, min(round(e / token_sec), n_codes))
        ends.append(prev)
    ends[-1] = n_codes
    durs = iter([b - a for a, b in zip([0] + ends[:-1], ends)])
    return [0] + [0 if t in PROSODY_MARKS else next(durs) for t in tokens] + [0]
