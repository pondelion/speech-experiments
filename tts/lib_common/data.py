"""JSUT(女性 1 人・約 10 時間の読み上げ音声)の読み込みと、学習・検証・テストへの分割。

前処理ファイルは作らず、起動時に全 wav を読んで log-Mel にしてメモリに置く。
データセットはリポジトリ直下の datasets/ に置く(datasets/README.md を参照)。
"""
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .audio import HOP, SR, load_wave, wave_to_logmel

REPO_ROOT = Path(__file__).resolve().parents[2]  # lib_common → tts → リポジトリ
JSUT_DIR = REPO_ROOT / "datasets" / "jsut" / "jsut_ver1.1"
JSUT_LAB_DIR = REPO_ROOT / "datasets" / "jsut_lab"  # 音素アライメント(jsut-lab)
TEST_SUBSET = "voiceactress100"  # JVS の parallel100 と同じ 100 文。後で多話者の実験と比べられる
N_VAL = 100


@dataclass
class Utterance:
    """1 発話のデータ。

    Attributes:
        id: 発話 ID(例 BASIC5000_0001)。
        subset: JSUT のサブセット名(例 basic5000)。
        path: wav のパス。
        text: 書き起こし。
        split: "train" / "val" / "test"。
        mel: 正規化前の log-Mel [T(フレーム数), 80(Mel の帯域)]。load_mels() で入る。
    """
    id: str
    subset: str
    path: str
    text: str
    split: str = ""
    mel: torch.Tensor | None = None


def list_jsut(seed: int = 0) -> list[Utterance]:
    """JSUT の全発話を列挙し、学習・検証・テストに分ける。

    test は voiceactress100 の 100 発話、val は残りからランダムに選んだ 100 発話、train はそれ以外。

    Args:
        seed: val を選ぶ乱数の seed。

    Returns:
        Utterance のリスト(mel はまだ空)。
    """
    utts = []
    for subset_dir in sorted(p for p in JSUT_DIR.iterdir() if p.is_dir()):
        for line in (subset_dir / "transcript_utf8.txt").read_text(encoding="utf-8").splitlines():
            uid, text = line.split(":", 1)
            utts.append(Utterance(uid, subset_dir.name, str(subset_dir / "wav" / f"{uid}.wav"), text))
    rest = [u for u in utts if u.subset != TEST_SUBSET]
    val_ids = {u.id for u in random.Random(seed).sample(rest, N_VAL)}
    for u in utts:
        u.split = "test" if u.subset == TEST_SUBSET else "val" if u.id in val_ids else "train"
    return utts


def load_mels(utts: list[Utterance], log=print) -> None:
    """全発話の log-Mel を計算して Utterance.mel に入れる。

    Args:
        utts: Utterance のリスト。その場で mel を書き込む。
        log: 進み具合を出力する関数。
    """
    t0 = time.time()
    for i, u in enumerate(utts):
        u.mel = wave_to_logmel(load_wave(u.path))
        if (i + 1) % 2000 == 0:
            log(f"  loaded {i + 1}/{len(utts)} ({time.time() - t0:.0f}s)")
    hours = sum(len(u.mel) for u in utts) * HOP / SR / 3600
    log(f"loaded {len(utts)} utterances, {hours:.2f} h of audio in {time.time() - t0:.0f}s")


def load_dataset(limit: int | None = None, log=print) -> list[Utterance]:
    """JSUT を列挙・分割し、log-Mel まで読み込む。

    Args:
        limit: 動作確認用。指定すると train は先頭 limit 発話、val・test は先頭 limit//10 発話(最低 2)だけ使う。
        log: 進み具合を出力する関数。

    Returns:
        mel を読み込んだ Utterance のリスト。
    """
    utts = list_jsut()
    if limit:
        n_small = max(2, limit // 10)
        utts = split(utts, "train")[:limit] + split(utts, "val")[:n_small] + split(utts, "test")[:n_small]
    load_mels(utts, log)
    return utts


def load_jsut_alignment(utt: Utterance) -> list[tuple[float, float, str]]:
    """jsut-lab の音素アライメント(どの時刻にどの音素があるか)を読む。

    時刻は Julius による強制アライメントで自動的に付けたもの。音素の表記は pyopenjtalk と同じ
    (無声化した母音は大文字、文頭・文末の無音は sil、文中のポーズは pau)。

    Args:
        utt: JSUT の発話。

    Returns:
        (開始の秒, 終了の秒, 音素) のリスト(時刻の順)。
    """
    rows = []
    for line in (JSUT_LAB_DIR / utt.subset / "lab" / f"{utt.id}.lab").read_text(encoding="utf-8").splitlines():
        start, end, label = line.split(maxsplit=2)
        phoneme = label.split("-", 1)[1].split("+", 1)[0]  # 「前^前-今+次=次」の「今」の部分
        rows.append((int(start) * 1e-7, int(end) * 1e-7, phoneme))  # 100 ナノ秒単位 → 秒
    return rows


def align_to_segments(align: list[tuple[float, float, str]], n_segments: int,
                      segment_sec: float) -> list[tuple[str, float]]:
    """音素アライメントを、一定の長さの区間(例: 音声トークン 1 個 = 42.7ms)ごとの音素に変換する。

    各区間と一番長く重なる音素を、その区間の音素にする。

    Args:
        align: load_jsut_alignment() の (開始の秒, 終了の秒, 音素) のリスト。
        n_segments: 区間の数(先頭から 0, segment_sec, 2 × segment_sec, ... 秒の区間)。
        segment_sec: 1 区間の長さ(秒)。

    Returns:
        区間ごとの (音素, 区間のうちその音素が占める割合) のリスト(長さ n_segments)。
        どの音素とも重ならない区間(音声の末尾など)は ("sil", 0.0)。
    """
    out = []
    for u in range(n_segments):
        t0, t1 = u * segment_sec, (u + 1) * segment_sec
        overlap = {}
        for s, e, p in align:
            if e > t0 and s < t1:
                overlap[p] = overlap.get(p, 0.0) + min(e, t1) - max(s, t0)
        p = max(overlap, key=overlap.get) if overlap else "sil"
        out.append((p, overlap.get(p, 0.0) / segment_sec))
    return out


def split(utts: list[Utterance], name: str) -> list[Utterance]:
    """指定した split の発話だけを返す。

    Args:
        utts: Utterance のリスト。
        name: "train" / "val" / "test"。

    Returns:
        その split の Utterance のリスト。
    """
    return [u for u in utts if u.split == name]
