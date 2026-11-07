"""exp0001 固有の小物(保存先、codec の読み込みとトークン化、4 フレーム単位での Mel の詰め方)。

shape の軸の記号:
    B: バッチ内の発話数、T: Mel のフレーム数、F: Mel の帯域数(80)、U: 音声トークン数(= T/4)
"""
from pathlib import Path

import torch

from lib_common.audio import MelNormalizer
from lib_common.utils import length_batches

from .codec import DOWNSAMPLE, MelCodec

EXP_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = EXP_DIR / "runs"  # checkpoint・ログ・全テスト文の音声の置き場(git の対象外)


def pad_mels(mels: list[torch.Tensor], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """長さの違う Mel を、0 で詰めて 1 つのテンソルにする。

    長さは codec の時間圧縮(4 フレーム = 1 トークン)に合わせ、4 の倍数に切り捨てる。

    Args:
        mels: 正規化した Mel [T_i(i 番目の発話のフレーム数), 80(Mel の帯域)] のリスト。
        device: 置き先のデバイス。

    Returns:
        Mel [B(発話数), T(一番長い発話のフレーム数), F(Mel の帯域数 = 80)] と、
        有効なフレームが True のマスク [B, T]。
    """
    mels = [m[: len(m) // DOWNSAMPLE * DOWNSAMPLE] for m in mels]  # 各 [T_i, F] の T_i を 4 の倍数に
    T = max(len(m) for m in mels)                                    # 一番長い発話に合わせる
    x = torch.zeros(len(mels), T, mels[0].shape[1])                  # [B, T, F]
    mask = torch.zeros(len(mels), T, dtype=torch.bool)               # [B, T]
    for i, m in enumerate(mels):
        x[i, : len(m)] = m
        mask[i, : len(m)] = True
    return x.to(device), mask.to(device)


def load_codec(path: Path, device: str) -> tuple[MelCodec, MelNormalizer]:
    """学習済みの codec と、Mel の正規化の値を読み込む(以降の学習では codec を固定して使う)。

    Args:
        path: train_codec.py が保存した codec.pt。
        device: 置き先のデバイス。

    Returns:
        評価モードの codec と MelNormalizer。
    """
    ckpt = torch.load(path, map_location="cpu")
    codec = MelCodec(**ckpt["config"])
    codec.load_state_dict(ckpt["model"])
    return codec.to(device).eval(), MelNormalizer.from_state_dict(ckpt["normalizer"])


@torch.no_grad()
def encode_all(codec: MelCodec, mels: list[torch.Tensor], device: str, max_frames: int = 32000) -> list[torch.Tensor]:
    """全発話の Mel を codec でトークン ID 列にする。

    Args:
        codec: 学習済みの codec。
        mels: 正規化した Mel [T_i(i 番目の発話のフレーム数), 80(Mel の帯域)] のリスト。
        device: デバイス。
        max_frames: 1 バッチのフレーム数の上限。

    Returns:
        発話ごとのトークン ID 列 [U_i(トークン数 = T_i // 4)](CPU)のリスト。
        (以下のコメントの記号: B 発話数、T フレーム数、F Mel の帯域数、U トークン数)
    """
    out = [None] * len(mels)
    for b in length_batches([len(m) for m in mels], max_frames, shuffle=False):
        mel, _ = pad_mels([mels[i] for i in b], device)  # [B, T, F]
        ids = codec.encode(mel).cpu()                     # [B, T, F] → [B, U](U = T/4)
        for j, i in enumerate(b):
            out[i] = ids[j, : len(mels[i]) // DOWNSAMPLE]  # 詰め物の分を除いて [U_i]
    return out
