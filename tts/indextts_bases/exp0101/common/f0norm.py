"""F0 の正規化の仕方(exp0102 の f0_per_speaker で替える)。

    jsut(既定、exp0101): 全話者で同じ JSUT の log2 F0 の平均・標準偏差で正規化する(exp0005 の f0_targets)
    ref_mean(exp0102): 平均は「話者ベクトルを取った参照の発話」の有声の log2 F0 の平均、標準偏差は JSUT の値で正規化する
        (推論では参照音声 1 発話から求めるので、1 発話では不安定な標準偏差は話者ごとにしない)。
        学習でも推論でも同じ発話(話者ベクトル・参照音声の発話)の平均を使うので、T2S の出力と S2M の入力は同じ正規化になる
"""
import torch

from exp0005.train_t2s import f0_targets


def raw_targets(utts: list, codes: list, f0: dict) -> list[torch.Tensor]:
    """各音声トークンの区間の (有声か, log2 F0) [U_i, 2](正規化しない。無声は 0)。"""
    return f0_targets(utts, codes, f0, (0.0, 1.0))[0]


def mean_lf0(raw: torch.Tensor) -> float:
    """raw_targets の 1 発話分から、有声のトークンの log2 F0 の平均。"""
    v = raw[:, 0] > 0.5
    return float(raw[v, 1].mean()) if v.any() else 0.0


def normalize(raw: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """raw_targets の 1 発話分を、(log2 F0 - mean) / std に正規化する(無声は 0 のまま)。"""
    v = raw[:, 0] > 0.5
    return torch.stack([raw[:, 0], torch.where(v, (raw[:, 1] - mean) / std, torch.zeros_like(raw[:, 1]))], 1)
