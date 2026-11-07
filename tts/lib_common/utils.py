"""学習スクリプトで共通に使う小物(乱数の固定、ログ、学習率のスケジュール、長さの近い発話をまとめたバッチ)。"""
import math
import random
import time
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """乱数の seed を Python・NumPy・PyTorch でそろえる。

    Args:
        seed: seed の値。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Logger:
    """画面とファイルの両方に、経過時間付きで出力する。

    Args:
        path: ログファイルのパス(追記する)。
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")
        self.t0 = time.time()

    def __call__(self, msg: str) -> None:
        """1 行出力する。

        Args:
            msg: 出力する文字列。
        """
        line = f"[{time.time() - self.t0:7.0f}s] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def warmup_cosine(opt: torch.optim.Optimizer, total_steps: int, warmup_steps: int = 1000):
    """学習率を、最初の warmup_steps で 0 から上げ(warmup)、その後 cosine で 0 まで下げるスケジューラ。

    学習の最初は重みがでたらめで勾配が荒れやすいので、小さい学習率から始めると安定する。

    Args:
        opt: optimizer(設定した lr が最大値になる)。
        total_steps: 学習の総 step 数。
        warmup_steps: warmup の step 数(総 step 数の 1/10 を超えないように切り詰める)。

    Returns:
        毎 step 後に .step() を呼ぶ LambdaLR。
    """
    warmup = max(1, min(warmup_steps, total_steps // 10))

    def factor(s: int) -> float:
        return min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps))

    return torch.optim.lr_scheduler.LambdaLR(opt, factor)


def length_batches(lengths: list[int], max_frames: int, shuffle: bool, seed: int = 0) -> list[list[int]]:
    """長さの近い発話をまとめ、1 バッチの「最長の長さ × 発話数」が max_frames 以下になるように分ける。

    長さがばらばらの発話を適当に混ぜると、短い発話にも長い発話の長さまで詰め物が入って無駄が多い。
    長さ順に並べてから区切ると詰め物が減る。学習時はバッチの順番をシャッフルする。

    Args:
        lengths: 各発話の長さ。
        max_frames: 1 バッチの「最長の長さ × 発話数」の上限。
        shuffle: バッチの順番と、同じ長さの発話の並びをシャッフルするか。
        seed: シャッフルの seed。

    Returns:
        バッチごとの発話インデックスのリスト。
    """
    rng = random.Random(seed)
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], rng.random() if shuffle else i))
    batches, cur = [], []
    for i in order:
        if cur and max(lengths[j] for j in cur + [i]) * (len(cur) + 1) > max_frames:
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    if shuffle:
        rng.shuffle(batches)
    return batches
