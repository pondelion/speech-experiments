"""分析: T2S の学習条件ごとに、学習データと検証データの loss の推移を 1 枚の図に並べる。

各学習の runs/{名前}/log.txt の「step N | train loss A ... | val loss B」の行を読む。
学習データの loss は直近 eval_every step の平均、検証データの loss はその時点の値。

使い方:
    python plot_t2s_curves.py t2s_ph_base t2s_ph_drop03 t2s_ph_small t2s_ph_small_drop03 t2s_char_drop03
出力: results/fig_t2s_loss_curves.jpg
"""
import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common  # noqa: F401  tts/lib_common を import できるようにする
from common.utils import EXP_DIR, RUNS_DIR

LINE_RE = re.compile(r"step\s+(\d+) \| train loss ([\d.]+) .*\| val loss ([\d.]+)")


def read_log(name: str) -> tuple[list[int], list[float], list[float]]:
    """log.txt から step・学習データの loss・検証データの loss の列を読む。

    Args:
        name: runs/ の中の名前。

    Returns:
        step のリスト、学習データの loss のリスト、検証データの loss のリスト。
    """
    steps, train, val = [], [], []
    for line in (RUNS_DIR / name / "log.txt").read_text(encoding="utf-8").splitlines():
        m = LINE_RE.search(line)
        if m:
            steps.append(int(m.group(1)))
            train.append(float(m.group(2)))
            val.append(float(m.group(3)))
    return steps, train, val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="runs/ の中の T2S の名前")
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for name in args.runs:
        steps, train, val = read_log(name)
        line, = axes[1].plot(steps, val, "o-", ms=3, label=name)
        axes[0].plot(steps, train, "o-", ms=3, color=line.get_color(), label=name)
        best = min(range(len(val)), key=val.__getitem__)
        axes[1].plot(steps[best], val[best], "*", ms=12, color=line.get_color())  # 検証 loss が一番小さい点
    for ax, title in zip(axes, ["train loss (mean of last eval interval)", "validation loss (star = best)"]):
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("cross entropy of next audio token")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    (EXP_DIR / "results").mkdir(exist_ok=True)
    fig.savefig(EXP_DIR / "results" / "fig_t2s_loss_curves.jpg", dpi=110)
    print(f"saved {EXP_DIR / 'results' / 'fig_t2s_loss_curves.jpg'}")


if __name__ == "__main__":
    main()
