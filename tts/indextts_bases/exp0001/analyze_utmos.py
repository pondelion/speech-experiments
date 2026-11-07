"""分析: 評価で作った音声の自然さを、UTMOS22 で推定した MOS(1〜5、高いほど自然)で比べる。

CER は読みの正しさしか測らないので、音の細部の違い(codec の decoder のぼやけた Mel と、S2M のくっきりした Mel など)を
こちらで見る。音声は evaluate.py などが runs/eval/ に作ったものを使う(作り直さない)。

使い方:
    python analyze_utmos.py                                  # 既定の条件(evaluate.py のテスト 100 文の 6 条件)
    python analyze_utmos.py --dirs s2m_compare/s2m_f3/gt     # runs/eval/ からの相対パスで条件を指定
出力: results/utmos.json(条件ごとの MOS の平均・標準偏差。前に測った条件の結果は残す)
"""
import argparse
import json

import numpy as np
import torch

import common  # noqa: F401  tts/lib_common を import できるようにする
from common.utils import EXP_DIR, RUNS_DIR
from lib_common.mos import UTMOS
from lib_common.utils import Logger

DEFAULT_DIRS = ["real", "mel_gl", "codec_rec", "s2m_gt_tokens", "tts_phoneme", "tts_char"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="+", default=DEFAULT_DIRS, help="runs/eval/ からの相対パス")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.set_num_threads(4)
    log = Logger(RUNS_DIR / "eval" / "log_utmos.txt")
    mos = UTMOS(args.device)
    out_path = EXP_DIR / "results" / "utmos.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    for d in args.dirs:
        paths = sorted(str(p) for p in (RUNS_DIR / "eval" / d).glob("*.wav"))
        s = np.array(mos.score(paths))
        out[d] = {"mean": float(s.mean()), "std": float(s.std()), "n": len(s)}
        log(f"{d:32s} UTMOS {s.mean():.3f} ± {s.std():.3f} (n={len(s)})")
        out_path.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
