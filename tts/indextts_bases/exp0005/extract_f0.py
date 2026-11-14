"""JSUT の全ての録音の F0 を求めて保存する(T2S の F0 の補助の損失・S2M の F0 の条件の正解に使う)。

F0 は評価と同じ lib_common.prosody.f0(librosa の pyin、Mel と同じフレーム)。1 文に約 1 秒かかるので、CPU で並列に求め、
runs/f0.pt(発話 ID → (F0 [T](無声は 0), 有声か [T])、float16 で合計 10MB 程度)に保存して使い回す。

使い方:
    python extract_f0.py --workers 12
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

import common  # noqa: F401
from lib_common.audio import load_wave
from lib_common.data import list_jsut
from lib_common.prosody import f0

EXP = Path(__file__).resolve().parent


def work(path: str):
    f, v = f0(load_wave(path))
    return np.nan_to_num(f).astype(np.float16), v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    utts = list_jsut()
    with Pool(args.workers) as pool:
        res = pool.map(work, [u.path for u in utts], chunksize=16)
    out = {u.id: (torch.from_numpy(f), torch.from_numpy(v)) for u, (f, v) in zip(utts, res)}
    (EXP / "runs").mkdir(exist_ok=True)
    torch.save(out, EXP / "runs" / "f0.pt")
    print(f"saved {len(out)} utterances")


if __name__ == "__main__":
    main()
