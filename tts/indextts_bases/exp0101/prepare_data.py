"""手順 1: JSUT + JVS の全発話(common/data.py の一覧)の、学習・評価に使うものを 1 回だけ作って runs/ に保存する。

段階(--stage で 1 つずつ。F0 は CPU の並列処理を使うので、GPU を使う段階とは別のプロセスで流す):
    mel     Mel(フェーズ A と同じ 24kHz の log-Mel、float16)→ runs/mels.pt
    tokens  音声トークン(フェーズ A と同じ日本語 HuBERT の 9 層目・256 種類の codec。exp0003 の tokenize と同じ手順)→ runs/tokens.pt
            (exp0003 の tokens.pt には書き足さない)
    f0      F0(フェーズ A と同じ lib_common.prosody.f0)→ runs/f0.pt
    spk     話者ベクトル(CAM++。公式と同じく 16kHz の Kaldi 形式の 80 帯域の fbank から時間方向の平均を引いて入れる)→ runs/spk.pt
音素ごとの長さの正解は、ラベルから学習のときに作る(common/data.py の phoneme_durations)。

使い方:
    python prepare_data.py --stage mel
    python prepare_data.py --stage tokens
    python prepare_data.py --stage f0 --workers 8
    python prepare_data.py --stage spk
    python prepare_data.py --stage mel --limit 50 --out_suffix _smoke   # 動作確認(先頭 50 発話)
"""
import argparse
import time
from multiprocessing import Pool

import numpy as np
import torch

import common
from common.data import list_all
from lib_common.audio import load_wave, wave_to_logmel
from lib_common.utils import Logger

E3 =common.EXP_DIR.parent / "exp0003" / "runs"


def f0_work(path: str):
    from lib_common.prosody import f0
    f, v = f0(load_wave(path))
    return np.nan_to_num(f).astype(np.float16), v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["mel", "tokens", "f0", "spk"], required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out_suffix", default="")
    args = parser.parse_args()
    torch.set_num_threads(4)
    log = Logger(common.RUNS_DIR / "log_prepare.txt")
    utts = list_all()[: args.limit]
    out_path = common.RUNS_DIR / f"{args.stage if args.stage != 'mel' else 'mels'}{args.out_suffix}.pt"
    t0 = time.time()
    if args.stage == "mel":
        out = {}
        for i, u in enumerate(utts):
            out[u.id] = wave_to_logmel(load_wave(u.path)).half()
            if (i + 1) % 2000 == 0:
                log(f"  mel {i + 1}/{len(utts)} ({time.time() - t0:.0f}s)")
    elif args.stage == "tokens":
        from exp0001.common.utils import encode_all
        from exp0003.common.ssl import SSLFeatures
        from exp0003.common.utils import codec_inputs, load_ssl_codec

        mels = torch.load(common.RUNS_DIR / f"mels{args.out_suffix}.pt")
        for u in utts:
            u.mel = mels[u.id].float()
        model, _, feat_norm, info = load_ssl_codec(E3 / "codec_hubert_ja_k256" / "codec.pt", "cuda")
        enc = SSLFeatures(info["ssl"], "cuda", info["layer"])
        out = {}
        for s in range(0, len(utts), 256):  # exp0003 の tokenize と同じく 256 発話ずつ
            us = utts[s: s + 256]
            out.update(zip([u.id for u in us], [c.short() for c in
                                                encode_all(model, codec_inputs([enc(u.path) for u in us], us, feat_norm), "cuda")]))
            log(f"  tokens {min(s + 256, len(utts))}/{len(utts)} ({time.time() - t0:.0f}s) "
                f"peak VRAM {torch.cuda.max_memory_allocated() / 2 ** 30:.1f}GiB")
    elif args.stage == "f0":
        with Pool(args.workers) as pool:
            res = pool.map(f0_work, [u.path for u in utts], chunksize=16)
        out = {u.id: (torch.from_numpy(f), torch.from_numpy(v)) for u, (f, v) in zip(utts, res)}
    else:
        from common.speaker import CAMPlus
        cam = CAMPlus("cpu")
        out = {}
        for i, u in enumerate(utts):
            out[u.id] = cam(u.path)
            if (i + 1) % 2000 == 0:
                log(f"  spk {i + 1}/{len(utts)} ({time.time() - t0:.0f}s)")
    torch.save(out, out_path)
    log(f"{args.stage}: saved {len(out)} utterances to {out_path.name} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
