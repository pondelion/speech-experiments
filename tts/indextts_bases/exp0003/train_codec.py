"""手順 1: SSL の特徴を量子化する codec を学習する(IndexTTS2.5 の semantic codec にあたる)。

構造・損失は exp0001 の Mel の codec(common/codec.py の MelCodec)と同じで、入力・復元する特徴を Mel から SSL の特徴に変える。
SSL の特徴は Mel のフレーム数に補間してから入れるので、トークン 1 個 = Mel 4 フレーム(42.7ms)は exp0001 と同じ。

使い方:
    python train_codec.py --ssl w2vbert2 --name codec_w2vbert2_k256
    python train_codec.py --ssl hubert_ja --name codec_hubert_ja_k256
    python train_codec.py --ssl hubert_ja --limit 20 --steps 4 --eval_every 2 --device cpu --name smoke   # 動作確認
出力: runs/{name}/codec.pt(codec、Mel と SSL の特徴の正規化の値、SSL の名前と層)と log.txt
"""
import argparse
import math
import time

import torch

import common  # noqa: F401  tts/lib_common と exp0001 を import できるようにする
from common.ssl import SSL_MODELS, SSLFeatures
from common.utils import RUNS_DIR, codec_inputs, extract_all, fit_normalizer
from exp0001.common.codec import DOWNSAMPLE, MelCodec
from exp0001.common.utils import pad_mels
from lib_common.audio import MelNormalizer
from lib_common.data import load_dataset, split
from lib_common.utils import Logger, length_batches, seed_everything, warmup_cosine


@torch.no_grad()
def evaluate(model: MelCodec, xs: list[torch.Tensor], device: str, max_frames: int) -> dict:
    """検証データでの復元誤差と、トークンの使用状況(使われた種類数・perplexity)。"""
    model.eval()
    counts = torch.zeros(model.codebook_size)
    recon, n = 0.0, 0
    for b in length_batches([len(x) for x in xs], max_frames, shuffle=False):
        x, mask = pad_mels([xs[i] for i in b], device)                   # [B, T, D]、[B, T]
        _, parts, ids, _ = model(x, mask)                                # ids [B, U]
        recon, n = recon + parts["recon"] * len(b), n + len(b)
        counts += torch.bincount(ids[mask[:, ::DOWNSAMPLE]].cpu(), minlength=model.codebook_size)
    p = counts / counts.sum()
    model.train()
    return {"recon": recon / n, "used": int((counts > 0).sum()), "perplexity": math.exp(-(p[p > 0] * p[p > 0].log()).sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssl", choices=list(SSL_MODELS), required=True)
    parser.add_argument("--layer", type=int, default=None, help="使う層(既定は SSL_MODELS の値)")
    parser.add_argument("--codebook_size", type=int, default=256, help="exp0001 で 1024 より 256 が良かった")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--max_frames", type=int, default=16000, help="1 バッチの (最長フレーム数 × 発話数) の上限")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--name", required=True, help="保存先 runs/{name}/")
    parser.add_argument("--limit", type=int, default=None, help="動作確認用に発話数を絞る")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    torch.set_num_threads(4)
    out = RUNS_DIR / args.name
    log = Logger(out / "log.txt")
    log(f"args: {vars(args)}")

    utts = load_dataset(args.limit, log)
    train, val = split(utts, "train"), split(utts, "val")
    t0 = time.time()
    ssl = SSLFeatures(args.ssl, args.device, args.layer)
    feats = {name: extract_all(ssl, us, log) for name, us in [("train", train), ("val", val)]}
    layer, dim = ssl.layer, ssl.dim
    del ssl
    torch.cuda.empty_cache()
    log(f"extracted {args.ssl} layer {layer} ({dim} dims) in {time.time() - t0:.0f}s")
    feat_norm = fit_normalizer(feats["train"])
    mel_norm = MelNormalizer.fit([u.mel for u in train])  # S2M の出力(Mel)の正規化。S2M・合成で使う
    val_x = codec_inputs(feats["val"], val, feat_norm)

    config = {"codebook_size": args.codebook_size, "in_dim": dim}
    model = MelCodec(**config).to(args.device)
    log(f"codec params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = warmup_cosine(opt, args.steps)
    lengths = [len(u.mel) // DOWNSAMPLE * DOWNSAMPLE for u in train]
    step, epoch, running = 0, 0, {}
    while step < args.steps:
        for b in length_batches(lengths, args.max_frames, shuffle=True, seed=epoch):
            x, mask = pad_mels(codec_inputs([feats["train"][i] for i in b], [train[i] for i in b], feat_norm), args.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                loss, parts, _, _ = model(x, mask)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            for k, v in {"loss": loss.item(), **parts}.items():
                running[k] = running.get(k, 0) + v
            if step % args.eval_every == 0 or step == args.steps:
                n = args.eval_every if step % args.eval_every == 0 else step % args.eval_every
                ev = evaluate(model, val_x, args.device, args.max_frames)
                log(f"step {step:6d} | train " + " ".join(f"{k} {v / n:.4f}" for k, v in running.items())
                    + f" | val recon {ev['recon']:.4f} used {ev['used']}/{args.codebook_size} perplexity {ev['perplexity']:.0f}")
                running = {}
            if step >= args.steps:
                break
        epoch += 1

    torch.save({"model": model.state_dict(), "config": config, "normalizer": mel_norm.state_dict(),
                "feat_normalizer": feat_norm.state_dict(), "ssl": args.ssl, "layer": layer}, out / "codec.pt")
    log(f"saved {out / 'codec.pt'}")


if __name__ == "__main__":
    main()
