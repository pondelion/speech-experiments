"""手順 1: Mel を離散トークンにする codec を学習する。

codec は T2S・S2M の前提になる(T2S は codec が作るトークンを当てる、S2M はトークンから Mel を作る)ので最初に学習し、
以降は固定する。学習後に codec を変えると、T2S・S2M の学習データ(トークン)の意味が変わるため作り直しになる。

途中経過として、検証データで次を測る。
  recon: 復元 Mel と元の Mel の二乗誤差(正規化した Mel で測る。小さいほどよく復元できている)
  used: 検証データで実際に使われたトークンの種類数(codebook の一部しか使われない「崩壊」が起きていないか)
  perplexity: トークンの使われ方の偏り。exp(エントロピー)。全種類が均等に使われると codebook_size に一致する

使い方:
    python train_codec.py                   # 本番
    python train_codec.py --limit 50 --steps 20 --device cpu   # 動作確認
出力: runs/codec/ に codec.pt(Mel の正規化の値も含む)と log.txt
"""
import argparse
import math

import torch

import common  # noqa: F401  tts/lib_common を import できるようにする
from common.codec import MelCodec
from common.utils import RUNS_DIR, pad_mels
from lib_common.audio import MelNormalizer
from lib_common.data import load_dataset, split
from lib_common.utils import Logger, length_batches, seed_everything, warmup_cosine


@torch.no_grad()
def evaluate(model: MelCodec, mels: list[torch.Tensor], device: str, max_frames: int) -> dict:
    """検証データでの復元誤差とトークンの使用状況を測る。

    Args:
        model: codec。
        mels: 正規化した Mel のリスト。
        device: デバイス。
        max_frames: 1 バッチのフレーム数の上限。

    Returns:
        {"recon": 平均復元誤差, "used": 使われたトークンの種類数, "perplexity": トークン分布の perplexity}。
        (以下のコメントの記号: B 発話数、T フレーム数、F Mel の帯域数 = 80、U トークン数 = T/4、K トークンの種類数)
    """
    model.eval()
    counts = torch.zeros(model.codebook_size)
    recon_sum, n = 0.0, 0
    for b in length_batches([len(m) for m in mels], max_frames, shuffle=False):
        mel, mask = pad_mels([mels[i] for i in b], device)  # [B, T, F]、[B, T]
        _, parts, ids, _ = model(mel, mask)                 # ids: [B, U]
        recon_sum += parts["recon"] * len(b)
        n += len(b)
        valid_ids = ids[mask[:, ::4]]                       # 詰め物以外のトークンを 1 列に [有効なトークン数]
        counts += torch.bincount(valid_ids.cpu(), minlength=model.codebook_size)  # 種類ごとの出現回数 [K]
    p = counts / counts.sum()
    entropy = -(p[p > 0] * p[p > 0].log()).sum().item()
    model.train()
    return {"recon": recon_sum / n, "used": int((counts > 0).sum()), "perplexity": math.exp(entropy)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--max_frames", type=int, default=16000, help="1 バッチの (最長フレーム数 × 発話数) の上限")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--codebook_size", type=int, default=1024)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=None, help="動作確認用に発話数を絞る")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="codec")
    args = parser.parse_args()

    seed_everything(args.seed)
    torch.set_num_threads(4)
    out = RUNS_DIR / args.out
    log = Logger(out / "log.txt")
    log(f"args: {vars(args)}")

    utts = load_dataset(args.limit, log)
    train, val = split(utts, "train"), split(utts, "val")
    norm = MelNormalizer.fit([u.mel for u in train])
    train_mels = [norm.normalize(u.mel) for u in train]
    val_mels = [norm.normalize(u.mel) for u in val]

    model = MelCodec(codebook_size=args.codebook_size).to(args.device)
    log(f"codec params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = warmup_cosine(opt, args.steps)

    step, epoch, running = 0, 0, {}
    while step < args.steps:
        for b in length_batches([len(m) for m in train_mels], args.max_frames, shuffle=True, seed=epoch):
            mel, mask = pad_mels([train_mels[i] for i in b], args.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                loss, parts, _, _ = model(mel, mask)
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
                train_str = " ".join(f"{k} {v / n:.4f}" for k, v in running.items())
                ev = evaluate(model, val_mels, args.device, args.max_frames)
                log(f"step {step:6d} | train {train_str} | val recon {ev['recon']:.4f} used {ev['used']}/"
                    f"{args.codebook_size} perplexity {ev['perplexity']:.0f}")
                running = {}
            if step >= args.steps:
                break
        epoch += 1

    torch.save({"model": model.state_dict(), "config": {"codebook_size": args.codebook_size},
                "normalizer": norm.state_dict()}, out / "codec.pt")
    log(f"saved {out / 'codec.pt'}")


if __name__ == "__main__":
    main()
