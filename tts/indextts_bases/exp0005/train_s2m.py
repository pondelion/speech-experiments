"""S2M に F0 の条件を足して、追加で学習する(exp0005 の対策 A)。

元にするのは exp0003 の日本語 HuBERT の S2M(`s2m_hubert_ja_k256`)。common/s2m_f0.py の F0Condition(重みは 0 から)を足し、
exp0001 の F3 と同じ追加学習の設定(10,000 step、学習率 1e-4、1,000 step ごとに検証して検証 loss が一番小さい時点を保存)で学習する。
F0 の条件は正解の F0(runs/f0.pt。T2S の補助の損失と同じ、各トークンの区間の有声・無声と正規化した log F0)。

使い方:
    python train_s2m.py --name s2m_f0cond
    python train_s2m.py --limit 30 --steps 2 --eval_every 1 --name smoke   # 動作確認
出力: runs/{name}/s2m.pt(S2M と F0Condition の重み、F0 の正規化の値)と log.txt
"""
import argparse
from pathlib import Path

import torch
from einops import rearrange

import common  # noqa: F401
from common.s2m_f0 import FPT, F0Condition
from exp0001.common.pipeline import load_s2m
from exp0001.common.utils import load_codec, pad_mels
from exp0003.common.utils import tokenize
from lib_common.data import load_dataset, split
from lib_common.utils import Logger, length_batches, seed_everything, warmup_cosine
from train_t2s import f0_targets

EXP = Path(__file__).resolve().parent
E3 = EXP.parent / "exp0003" / "runs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_frames", type=int, default=16000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--name", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(0)
    torch.set_num_threads(4)
    out = EXP / "runs" / args.name
    log = Logger(out / "log.txt")
    log(f"args: {vars(args)}")
    utts = load_dataset(args.limit, log)
    train, val = split(utts, "train"), split(utts, "val")
    codec_path = E3 / "codec_hubert_ja_k256" / "codec.pt"
    codes, _, _ = tokenize(codec_path, train + val, args.device, log)
    codec, mel_norm = load_codec(codec_path, args.device)
    f0 = torch.load(EXP / "runs" / "f0.pt")
    stats = torch.load(EXP / "runs" / "t2s_f0aux" / "t2s.pt", map_location="cpu")["f0_stats"]  # T2S と同じ正規化
    data = {}
    for name, us, cs in [("train", train, codes[: len(train)]), ("val", val, codes[len(train):])]:
        mels = [mel_norm.normalize(u.mel)[: len(c) * FPT] for u, c in zip(us, cs)]
        data[name] = (mels, cs, f0_targets(us, cs, f0, stats)[0])

    s2m = load_s2m(E3 / "s2m_hubert_ja_k256" / "s2m.pt", args.device).train()
    f0cond = F0Condition(codec.hidden_dim).to(args.device)
    params = list(s2m.parameters()) + list(f0cond.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)
    sched = warmup_cosine(opt, args.steps)

    def batch(d, b):
        mel, mask = pad_mels([d[0][i] for i in b], args.device)
        ids = torch.nn.utils.rnn.pad_sequence([d[1][i] for i in b], batch_first=True).to(args.device)
        f = torch.nn.utils.rnn.pad_sequence([d[2][i] for i in b], batch_first=True).to(args.device)
        with torch.no_grad():
            hidden = codec.decode_hidden(ids)                                                   # codec は固定
        return mel, mask, hidden + f0cond(f)

    @torch.no_grad()
    def evaluate(d) -> float:  # exp0001 の S2M と同じ検証 loss(ノイズと時刻を固定した速度の二乗誤差)
        s2m.eval()
        g = torch.Generator(device=args.device).manual_seed(0)
        total, n = 0.0, 0
        for b in length_batches([len(m) for m in d[0]], args.max_frames, shuffle=False):
            mel, mask, hidden = batch(d, b)
            x0 = torch.randn(mel.shape, device=args.device, generator=g)
            t = torch.rand(len(b), device=args.device, generator=g)
            xt = (1 - rearrange(t, "b -> b 1 1")) * x0 + rearrange(t, "b -> b 1 1") * mel
            v = s2m.velocity(xt, s2m.regulator(hidden, mel.shape[1]), t, ~mask)
            err = ((v - (mel - x0)) ** 2).mean(-1)
            total, n = total + ((err * mask).sum() / mask.sum()).item() * len(b), n + len(b)
        s2m.train()
        return total / n

    best = evaluate(data["val"])
    log(f"step 0 (元の S2M) | val loss {best:.4f}")
    torch.save({"s2m": s2m.state_dict(), "f0cond": f0cond.state_dict(), "f0_stats": stats, "step": 0}, out / "s2m.pt")
    tr = data["train"]
    step, epoch, running = 0, 0, 0.0
    while step < args.steps:
        for b in length_batches([len(m) for m in tr[0]], args.max_frames, shuffle=True, seed=epoch):
            mel, mask, hidden = batch(tr, b)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                loss = s2m(mel, hidden, mask)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            running += loss.item()
            if step % args.eval_every == 0 or step == args.steps:
                n = args.eval_every if step % args.eval_every == 0 else step % args.eval_every
                torch.cuda.empty_cache()
                val = evaluate(data["val"])
                torch.cuda.empty_cache()
                mark = ""
                if val < best:
                    best, mark = val, " *saved"
                    torch.save({"s2m": s2m.state_dict(), "f0cond": f0cond.state_dict(), "f0_stats": stats, "step": step},
                               out / "s2m.pt")
                log(f"step {step:6d} | train loss {running / n:.4f} | val loss {val:.4f}{mark}")
                running = 0.0
            if step >= args.steps:
                break
        epoch += 1
    log(f"best val loss {best:.4f}, checkpoint {out / 's2m.pt'}")


if __name__ == "__main__":
    main()
