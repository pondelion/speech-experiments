"""手順 2: semantic codec を学習し、全発話を音声トークンにする(common/codec.py。IndexTTS-2.5 の EnhancedCodec と同じ構成)。

学習の設定(公式の codec の学習コード・設定は非公開なので exp0003 の codec と同じ): AdamW、学習率 3e-4(warmup 1,000 step + cosine)、
30,000 step、1 バッチは「最長のフレーム数 × 発話数」が 16,000 以下(50 フレーム/秒で約 5.3 分)、勾配の大きさの上限 1.0、bf16。
1,000 step ごとに検証データの復元誤差・使われたトークンの種類数を測り、復元誤差が一番小さい時点を保存する。
特徴は extract_w2v.py の値を、学習データの平均・標準偏差で正規化してから入れる(公式の推論と同じ)。フレーム数は偶数に切り捨てる。

使い方:
    python train_codec.py --name codec
    python train_codec.py --feats w2v_smoke --steps 4 --eval_every 2 --name smoke   # 動作確認
出力: runs/{name}/codec.pt(重み・設定・特徴の平均と標準偏差)、runs/{name}/tokens.pt(全発話の音声トークン)、log.txt
"""
import argparse
import math

import torch

import common
from common.codec import SemanticCodec
from lib_common.data import list_jsut, split
from lib_common.utils import Logger, length_batches, seed_everything, warmup_cosine

CONFIG = {"dim": 192, "intermediate_dim": 1024, "num_layers": 12, "codebook_size": 8192}  # 19.0M(公式の比 5.33 倍の中間の幅)


def pad(xs: list[torch.Tensor], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """[T_i, D] のリスト → [B, T, D] と mask [B, T]。"""
    x = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True).to(device)
    mask = torch.arange(x.shape[1], device=device)[None] < torch.tensor([len(a) for a in xs], device=device)[:, None]
    return x, mask


def load_codec(path, device: str) -> tuple[SemanticCodec, torch.Tensor, torch.Tensor]:
    """codec.pt → (codec, 特徴の平均, 標準偏差)。"""
    ck = torch.load(path, map_location="cpu")
    codec = SemanticCodec(**ck["config"])
    codec.load_state_dict(ck["model"])
    return codec.to(device).eval(), ck["mean"].to(device), ck["std"].to(device)


@torch.no_grad()
def evaluate(model: SemanticCodec, xs: list[torch.Tensor], device: str, max_frames: int) -> dict:
    model.eval()
    counts = torch.zeros(model.codebook_size)
    recon, n = 0.0, 0
    for b in length_batches([len(x) for x in xs], max_frames, shuffle=False):
        x, mask = pad([xs[i] for i in b], device)
        _, parts, ids = model(x, mask)
        recon, n = recon + parts["recon"] * len(b), n + len(b)
        counts += torch.bincount(ids[mask[:, ::2]].cpu(), minlength=model.codebook_size)
    p = counts / counts.sum()
    model.train()
    return {"recon": recon / n, "used": int((counts > 0).sum()), "perplexity": math.exp(-(p[p > 0] * p[p > 0].log()).sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feats", default="w2v_l17", help="runs/{feats}.pt(extract_w2v.py の出力)")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--max_frames", type=int, default=16000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    seed_everything(args.seed)
    torch.set_num_threads(4)
    out = common.RUNS_DIR / args.name
    log = Logger(out / "log.txt")
    log(f"args: {vars(args)}")
    d = torch.load(common.RUNS_DIR / f"{args.feats}.pt")
    mean, std = d["mean"], d["var"].sqrt()
    utts = [u for u in list_jsut() if u.id in d["feats"]]

    def inputs(us):  # 正規化して、フレーム数を偶数に切り捨てる
        return [((d["feats"][u.id].float() - mean) / std)[: len(d["feats"][u.id]) // 2 * 2] for u in us]

    train, val = inputs(split(utts, "train")), inputs(split(utts, "val"))
    log(f"train {len(train)} / val {len(val)} utterances")
    model = SemanticCodec(**CONFIG).to(args.device)
    log(f"codec params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = warmup_cosine(opt, args.steps)
    best, step, epoch, running = float("inf"), 0, 0, {}
    while step < args.steps:
        for b in length_batches([len(x) for x in train], args.max_frames, shuffle=True, seed=epoch):
            x, mask = pad([train[i] for i in b], args.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                loss, parts, _ = model(x, mask)
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
                torch.cuda.empty_cache()
                ev = evaluate(model, val, args.device, args.max_frames)
                torch.cuda.empty_cache()
                mark = ""
                if ev["recon"] < best:
                    best, mark = ev["recon"], " *saved"
                    torch.save({"model": model.state_dict(), "config": CONFIG, "mean": mean, "std": std, "step": step},
                               out / "codec.pt")
                log(f"step {step:6d} | train " + " ".join(f"{k} {v / n:.4f}" for k, v in running.items())
                    + f" | val recon {ev['recon']:.4f} used {ev['used']}/{model.codebook_size} perplexity {ev['perplexity']:.0f}{mark}")
                running = {}
            if step >= args.steps:
                break
        epoch += 1

    codec, mean, std = load_codec(out / "codec.pt", args.device)  # 一番良い時点で、全発話をトークンにする
    tokens = {}
    with torch.no_grad():
        for u in utts:
            x = ((d["feats"][u.id].float().to(args.device) - mean) / std)[: len(d["feats"][u.id]) // 2 * 2]
            tokens[u.id] = codec.quantize(x[None])[0].short().cpu()
    torch.save(tokens, out / "tokens.pt")
    log(f"best val recon {best:.4f}; saved tokens of {len(tokens)} utterances")


if __name__ == "__main__":
    main()
