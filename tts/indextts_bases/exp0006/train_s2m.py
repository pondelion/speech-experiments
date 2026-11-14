"""手順 3: S2M を学習する(common/s2m.py。IndexTTS-2.5 の length regulator + DiT の Flow Matching と同じ構成)。

入力は codec のトークンを codec の decoder に通した特徴(公式の推論の semantic_codec.decode と同じ。codec は固定)、
正解は 22.05kHz の log-Mel(正規化しない)。参照の Mel・話者ベクトルは入れない(フェーズ A)。
学習の設定(公式の S2M の学習コード・設定は非公開なので exp0003 の S2M と同じ): AdamW、学習率 2e-4(warmup 1,000 step + cosine)、
40,000 step、1 バッチは「最長の Mel のフレーム数 × 発話数」が 16,000 以下、勾配の大きさの上限 1.0、bf16。
1,000 step ごとに検証 loss(ノイズと時刻を固定)を測り、一番小さい時点を保存する。

使い方:
    python train_s2m.py --codec codec --name s2m
    python train_s2m.py --codec codec --limit 30 --steps 4 --eval_every 2 --name smoke   # 動作確認
出力: runs/{name}/s2m.pt(重み・設定)と log.txt
"""
import argparse

import torch

import common
from common.s2m import S2M
from lib_common.data import load_dataset, split
from lib_common.utils import Logger, length_batches, seed_everything, warmup_cosine
from train_codec import load_codec, pad

CONFIG = {"dim": 320, "depth": 13, "heads": 5, "wavenet_layers": 8}  # 39.2M(公式は 512 次元・8 ヘッドで 97.8M)


def load_s2m(path, device: str) -> S2M:
    ck = torch.load(path, map_location="cpu")
    model = S2M(**ck["config"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec", required=True, help="runs/{codec}/(codec.pt と tokens.pt)")
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--max_frames", type=int, default=16000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--name", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    seed_everything(args.seed)
    torch.set_num_threads(4)
    out = common.RUNS_DIR / args.name
    log = Logger(out / "log.txt")
    log(f"args: {vars(args)}")
    utts = load_dataset(args.limit, log)
    codec, _, _ = load_codec(common.RUNS_DIR / args.codec / "codec.pt", args.device)
    tokens = torch.load(common.RUNS_DIR / args.codec / "tokens.pt")
    data = {name: [(u.mel, tokens[u.id].long()) for u in split(utts, name)] for name in ("train", "val")}
    model = S2M(**CONFIG).to(args.device)
    log(f"s2m params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = warmup_cosine(opt, args.steps)

    def batch(items):
        mel, mask = pad([m for m, _ in items], args.device)                       # [B, T, F]、[B, T]
        ids = torch.nn.utils.rnn.pad_sequence([c for _, c in items], batch_first=True).to(args.device)
        with torch.no_grad():
            feat = codec.decode(ids)                                              # [B, 2U, 1024]
        return mel, feat, mask

    @torch.no_grad()
    def evaluate() -> float:
        model.eval()
        total, n = 0.0, 0
        with torch.random.fork_rng(devices=[args.device] if args.device == "cuda" else []):
            torch.manual_seed(0)
            for b in length_batches([len(m) for m, _ in data["val"]], args.max_frames, shuffle=False):
                mel, feat, mask = batch([data["val"][i] for i in b])
                total, n = total + model(mel, feat, mask).item() * len(b), n + len(b)
        model.train()
        return total / n

    best, step, epoch, running = float("inf"), 0, 0, 0.0
    tr = data["train"]
    while step < args.steps:
        for b in length_batches([len(m) for m, _ in tr], args.max_frames, shuffle=True, seed=epoch):
            mel, feat, mask = batch([tr[i] for i in b])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                loss = model(mel, feat, mask)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            running += loss.item()
            if step % args.eval_every == 0 or step == args.steps:
                n = args.eval_every if step % args.eval_every == 0 else step % args.eval_every
                torch.cuda.empty_cache()
                val = evaluate()
                torch.cuda.empty_cache()
                mark = ""
                if val < best:
                    best, mark = val, " *saved"
                    torch.save({"model": model.state_dict(), "config": CONFIG, "step": step}, out / "s2m.pt")
                log(f"step {step:6d} | train loss {running / n:.4f} | val loss {val:.4f}{mark}")
                running = 0.0
            if step >= args.steps:
                break
        epoch += 1
    log(f"best val loss {best:.4f}")


if __name__ == "__main__":
    main()
