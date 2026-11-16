"""手順 4: T2S を学習する(common/t2s.py。IndexTTS-2.5 の UnifiedVoice と同じ構成の自己回帰の GPT-2)。

テキストは common/text.py の公式の日本語の処理(MeCab で単語に区切る、漢字のまま)。学習のときは、漢字を含む単語の 2 割を
読み(平仮名)に置き換える(バッチを作るたびに選び直す。検証は置き換えなし)。正解の音声トークンは codec(train_codec.py)のもの。
学習の設定(公式の T2S の学習コード・設定は非公開なので exp0003 の自己回帰の T2S と同じ): AdamW(weight decay 0.01)、学習率 3e-4
(warmup 1,000 step + cosine)、8,000 step、1 バッチは「(テキスト + トークン)の最長 × 発話数」が 12,000 以下、勾配の大きさの上限 1.0、bf16。
dropout は GPT-2 の既定の 0.1(公式と同じ)。500 step ごとに検証 loss(音声トークンの交差エントロピー)を測り、一番小さい時点を保存する。

使い方:
    python train_t2s.py --codec codec --name t2s
    python train_t2s.py --codec codec --limit 30 --steps 4 --eval_every 2 --name smoke   # 動作確認
出力: runs/{name}/t2s.pt(重み・設定・語彙)と log.txt
"""
import argparse
import random

import torch

import common
from common.t2s import UnifiedVoice
from common.text import CharVocab, frontend
from lib_common.data import list_jsut, split
from lib_common.utils import Logger, length_batches, seed_everything, warmup_cosine

CONFIG = {"dim": 512, "layers": 13, "heads": 8}  # 53.7M(公式は 1,280 次元・24 層・20 ヘッド)
G2P_RATIO = 0.2  # 公式の JapaneseG2PProcessor の既定の割合


def load_t2s(path, device: str) -> tuple[UnifiedVoice, CharVocab]:
    ck = torch.load(path, map_location="cpu")
    vocab = CharVocab(ck["vocab"][len(CharVocab.SPECIALS):])
    model = UnifiedVoice(len(vocab), **ck["config"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec", required=True, help="runs/{codec}/tokens.pt")
    parser.add_argument("--dim", type=int, default=CONFIG["dim"])
    parser.add_argument("--layers", type=int, default=CONFIG["layers"])
    parser.add_argument("--heads", type=int, default=CONFIG["heads"])
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--max_tokens", type=int, default=12000, help="1 回の更新に使う「最長 × 発話数」の上限")
    parser.add_argument("--grad_accum", type=int, default=1, help="1 回の更新を何個の小さいバッチに分けるか(VRAM を減らす)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--name", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    seed_everything(args.seed)
    rng = random.Random(args.seed)
    torch.set_num_threads(4)
    out = common.RUNS_DIR / args.name
    log = Logger(out / "log.txt")
    log(f"args: {vars(args)}")
    tokens = torch.load(common.RUNS_DIR / args.codec / "tokens.pt")
    utts = list_jsut()
    train, val = split(utts, "train"), split(utts, "val")
    if args.limit:
        train, val = train[: args.limit], val[:3]
    vocab = CharVocab.build([u.text for u in train])
    log(f"train {len(train)} / val {len(val)} utterances, vocab {len(vocab)}")
    codes = {u.id: tokens[u.id].long() for u in train + val}
    val_text = [torch.tensor(vocab.encode(frontend(u.text))) for u in val]
    lengths = [len(vocab.encode(frontend(u.text))) + len(codes[u.id]) for u in train]
    n_codes = torch.load(common.RUNS_DIR / args.codec / "codec.pt", map_location="cpu")["config"]["codebook_size"]
    config = {"dim": args.dim, "layers": args.layers, "heads": args.heads, "n_codes": n_codes}  # 音声トークンの種類数は codec に合わせる(公式の number_mel_codes = 種類数 + 2)
    model = UnifiedVoice(len(vocab), **config).to(args.device)
    log(f"t2s params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = warmup_cosine(opt, args.steps)

    @torch.no_grad()
    def evaluate() -> dict:
        model.eval()
        tot, n = {}, 0
        for b in length_batches([len(t) + len(codes[u.id]) for t, u in zip(val_text, val)], args.max_tokens, shuffle=False):
            _, parts = model([val_text[i] for i in b], [codes[val[i].id] for i in b])
            for k, v in parts.items():
                tot[k] = tot.get(k, 0) + v * len(b)
            n += len(b)
        model.train()
        return {k: v / n for k, v in tot.items()}

    best, step, epoch, running = float("inf"), 0, 0, {}
    while step < args.steps:
        micro, pending = 0, {}
        opt.zero_grad()  # 前の周回の最後に余った小さいバッチ(grad_accum 個に足りない分)の勾配は捨てる
        for b in length_batches(lengths, args.max_tokens // args.grad_accum, shuffle=True, seed=epoch):
            text = [torch.tensor(vocab.encode(frontend(train[i].text, G2P_RATIO, rng))) for i in b]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                loss, parts = model(text, [codes[train[i].id] for i in b])
            (loss / args.grad_accum).backward()                    # grad_accum 個の小さいバッチの勾配をためてから 1 回更新する
            for k, v in {"loss": loss.item(), **parts}.items():
                pending[k] = pending.get(k, 0) + v / args.grad_accum
            micro += 1
            if micro % args.grad_accum:
                continue
            for k, v in pending.items():                           # 更新に使った分だけをログの値に数える
                running[k] = running.get(k, 0) + v
            pending = {}
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            sched.step()
            step += 1
            if step == 20 and args.device == "cuda":
                log(f"peak VRAM {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB(最初の 20 step)")
            if step % args.eval_every == 0 or step == args.steps:
                n = args.eval_every if step % args.eval_every == 0 else step % args.eval_every
                torch.cuda.empty_cache()
                ev = evaluate()
                torch.cuda.empty_cache()
                mark = ""
                if ev["mel_ce"] < best:
                    best, mark = ev["mel_ce"], " *saved"
                    torch.save({"model": model.state_dict(), "config": config, "vocab": vocab.itos, "step": step}, out / "t2s.pt")
                log(f"step {step:6d} | train " + " ".join(f"{k} {v / n:.4f}" for k, v in running.items())
                    + " | val " + " ".join(f"{k} {v:.4f}" for k, v in ev.items())
                    + (f" | reserved {torch.cuda.max_memory_reserved() / 2 ** 30:.1f}GiB" if args.device == "cuda" else "") + mark)
                running = {}
            if step >= args.steps:
                break
        epoch += 1
    log(f"best val mel_ce {best:.4f}")


if __name__ == "__main__":
    main()
