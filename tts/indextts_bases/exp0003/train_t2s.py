"""手順 3: SSL の codec のトークンを予測する T2S を学習する。

T2S は exp0002 で一番良かったハイブリッド型(長さを一度に予測 + MaskGCT 系、exp0002 の common/t2s_hybrid.py)で、
設定も exp0002 の `t2s_hybrid_nar` と同じ(8,000 step、dropout 0.3、500 step ごとに検証して検証 loss が一番小さい時点を保存)。
音声トークンだけを SSL の codec のものに変える。音素ごとの長さの正解は exp0002 と同じく jsut-lab から作る
(トークン 1 個の長さは codec から受け取る。SSL の codec は exp0001・exp0002 と同じ 42.7ms、外部のコーデックは
1 / (1 秒あたりのトークン数))。--codec ext:{名前} で外部のコーデックのトークンにする。

使い方:
    python train_t2s.py --codec runs/codec_w2vbert2_k256/codec.pt --name t2s_w2vbert2_k256
    python train_t2s.py --codec runs/smoke/codec.pt --limit 60 --steps 4 --eval_every 2 --device cpu --name smoke_t2s   # 動作確認
出力: runs/{name}/t2s.pt(exp0002 の load_t2s で読める形)と log.txt
"""
import argparse

import torch

import common  # noqa: F401  tts/lib_common・exp0001・exp0002 を import できるようにする
from common.utils import RUNS_DIR, code_vectors, tokenize
from exp0002.common.t2s_hybrid import HybridT2S
from exp0002.common.utils import prepare
from lib_common.data import load_dataset, split
from lib_common.text import TextVocab
from lib_common.utils import Logger, length_batches, seed_everything, warmup_cosine


def run(model, d: dict, idx: list[int]):
    """発話 idx のバッチで損失を計算する。"""
    return model([d["texts"][i] for i in idx], [d["codes"][i] for i in idx], [d["durations"][i] for i in idx])


@torch.no_grad()
def evaluate(model, d: dict, max_tokens: int) -> dict:
    """検証データでの loss と正解率(隠し方の乱数を固定する。exp0002 と同じ)。"""
    model.eval()
    rng = torch.get_rng_state()
    torch.manual_seed(0)
    sums, n = {}, 0
    for b in length_batches([len(t) + len(c) for t, c in zip(d["texts"], d["codes"])], max_tokens, shuffle=False):
        loss, parts = run(model, d, b)
        for k, v in {"loss": loss.item(), **parts}.items():
            sums[k] = sums.get(k, 0.0) + v * len(b)
        n += len(b)
    torch.set_rng_state(rng)
    model.train()
    return {k: v / n for k, v in sums.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec", required=True, help="train_codec.py の codec.pt")
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--max_tokens", type=int, default=12000, help="1 バッチの (最長系列長 × 発話数) の上限")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--fixed_code_emb", action="store_true",
                        help="音声トークンの入力の埋め込みを codec の固定の代表ベクトル + 学習する射影にする(ablation)")
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
    codes, codec, _ = tokenize(args.codec, train + val, args.device, log)
    vocab = TextVocab.build("phoneme", [u.text for u in train])
    data = {"train": prepare(train, vocab, None, None, args.device, True, codes[: len(train)], codec.token_sec),
            "val": prepare(val, vocab, None, None, args.device, True, codes[len(train):], codec.token_sec)}
    tr = data["train"]
    log(f"train {len(tr['texts'])} utts (dropped {tr['n_dropped']}) | val {len(data['val']['texts'])} utts")

    config = {"dim": 512, "layers": 8, "heads": 8, "dropout": args.dropout, "text_drop": 0.1, "duration_model": "nar"}
    vecs = code_vectors(args.codec, args.device) if args.fixed_code_emb else None
    if vecs is not None:  # ablation: 音声トークンの入力を codec の固定の代表ベクトル + 学習する射影にする
        config["fixed_code_dim"] = vecs.shape[1]
    model = HybridT2S(len(vocab), codec.codebook_size, **config).to(args.device)
    if vecs is not None:
        model.load_code_vectors(vecs)
    log(f"t2s params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = warmup_cosine(opt, args.steps)
    lengths = [len(t) + len(c) for t, c in zip(tr["texts"], tr["codes"])]
    step, epoch, best, running = 0, 0, float("inf"), {}
    while step < args.steps:
        for b in length_batches(lengths, args.max_tokens, shuffle=True, seed=epoch):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                loss, parts = run(model, tr, b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            for k, v in {"loss": loss.item(), **parts}.items():
                running[k] = running.get(k, 0.0) + v
            if step % args.eval_every == 0 or step == args.steps:
                n = args.eval_every if step % args.eval_every == 0 else step % args.eval_every
                torch.cuda.empty_cache()
                ev = evaluate(model, data["val"], args.max_tokens)
                torch.cuda.empty_cache()
                mark = ""
                if ev["loss"] < best:
                    best, mark = ev["loss"], " *saved"
                    torch.save({"model": model.state_dict(), "config": config, "method": "hybrid",
                                "vocab": vocab.state_dict(), "n_codes": codec.codebook_size, "step": step}, out / "t2s.pt")
                log(f"step {step:6d} | train " + " ".join(f"{k} {v / n:.4f}" for k, v in running.items())
                    + " | val " + " ".join(f"{k} {v:.4f}" for k, v in ev.items()) + mark)
                running = {}
            if step >= args.steps:
                break
        epoch += 1
    log(f"best val loss {best:.4f}, checkpoint {out / 't2s.pt'}")


if __name__ == "__main__":
    main()
