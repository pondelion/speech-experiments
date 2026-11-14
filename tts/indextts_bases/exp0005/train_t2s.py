"""T2S(ハイブリッド型)を、各音声トークンの区間の F0 を当てる補助の損失をつけて学習する(exp0005 の対策 1)。

分析(結果 1)で、抑揚の崩れの大部分は T2S が選ぶトークンから来ていた。exp0001 で「今どの音素か」を当てる補助の損失が読みを
大きく改善したのと同じように、「このトークンの区間は有声か・声の高さはどれくらいか」を当てさせ、T2S の中に高さの情報を持たせる。
補助の出力は学習のときだけ使い、生成の手順は変えない。

それ以外は exp0003 の `t2s_hubert_ja_k256` と同じ設定(日本語 HuBERT のトークン、8,000 step、dropout 0.3、500 step ごとに検証して
検証 loss が一番小さい時点を保存)。検証 loss は補助の損失を含むので、補助の損失なしの学習とは比べられない(比べるのは評価の指標)。
F0 の正解は extract_f0.py が作った runs/f0.pt。各トークンの区間(Mel 4 フレーム)の半分以上が有声なら有声とし、
有声のフレームの log2 F0 の平均を、学習データ全体の平均・標準偏差で正規化したものを当てる。

使い方:
    python train_t2s.py --f0_weight 1.0 --name t2s_f0aux
    python train_t2s.py --f0_weight 1.0 --limit 60 --steps 4 --eval_every 2 --name smoke   # 動作確認
"""
import argparse
from pathlib import Path

import torch

import common  # noqa: F401
from exp0002.common.t2s_hybrid import HybridT2S
from exp0002.common.utils import prepare
from exp0003.common.utils import tokenize
from lib_common.data import load_dataset, split
from lib_common.text import TextVocab
from lib_common.utils import Logger, length_batches, seed_everything, warmup_cosine

EXP = Path(__file__).resolve().parent
CODEC = EXP.parent / "exp0003" / "runs" / "codec_hubert_ja_k256" / "codec.pt"
FPT = 4  # 音声トークン 1 個 = Mel 4 フレーム


def f0_targets(utts: list, codes: list, f0: dict, stats=None):
    """各音声トークンの区間の (有声か, 正規化した log2 F0) [U_i, 2] のリスト。stats は (平均, 標準偏差)(None なら求める)。"""
    raw = []
    for u, c in zip(utts, codes):
        f, v = f0[u.id]
        f, v = f.float(), v.float()
        n = len(c) * FPT
        f, v = torch.nn.functional.pad(f[:n], (0, max(0, n - len(f)))), torch.nn.functional.pad(v[:n], (0, max(0, n - len(v))))
        f, v = f.view(len(c), FPT), v.view(len(c), FPT)                                     # [U, 4]
        voiced = v.mean(1) >= 0.5
        lf0 = torch.where(v > 0, torch.log2(f.clamp(min=1)), torch.zeros_like(f)).sum(1) / v.sum(1).clamp(min=1)
        raw.append((voiced, lf0))
    if stats is None:
        allv = torch.cat([l[v] for v, l in raw])
        stats = (allv.mean().item(), allv.std().item())
    return [torch.stack([v.float(), torch.where(v, (l - stats[0]) / stats[1], torch.zeros_like(l))], 1) for v, l in raw], stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f0_weight", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--max_tokens", type=int, default=12000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--name", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    seed_everything(args.seed)
    torch.set_num_threads(4)
    out = EXP / "runs" / args.name
    log = Logger(out / "log.txt")
    log(f"args: {vars(args)}")
    utts = load_dataset(args.limit, log)
    train, val = split(utts, "train"), split(utts, "val")
    codes, codec, _ = tokenize(CODEC, train + val, args.device, log)
    vocab = TextVocab.build("phoneme", [u.text for u in train])
    f0 = torch.load(EXP / "runs" / "f0.pt")
    data, stats = {}, None
    for name, us, cs in [("train", train, codes[: len(train)]), ("val", val, codes[len(train):])]:
        d = prepare(us, vocab, None, None, args.device, True, cs, codec.token_sec)
        d["f0"], stats = f0_targets([us[i] for i in d["keep"]], d["codes"], f0, stats)
        data[name] = d
    log(f"train {len(data['train']['texts'])} utts | log2 F0 mean {stats[0]:.3f} std {stats[1]:.3f}")

    config = {"dim": 512, "layers": 8, "heads": 8, "dropout": 0.3, "text_drop": 0.1, "duration_model": "nar",
              "f0_weight": args.f0_weight}
    model = HybridT2S(len(vocab), codec.codebook_size, **config).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = warmup_cosine(opt, args.steps)
    run = lambda d, b: model(*[[d[k][i] for i in b] for k in ("texts", "codes", "durations", "f0")])  # noqa: E731

    @torch.no_grad()
    def evaluate(d):
        model.eval()
        rng = torch.get_rng_state()
        torch.manual_seed(0)
        sums, n = {}, 0
        for b in length_batches([len(t) + len(c) for t, c in zip(d["texts"], d["codes"])], args.max_tokens, shuffle=False):
            loss, parts = run(d, b)
            for k, v in {"loss": loss.item(), **parts}.items():
                sums[k] = sums.get(k, 0.0) + v * len(b)
            n += len(b)
        torch.set_rng_state(rng)
        model.train()
        return {k: v / n for k, v in sums.items()}

    tr = data["train"]
    lengths = [len(t) + len(c) for t, c in zip(tr["texts"], tr["codes"])]
    step, epoch, best, running = 0, 0, float("inf"), {}
    while step < args.steps:
        for b in length_batches(lengths, args.max_tokens, shuffle=True, seed=epoch):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                loss, parts = run(tr, b)
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
                ev = evaluate(data["val"])
                torch.cuda.empty_cache()
                mark = ""
                if ev["loss"] < best:
                    best, mark = ev["loss"], " *saved"
                    torch.save({"model": model.state_dict(), "config": config, "method": "hybrid", "vocab": vocab.state_dict(),
                                "n_codes": codec.codebook_size, "step": step, "f0_stats": stats}, out / "t2s.pt")
                log(f"step {step:6d} | train " + " ".join(f"{k} {v / n:.4f}" for k, v in running.items())
                    + " | val " + " ".join(f"{k} {v:.4f}" for k, v in ev.items()) + mark)
                running = {}
            if step >= args.steps:
                break
        epoch += 1
    log(f"best val loss {best:.4f}, checkpoint {out / 't2s.pt'}")


if __name__ == "__main__":
    main()
