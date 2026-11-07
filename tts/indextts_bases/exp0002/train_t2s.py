"""非自己回帰の T2S を学習する(codec・S2M は exp0001 の学習済みのものを使い回す)。

--method で方式を選ぶ:
  duration: FastSpeech2 系。テキストの各トークンの長さを予測し、全ての音声トークンを一度に予測する
            (正解の長さは jsut-lab の音素の時刻から作る。pyopenjtalk と音素の並びが合わない約 5% の発話は使わない)
  masked:   MaskGCT 系。音声トークンの一部を隠して当てることを学び、生成では数回に分けて埋める
テキストの入力は音素 + アクセント記号(exp0001 で文字入力より良かった)。

途中経過として、検証データで次を測る(どちらも正解の長さ・正解のトークンを使った値で、実際に読めるかは evaluate.py の CER で見る)。
  loss: 学習の損失(方式ごとに意味が違うので、方式どうしでは比べられない)
  acc: duration は全位置のトークンの正解率、masked は隠した位置のトークンの正解率
  dur_loss: duration の長さの損失

使い方:
    python train_t2s.py --method duration --name t2s_duration
    python train_t2s.py --method masked --name t2s_masked
    python train_t2s.py --method masked --limit 50 --steps 20 --eval_every 10 --name smoke   # 動作確認
出力: runs/{name}/ に t2s.pt(検証 loss が一番小さかった時点)と log.txt
"""
import argparse

import torch

import common  # noqa: F401  tts/lib_common と exp0001 を import できるようにする
from common.t2s_duration import DurationT2S
from common.t2s_masked import MaskedT2S
from common.utils import BASE_CODEC, PROSODY_MARKS, RUNS_DIR, prepare
from exp0001.common.utils import load_codec
from lib_common.data import load_dataset, split
from lib_common.text import TextVocab
from lib_common.utils import Logger, length_batches, seed_everything, warmup_cosine


def run(model, d: dict, idx: list[int]):
    """発話 idx のバッチで損失を計算する(方式によって渡す正解が違う)。"""
    args = [[d["texts"][i] for i in idx], [d["codes"][i] for i in idx]]
    if d["durations"] is not None:
        args.append([d["durations"][i] for i in idx])
    return model(*args)


@torch.no_grad()
def evaluate(model, d: dict, max_tokens: int) -> dict:
    """検証データでの loss と正解率(発話数で重み付けした平均)。masked は隠し方が乱数なので、乱数を固定する。"""
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
    parser.add_argument("--method", choices=["duration", "masked"], required=True)
    parser.add_argument("--codec", default=str(BASE_CODEC))
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--max_tokens", type=int, default=12000, help="1 バッチの (最長系列長 × 発話数) の上限")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--dim", type=int, default=512, help="隠れ層の次元")
    parser.add_argument("--layers", type=int, default=8, help="Transformer の層数(duration は encoder・decoder に半分ずつ)")
    parser.add_argument("--heads", type=int, default=8, help="attention のヘッド数")
    parser.add_argument("--dropout", type=float, default=0.3, help="exp0001 の自己回帰の T2S で一番良かった値")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--text_drop", type=float, default=0.1, help="masked: 学習でテキストを除く確率(CFG 用)")
    parser.add_argument("--name", required=True, help="保存先 runs/{name}/ の名前")
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
    codec, norm = load_codec(args.codec, args.device)
    train, val = split(utts, "train"), split(utts, "val")
    vocab = TextVocab.build("phoneme", [u.text for u in train])
    n_codes = codec.codebook_size
    data = {name: prepare(us, vocab, codec, norm, args.device, with_durations=args.method == "duration")
            for name, us in [("train", train), ("val", val)]}
    tr = data["train"]
    log(f"text vocab {len(vocab)} | audio vocab {n_codes} | train {len(tr['texts'])} utts (dropped {tr['n_dropped']}) "
        f"| val {len(data['val']['texts'])} utts (dropped {data['val']['n_dropped']})")

    if args.method == "duration":
        config = {"dim": args.dim, "enc_layers": args.layers // 2, "dec_layers": args.layers - args.layers // 2,
                  "heads": args.heads, "dropout": args.dropout}
        model = DurationT2S(len(vocab), n_codes, **config)
    else:
        # 長さを決める「1 音素あたりの音声トークン数」は、学習データ全体の平均にする(公式は参照音声から求める)
        n_phones = sum(len([t for t in vocab.tokenize(u.text) if t not in PROSODY_MARKS]) for u in train)
        config = {"dim": args.dim, "layers": args.layers, "heads": args.heads, "dropout": args.dropout,
                  "text_drop": args.text_drop, "tokens_per_phone": sum(map(len, tr["codes"])) / n_phones}
        model = MaskedT2S(len(vocab), n_codes, **config)
        log(f"tokens per phone {config['tokens_per_phone']:.3f}")
    model.to(args.device)
    log(f"t2s params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
                if ev["loss"] < best:  # 検証 loss が一番小さい時点を保存する
                    best, mark = ev["loss"], " *saved"
                    torch.save({"model": model.state_dict(), "config": config, "method": args.method,
                                "vocab": vocab.state_dict(), "n_codes": n_codes, "step": step}, out / "t2s.pt")
                log(f"step {step:6d} | train " + " ".join(f"{k} {v / n:.4f}" for k, v in running.items())
                    + " | val " + " ".join(f"{k} {v:.4f}" for k, v in ev.items()) + mark)
                running = {}
            if step >= args.steps:
                break
        epoch += 1
    log(f"best val loss {best:.4f}, checkpoint {out / 't2s.pt'}")


if __name__ == "__main__":
    main()
