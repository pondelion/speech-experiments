"""手順 2: テキストから音声トークン列を作る T2S(GPT)を学習する。

学習済みの codec(手順 1)で全発話の Mel をトークン列にし、それを正解として「次の音声トークン予測」を学習する。
テキストの入力形式は --text で選ぶ(char: 文字そのまま / phoneme: pyopenjtalk の音素 + アクセント記号)。

学習の仕方の比較用の設定(既定値は、どれも使わない = IndexTTS2.5 と同じ分類の損失):
  --objective soft_ce:     代表ベクトルが正解に近いトークンにも確率を配った目標で学習する(--soft_tau で配り方を決める)
  --objective regression:  トークンの分類ではなく、正解のトークンの代表ベクトルを cosine 距離で回帰する
  --aux_phoneme W:         音声側の各位置で「今どの音素か」を当てる補助の損失を、重み W で足す
                           (正解の音素は jsut-lab の音素アライメントから、トークンの区間ごとに作る)
  --coarse PATH:           codec のトークンを k-means でまとめた粗いトークンで学習する(analyze_coarse_tokens.py が作る対応表)

途中経過として、検証データで次を測る(teacher forcing、つまり正解のトークン列を入力したときの値)。
  loss: 学習の損失(--objective ごとに意味が違うので、違う設定どうしでは比べられない)
  acc: 次のトークンの正解率(regression は予測したベクトルに一番近いトークンで判定)
  aux_acc: --aux_phoneme を使うときの、音素の正解率
  (実際に読めるかは analyze_t2s_teacher_forced.py・evaluate.py の CER で見る)

使い方:
    python train_t2s.py --text phoneme
    python train_t2s.py --text phoneme --dim 256 --layers 4 --heads 4 --dropout 0.3 --name t2s_ph_small_drop03
    python train_t2s.py --text phoneme --dropout 0.3 --aux_phoneme 1.0 --name t2s_ph_aux
    python train_t2s.py --text char --limit 50 --steps 20 --device cpu   # 動作確認
出力: runs/{name}/(既定は runs/t2s_{text}/)に t2s.pt(検証 loss が一番小さかった時点)と log.txt
"""
import argparse

import torch
import torch.nn.functional as F

import common  # noqa: F401  tts/lib_common を import できるようにする
from common.codec import DOWNSAMPLE
from common.t2s import OBJECTIVES, TextToSemantic
from common.utils import RUNS_DIR, encode_all, load_codec
from lib_common.audio import HOP, SR
from lib_common.data import align_to_segments, load_dataset, load_jsut_alignment, split
from lib_common.text import TextVocab
from lib_common.utils import Logger, length_batches, seed_everything, warmup_cosine

TOKEN_SEC = DOWNSAMPLE * HOP / SR  # 音声トークン 1 個が受け持つ時間(42.7ms)


@torch.no_grad()
def evaluate(model: TextToSemantic, texts: list[torch.Tensor], codes: list[torch.Tensor],
             phonemes: list[torch.Tensor] | None, max_tokens: int) -> dict:
    """検証データでの loss と正解率(発話数で重み付けした平均)。"""
    model.eval()
    sums, n = {}, 0
    for b in length_batches([len(texts[i]) + len(codes[i]) for i in range(len(texts))], max_tokens, shuffle=False):
        loss, parts = model([texts[i] for i in b], [codes[i] for i in b],
                            [phonemes[i] for i in b] if phonemes else None)
        for k, v in {"loss": loss.item(), **parts}.items():
            sums[k] = sums.get(k, 0.0) + v * len(b)
        n += len(b)
    model.train()
    return {k: v / n for k, v in sums.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", choices=["char", "phoneme"], required=True)
    parser.add_argument("--codec", default=str(RUNS_DIR / "codec" / "codec.pt"))
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--max_tokens", type=int, default=12000, help="1 バッチの (最長系列長 × 発話数) の上限")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--dim", type=int, default=512, help="隠れ層の次元")
    parser.add_argument("--layers", type=int, default=8, help="Transformer の層数")
    parser.add_argument("--heads", type=int, default=8, help="attention のヘッド数")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--objective", choices=OBJECTIVES, default="ce", help="学習の損失の種類")
    parser.add_argument("--soft_tau", type=float, default=0.1, help="soft_ce で確率を配るときの温度")
    parser.add_argument("--aux_phoneme", type=float, default=0.0, help="音素を当てる補助の損失の重み(0 なら使わない)")
    parser.add_argument("--coarse", default=None, help="粗いトークンの対応表(analyze_coarse_tokens.py の出力)")
    parser.add_argument("--name", default=None, help="保存先 runs/{name}/ の名前(既定は t2s_{text})")
    parser.add_argument("--limit", type=int, default=None, help="動作確認用に発話数を絞る")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    torch.set_num_threads(4)
    out = RUNS_DIR / (args.name or f"t2s_{args.text}")
    log = Logger(out / "log.txt")
    log(f"args: {vars(args)}")

    utts = load_dataset(args.limit, log)
    codec, norm = load_codec(args.codec, args.device)
    train, val = split(utts, "train"), split(utts, "val")
    vocab = TextVocab.build(args.text, [u.text for u in train])

    # 音声トークンの語彙: codec の 1024 種類か、k-means でまとめた粗いトークン
    from_fine = to_fine = None
    code_vecs = F.normalize(codec.codebook.weight.detach().cpu(), dim=-1)            # 代表ベクトル [1024, C(8)]
    if args.coarse:
        coarse = torch.load(args.coarse)
        from_fine, to_fine, code_vecs = coarse["from_fine"], coarse["to_fine"], coarse["centroids"]
    n_codes = len(code_vecs)

    data = {}
    for name, us in [("train", train), ("val", val)]:
        texts = [torch.tensor(vocab.encode(u.text)) for u in us]
        codes = encode_all(codec, [norm.normalize(u.mel) for u in us], args.device)   # codec のトークン、各 [U_i]
        if from_fine is not None:
            codes = [from_fine[c] for c in codes]                                     # 粗いトークンに置き換える
        data[name] = [texts, codes, None]
    phone_vocab = []
    if args.aux_phoneme > 0:  # 各トークンの音素(jsut-lab のアライメントから)
        labels = {name: [[p for p, _ in align_to_segments(load_jsut_alignment(u), len(c), TOKEN_SEC)]
                         for u, c in zip(us, data[name][1])] for name, us in [("train", train), ("val", val)]}
        phone_vocab = sorted({p for ls in labels["train"] for p in ls})
        stoi = {p: i for i, p in enumerate(phone_vocab)}
        for name in labels:
            data[name][2] = [torch.tensor([stoi.get(p, stoi["sil"]) for p in ls]) for ls in labels[name]]
        log(f"aux phoneme labels: {len(phone_vocab)} phonemes")
    tr_t, tr_c, tr_p = data["train"]
    log(f"text vocab {len(vocab)} | audio vocab {n_codes} | text length mean {sum(map(len, tr_t)) / len(tr_t):.0f} "
        f"max {max(map(len, tr_t))} | audio tokens mean {sum(map(len, tr_c)) / len(tr_c):.0f} max {max(map(len, tr_c))}")

    model_config = {"dim": args.dim, "layers": args.layers, "heads": args.heads, "dropout": args.dropout,
                    "objective": args.objective, "soft_tau": args.soft_tau, "n_phonemes": len(phone_vocab),
                    "aux_weight": args.aux_phoneme}
    model = TextToSemantic(len(vocab), n_codes, **model_config, code_vecs=code_vecs,
                           from_fine=from_fine, to_fine=to_fine).to(args.device)
    log(f"t2s params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = warmup_cosine(opt, args.steps)

    lengths = [len(t) + len(c) for t, c in zip(tr_t, tr_c)]
    step, epoch, best, running = 0, 0, float("inf"), {}
    while step < args.steps:
        for b in length_batches(lengths, args.max_tokens, shuffle=True, seed=epoch):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                loss, parts = model([tr_t[i] for i in b], [tr_c[i] for i in b], [tr_p[i] for i in b] if tr_p else None)
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
                ev = evaluate(model, *data["val"], args.max_tokens)
                mark = ""
                if ev["loss"] < best:  # 検証 loss が一番小さい時点を保存する
                    best, mark = ev["loss"], " *saved"
                    torch.save({"model": model.state_dict(), "config": model_config, "vocab": vocab.state_dict(),
                                "n_codes": n_codes, "phone_vocab": phone_vocab, "step": step}, out / "t2s.pt")
                fmt = lambda d: " ".join(f"{k} {v / (n if d is running else 1):.4f}" for k, v in d.items())
                log(f"step {step:6d} | train {fmt(running)} | val {fmt(ev)}{mark}")
                running = {}
            if step >= args.steps:
                break
        epoch += 1
    log(f"best val loss {best:.4f}, checkpoint {out / 't2s.pt'}")


if __name__ == "__main__":
    main()
