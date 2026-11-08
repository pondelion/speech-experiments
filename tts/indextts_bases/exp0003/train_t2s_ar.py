"""手順 3(exp0001 の構成で比べる場合): SSL の codec のトークンを予測する自己回帰の T2S を学習する。

T2S は exp0001 で一番良かった自己回帰の GPT + 音素の補助の損失(exp0001 の common/t2s.py)で、設定も exp0001 の
`t2s_ph_k256_aux` と同じ(音素入力、dropout 0.3、補助の損失の重み 1.0、6,000 step、500 step ごとに検証)。
--codec ext:{名前} で外部のコーデックのトークンにする(補助の損失の音素の区間は、トークン 1 個の長さに合わせる)。

使い方:
    python train_t2s_ar.py --codec runs/codec_w2vbert2_k256/codec.pt --name t2s_ar_w2vbert2_k256
出力: runs/{name}/t2s.pt(exp0001 の load_t2s で読める形)と log.txt
"""
import argparse

import torch
import torch.nn.functional as F

import common  # noqa: F401  tts/lib_common・exp0001 を import できるようにする
from common.utils import RUNS_DIR, tokenize
from exp0001.common.t2s import TextToSemantic
from lib_common.data import align_to_segments, load_dataset, load_jsut_alignment, split
from lib_common.text import TextVocab
from lib_common.utils import Logger, length_batches, seed_everything, warmup_cosine


@torch.no_grad()
def evaluate(model, d: tuple, max_tokens: int) -> dict:
    """検証データでの loss と正解率(teacher forcing)。"""
    model.eval()
    texts, codes, phones = d
    sums, n = {}, 0
    for b in length_batches([len(t) + len(c) for t, c in zip(texts, codes)], max_tokens, shuffle=False):
        loss, parts = model([texts[i] for i in b], [codes[i] for i in b], [phones[i] for i in b])
        for k, v in {"loss": loss.item(), **parts}.items():
            sums[k] = sums.get(k, 0.0) + v * len(b)
        n += len(b)
    model.train()
    return {k: v / n for k, v in sums.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec", required=True, help="train_codec.py の codec.pt")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--max_tokens", type=int, default=12000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval_every", type=int, default=500)
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
    # 補助の損失の正解: 各トークンの区間と一番長く重なる音素(jsut-lab から。exp0001 と同じ)
    labels = [[p for p, _ in align_to_segments(load_jsut_alignment(u), len(c), codec.token_sec)]
              for u, c in zip(train + val, codes)]
    phone_vocab = sorted({p for ls in labels[: len(train)] for p in ls})
    stoi = {p: i for i, p in enumerate(phone_vocab)}
    phones = [torch.tensor([stoi.get(p, stoi["sil"]) for p in ls]) for ls in labels]
    texts = [torch.tensor(vocab.encode(u.text)) for u in train + val]
    k = len(train)
    tr, va = (texts[:k], codes[:k], phones[:k]), (texts[k:], codes[k:], phones[k:])

    config = {"dim": 512, "layers": 8, "heads": 8, "dropout": 0.3, "objective": "ce", "soft_tau": 0.1,
              "n_phonemes": len(phone_vocab), "aux_weight": 1.0,
              # 音声トークンの位置埋め込みの数: exp0001 と同じ 1024 か、トークンが細かい外部のコーデックでは一番長い発話の 1.5 倍
              "max_audio": max(1024, int(max(map(len, codes)) * 1.5))}
    code_vecs = F.normalize(codec.codebook.weight.detach().cpu(), dim=-1)
    model = TextToSemantic(len(vocab), codec.codebook_size, **config, code_vecs=code_vecs).to(args.device)
    log(f"t2s params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = warmup_cosine(opt, args.steps)
    step, epoch, best, running = 0, 0, float("inf"), {}
    while step < args.steps:
        for b in length_batches([len(t) + len(c) for t, c in zip(tr[0], tr[1])], args.max_tokens, shuffle=True, seed=epoch):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                loss, parts = model(*[[x[i] for i in b] for x in tr])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            for key, v in {"loss": loss.item(), **parts}.items():
                running[key] = running.get(key, 0.0) + v
            if step % args.eval_every == 0 or step == args.steps:
                n = args.eval_every if step % args.eval_every == 0 else step % args.eval_every
                ev = evaluate(model, va, args.max_tokens)
                mark = ""
                if ev["loss"] < best:
                    best, mark = ev["loss"], " *saved"
                    torch.save({"model": model.state_dict(), "config": config, "vocab": vocab.state_dict(),
                                "n_codes": codec.codebook_size, "phone_vocab": phone_vocab, "step": step}, out / "t2s.pt")
                log(f"step {step:6d} | train " + " ".join(f"{key} {v / n:.4f}" for key, v in running.items())
                    + " | val " + " ".join(f"{key} {v:.4f}" for key, v in ev.items()) + mark)
                running = {}
            if step >= args.steps:
                break
        epoch += 1
    log(f"best val loss {best:.4f}, checkpoint {out / 't2s.pt'}")


if __name__ == "__main__":
    main()
