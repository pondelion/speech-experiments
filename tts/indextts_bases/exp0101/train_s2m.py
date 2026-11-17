"""手順 3: 話者の条件を入れた S2M(common/s2m_spk.py、F0 の条件つき)を学習する。--init でスクラッチか、フェーズ A の一番良い S2M からかを選ぶ。

比べる 2 つの条件(scratch・finetune)で違うのは初期値だけ: 構成・データ・学習の設定は同じ。
    構成: フェーズ A の一番良い S2M(exp0005 の s2m_f0cond: exp0003 の日本語 HuBERT の S2M + F0 の条件)と同じ + プロンプト・話者ベクトルの射影
    codec: フェーズ A と同じ日本語 HuBERT の codec(固定)。Mel の正規化もフェーズ A と同じ(JSUT の値)
    F0 の条件: 正解の F0(T2S の F0 の補助の損失と同じ形・同じ正規化)
    学習の設定: 学習率 2e-4(warmup 1,000 step + cosine)、50,000 step(フェーズ A のスクラッチの S2M は 40,000 step。データが約 2.2 倍)、
              1 バッチは「最長の Mel のフレーム数 × 発話数」が 16,000 以下、勾配の大きさの上限 1.0、bf16。
              1,000 step ごとに検証 loss(ノイズ・時刻・参照の長さを固定)を測り、一番小さい時点を保存する
    話者ベクトル: T2S と同じく、同じ話者の別の発話の CAM++ のベクトル。プロンプトは同じ発話の先頭の一部(common/s2m_spk.py)

使い方:
    python train_s2m.py --init scratch --name s2m_scratch
    python train_s2m.py --init finetune --name s2m_finetune
    python train_s2m.py --init finetune --limit 200 --steps 4 --eval_every 2 --name smoke   # 動作確認
出力: runs/{name}/s2m.pt と log.txt
"""
import argparse
import random

import torch

import common
from common.data import list_all
from common.s2m_spk import SpeakerS2M
from exp0001.common.utils import load_codec, pad_mels
from exp0005.common.s2m_f0 import FPT, F0Condition
from exp0005.train_t2s import f0_targets
from lib_common.utils import Logger, length_batches, seed_everything, warmup_cosine
from train_t2s import PHASE_A as PHASE_A_T2S
from train_t2s import other_utt

E3, E5 = common.EXP_DIR.parent / "exp0003" / "runs", common.EXP_DIR.parent / "exp0005" / "runs"
CODEC = E3 / "codec_hubert_ja_k256" / "codec.pt"
BASE_S2M, PHASE_A = E3 / "s2m_hubert_ja_k256" / "s2m.pt", E5 / "s2m_f0cond" / "s2m.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", choices=["scratch", "finetune"], required=True)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_frames", type=int, default=16000)
    parser.add_argument("--eval_every", type=int, default=1000)
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
    R = common.RUNS_DIR
    mels, tokens, f0, spk = (torch.load(R / f) for f in ("mels.pt", "tokens.pt", "f0.pt", "spk.pt"))
    stats = torch.load(PHASE_A_T2S, map_location="cpu")["f0_stats"]
    codec, mel_norm = load_codec(CODEC, args.device)
    utts = list_all()
    data = {}
    for name in ("train", "val"):
        us = [u for u in utts if u.split == name]
        if name == "train" and args.limit:
            us = us[: args.limit]
        cs = [tokens[u.id].long() for u in us]
        data[name] = {"utts": us, "codes": cs, "f0": f0_targets(us, cs, f0, stats)[0],
                      "mels": [mel_norm.normalize(mels[u.id].float())[: len(c) * FPT] for u, c in zip(us, cs)]}
        log(f"{name}: {len(us)} utts, speakers {len({u.speaker for u in us})}")
    by_spk = {}
    for u in data["train"]["utts"]:
        by_spk.setdefault(u.speaker, []).append(u.id)
    val_rng = random.Random(0)
    val_by_spk = {s: by_spk.get(s, []) + [u.id for u in data["val"]["utts"] if u.speaker == s] for s in {u.speaker for u in data["val"]["utts"]}}
    data["val"]["spk_ids"] = [other_utt(val_rng, val_by_spk, u) for u in data["val"]["utts"]]

    bck = torch.load(BASE_S2M, map_location="cpu")
    s2m = SpeakerS2M(bck["cond_in_dim"], **bck["config"])
    f0cond = F0Condition(codec.hidden_dim)
    if args.init == "finetune":
        ck = torch.load(PHASE_A, map_location="cpu")
        r = s2m.load_state_dict(ck["s2m"], strict=False)
        f0cond.load_state_dict(ck["f0cond"])
        log(f"init from phase A {PHASE_A.parent.name}; new (zero-init) params: {r.missing_keys}")
    s2m.to(args.device).train()
    f0cond.to(args.device)
    params = list(s2m.parameters()) + list(f0cond.parameters())
    log(f"s2m params: {sum(p.numel() for p in params) / 1e6:.2f}M")
    opt = torch.optim.AdamW(params, lr=args.lr)
    sched = warmup_cosine(opt, args.steps)

    def batch(d, b, spk_ids):
        mel, mask = pad_mels([d["mels"][i] for i in b], args.device)
        ids = torch.nn.utils.rnn.pad_sequence([d["codes"][i] for i in b], batch_first=True).to(args.device)
        f = torch.nn.utils.rnn.pad_sequence([d["f0"][i] for i in b], batch_first=True).to(args.device)
        with torch.no_grad():
            hidden = codec.decode_hidden(ids)                                                 # codec は固定
        return mel, hidden + f0cond(f), mask, torch.stack([spk[i] for i in spk_ids]).to(args.device)

    @torch.no_grad()
    def evaluate(d) -> float:
        s2m.eval()
        total, n = 0.0, 0
        with torch.random.fork_rng(devices=[args.device] if args.device == "cuda" else []):
            torch.manual_seed(0)
            for b in length_batches([len(m) for m in d["mels"]], args.max_frames, shuffle=False):
                mel, hidden, mask, sp = batch(d, b, [d["spk_ids"][i] for i in b])
                total, n = total + s2m(mel, hidden, mask, sp).item() * len(b), n + len(b)
        s2m.train()
        return total / n

    best, step, epoch, running = float("inf"), 0, 0, 0.0
    tr = data["train"]
    while step < args.steps:
        for b in length_batches([len(m) for m in tr["mels"]], args.max_frames, shuffle=True, seed=epoch):
            mel, hidden, mask, sp = batch(tr, b, [other_utt(rng, by_spk, tr["utts"][i]) for i in b])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                loss = s2m(mel, hidden, mask, sp)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            running += loss.item()
            if step == 20 and args.device == "cuda":
                log(f"peak VRAM {torch.cuda.max_memory_allocated() / 2 ** 30:.1f}GiB(最初の 20 step)")
            if step % args.eval_every == 0 or step == args.steps:
                n = args.eval_every if step % args.eval_every == 0 else step % args.eval_every
                torch.cuda.empty_cache()
                val = evaluate(data["val"])
                torch.cuda.empty_cache()
                mark = ""
                if val < best:
                    best, mark = val, " *saved"
                    torch.save({"s2m": s2m.state_dict(), "f0cond": f0cond.state_dict(), "cond_in_dim": bck["cond_in_dim"],
                                "config": bck["config"], "f0_stats": stats, "init": args.init, "step": step}, out / "s2m.pt")
                log(f"step {step:6d} | train loss {running / n:.4f} | val loss {val:.4f}"
                    + (f" | reserved {torch.cuda.max_memory_reserved() / 2 ** 30:.1f}GiB" if args.device == "cuda" else "") + mark)
                running = 0.0
            if step >= args.steps:
                break
        epoch += 1
    log(f"best val loss {best:.4f}")


if __name__ == "__main__":
    main()
