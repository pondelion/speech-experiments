"""手順 2: 話者の条件を入れた T2S(common/t2s_spk.py)を学習する。--init でスクラッチか、フェーズ A の一番良い T2S からかを選ぶ。

比べる 2 つの条件(scratch・finetune)で違うのは初期値だけ: 構成・データ・語彙・学習の設定は同じ。
    構成: フェーズ A の一番良い T2S(exp0005 の t2s_f0aux_w2: ハイブリッド型、F0 の補助の損失 2.0)と同じ + 話者ベクトルの射影 2 つ
    語彙: フェーズ A の T2S の語彙(音素 + アクセント記号、50 種類)。F0 の正規化の値もフェーズ A と同じ(JSUT の値)
    学習の設定: フェーズ A と同じ(AdamW、学習率 3e-4、weight decay 0.01、warmup 1,000 step + cosine、1 バッチの上限 12,000、
              勾配の大きさの上限 1.0、bf16)。step 数はデータが約 2.2 倍になるので 12,000(フェーズ A は 8,000)。
              500 step ごとに検証 loss を測り、一番小さい時点を保存する
    話者ベクトル: 学習では、同じ話者の別の発話(学習データからランダム)の CAM++ のベクトルを使う(その発話の中身・抑揚が漏れないように)。
              検証では、同じ話者の別の発話を乱数の種 0 で 1 つ決めて使う
音素ごとの長さの正解が作れない発話は使わない(フェーズ A と同じ)。

使い方:
    python train_t2s.py --init scratch --name t2s_scratch
    python train_t2s.py --init finetune --name t2s_finetune
    python train_t2s.py --init finetune --limit 200 --steps 4 --eval_every 2 --name smoke   # 動作確認
出力: runs/{name}/t2s.pt と log.txt
"""
import argparse
import random
from pathlib import Path

import torch

import common
from common.data import list_all, phoneme_durations
from common.f0norm import mean_lf0, normalize, raw_targets
from common.t2s_spk import SpeakerHybridT2S, load_phase_a_state
from exp0005.train_t2s import f0_targets
from lib_common.text import TextVocab
from lib_common.utils import Logger, length_batches, seed_everything, warmup_cosine

PHASE_A = common.EXP_DIR.parent / "exp0005" / "runs" / "t2s_f0aux_w2" / "t2s.pt"


def other_utt(rng: random.Random, by_spk: dict, u) -> str:
    """同じ話者の、u 以外の発話の ID をランダムに 1 つ。"""
    cands = by_spk[u.speaker]
    while True:
        c = rng.choice(cands)
        if c != u.id or len(cands) == 1:
            return c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", choices=["scratch", "finetune"], required=True)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--max_tokens", type=int, default=12000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--name", required=True)
    parser.add_argument("--limit", type=int, default=None, help="動作確認用に、学習データを先頭から絞る")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_spk", action="store_true", help="exp0102 の no_t2s_spk: 話者ベクトルを 0 にする(話者の条件を外す)")
    parser.add_argument("--f0_norm", choices=["jsut", "ref_mean"], default="jsut", help="F0 の正規化(common/f0norm.py)")
    parser.add_argument("--ref_encoder", action="store_true",
                        help="exp0102 の t2s_ref_encoder: 話者の条件を、参照の発話の Mel を読む encoder にする(common/t2s_ref.py)")
    parser.add_argument("--runs_dir", default=None, help="保存先の runs/(既定は exp0101/runs。exp0102 は exp0102/runs)")
    args = parser.parse_args()
    assert not (args.ref_encoder and args.no_spk), "--ref_encoder と --no_spk は一緒に使わない"
    seed_everything(args.seed)
    rng = random.Random(args.seed)
    torch.set_num_threads(4)
    out = (Path(args.runs_dir) if args.runs_dir else common.RUNS_DIR) / args.name
    log = Logger(out / "log.txt")
    log(f"args: {vars(args)}")
    R = common.RUNS_DIR
    tokens, f0, spk = torch.load(R / "tokens.pt"), torch.load(R / "f0.pt"), torch.load(R / "spk.pt")
    ck = torch.load(PHASE_A, map_location="cpu")
    vocab, stats = TextVocab.from_state_dict(ck["vocab"]), ck["f0_stats"]
    utts = list_all()
    data = {}
    for name in ("train", "val"):
        us = [u for u in utts if u.split == name]
        if name == "train" and args.limit:
            us = us[: args.limit]
        codes = [tokens[u.id].long() for u in us]
        durs = [phoneme_durations(u, vocab.tokenize(u.text), len(c)) for u, c in zip(us, codes)]
        keep = [i for i, d in enumerate(durs) if d is not None]
        us, codes = [us[i] for i in keep], [codes[i] for i in keep]
        data[name] = {"utts": us, "texts": [torch.tensor(vocab.encode(u.text)) for u in us], "codes": codes,
                      "durations": [torch.tensor(durs[i]) for i in keep], "f0": f0_targets(us, codes, f0, stats)[0],
                      "f0_raw": raw_targets(us, codes, f0)}
        log(f"{name}: {len(us)} utts (dropped {len(durs) - len(keep)} without durations), "
            f"speakers {len({u.speaker for u in us})}")
    by_spk = {}
    for u in data["train"]["utts"]:
        by_spk.setdefault(u.speaker, []).append(u.id)
    val_rng = random.Random(0)
    val_by_spk = {s: by_spk.get(s, []) + [u.id for u in data["val"]["utts"] if u.speaker == s] for s in {u.speaker for u in data["val"]["utts"]}}
    data["val"]["spk_ids"] = [other_utt(val_rng, val_by_spk, u) for u in data["val"]["utts"]]

    ref_mels = {}  # 発話 ID → 正規化した Mel(--ref_encoder のとき、話者の条件を取る参照の発話の Mel。S2M と同じ正規化)
    if args.ref_encoder:
        from common.t2s_ref import RefEncoderT2S
        from lib_common.audio import MelNormalizer
        codec_ck = common.EXP_DIR.parent / "exp0003" / "runs" / "codec_hubert_ja_k256" / "codec.pt"   # train_s2m.py の CODEC
        norm = MelNormalizer.from_state_dict(torch.load(codec_ck, map_location="cpu")["normalizer"])
        ref_mels = {k: norm.normalize(v.float()) for k, v in torch.load(R / "mels.pt").items()}
        model = RefEncoderT2S(len(vocab), ck["n_codes"], **ck["config"])
    else:
        model = SpeakerHybridT2S(len(vocab), ck["n_codes"], **ck["config"])
    if args.init == "finetune":
        log(f"init from phase A {PHASE_A.parent.name}; new (zero-init) params: {load_phase_a_state(model, ck['model'])}")
    model.to(args.device)
    log(f"t2s params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = warmup_cosine(opt, args.steps)

    lf0_mean = {}  # 発話 ID → 有声の log2 F0 の平均(f0_norm=ref_mean のとき、参照の発話の値で正規化する)
    if args.f0_norm == "ref_mean":
        all_ids = [u.id for d in data.values() for u in d["utts"]]
        all_raw = [r for d in data.values() for r in d["f0_raw"]]
        lf0_mean = {i: mean_lf0(r) for i, r in zip(all_ids, all_raw)}

    def run(d, b, spk_ids):
        if args.ref_encoder:
            s = [ref_mels[i] for i in spk_ids]
        else:
            s = torch.stack([spk[i] for i in spk_ids])
        if args.no_spk:
            s = torch.zeros_like(s)
        if args.f0_norm == "ref_mean":
            f0b = [normalize(d["f0_raw"][i], lf0_mean[r], stats[1]) for i, r in zip(b, spk_ids)]
        else:
            f0b = [d["f0"][i] for i in b]
        with model.speaker(s):
            return model(*[[d[k][i] for i in b] for k in ("texts", "codes", "durations")], f0b)

    @torch.no_grad()
    def evaluate(d):
        model.eval()
        state = torch.get_rng_state()
        torch.manual_seed(0)
        sums, n = {}, 0
        for b in length_batches([len(t) + len(c) for t, c in zip(d["texts"], d["codes"])], args.max_tokens, shuffle=False):
            loss, parts = run(d, b, [d["spk_ids"][i] for i in b])
            for k, v in {"loss": loss.item(), **parts}.items():
                sums[k] = sums.get(k, 0.0) + v * len(b)
            n += len(b)
        torch.set_rng_state(state)
        model.train()
        return {k: v / n for k, v in sums.items()}

    tr = data["train"]
    lengths = [len(t) + len(c) for t, c in zip(tr["texts"], tr["codes"])]
    step, epoch, best, running = 0, 0, float("inf"), {}
    while step < args.steps:
        for b in length_batches(lengths, args.max_tokens, shuffle=True, seed=epoch):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                loss, parts = run(tr, b, [other_utt(rng, by_spk, tr["utts"][i]) for i in b])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step == 20 and args.device == "cuda":
                log(f"peak VRAM {torch.cuda.max_memory_allocated() / 2 ** 30:.1f}GiB(最初の 20 step)")
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
                    torch.save({"model": model.state_dict(), "config": ck["config"], "vocab": ck["vocab"], "n_codes": ck["n_codes"],
                                "f0_stats": stats, "init": args.init, "step": step,
                                "options": {"no_spk": args.no_spk, "f0_norm": args.f0_norm, "ref_encoder": args.ref_encoder}},
                               out / "t2s.pt")
                log(f"step {step:6d} | train " + " ".join(f"{k} {v / n:.4f}" for k, v in running.items())
                    + " | val " + " ".join(f"{k} {v:.4f}" for k, v in ev.items())
                    + (f" | reserved {torch.cuda.max_memory_reserved() / 2 ** 30:.1f}GiB" if args.device == "cuda" else "") + mark)
                running = {}
            if step >= args.steps:
                break
        epoch += 1
    log(f"best val loss {best:.4f}")


if __name__ == "__main__":
    main()
