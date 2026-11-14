"""NN ボコーダ(HiFi-GAN V1・V2、Vocos。tts/lib_common/neural_vocoder)を JSUT で学習する。

学習データ: JSUT の学習データの録音から、ランダムな位置で segment サンプル(HiFi-GAN 8192、Vocos 16384)を切り出し、
            その log-Mel(lib_common.audio と同じ。S2M の出力と同じ形)を入力、元の波形を正解にする。
1 step: (1) 判別器を更新(本物と、生成器が作った波形を見分ける)、(2) 生成器を更新(敵対的な損失 + feature matching + Mel の L1 × 45)。
学習率などの設定は lib_common.neural_vocoder.RECIPES(論文の値)。
検証: eval_every step ごとに、検証データ 100 文の全体の Mel から波形を作り、その Mel と元の Mel の L1 を測り、一番小さい時点を
      vocoder.pt に保存する(テスト文は使わない)。途中から続けられるように、最新の状態を last.pt にも保存する。

使い方:
    python train_vocoder.py --model vocos --steps 200000 --name vocos
    python train_vocoder.py --model hifigan_v2 --limit 30 --steps 20 --eval_every 10 --name smoke   # 動作確認
    python train_vocoder.py --model vocos --steps 200000 --name vocos --resume                      # 続きから
出力: runs/{name}/vocoder.pt(検証の Mel の L1 が一番小さい時点)、last.pt(続き用)、log.txt、
      samples/step{N}.wav(sample_every step ごとの検証データの 1 文。耳で改善の推移を確かめる用)と original.wav(元の録音)
"""
import argparse
import math
import random
import time
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F

import common
from lib_common.audio import SR, load_wave, logmel_torch, n_samples
from lib_common.data import load_dataset, split
from lib_common.neural_vocoder import (MEL_WEIGHT, RECIPES, MultiScaleMelLoss, adversarial_d, adversarial_g,
                                       build_discriminators, build_generator, feature_matching)
from lib_common.utils import Logger, seed_everything


def batch(waves: list[torch.Tensor], size: int, segment: int, device: str) -> torch.Tensor:
    """ランダムな発話のランダムな位置から segment サンプルを切り出す(短い発話は 0 で詰める)。[B, segment]。"""
    out = []
    for w in random.choices(waves, k=size):
        s = random.randint(0, max(0, len(w) - segment))
        out.append(F.pad(w[s: s + segment].float(), (0, max(0, segment - len(w)))))
    return torch.stack(out).to(device)


@torch.no_grad()
def validate(gen, val, device: str) -> float:
    """検証データの全体の Mel → 波形 → その Mel と元の Mel の L1(発話の平均)。"""
    torch.cuda.empty_cache()
    gen.eval()
    total = 0.0
    for u in val:
        mel = u.mel.to(device)[None]                                                # [1, T, F]
        wave = gen(mel.transpose(1, 2))[:, 0, : n_samples(mel.shape[1])]           # [1, N](元の録音の長さ)
        total += F.l1_loss(logmel_torch(wave), mel).item()
    gen.train()
    torch.cuda.empty_cache()
    return total / len(val)


@torch.no_grad()
def save_sample(gen, utt, path, device: str):
    """1 文の Mel を波形にして保存する(最初の 1 回は、比べる用に元の録音も保存する)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not (path.parent / "original.wav").exists():
        sf.write(path.parent / "original.wav", load_wave(utt.path), SR)
    gen.eval()
    mel = utt.mel.to(device)[None]
    sf.write(path, gen(mel.transpose(1, 2))[0, 0, : n_samples(mel.shape[1])].cpu().numpy(), SR)
    gen.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(RECIPES), required=True)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--eval_every", type=int, default=5000)
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--sample_every", type=int, default=10000, help="検証データの 1 文を音声にして残す間隔")
    parser.add_argument("--name", required=True)
    parser.add_argument("--runs_dir", default=None, help="保存先の runs/ (既定は exp0004/runs。exp0006 の 22.05kHz のボコーダは exp0006/runs)")
    parser.add_argument("--resume", action="store_true", help="runs/{name}/last.pt から続ける")
    parser.add_argument("--limit", type=int, default=None, help="動作確認用に発話数を絞る")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(4)
    out = (Path(args.runs_dir) if args.runs_dir else common.RUNS_DIR) / args.name
    log = Logger(out / "log.txt")
    log(f"args: {vars(args)}")
    r = RECIPES[args.model]
    t0 = time.time()
    utts = load_dataset(args.limit, log)
    train, val = split(utts, "train"), split(utts, "val")
    waves = [torch.from_numpy(load_wave(u.path)).half() for u in train]             # 学習データの波形(float16 で RAM に置く)
    log(f"loaded {len(waves)} train waves ({sum(map(len, waves)) / SR / 3600:.1f} h) in {time.time() - t0:.0f}s")

    gen, discs = build_generator(args.model).to(args.device), build_discriminators(args.model).to(args.device)
    log(f"generator {sum(p.numel() for p in gen.parameters()) / 1e6:.2f}M, discriminators "
        f"{sum(p.numel() for p in discs.parameters()) / 1e6:.1f}M")
    opt_g = torch.optim.AdamW(gen.parameters(), lr=r["lr"], betas=r["betas"], weight_decay=0.01)
    opt_d = torch.optim.AdamW(discs.parameters(), lr=r["lr"], betas=r["betas"], weight_decay=0.01)
    if r["schedule"] == "exponential":  # HiFi-GAN: 1 epoch ごとに 0.999 倍(ここでは step ごとに同じ割合で下げる)
        gamma = 0.999 ** (r["batch"] / len(waves))
        factor = lambda s: gamma ** s  # noqa: E731
    elif r["schedule"] == "exponential_step":  # BigVGAN-v2: step ごとに gamma 倍
        factor = lambda s: r["gamma"] ** s  # noqa: E731
    else:  # Vocos: 学習の全体で cosine に 0 まで下げる
        factor = lambda s: 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps))  # noqa: E731
    sched_g, sched_d = torch.optim.lr_scheduler.LambdaLR(opt_g, factor), torch.optim.lr_scheduler.LambdaLR(opt_d, factor)
    step, best = 0, float("inf")
    if args.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cpu")
        for m, k in [(gen, "generator"), (discs, "discriminators"), (opt_g, "opt_g"), (opt_d, "opt_d"),
                     (sched_g, "sched_g"), (sched_d, "sched_d")]:
            m.load_state_dict(ck[k])
        step, best = ck["step"], ck["best"]
        log(f"resumed from step {step}")

    def disc_loss(fn, w):  # 判別器ごとの重みと、Vocos は判別器の中の小さな判別器の数で割る
        return sum(wi * fn(i) / (n if r["mean_over_discs"] else 1) for i, (wi, n) in enumerate(zip(w, counts)))

    counts = [len(d.discs) for d in discs]
    # Mel の損失: 1 つの Mel の L1 × 45(HiFi-GAN・Vocos)か、複数の解像度の Mel の L1 × 15(BigVGAN-v2)
    multi_mel = MultiScaleMelLoss().to(args.device) if r.get("mel_loss") == "multiscale" else None
    mel_weight = r.get("mel_weight", MEL_WEIGHT)
    clip = r.get("clip_grad")  # 勾配の大きさの上限(BigVGAN-v2 は 500。他は使わない)
    running, t_last = {}, time.time()
    while step < args.steps:
        x = batch(waves, r["batch"], r["segment"], args.device)                      # 本物の波形 [B, L]
        mel = logmel_torch(x)                                                        # [B, T, F]
        y = gen(mel.transpose(1, 2))[..., : x.shape[-1]]                             # 生成した波形 [B, 1, L]
        real = x[:, None]
        # (1) 判別器
        outs = [(d(real), d(y.detach())) for d in discs]
        loss_d = disc_loss(lambda i: adversarial_d(outs[i][0][0], outs[i][1][0], r["gan"]), r["disc_weights"])
        opt_d.zero_grad()
        loss_d.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(discs.parameters(), clip)
        opt_d.step()
        # (2) 生成器
        mel_loss = multi_mel(y[:, 0], x) if multi_mel is not None else F.l1_loss(logmel_torch(y[:, 0]), mel)
        outs = []
        for d in discs:
            with torch.no_grad():
                _, feats_real = d(real)
            o_fake, feats_fake = d(y)
            outs.append((o_fake, feats_real, feats_fake))
        adv = disc_loss(lambda i: adversarial_g(outs[i][0], r["gan"]), r["disc_weights"])
        fm = disc_loss(lambda i: feature_matching(outs[i][1], outs[i][2]), r["disc_weights"])
        loss_g = adv + r["fm_weight"] * fm + mel_weight * mel_loss
        opt_g.zero_grad()
        loss_g.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(gen.parameters(), clip)
        opt_g.step()
        sched_g.step()
        sched_d.step()
        step += 1
        for k, v in {"d": loss_d, "adv": adv, "fm": fm, "mel": mel_loss}.items():
            running[k] = running.get(k, 0.0) + v.item()
        if step % args.log_every == 0:
            n = args.log_every
            log(f"step {step:7d} | " + " ".join(f"{k} {v / n:.4f}" for k, v in running.items())
                + f" | {(time.time() - t_last) / n:.3f}s/step")
            running, t_last = {}, time.time()
        if step % args.eval_every == 0 or step == args.steps:
            val_l1 = validate(gen, val, args.device)
            mark = ""
            if val_l1 < best:
                best, mark = val_l1, " *saved"
                torch.save({"name": args.model, "generator": gen.state_dict(), "step": step}, out / "vocoder.pt")
            torch.save({"generator": gen.state_dict(), "discriminators": discs.state_dict(), "opt_g": opt_g.state_dict(),
                        "opt_d": opt_d.state_dict(), "sched_g": sched_g.state_dict(), "sched_d": sched_d.state_dict(),
                        "step": step, "best": best}, out / "last.pt")
            log(f"step {step:7d} | val mel L1 {val_l1:.4f}{mark}")
            t_last = time.time()
        if step % args.sample_every == 0 or step == args.steps:  # 耳で改善の推移を確かめる用: 検証データの 1 文
            save_sample(gen, val[0], out / "samples" / f"step{step:07d}.wav", args.device)
    log(f"best val mel L1 {best:.4f}, checkpoint {out / 'vocoder.pt'}")


if __name__ == "__main__":
    main()
