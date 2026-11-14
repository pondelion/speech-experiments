"""手順 6: T2S を GRPO で事後学習する(IndexTTS-2.5 の論文: 1 つの入力から確率的な生成で 4 つ作り、ASR の誤り率と話者の類似度を報酬にする)。

報酬: 合成した音声(T2S → codec の decoder → S2M → BigVGAN-v2)を ASR で書き起こした、読みの CER の符号を変えたもの(-CER)。
      ASR は評価(parakeet)と別の Whisper large-v3-turbo を使う(評価の ASR だけに合わせて数値を上げてしまうのを避けるため)。
      話者の類似度は、1 話者なので入れない(フェーズ A)。
アルゴリズム: TRL(1.18)の GRPO の既定の設定と同じ形:
    - 利点 = (報酬 - グループの平均) / (グループの標準偏差 + 1e-4)(scale_rewards="group")
    - 損失 = -min(比 × 利点, clip(比, 1 - 0.2, 1 + 0.2) × 利点) の、バッチの全トークンでの平均(loss_type="dapo")
      比 = exp(今の対数確率 - 生成したときの対数確率)。1 回の生成あたりの更新は num_iterations 回(既定 1)
    - KL の重み beta(既定 0。0 より大きいときは、元の T2S を参照にして k3 推定の KL を足す)
    - 対数確率は logits を生成の温度で割ってから求める。AdamW、学習率 1e-6 から線形に 0 へ、勾配の大きさの上限 1.0
生成(候補を作るとき): 公式の推論のサンプリング(温度 0.8、top_p 0.8、top_k 30、繰り返しのペナルティ 10)。候補を多様にするためビームサーチは使わない。
1 step = 学習データからランダムに選んだ --prompts 文 × 4 つ。--eval_every step ごとに検証 100 文を公式の推論の設定(ビームサーチ 3)で
合成して parakeet の CER を測り、一番低い時点を保存する(step 0 = 元の T2S も測る)。

使い方:
    python train_grpo.py --t2s t2s --vocoder bigvgan_v2_22k --name grpo
    python train_grpo.py --t2s t2s --vocoder bigvgan_v2_22k --steps 2 --prompts 2 --eval_every 2 --val_limit 3 --name smoke   # 動作確認
出力: runs/{name}/t2s.pt(train_t2s.py と同じ形。evaluate.py --t2s {name} で評価できる)と log.txt
"""
import argparse
import copy
import random
import time

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import pipeline

import common
from common.evaluator import Evaluator  # noqa: F401  (音声の設定を exp0006 に合わせて import しておく)
from common.s2m import MEL_PER_CODEC_FRAME
from common.text import frontend
from lib_common.asr_eval import ASR, cer_scores, utterance_cer
from lib_common.audio import SR
from lib_common.data import list_jsut, split
from lib_common.neural_vocoder import load_vocoder
from lib_common.utils import Logger, seed_everything
from train_codec import load_codec
from train_s2m import load_s2m
from train_t2s import load_t2s

SAMPLE_KW = {"do_sample": True, "temperature": 0.8, "top_p": 0.8, "top_k": 30, "repetition_penalty": 10.0, "num_beams": 1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t2s", default="t2s")
    parser.add_argument("--codec", default="codec")
    parser.add_argument("--s2m", default="s2m")
    parser.add_argument("--vocoder", default="bigvgan_v2_22k")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--prompts", type=int, default=16, help="1 step の文の数(× num_generations 個の候補)")
    parser.add_argument("--num_generations", type=int, default=4, help="公式の論文の値")
    parser.add_argument("--lr", type=float, default=1e-6, help="TRL の既定")
    parser.add_argument("--beta", type=float, default=0.0, help="TRL の既定")
    parser.add_argument("--epsilon", type=float, default=0.2, help="TRL の既定")
    parser.add_argument("--num_iterations", type=int, default=1, help="TRL の既定")
    parser.add_argument("--max_tokens", type=int, default=600, help="候補の最大の長さ(24 秒。学習データの最長は約 15 秒)")
    parser.add_argument("--eval_every", type=int, default=25)
    parser.add_argument("--val_limit", type=int, default=None)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    seed_everything(args.seed)
    rng = random.Random(args.seed)
    torch.set_num_threads(4)
    out = common.RUNS_DIR / args.name
    log = Logger(out / "log.txt")
    log(f"args: {vars(args)}")
    R = common.RUNS_DIR
    ck = torch.load(R / args.t2s / "t2s.pt", map_location="cpu")
    policy, vocab = load_t2s(R / args.t2s / "t2s.pt", "cuda")
    ref = copy.deepcopy(policy).eval() if args.beta > 0 else None
    codec, _, _ = load_codec(R / args.codec / "codec.pt", "cuda")
    s2m = load_s2m(R / args.s2m / "s2m.pt", "cuda")
    voc = load_vocoder(R / args.vocoder / "vocoder.pt", "cuda")
    whisper = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", torch_dtype=torch.bfloat16,
                       device="cuda", model_kwargs={"attn_implementation": "sdpa"})
    parakeet = ASR("cuda")
    utts = list_jsut()
    train, val = split(utts, "train"), split(utts, "val")[: args.val_limit]
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1 - s / args.steps)   # TRL(Trainer)の既定: 線形に 0 へ

    @torch.no_grad()
    def synth(codes_list: list[torch.Tensor]) -> list[np.ndarray]:
        """音声トークンのリスト → 波形のリスト(長さの近いものを 16 個ずつまとめて S2M に通す)。"""
        waves = [None] * len(codes_list)
        order = sorted(range(len(codes_list)), key=lambda i: len(codes_list[i]))
        for s in range(0, len(order), 16):
            idx = [i for i in order[s: s + 16] if len(codes_list[i]) > 0]
            if not idx:
                continue
            ids = torch.nn.utils.rnn.pad_sequence([codes_list[i].long() for i in idx], batch_first=True).cuda()
            feat = codec.decode(ids)
            n = torch.tensor([int(2 * len(codes_list[i]) * MEL_PER_CODEC_FRAME) for i in idx], device="cuda")
            mels = s2m.generate(feat, n)
            for k, i in enumerate(idx):
                waves[i] = voc.generate(mels[k: k + 1, : n[k]])[0].float().cpu().numpy()
        return [w if w is not None else np.zeros(SR // 10, dtype=np.float32) for w in waves]

    def whisper_text(waves: list[np.ndarray]) -> list[str]:
        xs = [{"raw": librosa.resample(w, orig_sr=SR, target_sr=16000), "sampling_rate": 16000} for w in waves]
        return [o["text"] for o in whisper(xs, batch_size=16, generate_kwargs={"language": "ja", "task": "transcribe"})]

    def validate(step: int) -> float:
        policy.eval()
        torch.cuda.empty_cache()
        codes = [policy.generate(torch.tensor(vocab.encode(frontend(u.text)))) for u in val]
        d = out / "val" / f"step{step}"
        d.mkdir(parents=True, exist_ok=True)
        paths = []
        for u, w in zip(val, synth(codes)):
            sf.write(d / f"{u.id}.wav", w, SR)
            paths.append(str(d / f"{u.id}.wav"))
        cer = cer_scores([u.text for u in val], parakeet.transcribe(paths))["reading"]
        torch.cuda.empty_cache()
        return cer

    best = validate(0)
    log(f"step 0 (元の T2S) | val CER {best:.4f}")
    torch.save({**ck, "step": 0}, out / "t2s.pt")
    G = args.num_generations
    for step in range(1, args.steps + 1):
        t0 = time.time()
        batch = rng.sample(train, args.prompts)
        texts = [torch.tensor(vocab.encode(frontend(u.text))) for u in batch]
        policy.eval()
        cands = [c for t in texts for c in policy.sample(t, args.max_tokens, num_return_sequences=G, **SAMPLE_KW)]
        t_gen = time.time() - t0
        hyps = whisper_text(synth([c for c, _ in cands]))
        cer = torch.tensor([utterance_cer(batch[i // G].text, h) for i, h in enumerate(hyps)])
        reward = -cer                                                                         # [prompts × G]
        r = reward.view(-1, G)
        adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)).view(-1).cuda()
        t_rew = time.time() - t0 - t_gen
        text_b = [texts[i // G] for i in range(len(cands))]
        codes_b, ended_b = [c for c, _ in cands], [e for _, e in cands]
        policy.train()
        old = None  # 生成したときの対数確率。1 回しか更新しないときは、TRL と同じく今の値(勾配を切ったもの)を使う(比 = 1)
        with torch.no_grad():
            if args.num_iterations > 1:
                old, _ = policy.token_logps(text_b, codes_b, ended_b, SAMPLE_KW["temperature"])
            ref_lp = ref.token_logps(text_b, codes_b, ended_b, SAMPLE_KW["temperature"])[0] if ref else None
        for _ in range(args.num_iterations):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lp, mask = policy.token_logps(text_b, codes_b, ended_b, SAMPLE_KW["temperature"])
            ratio = torch.exp(lp - (lp.detach() if old is None else old))
            per_tok = -torch.min(ratio * adv[:, None], ratio.clamp(1 - args.epsilon, 1 + args.epsilon) * adv[:, None])
            if ref is not None:  # k3 推定の KL(TRL と同じ)
                per_tok = per_tok + args.beta * (torch.exp(ref_lp - lp) - (ref_lp - lp) - 1)
            loss = (per_tok * mask).sum() / mask.sum()
            opt.zero_grad()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
        sched.step()
        lens = [len(c) for c in codes_b]
        log(f"step {step:4d} | reward {reward.mean():.4f} (std in group {r.std(1).mean():.4f}) whisper CER {cer.mean():.4f} "
            f"| len {np.mean(lens):.0f} truncated {1 - np.mean(ended_b):.2f} | loss {loss.item():.4f} grad {gnorm:.3f} "
            f"| gen {t_gen:.0f}s reward {t_rew:.0f}s total {time.time() - t0:.0f}s")
        if step % args.eval_every == 0 or step == args.steps:
            v = validate(step)
            mark = ""
            if v < best:
                best, mark = v, " *saved"
                torch.save({**ck, "model": policy.state_dict(), "step": step}, out / "t2s.pt")
            log(f"step {step:4d} | val CER {v:.4f}{mark}")
    log(f"best val CER {best:.4f}")


if __name__ == "__main__":
    main()
