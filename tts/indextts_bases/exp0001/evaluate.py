"""手順 4: テスト 100 文を合成し、ASR の文字誤り率(CER)で読み上げの正確さを測る。

TTS 全体の誤りが、どの部品で増えているかを切り分けるため、途中の段階から合成した音声も同じ方法で測る。
合成は common/pipeline.py の TTSPipeline の各段階を使う(synthesize.py と同じ部品)。

    条件               経路                                                   分かること
    real               元の録音                                                ASR 自体の誤り(下限)
    mel_gl             正解 Mel → Griffin-Lim                                  Griffin-Lim による劣化
    codec_rec          正解トークン → codec decoder → Griffin-Lim               codec(量子化)による劣化
    s2m_gt_tokens      正解トークン → S2M → Griffin-Lim                         S2M による劣化
    tts_char           文字 → T2S → S2M → Griffin-Lim                          TTS 全体(文字入力)
    tts_phoneme        音素+アクセント → T2S → S2M → Griffin-Lim               TTS 全体(音素入力)

さらに、全実験で共通の評価用文章(datasets/eval_texts/。学習データと違う種類の文章)も、テキストからの合成
(tts_char / tts_phoneme)で読み上げて CER を測る。既定は実用文 55 文で、--with_ita で ITA コーパス 424 文も加える。

使い方:
    python evaluate.py                       # runs/ にある学習済みモデルを全部使う
    python evaluate.py --with_ita            # ITA コーパスも評価する
    python evaluate.py --limit 5 --device cpu   # 動作確認
出力:
    runs/eval/{条件}/{発話ID}.wav           全テスト文の音声(git の対象外)
    runs/eval_texts/{条件}/{文章ID}.wav     評価用文章の音声(git の対象外)
    results/metrics.json                    条件ごとの CER と、T2S が作った音声の長さ
    results/per_utterance.csv               テスト文の発話ごとの書き起こしと CER
    results/per_utterance_eval_texts.csv    評価用文章ごとの書き起こしと CER
    results/samples/test/{条件}/            試聴用: テスト文の先頭 N 文(GitHub に載せる分。48kbps の MP3)
    results/samples/eval_texts/{条件}/      試聴用: 実用文の各カテゴリの先頭 1 文(ITA を含めたら各セットの先頭 3 文も)
    results/samples/index.json              試聴用の音声の一覧(条件・文章・カテゴリ)。試聴ページはこれから作る
    results/fig_mels.jpg                    Mel の比較図
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf
import torch
from einops import rearrange

import common  # noqa: F401  tts/lib_common を import できるようにする
from common.pipeline import TTSPipeline, load_t2s
from common.utils import EXP_DIR, RUNS_DIR, encode_all
from lib_common.asr_eval import ASR, cer_scores, utterance_cer
from lib_common.audio import SR, load_wave, save_preview
from lib_common.data import load_dataset, split
from lib_common.eval_sets import load_eval_texts
from lib_common.utils import Logger, seed_everything

RESULTS_DIR = EXP_DIR / "results"
N_ITA_PREVIEW = 3  # ITA コーパスの各セットから試聴用にコピーする文数


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_steps", type=int, default=25, help="S2M の Euler 法のステップ数")
    parser.add_argument("--cfg", type=float, default=0.7, help="S2M の CFG の強さ")
    parser.add_argument("--repetition_penalty", type=float, default=10.0)
    parser.add_argument("--n_samples", type=int, default=5, help="results/samples/test にコピーするテスト文の数")
    parser.add_argument("--with_ita", action="store_true", help="評価用文章に ITA コーパス(424 文)も加える")
    parser.add_argument("--t2s_char", default="t2s_char", help="文字入力の T2S の runs/ の中の名前")
    parser.add_argument("--t2s_phoneme", default="t2s_phoneme", help="音素入力の T2S の runs/ の中の名前")
    parser.add_argument("--limit", type=int, default=None, help="動作確認用に発話数を絞る")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    torch.set_num_threads(4)
    RESULTS_DIR.mkdir(exist_ok=True)
    log = Logger(RUNS_DIR / "eval" / "log.txt")
    log(f"args: {vars(args)}")

    utts = load_dataset(args.limit, log)
    test = split(utts, "test")
    pipe = TTSPipeline.from_checkpoints(RUNS_DIR / "codec" / "codec.pt", RUNS_DIR / "s2m" / "s2m.pt",
                                        device=args.device)
    # 正解の音声から作ったトークン列(各 [U_i])。Mel の長さは 4 の倍数に切りそろえる
    test_mels = [pipe.normalizer.normalize(u.mel)[: len(u.mel) // 4 * 4] for u in test]
    gt_codes = encode_all(pipe.codec, test_mels, args.device)

    # 条件ごとに、正規化前の log-Mel [T_i, F] を用意する(real だけは元の波形をそのまま使う)
    mels = {"mel_gl": [u.mel for u in test],
            "codec_rec": pipe.codes_to_mel_codec(gt_codes),
            "s2m_gt_tokens": pipe.codes_to_mel(gt_codes, n_steps=args.n_steps, cfg=args.cfg)}
    eval_texts = load_eval_texts(["practical"] + (["ita_emotion", "ita_recitation"] if args.with_ita else []))
    if args.limit:
        eval_texts = eval_texts[: args.limit]
    gen_lengths, eval_text_mels = {}, {}
    for kind in ["char", "phoneme"]:
        path = RUNS_DIR / getattr(args, f"t2s_{kind}") / "t2s.pt"
        if not path.exists():
            log(f"skip tts_{kind}: {path.name} がない")
            continue
        pipe.t2s, pipe.vocab = load_t2s(path, args.device)  # 入力形式ごとに T2S を差し替える
        codes = pipe.text_to_codes([u.text for u in test], repetition_penalty=args.repetition_penalty)
        gen_lengths[f"tts_{kind}"] = [len(c) for c in codes]
        mels[f"tts_{kind}"] = pipe.codes_to_mel(codes, n_steps=args.n_steps, cfg=args.cfg)
        # 評価用文章(正解の音声は無いので、テキストからの合成だけ)
        codes = pipe.text_to_codes([t.text for t in eval_texts], repetition_penalty=args.repetition_penalty)
        eval_text_mels[f"tts_{kind}"] = pipe.codes_to_mel(codes, n_steps=args.n_steps, cfg=args.cfg)
        log(f"tts_{kind}: generated {len(test)} test utterances and {len(eval_texts)} eval texts")

    # 波形に戻して保存する
    conditions = ["real"] + list(mels)
    for cond in conditions:
        d = RUNS_DIR / "eval" / cond
        d.mkdir(parents=True, exist_ok=True)
        for i, u in enumerate(test):
            wave = load_wave(u.path) if cond == "real" else pipe.mel_to_wave(mels[cond][i]).numpy()
            sf.write(d / f"{u.id}.wav", wave, SR)
    log("wrote wavs")

    # ASR で書き起こして CER を測る
    asr = ASR(args.device)
    rows, metrics = [], {}
    for cond in conditions:
        paths = [str(RUNS_DIR / "eval" / cond / f"{u.id}.wav") for u in test]
        hyps = asr.transcribe(paths)
        metrics[cond] = cer_scores([u.text for u in test], hyps)
        for u, h in zip(test, hyps):
            rows.append({"condition": cond, "id": u.id, "ref": u.text, "hyp": h, "reading_cer": utterance_cer(u.text, h)})
        log(f"{cond:15s} CER raw {metrics[cond]['raw']:.4f} normalized {metrics[cond]['normalized']:.4f} "
            f"reading {metrics[cond]['reading']:.4f}")
    # T2S が決めた長さ(トークン数)と正解の長さの比
    for cond, lens in gen_lengths.items():
        ratio = [g / max(len(c), 1) for g, c in zip(lens, gt_codes)]
        metrics[cond]["length_ratio_mean"] = sum(ratio) / len(ratio)
        metrics[cond]["n_hit_max_len"] = sum(g >= 600 for g in lens)
    # 評価用文章: 波形に戻して保存し、セットごと・カテゴリごとの CER を測る
    et_rows = []
    metrics["eval_texts"] = {}
    for cond, et_mels in eval_text_mels.items():
        d = RUNS_DIR / "eval_texts" / cond
        d.mkdir(parents=True, exist_ok=True)
        for t, m in zip(eval_texts, et_mels):
            sf.write(d / f"{t.id}.wav", pipe.mel_to_wave(m).numpy(), SR)
        hyps = asr.transcribe([str(d / f"{t.id}.wav") for t in eval_texts])
        for t, h in zip(eval_texts, hyps):
            et_rows.append({"condition": cond, "set": t.set, "category": t.category, "id": t.id, "ref": t.text,
                            "hyp": h, "reading_cer": utterance_cer(t.text, h)})
        groups = {}  # セットごと(practical はカテゴリごとも)
        for t, h in zip(eval_texts, hyps):
            for key in {t.set, f"{t.set}/{t.category}"}:
                groups.setdefault(key, ([], []))
                groups[key][0].append(t.text)
                groups[key][1].append(h)
        metrics["eval_texts"][cond] = {k: cer_scores(r, h) for k, (r, h) in sorted(groups.items())}
        for k in sorted(k for k in groups if "/" not in k):
            log(f"{cond:15s} eval_texts {k:15s} reading CER {metrics['eval_texts'][cond][k]['reading']:.4f}")

    metrics["t2s_runs"] = {"tts_char": args.t2s_char, "tts_phoneme": args.t2s_phoneme}  # どの学習結果を使ったか
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "per_utterance.csv", index=False)
    pd.DataFrame(et_rows).to_csv(RESULTS_DIR / "per_utterance_eval_texts.csv", index=False)

    # 試聴用のコピーと、その一覧(index.json)
    index = []
    for cond in conditions:
        d = RESULTS_DIR / "samples" / "test" / cond
        d.mkdir(parents=True, exist_ok=True)
        for u in test[: args.n_samples]:
            save_preview(RUNS_DIR / "eval" / cond / f"{u.id}.wav", d / f"{u.id}.mp3")
            index.append({"group": "test", "condition": cond, "id": u.id, "set": "jsut_test",
                          "category": "jsut_test", "text": u.text, "file": f"test/{cond}/{u.id}.mp3"})
    # 試聴用に選ぶ文: 実用文はカテゴリごとの先頭 1 文、ITA は各セットの先頭 N_ITA_PREVIEW 文
    # (ITA の category はセットの名前と同じなので、(set, category) ごとに数えればよい)
    preview, taken = [], {}
    for t in eval_texts:
        key = (t.set, t.category)
        if taken.get(key, 0) < (1 if t.set == "practical" else N_ITA_PREVIEW):
            preview.append(t)
            taken[key] = taken.get(key, 0) + 1
    for cond in eval_text_mels:
        d = RESULTS_DIR / "samples" / "eval_texts" / cond
        d.mkdir(parents=True, exist_ok=True)
        for t in preview:
            save_preview(RUNS_DIR / "eval_texts" / cond / f"{t.id}.wav", d / f"{t.id}.mp3")
            index.append({"group": "eval_texts", "condition": cond, "id": t.id, "set": t.set, "category": t.category,
                          "text": t.text, "file": f"eval_texts/{cond}/{t.id}.mp3"})
    (RESULTS_DIR / "samples" / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False))

    # Mel の比較図
    show = [c for c in conditions if c != "real"]
    fig, axes = plt.subplots(len(show), 3, figsize=(15, 2.2 * len(show)), squeeze=False)
    for col, i in enumerate(range(min(3, len(test)))):
        for row, cond in enumerate(show):
            ax = axes[row, col]
            ax.imshow(rearrange(mels[cond][i], "t f -> f t"), origin="lower", aspect="auto", cmap="magma")  # 縦軸が周波数
            ax.set_title(f"{cond} | {test[i].id}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_mels.jpg", dpi=100)
    log(f"saved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
