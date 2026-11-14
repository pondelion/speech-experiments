"""NN ボコーダの評価: 読みの正確さ(CER)・自然さの推定(UTMOS)・抑揚(F0・アクセント・長さ)・推論の速さを、Griffin-Lim と比べる。

条件(テスト 100 文。TTS は実用文 55 文も):
    real                元の録音(上限)
    copy/{ボコーダ}      正解の Mel → ボコーダ(ボコーダ単独の性能)
    tts/{ボコーダ}       テキスト → exp0003 の日本語 HuBERT の構成(ハイブリッド型の T2S + S2M)の Mel → ボコーダ。
                        Mel は 1 回だけ作り、全てのボコーダで同じものを使う(違いはボコーダだけ)
ボコーダ: griffinlim(これまでの 64 回)・vocos・hifigan_v2(runs/{名前}/vocoder.pt)
指標(lib_common.prosody):
    CER(読み)、UTMOS(自然さの MOS の推定)、F0 の GPE・相関・誤差(copy はフレームをそのまま、tts は DTW で時間を合わせる)、
    アクセントの正しさ(記号の前後の母音の F0 の向き。real・copy は jsut-lab の時刻、tts は T2S の予測した長さ)、
    音素ごとの長さの誤差(tts の T2S の予測と jsut-lab。ボコーダに依らない)、推論の速さ(RTF = 生成時間 / 音声の長さ。GPU と CPU)

使い方:
    python evaluate.py --vocoders vocos hifigan_v2
出力: results/metrics.json、results/samples/(48kbps の MP3)と index.json。全ての音声は runs/eval/(git の対象外)
"""
import argparse
import json
import time

import soundfile as sf
import torch

import common
from exp0001.common.pipeline import TTSPipeline
from exp0002.common.utils import PROSODY_MARKS, TOKEN_SEC, load_t2s, phoneme_durations
from lib_common.asr_eval import ASR, cer_scores
from lib_common.audio import SR, load_wave, save_preview
from lib_common.data import load_dataset, split
from lib_common.eval_sets import load_eval_texts
from lib_common.mos import UTMOS
from lib_common.neural_vocoder import load_vocoder
from lib_common.prosody import accent_accuracy, compare_f0, compare_f0_dtw, duration_error, f0
from lib_common.utils import Logger, seed_everything
from lib_common.vocoder import GriffinLim

E3 = common.EXP_DIR.parent / "exp0003" / "runs"
RESULTS = common.EXP_DIR / "results"
FRAMES_PER_TOKEN = 4  # 音声トークン 1 個 = Mel 4 フレーム


def rtf(fn, mels: list[torch.Tensor]) -> float:
    """生成時間の合計 / 音声の長さの合計(1 より小さいほど実時間より速い)。最初の 1 回は準備の分なので測らない。"""
    fn(mels[0])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time.time()
    for m in mels:
        fn(m)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.time() - t) / (sum(len(m) for m in mels) * 256 / SR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocoders", nargs="+", default=["vocos", "hifigan_v2"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None, help="動作確認用に文を絞る(results/ は書き換えない)")
    args = parser.parse_args()
    seed_everything(0)
    torch.set_num_threads(4)
    log = Logger(common.RUNS_DIR / "eval" / "log.txt")
    test = split(load_dataset(None, log), "test")[: args.limit]
    ets = load_eval_texts(["practical"])[: args.limit]
    global RESULTS
    if args.limit:
        RESULTS = common.RUNS_DIR / "eval_smoke_results"

    # ボコーダ: Mel [T, F](正規化前の log-Mel)→ 24kHz の波形 [N]
    gl = GriffinLim(args.device, n_iter=64)
    vocs = {"griffinlim": lambda m: gl(m.to(args.device)).cpu()}
    for name in args.vocoders:
        v = load_vocoder(common.RUNS_DIR / name / "vocoder.pt", args.device)
        vocs[name] = lambda m, v=v: v.generate(m.to(args.device)[None])[0].cpu()

    # TTS の Mel(exp0003 の日本語 HuBERT の構成。1 回だけ作る)と、T2S が予測した音素ごとの長さ
    pipe = TTSPipeline.from_checkpoints(E3 / "codec_hubert_ja_k256" / "codec.pt", E3 / "s2m_hubert_ja_k256" / "s2m.pt",
                                        device=args.device)
    t2s, vocab, _ = load_t2s(E3 / "t2s_hubert_ja_k256" / "t2s.pt", args.device)

    def tts(texts):
        mels, durs = [], []
        for t in texts:
            ids = [torch.tensor(vocab.encode(t))]
            d = t2s.predict_durations(ids)
            mels += pipe.codes_to_mel(t2s.generate(ids, durations=d))
            durs.append(d[0].cpu())
        return mels, durs

    tts_test, tts_durs = tts([u.text for u in test])
    tts_et, _ = tts([t.text for t in ets])
    toks = [vocab.tokenize(u.text) for u in test]
    gt_durs = [phoneme_durations(u, tk, len(u.mel) // FRAMES_PER_TOKEN) for u, tk in zip(test, toks)]
    log(f"TTS mels done; ground-truth durations available for {sum(d is not None for d in gt_durs)}/{len(test)}")

    # 音声を作って保存する: {条件: (文のリスト, 波形のリスト)}
    conds = {"real": (test, [torch.from_numpy(load_wave(u.path)) for u in test])}
    for name, fn in vocs.items():
        conds[f"copy/{name}"] = (test, [fn(u.mel) for u in test])
        conds[f"tts/{name}"] = (test, [fn(m) for m in tts_test])
        conds[f"tts/{name}@eval_texts"] = (ets, [fn(m) for m in tts_et])
    asr, mos = ASR(args.device), UTMOS(args.device)
    ref_f0 = [f0(w.numpy()) for w in conds["real"][1]]
    metrics, index = {}, []
    for cond, (items, waves) in conds.items():
        group = "eval_texts" if cond.endswith("@eval_texts") else "test"
        name = cond.replace("@eval_texts", "")
        d = common.RUNS_DIR / "eval" / group / name
        d.mkdir(parents=True, exist_ok=True)
        paths = []
        for x, w in zip(items, waves):
            sf.write(d / f"{x.id}.wav", w.numpy(), SR)
            paths.append(str(d / f"{x.id}.wav"))
        res = {"cer": cer_scores([x.text for x in items], asr.transcribe(paths))["reading"],
               "utmos": sum(mos.score(paths)) / len(paths)}
        if group == "test":
            f0s = [f0(w.numpy()) for w in waves]
            if name.startswith("tts/"):
                res.update(compare_f0_dtw(ref_f0, f0s, [u.mel for u in test], tts_test))
                res.update(accent_accuracy(f0s, toks, [x.numpy() for x in tts_durs], FRAMES_PER_TOKEN))
            else:  # 元の録音・正解の Mel から戻した音声は、元の録音と同じ時刻なので jsut-lab の長さを使う
                if name != "real":
                    res.update(compare_f0(ref_f0, f0s))
                ok = [i for i, g in enumerate(gt_durs) if g is not None]
                res.update(accent_accuracy([f0s[i] for i in ok], [toks[i] for i in ok], [gt_durs[i] for i in ok],
                                           FRAMES_PER_TOKEN))
        metrics.setdefault(group, {})[name] = res
        log(f"{group:10s} {name:22s} " + " ".join(f"{k} {v:.3f}" for k, v in res.items() if k != "n"))
        # 試聴用: テスト文の先頭 3 文、実用文はカテゴリごとの先頭 1 文
        firsts = {}
        for x in items:
            firsts.setdefault(getattr(x, "category", len(firsts)), x)
        preview = items[:3] if group == "test" else list(firsts.values())
        sd = RESULTS / "samples" / group / name
        sd.mkdir(parents=True, exist_ok=True)
        for x in preview:
            save_preview(d / f"{x.id}.wav", sd / f"{x.id}.mp3")
            index.append({"group": group, "condition": name, "id": x.id, "text": x.text,
                          "category": getattr(x, "category", "jsut_test"), "file": f"{group}/{name}/{x.id}.mp3"})
    metrics["tts_duration_error_ms"] = duration_error([x.numpy() for x in tts_durs], gt_durs, toks, TOKEN_SEC, PROSODY_MARKS)
    metrics["tts_length_ratio"] = sum(len(m) / len(u.mel) for m, u in zip(tts_test, test)) / len(test)

    # 推論の速さ: GPU はテスト文 20 文、CPU は 5 文(CPU は 4 スレッド)
    speed = {}
    for name, fn in vocs.items():
        speed[name] = {"rtf_gpu": rtf(fn, [u.mel for u in test[:20]])}
    for name in vocs:
        if name == "griffinlim":
            g = GriffinLim("cpu", n_iter=64)
            fn = lambda m: g(m)  # noqa: E731
        else:
            v = load_vocoder(common.RUNS_DIR / name / "vocoder.pt", "cpu")
            fn = lambda m, v=v: v.generate(m[None])[0]  # noqa: E731
        speed[name]["rtf_cpu"] = rtf(fn, [u.mel for u in test[:5]])
    metrics["speed"] = speed
    log(f"speed {speed}")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    (RESULTS / "samples" / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False))
    log("saved")


if __name__ == "__main__":
    main()
