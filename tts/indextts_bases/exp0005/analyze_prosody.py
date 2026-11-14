"""抑揚の崩れの切り分け(学習し直さずにできる分析。README の分析 1〜5)。基準は exp0004 の構成(日本語 HuBERT + ハイブリッド型 + Vocos)。

    1. 段階ごとの F0 のずれ: テスト 100 文で、次の経路の F0 を同じ比べ方(DTW)で元の録音と比べる
         s1 正解の Mel → Vocos / s2 正解のトークン → S2M → Vocos / s3 正解の音素ごとの長さ + T2S → S2M → Vocos /
         s4 テキスト → T2S → S2M → Vocos(実際に使うときと同じ)
       あわせて、モーラ単位の指標(元の録音との F0 の動きの相関、辞書のアクセントとの一致率)も測る
    2. S2M の乱数のばらつき: 同じトークン列(正解のトークン / T2S の 1 回の生成)を、S2M の乱数の種だけ変えて 3 回 Mel にし、
       同じ位置の F0 がどれだけ違うか(3 回のうちの 2 つずつの組の F0 の誤差・GPE)を測る
    3. トークンが持つ高さの情報: 学習データの 500 文で「トークンの番号 → そのトークンの区間の log F0 の平均」の表を作り、
       テスト文で、その表で F0 をどれだけ当てられるか(相関)を、Mel の codec・日本語 HuBERT・XCodec2 のトークンで比べる
    5. 悪い例: s4 で F0 の誤差が大きい 5 文を、元の録音と並べて試聴用に保存する
(4. のアクセントの指標の作り直しは lib_common.prosody の mora_phrases・mora_accent_scores)

使い方:
    python analyze_prosody.py
出力: results/prosody_analysis.json、results/samples/(悪い例の試聴用と index.json)。音声は runs/(git の対象外)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import common  # noqa: F401
from exp0001.common.pipeline import TTSPipeline
from exp0001.common.utils import encode_all
from exp0002.common.utils import load_t2s, phoneme_durations
from lib_common.audio import SR, load_wave, save_preview
from lib_common.data import load_dataset, split
from lib_common.neural_vocoder import load_vocoder
from lib_common.prosody import compare_f0, compare_f0_dtw, f0, mora_accent_scores, mora_phrases
from lib_common.text import phoneme_tokens
from lib_common.utils import Logger, seed_everything

EXP = Path(__file__).resolve().parent
BASES = EXP.parent
E3, E4 = BASES / "exp0003" / "runs", BASES / "exp0004" / "runs"
FPT = 4  # 音声トークン 1 個 = Mel 4 フレーム


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="動作確認用にテスト文を絞る(results/ は書き換えない)")
    args = parser.parse_args()
    seed_everything(0)
    torch.set_num_threads(4)
    log = Logger(EXP / "runs" / "log_analyze.txt")
    utts = load_dataset(None, log)
    train, test = split(utts, "train"), split(utts, "test")[: args.limit]
    results = EXP / ("runs/smoke_results" if args.limit else "results")
    pipe = TTSPipeline.from_checkpoints(E3 / "codec_hubert_ja_k256" / "codec.pt", E3 / "s2m_hubert_ja_k256" / "s2m.pt")
    t2s, vocab, _ = load_t2s(E3 / "t2s_hubert_ja_k256" / "t2s.pt", "cuda")
    voc = load_vocoder(E4 / "vocos" / "vocoder.pt", "cuda")
    wave = lambda m: voc.generate(m.cuda()[None])[0].cpu().numpy()  # noqa: E731
    tokens_hubert = torch.load(E3 / "codec_hubert_ja_k256" / "tokens.pt")
    toks = [phoneme_tokens(u.text) for u in test]
    gt_durs = [phoneme_durations(u, tk, len(u.mel) // FPT) for u, tk in zip(test, toks)]
    ok = [i for i, d in enumerate(gt_durs) if d is not None]
    log(f"ground-truth durations: {len(ok)}/{len(test)}")
    ref_f0 = [f0(load_wave(u.path)) for u in test]
    ref_ph = [mora_phrases(toks[i], gt_durs[i], FPT) if gt_durs[i] is not None else None for i in range(len(test))]
    gt_tokens = [tokens_hubert[u.id].long() for u in test]

    def t2s_codes(durations=None):
        out, durs = [], []
        for i, u in enumerate(test):
            ids = [torch.tensor(vocab.encode(u.text))]
            d = [torch.tensor(durations[i])] if durations is not None and durations[i] is not None else t2s.predict_durations(ids)
            out += t2s.generate(ids, durations=d)
            durs.append(d[0].cpu().numpy())
        return out, durs

    def measure(name, mels, durs):
        """F0(DTW)とモーラ単位の指標(正解の長さがある文だけ)。"""
        f0s = [f0(wave(m)) for m in mels]
        res = compare_f0_dtw(ref_f0, f0s, [u.mel for u in test], mels)
        ph = [mora_phrases(toks[i], durs[i], FPT) for i in ok]
        res.update(mora_accent_scores([f0s[i] for i in ok], ph, [ref_f0[i] for i in ok], [ref_ph[i] for i in ok]))
        log(f"{name:28s} " + " ".join(f"{k} {v:.3f}" for k, v in res.items() if not k.startswith("n_")))
        return res, f0s

    out = {}
    # 0. 元の録音(辞書のアクセントとの一致率の上限)
    out["real"] = mora_accent_scores([ref_f0[i] for i in ok], [ref_ph[i] for i in ok])
    log(f"real hl_agreement {out['real']['hl_agreement']:.3f}")
    # 1. 段階ごと
    out["s1_gt_mel"], _ = measure("s1 gt mel", [u.mel for u in test], gt_durs)
    mel_gt_tok = pipe.codes_to_mel(gt_tokens)
    out["s2_gt_tokens"], _ = measure("s2 gt tokens -> S2M", mel_gt_tok, gt_durs)
    codes3, durs3 = t2s_codes(gt_durs)
    out["s3_gt_durations"], _ = measure("s3 gt durations + T2S", pipe.codes_to_mel(codes3), durs3)
    codes4, durs4 = t2s_codes()
    mel4 = pipe.codes_to_mel(codes4)
    out["s4_full"], f0s4 = measure("s4 text -> T2S -> S2M", mel4, durs4)
    # 2. S2M の乱数のばらつき(同じトークン列、S2M の乱数の種だけ変える)
    for name, codes, durs in [("gt_tokens", gt_tokens, gt_durs), ("t2s_tokens", codes4, durs4)]:
        runs = []
        for seed in range(3):
            torch.manual_seed(100 + seed)
            runs.append([f0(wave(m)) for m in pipe.codes_to_mel(codes)])
        pairs = [compare_f0(runs[a], runs[b]) for a, b in [(0, 1), (0, 2), (1, 2)]]  # 同じ長さなのでフレームをそのまま対応
        out[f"s2m_seed_spread_{name}"] = {k: float(np.mean([p[k] for p in pairs])) for k in pairs[0]}
        log(f"S2M seed spread ({name}) {out[f's2m_seed_spread_{name}']}")
    # 3. トークンが持つ高さの情報(トークンの番号 → log F0 の平均の表で、テスト文の F0 を当てる)
    out["token_f0_probe"] = token_probe(train[: (args.limit or 100) * 5], test, tokens_hubert, log)
    # 5. 悪い例(s4 の F0 の誤差が大きい 5 文)
    errs = []
    for i, u in enumerate(test):
        e = compare_f0_dtw([ref_f0[i]], [f0s4[i]], [u.mel], [mel4[i]])
        errs.append((e["f0_rmse_cent"] + 1000 * e["gpe"], i))
    worst = [i for _, i in sorted(errs, reverse=True)[:5]]
    out["worst"] = [{"id": test[i].id, "text": test[i].text} for i in worst]
    index = []
    for i in worst:
        u = test[i]
        for cond, w in [("worst/real", load_wave(u.path)), ("worst/tts_vocos", wave(mel4[i]))]:
            d = EXP / "runs" / "eval" / cond
            d.mkdir(parents=True, exist_ok=True)
            sf.write(d / f"{u.id}.wav", w, SR)
            sd = results / "samples" / "test" / cond
            sd.mkdir(parents=True, exist_ok=True)
            save_preview(d / f"{u.id}.wav", sd / f"{u.id}.mp3")
            index.append({"group": "worst", "condition": cond, "id": u.id, "text": u.text, "category": "jsut_test",
                          "file": f"test/{cond}/{u.id}.mp3"})
    (results / "samples" / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False))
    (results / "prosody_analysis.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log("saved")


def token_probe(train, test, tokens_hubert, log) -> dict:
    """トークンの番号 → そのトークンの区間の log F0 の平均、の表で、テスト文の F0 をどれだけ当てられるか(有声のトークンの相関)。"""
    xcodec = torch.load(E3 / "codec_xcodec2" / "tokens.pt")  # XCodec2 のトークンは exp0003 で作ったもの
    mel_codec = TTSPipeline.from_checkpoints(BASES / "exp0001" / "runs" / "codec_k256" / "codec.pt",
                                             BASES / "exp0001" / "runs" / "s2m_k256" / "s2m.pt")
    sources = {
        "mel_codec_k256": (lambda us: encode_all(mel_codec.codec, [mel_codec.normalizer.normalize(u.mel) for u in us], "cuda"), FPT),
        "hubert_ja_k256": (lambda us: [tokens_hubert[u.id] for u in us], FPT),
        "xcodec2": (lambda us: [xcodec[u.id].long() for u in us], 93.75 / 50),
    }
    f0_train = [f0(load_wave(u.path)) for u in train]
    f0_test = [f0(load_wave(u.path)) for u in test]
    res = {}
    for name, (get, fpt) in sources.items():
        def per_token(f0s, codes):  # 各トークンの区間の log F0 の平均(有声のフレームが無いトークンは除く)
            ids, vals = [], []
            for (f, v), c in zip(f0s, codes):
                for j, t in enumerate(c.tolist()):
                    a, b = int(j * fpt), int((j + 1) * fpt)
                    if v[a:b].any():
                        ids.append(t)
                        vals.append(np.log2(np.nanmean(f[a:b][v[a:b]])))
            return np.array(ids), np.array(vals)
        tr_ids, tr_vals = per_token(f0_train, get(train))
        te_ids, te_vals = per_token(f0_test, get(test))
        table, mean = {}, tr_vals.mean()
        for t in np.unique(tr_ids):
            table[t] = tr_vals[tr_ids == t].mean()
        pred = np.array([table.get(t, mean) for t in te_ids])
        res[name] = {"corr": float(np.corrcoef(pred, te_vals)[0, 1]),
                     "rmse_cent": float(np.sqrt(((pred - te_vals) ** 2).mean()) * 1200),
                     "std_cent": float(te_vals.std() * 1200)}  # 参考: 何も当てない(平均で当てる)ときの誤差
        log(f"token F0 probe {name}: {res[name]}")
    return res


if __name__ == "__main__":
    main()
