"""分析: 音声トークン(codec)の質を、CER とは別の定量的な指標で比べる。

    A. 発音の情報(HuBERT の論文の指標。音素は jsut-lab の時刻から、各トークンの区間と一番長く重なる音素)
       phone purity:   トークンから音素を当てたときの正解率(トークン → 音素の対応は学習データの多数決で決め、テスト文で測る)
       cluster purity: 音素から、その音素で一番多いトークンを当てたときの正解率(テスト文)。1 つの音素が少数のトークンに
                       まとまっているほど高い(種類の多いトークンほど低くなりやすい)
       PNMI:           トークンと音素の相互情報量 / 音素のエントロピー(0〜1、テスト文)。トークンが音素の情報をどれだけ持つか
    B. トークンの統計: 1 秒あたりのトークン数、ビットレート(1 秒あたりのトークン数 × log2(種類数))、使われた種類数、
       perplexity(テスト文のトークンの分布)、1 秒あたりの切り替わりの回数(隣と違うトークンになる回数)
    D. 抑揚の情報: 正解トークンから戻した音声と元の録音で、F0(声の高さ。librosa の pyin)を比べる。
       GPE・FFE(F0 が 20% 以上ずれたフレームの割合など)と、それ以外のフレームでの相関・誤差(セント、1200 セント = 1 オクターブ)。
       (Griffin-Lim の音声では F0 の推定が数オクターブ低い値を拾うフレームがあり、全フレームの相関・二乗誤差はその外れ値に
       引っ張られて意味のない値になったので、外れ値を分けて数える)
       戻し方は S2M + Griffin-Lim(学習済みのもの)と、外部のコーデックではコーデック自身の decoder。基準として、
       正解の Mel を Griffin-Lim で戻しただけの音声(mel_gl)も測る

使い方:
    python analyze_codec_quality.py --variants mel_k256 w2vbert2_k256 hubert_ja_k256 bicodec wavtokenizer xcodec2
    (mel_k256 は exp0001 の codec_k256・s2m_k256。外部のコーデックは、学習済みの S2M が無ければ decoder だけ測る)
出力: results/codec_quality.json(前に測った種類の結果は残す)
"""
import argparse
import json
import math
from collections import Counter

import librosa
import numpy as np
import torch

import common  # noqa: F401  tts/lib_common・exp0001・exp0002 を import できるようにする
from common.ext_codec import EXT_CODECS
from common.utils import EXP_DIR, RUNS_DIR, tokenize
from evaluate import build_pipe
from exp0001.common.pipeline import TTSPipeline
from exp0001.common.utils import encode_all
from exp0002.common.utils import BASE_CODEC, BASE_S2M, TOKEN_SEC
from lib_common.audio import SR, load_wave
from lib_common.data import align_to_segments, load_dataset, load_jsut_alignment, split
from lib_common.utils import Logger, seed_everything

F0_SR, F0_HOP = 16000, 160  # F0 は 16kHz・10ms ごとに測る


def phone_labels(utts, codes, token_sec):
    return [[p for p, _ in align_to_segments(load_jsut_alignment(u), len(c), token_sec)] for u, c in zip(utts, codes)]


def phonetic(train_codes, train_ph, test_codes, test_ph) -> dict:
    """A. phone purity(学習データで対応を決めてテスト文で測る)・cluster purity・PNMI(テスト文)。"""
    by_tok = {}
    for cs, ps in zip(train_codes, train_ph):
        for c, p in zip(cs.tolist(), ps):
            by_tok.setdefault(c, Counter())[p] += 1
    tok2ph = {c: cnt.most_common(1)[0][0] for c, cnt in by_tok.items()}
    joint = Counter((c, p) for cs, ps in zip(test_codes, test_ph) for c, p in zip(cs.tolist(), ps))
    n = sum(joint.values())
    hit = sum(v for (c, p), v in joint.items() if tok2ph.get(c) == p)
    pc, pp = Counter(), Counter()
    for (c, p), v in joint.items():
        pc[c] += v
        pp[p] += v
    best_tok = Counter()
    for (c, p), v in joint.items():
        best_tok[p] = max(best_tok[p], v)
    mi = sum(v / n * math.log(v * n / (pc[c] * pp[p])) for (c, p), v in joint.items())
    h_ph = -sum(v / n * math.log(v / n) for v in pp.values())
    return {"phone_purity": hit / n, "cluster_purity": sum(best_tok.values()) / n, "pnmi": mi / h_ph,
            "unseen_token_rate": sum(v for (c, _), v in joint.items() if c not in tok2ph) / n}


def stats(all_codes, test_codes, token_sec: float, codebook_size: int) -> dict:
    """B. トークンの統計。"""
    cnt = Counter(c for cs in test_codes for c in cs.tolist())
    n = sum(cnt.values())
    ent = -sum(v / n * math.log(v / n) for v in cnt.values())
    changes = sum(int((cs[1:] != cs[:-1]).sum()) for cs in test_codes)
    sec = sum(len(cs) for cs in test_codes) * token_sec
    return {"tokens_per_sec": 1 / token_sec, "bitrate_bps": math.log2(codebook_size) / token_sec,
            "codebook_size": codebook_size, "used": len({c for cs in all_codes for c in cs.tolist()}),
            "perplexity_test": math.exp(ent), "changes_per_sec": changes / sec}


def f0(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """24kHz の波形 → (F0 [フレーム](無声は nan), 有声か [フレーム])。"""
    y = librosa.resample(y, orig_sr=SR, target_sr=F0_SR)
    f, voiced, _ = librosa.pyin(y, fmin=60, fmax=500, sr=F0_SR, frame_length=1024, hop_length=F0_HOP)
    return f, voiced


def prosody(ref_f0s: list, waves: list) -> dict:
    """D. 元の録音と比べた F0(フレームは短い方に合わせる)。TTS の評価で標準的な、外れ値に強い指標にする。

    GPE(gross pitch error): 両方が有声のフレームのうち、F0 が 20% 以上ずれた割合(F0 の推定が倍・半分を拾う誤りも含む)
    FFE(F0 frame error): 有声・無声の判定の不一致か、GPE にあたるフレームの、全フレームに対する割合
    f0_corr・f0_rmse_cent: GPE にあたらないフレームでの、log F0 の相関と誤差(セント)。抑揚の形がどれだけ合うか
    vuv_error: 有声・無声の判定の不一致率
    """
    xs, ys, vuv, gross, voiced_both, total = [], [], 0, 0, 0, 0
    for (rf, rv), w in zip(ref_f0s, waves):
        f, v = f0(w)
        n = min(len(rf), len(f))
        rf, rv, f, v = rf[:n], rv[:n], f[:n], v[:n]
        both = rv & v
        d = 1200 * np.log2(f[both] / rf[both])            # 両方が有声のフレームの差(セント)
        ok = np.abs(d) < 1200 * np.log2(1.2)              # 20% 未満のずれ
        xs.append(np.log2(rf[both][ok]))
        ys.append(np.log2(f[both][ok]))
        vuv, gross, voiced_both, total = vuv + int((rv != v).sum()), gross + int((~ok).sum()), voiced_both + int(both.sum()), total + n
    x, y = np.concatenate(xs), np.concatenate(ys)
    return {"gpe": gross / voiced_both, "ffe": (vuv + gross) / total, "vuv_error": vuv / total,
            "f0_corr": float(np.corrcoef(x, y)[0, 1]), "f0_rmse_cent": float(np.sqrt((((x - y) * 1200) ** 2).mean()))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(0)
    torch.set_num_threads(4)
    log = Logger(RUNS_DIR / "eval" / "log_codec_quality.txt")
    utts = load_dataset(None, log)
    train, test = split(utts, "train"), split(utts, "test")
    out_path = EXP_DIR / "results" / "codec_quality.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    log("F0 of the original recordings ...")
    ref = [f0(load_wave(u.path)) for u in test]
    if "mel_gl" not in out:  # 基準: 正解の Mel を Griffin-Lim で戻しただけ
        pipe = TTSPipeline.from_checkpoints(BASE_CODEC, BASE_S2M, device=args.device)
        out["mel_gl"] = {"prosody_s2m": prosody(ref, [pipe.mel_to_wave(u.mel).numpy() for u in test])}
        log(f"mel_gl {out['mel_gl']}")

    for v in args.variants:
        res, ext = {}, v in EXT_CODECS
        if v == "mel_k256":  # exp0001 の Mel の codec(codebook 256)
            pipe = TTSPipeline.from_checkpoints(BASE_CODEC, BASE_S2M, device=args.device)
            norm = lambda us: [pipe.normalizer.normalize(u.mel) for u in us]  # noqa: E731
            tr, te = encode_all(pipe.codec, norm(train), args.device), encode_all(pipe.codec, norm(test), args.device)
            token_sec, K = TOKEN_SEC, pipe.codec.codebook_size
        else:
            spec = f"ext:{v}" if ext else RUNS_DIR / f"codec_{v}" / "codec.pt"
            tr, codec, _ = tokenize(spec, train, args.device, log)
            te, _, _ = tokenize(spec, test, args.device, log)
            token_sec, K = codec.token_sec, codec.codebook_size
            s2m_dir = RUNS_DIR / f"s2m_{v}"  # 学習が終わった S2M があるときだけ使う(学習中の途中の checkpoint は使わない)
            done = (s2m_dir / "log.txt").exists() and "best val loss" in (s2m_dir / "log.txt").read_text()
            pipe = build_pipe(v, s2m_dir / "s2m.pt", None, args.device) if done else None
        res["phonetic"] = phonetic(tr, phone_labels(train, tr, token_sec), te, phone_labels(test, te, token_sec))
        res["stats"] = stats(tr + te, te, token_sec, K)
        if pipe is not None:  # 正解トークン → S2M + Griffin-Lim
            waves = [pipe.mel_to_wave(m).numpy() for m in pipe.codes_to_mel(te, texts=[u.text for u in test])]
            res["prosody_s2m"] = prosody(ref, waves)
        if ext:  # 正解トークン → コーデック自身の decoder
            dec = EXT_CODECS[v](args.device)
            if hasattr(dec, "set_reference"):
                dec.set_reference(train[0].path)
            res["prosody_decoder"] = prosody(ref, [dec.decode(c) for c in te])
            del dec
        out[v] = res
        log(f"{v} {json.dumps(res)}")
        out_path.write_text(json.dumps(out, indent=2))
        del pipe
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
