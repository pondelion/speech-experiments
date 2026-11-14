"""抑揚(声の高さ・長さ・アクセント)の評価。CER に出ない「読み方の自然さ」を、正解の録音と比べて数値にする。

F0(声の高さ)は librosa の pyin で、Mel と同じフレーム(24kHz、HOP 256)ごとに求める。
    compare_f0: 同じ長さの音声(正解の Mel から戻した音声など)を、フレームをそのまま対応させて比べる
    compare_f0_dtw: 長さの違う音声(TTS)を、Mel の近さで DTW により時間を合わせてから比べる
    指標: GPE(両方が有声のフレームのうち F0 が 20% 以上ずれた割合)、FFE(有声・無声の判定の不一致か GPE にあたる
          フレームの割合)、それ以外のフレームでの log F0 の相関と誤差(セント。100 セント = 半音)
accent_accuracy: pyopenjtalk のアクセント記号の前後で、F0 が正しい向きに動いているかの割合。
    「[」(声が上がる): 記号の後の母音の F0 > 前の母音の F0、「]」(アクセント核の後で下がる): 後 < 前 なら正解。
    各母音の F0 は、その母音の区間の有声のフレームの中央値。音素の区間は、正解の録音では jsut-lab の時刻、
    TTS では T2S が予測した音素ごとの長さから決める。日本語は高さのアクセント(高低)で単語を区別するので、
    ここが崩れると「外国人が話しているような」不自然さになる
"""
import librosa
import numpy as np
import torch

from .audio import HOP, SR

VOWELS = set("aiueo")  # 有声の母音(大文字は無声化した母音なので使わない)


def f0(wave: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """24kHz の波形 → (F0 [フレーム](無声は nan), 有声か [フレーム])。フレームは Mel と同じ(HOP 256、中心をそろえる)。"""
    f, voiced, _ = librosa.pyin(wave, fmin=60, fmax=500, sr=SR, frame_length=1024, hop_length=HOP)
    return f, voiced


def _summary(pairs_ref: list, pairs_hyp: list, ref_v: list, hyp_v: list) -> dict:
    """対応させたフレームの F0 と有声の判定から、GPE・FFE・相関・誤差を求める。"""
    rf, hf, rv, hv = map(np.concatenate, (pairs_ref, pairs_hyp, ref_v, hyp_v))
    both = rv & hv
    d = 1200 * np.log2(hf[both] / rf[both])
    ok = np.abs(d) < 1200 * np.log2(1.2)
    x, y = np.log2(rf[both][ok]), np.log2(hf[both][ok])
    return {"gpe": float((~ok).mean()), "ffe": float(((rv != hv).sum() + (~ok).sum()) / len(rv)),
            "f0_corr": float(np.corrcoef(x, y)[0, 1]), "f0_rmse_cent": float(np.sqrt((((x - y) * 1200) ** 2).mean()))}


def compare_f0(refs: list, hyps: list) -> dict:
    """同じ長さの音声どうしの F0 の比較。refs・hyps: f0() の結果のリスト(フレームは短い方に合わせる)。"""
    out = [[], [], [], []]
    for (rf, rv), (hf, hv) in zip(refs, hyps):
        n = min(len(rf), len(hf))
        for lst, a in zip(out, (rf[:n], hf[:n], rv[:n], hv[:n])):
            lst.append(a)
    return _summary(*out)


def compare_f0_dtw(refs: list, hyps: list, ref_mels: list, hyp_mels: list) -> dict:
    """長さの違う音声どうしの F0 の比較。Mel [T, F] の近さで DTW の対応をとり、対応したフレームどうしを比べる。"""
    out = [[], [], [], []]
    for (rf, rv), (hf, hv), rm, hm in zip(refs, hyps, ref_mels, hyp_mels):
        _, path = librosa.sequence.dtw(X=np.asarray(rm).T, Y=np.asarray(hm).T, metric="euclidean")
        i, j = path[::-1, 0], path[::-1, 1]
        i, j = np.minimum(i, len(rf) - 1), np.minimum(j, len(hf) - 1)
        for lst, a in zip(out, (rf[i], hf[j], rv[i], hv[j])):
            lst.append(a)
    return _summary(*out)


def accent_accuracy(f0s: list, tokens: list[list[str]], durations: list, frames_per_unit: float) -> dict:
    """アクセント記号の前後の母音の F0 の向きが正しい割合。

    Args:
        f0s: 発話ごとの f0() の結果。
        tokens: 発話ごとのテキストのトークン(lib_common.text.phoneme_tokens。<bos>・<eos> は含まない)。
        durations: 発話ごとの各トークンの長さ [<bos> + トークン + <eos>](長さの単位は 1 つの音声トークン)。
        frames_per_unit: 長さの単位 1 つあたりの Mel のフレーム数(exp0001〜0003 の音声トークンなら 4)。

    Returns:
        {"accent_acc": 正解の割合, "rise_acc": 「[」の正解の割合, "fall_acc": 「]」の正解の割合, "n": 数えた記号の数}。
    """
    hits = {"[": [0, 0], "]": [0, 0]}
    for (f, v), toks, dur in zip(f0s, tokens, durations):
        dur = np.asarray(dur)[1:-1]                                                  # <bos>・<eos> を除く
        ends = np.cumsum(dur) * frames_per_unit
        starts = ends - dur * frames_per_unit

        def vowel_f0(idx):  # 母音のトークン idx の区間の、有声のフレームの F0 の中央値(無ければ None)
            a, b = int(starts[idx]), int(np.ceil(ends[idx]))
            seg = f[a:b][v[a:b]]
            return float(np.median(seg)) if len(seg) else None

        for k, t in enumerate(toks):
            if t not in hits:
                continue
            before = next((vowel_f0(i) for i in range(k - 1, -1, -1) if toks[i] in VOWELS), None)
            after = next((vowel_f0(i) for i in range(k + 1, len(toks)) if toks[i] in VOWELS), None)
            if before is None or after is None:
                continue
            hits[t][0] += int(after > before if t == "[" else after < before)
            hits[t][1] += 1
    n = hits["["][1] + hits["]"][1]
    return {"accent_acc": (hits["["][0] + hits["]"][0]) / max(n, 1), "rise_acc": hits["["][0] / max(hits["["][1], 1),
            "fall_acc": hits["]"][0] / max(hits["]"][1], 1), "n": n}


def duration_error(pred: list, gt: list, tokens: list[list[str]], sec_per_unit: float, marks: set) -> float:
    """音を持つトークン(アクセントの記号・<bos>・<eos> 以外)の長さの誤差の平均(ミリ秒)。

    Args:
        pred, gt: 発話ごとの予測・正解の長さ [<bos> + トークン + <eos>](正解が作れない発話は None で、除く)。
        tokens: 発話ごとのテキストのトークン。
        sec_per_unit: 長さの単位 1 つの秒数。
        marks: 音を持たない記号(長さ 0)。
    """
    errs = []
    for p, g, toks in zip(pred, gt, tokens):
        if g is None:
            continue
        p, g = np.asarray(p)[1:-1], np.asarray(g)[1:-1]
        keep = np.array([t not in marks for t in toks])
        errs.append(np.abs(p[keep] - g[keep]) * sec_per_unit * 1000)
    return float(np.concatenate(errs).mean()) if errs else float("nan")
