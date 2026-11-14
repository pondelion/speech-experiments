"""テスト文・実用文での評価(exp0005 の Evaluator と同じ指標を、22.05kHz・自己回帰の T2S で測れるようにしたもの)。

- 読み: ASR の CER(読みの文字誤り率)
- 自然さ: UTMOS
- 長さ: 合成した Mel の長さ / 元の録音の長さ
- 高さ(F0): 元の録音と Mel の近さで DTW の対応をとり、対応したフレームどうしで GPE・FFE・相関・誤差(セント)
- モーラの抑揚: 元の録音のモーラの区間(jsut-lab の時刻)を DTW の対応で合成音声の時間に写し、モーラごとの F0 で
  「辞書のアクセントとの一致率」と「元の録音との F0 の動きの相関」を測る(自己回帰の T2S は音素ごとの長さを出さないため。
  exp0005 は T2S が予測した長さでモーラの区間を作っていた。DTW の対応がずれると、その分だけ指標も悪く出る)
ボコーダは BigVGAN-v2(22.05kHz)。まだ無いときは Griffin-Lim を使う(途中の確認用)。
"""
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from exp0002.common.utils import phoneme_durations
from lib_common.asr_eval import ASR, cer_scores
from lib_common.audio import HOP, SR, load_wave
from lib_common.mos import UTMOS
from lib_common.neural_vocoder import load_vocoder
from lib_common.prosody import compare_f0_dtw, f0, mora_accent_scores, mora_phrases
from lib_common.text import phoneme_tokens
from lib_common.vocoder import GriffinLim


def _dtw_map(ref_mel: np.ndarray, hyp_mel: np.ndarray) -> np.ndarray:
    """元の録音のフレーム i → 合成音声のフレーム(DTW で i に対応したフレームの中央値)[T_ref]。"""
    _, path = librosa.sequence.dtw(X=ref_mel.T, Y=hyp_mel.T, metric="euclidean")
    path = path[::-1]
    out = np.zeros(len(ref_mel), dtype=np.int64)
    for i in range(len(ref_mel)):
        out[i] = int(np.median(path[path[:, 0] == i, 1]))
    return out


class Evaluator:
    def __init__(self, test: list, vocoder_path: Path | None, device: str = "cuda"):
        self.test, self.device = test, device
        self.voc = load_vocoder(vocoder_path, device) if vocoder_path else None
        self.gl = None if self.voc else GriffinLim(device)
        self.asr, self.mos = ASR(device), UTMOS(device)
        toks = [phoneme_tokens(u.text) for u in test]
        # jsut-lab の時刻 → Mel のフレームごとの長さ(1 単位 = 1 フレーム)。音素の並びが pyopenjtalk と違う発話は除く
        durs = [phoneme_durations(u, tk, len(u.mel), token_sec=HOP / SR) for u, tk in zip(test, toks)]
        self.ok = [i for i, d in enumerate(durs) if d is not None]
        self.ref_ph = {i: mora_phrases(toks[i], durs[i], 1.0) for i in self.ok}
        self.ref_f0 = [f0(load_wave(u.path)) for u in test]

    def wave(self, mel: torch.Tensor) -> np.ndarray:
        if self.voc:
            return self.voc.generate(mel.to(self.device)[None])[0].cpu().numpy()
        return self.gl(mel).cpu().numpy()

    def save(self, mels: list, items: list, out_dir: Path) -> list[str]:
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for x, m in zip(items, mels):
            sf.write(out_dir / f"{x.id}.wav", self.wave(m), SR)
            paths.append(str(out_dir / f"{x.id}.wav"))
        return paths

    def practical(self, items: list, mels: list, out_dir: Path) -> dict:
        """実用文(正解の録音が無い): CER と UTMOS。"""
        paths = self.save(mels, items, out_dir)
        return {"cer": cer_scores([t.text for t in items], self.asr.transcribe(paths))["reading"],
                "utmos": float(np.mean(self.mos.score(paths)))}

    def __call__(self, mels: list[torch.Tensor], out_dir: Path) -> dict:
        """テスト文の Mel [T_i, F](log-Mel)のリスト → 指標の辞書。"""
        paths = self.save(mels, self.test, out_dir)
        f0s = [f0(load_wave(p)) for p in paths]
        res = {"cer": cer_scores([u.text for u in self.test], self.asr.transcribe(paths))["reading"],
               "utmos": float(np.mean(self.mos.score(paths))),
               "length_ratio": float(np.mean([len(m) / len(u.mel) for m, u in zip(mels, self.test)]))}
        hyp_mels = [m.float().cpu().numpy() for m in mels]
        res.update(compare_f0_dtw(self.ref_f0, f0s, [u.mel.numpy() for u in self.test], hyp_mels))
        hyp_ph = []
        for i in self.ok:
            mp = _dtw_map(self.test[i].mel.numpy(), hyp_mels[i])
            hyp_ph.append([[(int(mp[a]), int(mp[min(b, len(mp)) - 1]) + 1, h) for a, b, h in ph] for ph in self.ref_ph[i]])
        res.update(mora_accent_scores([f0s[i] for i in self.ok], hyp_ph, [self.ref_f0[i] for i in self.ok],
                                      [self.ref_ph[i] for i in self.ok]))
        return res
