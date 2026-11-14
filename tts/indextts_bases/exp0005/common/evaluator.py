"""exp0005 の評価: テスト 100 文の TTS の Mel を Vocos で波形にし、読みの正確さ・自然さ・抑揚を測る。

指標: CER(読み)、UTMOS、F0 の GPE・誤差(DTW で元の録音と時間を合わせる)、モーラの F0 の動きの元の録音との相関、
辞書のアクセントとの一致率(lib_common.prosody。モーラの指標は正解の音素ごとの長さを作れる文で測る)、生成の長さの比。
"""
from pathlib import Path

import soundfile as sf
import torch

from exp0002.common.utils import phoneme_durations
from lib_common.asr_eval import ASR, cer_scores
from lib_common.audio import SR, load_wave
from lib_common.mos import UTMOS
from lib_common.neural_vocoder import load_vocoder
from lib_common.prosody import compare_f0_dtw, f0, mora_accent_scores, mora_phrases
from lib_common.text import phoneme_tokens

FPT = 4  # 音声トークン 1 個 = Mel 4 フレーム


class Evaluator:
    """テスト文と元の録音の F0 などを最初に 1 回だけ用意し、条件ごとの Mel を評価する。

    Args:
        test: テスト文(lib_common.data の発話)のリスト。
        vocoder_path: Vocos の vocoder.pt。
        out_dir: 評価用の音声の置き場(git の対象外)。
    """

    def __init__(self, test: list, vocoder_path: Path, out_dir: Path, device: str = "cuda"):
        self.test, self.out_dir, self.device = test, out_dir, device
        self.voc = load_vocoder(vocoder_path, device)
        self.asr, self.mos = ASR(device), UTMOS(device)
        self.toks = [phoneme_tokens(u.text) for u in test]
        self.gt_durs = [phoneme_durations(u, tk, len(u.mel) // FPT) for u, tk in zip(test, self.toks)]
        self.ok = [i for i, d in enumerate(self.gt_durs) if d is not None]
        self.ref_f0 = [f0(load_wave(u.path)) for u in test]
        self.ref_ph = {i: mora_phrases(self.toks[i], self.gt_durs[i], FPT) for i in self.ok}

    def wave(self, mel: torch.Tensor):
        return self.voc.generate(mel.to(self.device)[None])[0].cpu().numpy()

    def __call__(self, name: str, mels: list[torch.Tensor], durs: list) -> dict:
        """
        Args:
            name: 条件の名前(音声の保存先のディレクトリ名)。
            mels: テスト文ごとの Mel [T_i, F](正規化前の log-Mel)。
            durs: テスト文ごとの、T2S が予測した音素ごとの長さ [<bos> + トークン + <eos>]。

        Returns:
            指標の辞書。
        """
        d = self.out_dir / name
        d.mkdir(parents=True, exist_ok=True)
        waves = [self.wave(m) for m in mels]
        paths = []
        for u, w in zip(self.test, waves):
            sf.write(d / f"{u.id}.wav", w, SR)
            paths.append(str(d / f"{u.id}.wav"))
        f0s = [f0(w) for w in waves]
        res = {"cer": cer_scores([u.text for u in self.test], self.asr.transcribe(paths))["reading"],
               "utmos": sum(self.mos.score(paths)) / len(paths),
               "length_ratio": sum(len(m) / len(u.mel) for m, u in zip(mels, self.test)) / len(mels)}
        res.update(compare_f0_dtw(self.ref_f0, f0s, [u.mel for u in self.test], mels))
        ph = [mora_phrases(self.toks[i], durs[i], FPT) for i in self.ok]
        res.update(mora_accent_scores([f0s[i] for i in self.ok], ph, [self.ref_f0[i] for i in self.ok],
                                      [self.ref_ph[i] for i in self.ok]))
        return res
