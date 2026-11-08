"""torchaudio.functional.functional の代わり(Hz ⇔ Mel の変換だけ。torchaudio と同じ式)。"""
import math


def _hz_to_mel(freq: float, mel_scale: str = "htk") -> float:
    if mel_scale == "htk":
        return 2595.0 * math.log10(1.0 + freq / 700.0)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    mels = freq / f_sp
    min_log_mel, logstep = min_log_hz / f_sp, math.log(6.4) / 27.0
    return min_log_mel + math.log(freq / min_log_hz) / logstep if freq >= min_log_hz else mels


def _mel_to_hz(mels, mel_scale: str = "htk"):
    if mel_scale == "htk":
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    raise NotImplementedError("torchaudio_shim: _mel_to_hz は htk だけ扱う")
