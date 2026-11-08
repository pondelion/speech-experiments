"""torchaudio.functional の代わり(resample だけ。他は呼ぶと NotImplementedError)。"""
import librosa
import torch

from . import functional  # noqa: F401


def resample(waveform: torch.Tensor, orig_freq: int, new_freq: int, **kwargs) -> torch.Tensor:
    """librosa(soxr)でサンプリング周波数を変える(torchaudio の resample とは補間の方法が違う)。"""
    if orig_freq == new_freq:
        return waveform
    y = librosa.resample(waveform.detach().cpu().numpy(), orig_sr=orig_freq, target_sr=new_freq)
    return torch.from_numpy(y).to(waveform.device, waveform.dtype)


def __getattr__(name):
    raise NotImplementedError(f"torchaudio_shim: torchaudio.functional.{name} は扱わない")
