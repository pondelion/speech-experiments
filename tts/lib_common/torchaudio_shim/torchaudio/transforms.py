"""torchaudio.transforms の代わり(MelSpectrogram だけ)。"""
import librosa
import torch
import torch.nn as nn


class MelSpectrogram(nn.Module):
    """torchaudio.transforms.MelSpectrogram と同じ計算(中心をそろえた STFT・反射の詰め物・Hann 窓 → Mel フィルタ)。

    Mel フィルタは librosa(Slaney 型の Mel 尺度・Slaney の正規化)で作る。これは torchaudio の
    mel_scale="slaney", norm="slaney" と同じ。それ以外の設定は使われていないので扱わない。
    """

    def __init__(self, sample_rate=16000, n_fft=400, win_length=None, hop_length=None, f_min=0.0, f_max=None,
                 pad=0, n_mels=128, window_fn=torch.hann_window, power=2.0, normalized=False, wkwargs=None,
                 center=True, pad_mode="reflect", onesided=None, norm=None, mel_scale="htk"):
        super().__init__()
        if norm != "slaney" or mel_scale != "slaney" or pad != 0 or normalized:
            raise NotImplementedError("torchaudio_shim: Slaney 型の Mel(norm・mel_scale とも slaney)だけ扱う")
        self.n_fft, self.win_length = n_fft, win_length or n_fft
        self.hop_length = hop_length or self.win_length // 2
        self.power, self.center, self.pad_mode = power, center, pad_mode
        self.register_buffer("window", window_fn(self.win_length))
        fb = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=f_min, fmax=f_max, htk=False, norm="slaney")
        self.register_buffer("fb", torch.from_numpy(fb).float())  # [n_mels, n_fft/2 + 1]

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """[..., 時間] → [..., n_mels, フレーム数]。"""
        shape = waveform.shape
        spec = torch.stft(waveform.reshape(-1, shape[-1]), self.n_fft, self.hop_length, self.win_length, self.window,
                          center=self.center, pad_mode=self.pad_mode, return_complex=True).abs().pow(self.power)
        mel = torch.matmul(self.fb, spec)
        return mel.reshape(*shape[:-1], *mel.shape[-2:])
