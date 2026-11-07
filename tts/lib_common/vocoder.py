"""Mel から波形に戻す(ボコーダ)。今回は学習不要の Griffin-Lim を使う。

Mel には各周波数の「強さ」しか無く、波の「位相」(波がいつ山・谷になるか)と細かい周波数の情報が
失われている。波形に戻すにはこれらを推定する必要がある。
Griffin-Lim は、STFT と逆 STFT を繰り返して「その強さと矛盾しない位相」を探す古典的な方法。
学習は要らないが、音声らしさの知識は使わないので、ザラつき・こもりなどの劣化が残る
(IndexTTS は学習済みの NN ボコーダ BigVGAN を使う。NN ボコーダのスクラッチ学習は後の実験で行う)。

shape の軸の記号:
    T: フレーム数、F: Mel の帯域数(80)、Fq: STFT の周波数の数(513)、N: 波形のサンプル数
"""
import torch
from einops import rearrange

from .audio import HOP, N_FFT, mel_basis


class GriffinLim:
    """GPU で動く Griffin-Lim。

    Args:
        device: 計算に使うデバイス。
        n_iter: 位相を探す反復回数。
    """

    def __init__(self, device: str = "cuda", n_iter: int = 64):
        self.device, self.n_iter = device, n_iter
        basis = torch.from_numpy(mel_basis()).to(device)  # [F, Fq]
        self.inv_basis = torch.linalg.pinv(basis)          # Mel → 線形周波数の振幅に引き伸ばす擬似逆行列 [Fq, F]
        self.window = torch.hann_window(N_FFT, device=device)

    @torch.no_grad()
    def __call__(self, logmel: torch.Tensor) -> torch.Tensor:
        """log-Mel から波形を作る。

        Args:
            logmel: 正規化前の log-Mel [T(フレーム数), F(Mel の帯域数 = 80)]。

        Returns:
            波形 [N(サンプル数)](CPU の float32)。
            (以下のコメントの記号: Fq STFT の周波数の数 = 513)
        """
        # log-Mel [T, F] → exp で振幅に戻して [F, T] → [Fq, F] @ [F, T] → 線形周波数の振幅 [Fq, T]
        mel_amp = rearrange(logmel.to(self.device).float().exp(), "t f -> f t")
        mag = (self.inv_basis @ mel_amp).clamp(min=0)
        # 位相をランダムに始めて、「逆 STFT → STFT → 振幅だけ正しい値に戻す」を繰り返す
        phase = torch.exp(2j * torch.pi * torch.rand_like(mag))                       # 大きさ 1 の複素数 [Fq, T]
        for _ in range(self.n_iter):
            wave = torch.istft(mag * phase, N_FFT, HOP, window=self.window, center=True)  # [Fq, T] → 波形 [N]
            spec = torch.stft(wave, N_FFT, HOP, window=self.window, center=True, return_complex=True)  # [N] → [Fq, T]
            phase = spec / spec.abs().clamp(min=1e-8)                                     # 位相だけ取り出す [Fq, T]
        wave = torch.istft(mag * phase, N_FFT, HOP, window=self.window, center=True)      # [N]
        return (wave / wave.abs().max().clamp(min=1e-8) * 0.9).cpu()  # 音量をピーク 0.9 にそろえる
