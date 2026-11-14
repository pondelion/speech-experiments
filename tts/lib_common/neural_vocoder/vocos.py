"""Vocos(Siuzdak, 2023)の生成器と、Vocos が使う判別器 MRD。論文の設定(mel-24khz 相当)に合わせて、ここで実装したもの。

生成器: log-Mel [B, F, T] → ConvNeXt ブロック 8 個(時間の解像度は Mel のフレームのまま)→ 各フレームの STFT の振幅と位相を予測
        → 逆 STFT で波形 [B, N] にする。HiFi-GAN のように波形のサンプル単位まで畳み込みで引き伸ばさないので、計算が軽い
        (論文では HiFi-GAN の約 13 倍速い)。
判別器 MRD(multi-resolution、UnivNet と同じ形): 波形を 3 通りの窓の長さで STFT し、振幅 [時間, 周波数] に 2 次元の畳み込みをかける。
        周波数ごとの細かい構造の不自然さを見る。Vocos は HiFi-GAN の MPD と MRD を組み合わせて使う。

shape の軸の記号:
    B: バッチ内の発話数、F: Mel の帯域数、T: Mel のフレーム数、D: 隠れ層の次元、Fq: STFT の周波数の数(n_fft/2 + 1)、
    N: 波形のサンプル数(= HOP × (T - 1))
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils.parametrizations import weight_norm

from .hifigan import LRELU


class ConvNeXtBlock(nn.Module):
    """ConvNeXt ブロック: 時間方向の depthwise 畳み込み(カーネル 7)→ LayerNorm → 全結合(D → 3D → D)、残差の強さを学習する。"""

    def __init__(self, dim: int, hidden: int, layer_scale: float):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.ff = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.gamma = nn.Parameter(torch.full((dim,), layer_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, D, T] → [B, D, T]。"""
        h = rearrange(self.dwconv(x), "b d t -> b t d")
        return x + rearrange(self.gamma * self.ff(self.norm(h)), "b t d -> b d t")


class VocosGenerator(nn.Module):
    """log-Mel → 波形。

    Args:
        n_mels: Mel の帯域数 F。
        dim, hidden, layers: ConvNeXt の次元・全結合の中間の次元・ブロックの数(論文の mel-24khz は 512・1536・8)。
        n_fft, hop: 逆 STFT の窓の長さと間隔(Mel と同じ 1024・256)。
    """

    def __init__(self, n_mels: int, dim: int = 512, hidden: int = 1536, layers: int = 8, n_fft: int = 1024, hop: int = 256):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.embed = nn.Conv1d(n_mels, dim, 7, padding=3)
        self.norm_in = nn.LayerNorm(dim, eps=1e-6)
        self.blocks = nn.ModuleList([ConvNeXtBlock(dim, hidden, 1 / layers) for _ in range(layers)])
        self.norm_out = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, n_fft + 2)  # 各フレームの振幅(log)と位相、それぞれ Fq = n_fft/2 + 1 個
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: log-Mel [B, F, T]。

        Returns:
            波形 [B, 1, N](N = HOP × (T - 1)。Mel を中心をそろえた STFT で作ったときの元の長さ)。
        """
        x = rearrange(self.norm_in(rearrange(self.embed(mel), "b d t -> b t d")), "b t d -> b d t")  # [B, D, T]
        for block in self.blocks:
            x = block(x)
        h = self.head(self.norm_out(rearrange(x, "b d t -> b t d")))                 # [B, T, 2Fq]
        log_mag, phase = rearrange(h, "b t (two fq) -> two b fq t", two=2)           # 2 × [B, Fq, T]
        spec = torch.exp(log_mag).clamp(max=1e2) * torch.exp(1j * phase)             # 振幅 × e^{i 位相} [B, Fq, T]
        wave = torch.istft(spec, self.n_fft, self.hop, window=self.window, center=True)  # [B, N]
        return rearrange(wave, "b n -> b 1 n")


class ResolutionDiscriminator(nn.Module):
    """1 つの窓の長さの STFT の振幅 [B, 1, 時間, Fq] に 2 次元の畳み込みをかける判別器(周波数方向を縮めていく)。"""

    def __init__(self, n_fft: int, hop: int):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.convs = nn.ModuleList([weight_norm(nn.Conv2d(1, 32, (3, 9), padding=(1, 4)))] +
                                   [weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))) for _ in range(3)] +
                                   [weight_norm(nn.Conv2d(32, 32, (3, 3), padding=(1, 1)))])
        self.post = weight_norm(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        spec = torch.stft(rearrange(x, "b 1 n -> b n"), self.n_fft, self.hop, window=self.window, center=True,
                          return_complex=True).abs()                                 # [B, Fq, 時間]
        x = rearrange(spec, "b f t -> b 1 t f")
        feats = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU)
            feats.append(x)
        x = self.post(x)
        feats.append(x)
        return rearrange(x, "b c t f -> b (c t f)"), feats


class MultiResolutionDiscriminator(nn.Module):
    """窓の長さ 512・1024・2048(間隔は 1/4)の 3 つの判別器。"""

    def __init__(self, resolutions=((512, 128), (1024, 256), (2048, 512))):
        super().__init__()
        self.discs = nn.ModuleList([ResolutionDiscriminator(n, h) for n, h in resolutions])

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        outs = [d(x) for d in self.discs]
        return [o for o, _ in outs], [f for _, f in outs]
