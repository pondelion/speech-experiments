"""BigVGAN(Lee et al., 2023)の生成器。論文と公式の設定(v2 の 24kHz 版・base)に合わせて、ここで実装したもの。

HiFi-GAN の生成器との違い:
    - 活性化関数が SnakeBeta: x + (1/β) sin²(αx)。チャンネルごとに学習する周期 α を持ち、声の高さのような周期的な成分を作りやすい
    - 活性化の前後で、時間を 2 倍に細かくしてから活性化し、元に戻す(anti-aliased activation)。非線形な活性化で生まれる
      高い周波数が、元の時間の細かさで表せずに低い周波数に折り返す雑音(エイリアシング)を防ぐ。
      細かくする・戻すときは、Kaiser 窓をかけた sinc 関数の低域通過フィルタを使う
    - 残差ブロック(AMP ブロック)は HiFi-GAN の ResBlock1 と同じ形で、活性化だけを上のものにする
    - v2 は最後の tanh と、最後の畳み込みのバイアスを使わない(値は [-1, 1] に切り詰める)

shape の軸の記号:
    B: バッチ内の発話数、C: チャンネル数、T: Mel のフレーム数、N: 波形のサンプル数(= 256T)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils.parametrizations import weight_norm

from ..audio import SR
from .hifigan import LRELU, _pad

CONFIGS = {
    "bigvgan_v2": {"channels": 1536, "upsample_rates": (4, 4, 2, 2, 2, 2), "upsample_kernels": (8, 8, 4, 4, 4, 4)},
    "bigvgan_base": {"channels": 512, "upsample_rates": (8, 8, 2, 2), "upsample_kernels": (16, 16, 4, 4)},
}
RES_KERNELS, RES_DILATIONS = (3, 7, 11), ((1, 3, 5), (1, 3, 5), (1, 3, 5))


def lowpass_filter(cutoff: float, half_width: float, kernel: int) -> torch.Tensor:
    """Kaiser 窓をかけた sinc 関数の低域通過フィルタ [kernel]。

    Args:
        cutoff: 遮断周波数(サンプリング周波数に対する割合。0.5 がナイキスト周波数)。
        half_width: 通過域から阻止域に移る幅の半分(同じく割合)。
        kernel: フィルタの長さ(偶数)。
    """
    # Kaiser 窓の β を、移行の幅から決める(Kaiser の経験式。A は阻止域の減衰量 [dB])
    a = 2.285 * (kernel // 2 - 1) * math.pi * 4 * half_width + 7.95
    beta = 0.1102 * (a - 8.7) if a > 50 else (0.5842 * (a - 21) ** 0.4 + 0.07886 * (a - 21) if a >= 21 else 0.0)
    t = torch.arange(-kernel // 2, kernel // 2) + 0.5                                  # フィルタの中心からの位置
    f = 2 * cutoff * torch.sinc(2 * cutoff * t) * torch.kaiser_window(kernel, periodic=False, beta=beta)
    return f / f.sum()                                                                 # 直流を変えないように正規化


class SnakeBeta(nn.Module):
    """x + (1/β) sin²(αx)。α・β はチャンネルごとに学習し、log で持つ(公式の snake_logscale と同じ)。"""

    def __init__(self, ch: int):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.zeros(ch, 1))
        self.log_beta = nn.Parameter(torch.zeros(ch, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha, beta = self.log_alpha.exp(), self.log_beta.exp()
        return x + torch.sin(alpha * x) ** 2 / (beta + 1e-9)


_KERNEL = 12  # 時間を 2 倍・1/2 にするときの低域通過フィルタの長さ(公式と同じ)


def resample2(x: torch.Tensor, filt: torch.Tensor, up: bool) -> torch.Tensor:
    """時間を 2 倍に細かくする(up=True)/ 1/2 に間引く(up=False)。[B, C, N] → [B, C, 2N] / [B, C, N/2]。

    filt は lowpass_filter(0.25, 0.3, 12) を [1, 1, 12] にしたもの(遮断周波数は、細かい方のサンプリング周波数の 1/4)。
    """
    b, c, n = x.shape
    x = x.reshape(b * c, 1, n)                                                             # チャンネルごとに同じフィルタ
    if up:  # 0 を挟んで(転置畳み込み)低域通過フィルタをかける。0 を挟んだ分だけ振幅が半分になるので 2 倍する
        p = _KERNEL // 2 - 1
        x = F.conv_transpose1d(F.pad(x, (p, p), mode="replicate"), filt * 2, stride=2)
        crop = (_KERNEL - 2) // 2 + p * 2
        x = x[..., crop: crop + n * 2]
    else:  # 低域通過フィルタをかけてから 1 個おきに間引く
        x = F.conv1d(F.pad(x, (_KERNEL // 2 - 1, _KERNEL // 2), mode="replicate"), filt, stride=2)
    return x.reshape(b, c, -1)


class AntiAliasedSnake(nn.Module):
    """時間を 2 倍に細かくしてから SnakeBeta をかけ、低域通過フィルタで元の細かさに戻す。[B, C, N] → [B, C, N]。"""

    def __init__(self, ch: int):
        super().__init__()
        self.act = SnakeBeta(ch)
        self.register_buffer("filt", lowpass_filter(0.25, 0.3, _KERNEL)[None, None], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return resample2(self.act(resample2(x, self.filt, up=True)), self.filt, up=False)


class AMPBlock(nn.Module):
    """HiFi-GAN の ResBlock1 と同じ形で、活性化を AntiAliasedSnake にした残差ブロック。"""

    def __init__(self, ch: int, kernel: int, dilations: tuple):
        super().__init__()
        self.convs1 = nn.ModuleList([weight_norm(nn.Conv1d(ch, ch, kernel, padding=_pad(kernel, d), dilation=d))
                                     for d in dilations])
        self.convs2 = nn.ModuleList([weight_norm(nn.Conv1d(ch, ch, kernel, padding=_pad(kernel))) for _ in dilations])
        self.acts1 = nn.ModuleList([AntiAliasedSnake(ch) for _ in dilations])
        self.acts2 = nn.ModuleList([AntiAliasedSnake(ch) for _ in dilations])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, self.acts1, self.acts2):
            x = x + c2(a2(c1(a1(x))))
        return x


class BigVGANGenerator(nn.Module):
    """log-Mel [B, F, T] → 波形 [B, 1, N]。

    Args:
        n_mels: Mel の帯域数。
        channels: 最初のチャンネル数(v2 1536、base 512)。段ごとに半分にする。
        upsample_rates, upsample_kernels: 各段の時間の引き伸ばしの倍率と、転置畳み込みのカーネルの大きさ(倍率の積 = 256)。
    """

    def __init__(self, n_mels: int, channels: int, upsample_rates, upsample_kernels):
        super().__init__()
        self.conv_pre = weight_norm(nn.Conv1d(n_mels, channels, 7, padding=3))
        self.ups, self.mrfs = nn.ModuleList(), nn.ModuleList()
        ch = channels
        for r, k in zip(upsample_rates, upsample_kernels):
            self.ups.append(weight_norm(nn.ConvTranspose1d(ch, ch // 2, k, stride=r, padding=(k - r) // 2)))
            ch //= 2
            self.mrfs.append(nn.ModuleList([AMPBlock(ch, rk, rd) for rk, rd in zip(RES_KERNELS, RES_DILATIONS)]))
        self.act_post = AntiAliasedSnake(ch)
        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, padding=3, bias=False))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(mel)                                                         # [B, C, T]
        for up, mrf in zip(self.ups, self.mrfs):
            x = up(x)                                                                  # 時間を r 倍、チャンネルを半分に
            x = sum(block(x) for block in mrf) / len(mrf)
        return self.conv_post(self.act_post(x)).clamp(-1, 1)                           # [B, 1, N](v2: tanh を使わない)


class CQT(nn.Module):
    """定 Q 変換(CQT): 周波数のビンを対数の間隔(1 オクターブに bins 個)に並べ、各ビンの窓の長さを周波数に反比例させる
    (低い音ほど長い窓で細かい周波数を、高い音ほど短い窓で細かい時間を見る)。

    低いオクターブの長い窓を直接使うと計算が重いので、一番上のオクターブの短いフィルタだけを作り、
    波形を 1 オクターブ下がるごとに 1/2 に間引いて、同じフィルタを使い回す(nnAudio の CQT2010v2 と同じ考え方)。
    各ビンのフィルタ = Hann 窓 × e^{2πi f t} / 窓の長さ。出力に sqrt(窓の長さ) を掛ける(librosa と同じ大きさのそろえ方)。

    Args:
        sr: 入力のサンプリング周波数。
        hop: フレームの間隔(2^(n_octaves - 1) の倍数)。
        n_octaves, bins: オクターブの数と、1 オクターブあたりのビンの数。
        fmin: 一番低いビンの周波数(C1 = 32.70Hz。nnAudio の既定)。
    """

    def __init__(self, sr: int, hop: int, n_octaves: int, bins: int, fmin: float = 32.70):
        super().__init__()
        assert hop % 2 ** (n_octaves - 1) == 0
        self.hop, self.n_octaves = hop, n_octaves
        q = 1 / (2 ** (1 / bins) - 1)                                                  # Q 値(中心周波数 / 帯域幅)
        freqs = fmin * 2 ** (torch.arange(bins) / bins + n_octaves - 1)               # 一番上のオクターブの中心周波数 [bins]
        lengths = torch.ceil(q * sr / freqs).long()                                    # 各ビンの窓の長さ
        k = int(lengths.max()) + (int(lengths.max()) % 2 == 0)                          # フィルタの長さ(奇数)
        kernels = torch.zeros(bins, k, dtype=torch.cfloat)
        for i, (f, n) in enumerate(zip(freqs, lengths.tolist())):
            t = torch.arange(n) - n // 2
            s = (k - n) // 2
            kernels[i, s: s + n] = torch.hann_window(n, periodic=False) * torch.exp(2j * torch.pi * f * t / sr) / n
        kernels = kernels * lengths.sqrt()[:, None]
        self.register_buffer("real", kernels.real[:, None], persistent=False)          # [bins, 1, k]
        self.register_buffer("imag", kernels.imag[:, None], persistent=False)
        self.register_buffer("filt", lowpass_filter(0.25, 0.3, _KERNEL)[None, None], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 波形 [B, 1, N]。

        Returns:
            [B, 2(実部・虚部), フレーム数, n_octaves × bins](周波数は低い方から)。
        """
        outs, hop = [], self.hop
        for o in range(self.n_octaves):                                                # 上のオクターブから
            if o > 0:
                x = resample2(x, self.filt, up=False)                                  # 1 オクターブ下げる: 1/2 に間引く
                hop //= 2
            pad = self.real.shape[-1] // 2
            xp = F.pad(x, (pad, pad))
            outs.append(torch.stack([F.conv1d(xp, self.real, stride=hop), F.conv1d(xp, self.imag, stride=hop)], 1))
        n = min(o.shape[-1] for o in outs)
        z = torch.cat([o[..., :n] for o in reversed(outs)], dim=2)                     # [B, 2, n_octaves × bins, フレーム数]
        return rearrange(z, "b c f t -> b c t f")


class CQTDiscriminator(nn.Module):
    """1 つの CQT の設定の判別器(公式の v2 と同じ形)。

    波形を 2 倍のサンプリング周波数に細かくしてから CQT をとり、実部・虚部を 2 チャンネルにする。オクターブごとに別の畳み込みを
    かけてからつなぎ、時間方向に dilation 1・2・4 の 2 次元の畳み込み(周波数方向は stride 2 で縮める)を重ねる。
    """

    def __init__(self, sr: int, hop: int, n_octaves: int, bins: int, filters: int = 128, dilations=(1, 2, 4)):
        super().__init__()
        self.bins = bins
        self.register_buffer("filt", lowpass_filter(0.25, 0.3, _KERNEL)[None, None], persistent=False)
        self.cqt = CQT(sr * 2, hop, n_octaves, bins)
        p = lambda k, d=(1, 1): ((k[0] - 1) * d[0] // 2, (k[1] - 1) * d[1] // 2)  # noqa: E731  長さを変えない詰め物
        self.conv_pres = nn.ModuleList([weight_norm(nn.Conv2d(2, 2, (3, 9), padding=p((3, 9)))) for _ in range(n_octaves)])
        self.convs = nn.ModuleList([weight_norm(nn.Conv2d(2, filters, (3, 9), padding=p((3, 9))))])
        for d in dilations:
            self.convs.append(weight_norm(nn.Conv2d(filters, filters, (3, 9), stride=(1, 2), dilation=(d, 1),
                                                    padding=p((3, 9), (d, 1)))))
        self.convs.append(weight_norm(nn.Conv2d(filters, filters, (3, 3), padding=p((3, 3)))))
        self.post = weight_norm(nn.Conv2d(filters, 1, (3, 3), padding=p((3, 3))))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        z = self.cqt(resample2(x, self.filt, up=True))                                 # [B, 2, フレーム数, オクターブ × bins]
        z = torch.cat([conv(z[..., i * self.bins: (i + 1) * self.bins]) for i, conv in enumerate(self.conv_pres)], dim=-1)
        feats = []
        for conv in self.convs:
            z = F.leaky_relu(conv(z), LRELU)
            feats.append(z)
        z = self.post(z)
        return rearrange(z, "b c t f -> b (c t f)"), feats


class MultiScaleSubbandCQTDiscriminator(nn.Module):
    """CQT の判別器 3 つ(hop 512・256・256、1 オクターブあたりのビン 24・36・48、9 オクターブ。公式の v2 の設定)。"""

    def __init__(self, sr: int = SR, hops=(512, 256, 256), n_octaves=(9, 9, 9), bins=(24, 36, 48)):
        super().__init__()
        self.discs = nn.ModuleList([CQTDiscriminator(sr, h, o, b) for h, o, b in zip(hops, n_octaves, bins)])

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        outs = [d(x) for d in self.discs]
        return [o for o, _ in outs], [f for _, f in outs]
