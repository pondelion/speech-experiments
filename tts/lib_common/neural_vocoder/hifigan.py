"""HiFi-GAN(Kong et al., 2020)の生成器と判別器。論文と公式の設定(V1・V2)に合わせて、ここで実装したもの。

生成器: log-Mel [B, F, T] → 転置畳み込みで時間を 8・8・2・2 倍(合計 256 倍 = HOP)に引き伸ばしながら、
        各段で MRF(カーネルの大きさと dilation の違う残差ブロックの平均)を通し、最後に tanh で波形 [B, 1, N] にする。
        V1 と V2 は最初のチャンネル数だけが違う(V1 512、V2 128。V2 は約 1/15 のパラメータ)。
判別器: MPD(multi-period)= 波形を周期 p ごとに折りたたんだ 2 次元の畳み込みで、周期的な構造(声の高さ)の不自然さを見る。
        MSD(multi-scale)= 波形と、それを 1/2・1/4 に縮めたものに 1 次元の畳み込みをかけ、長い範囲の不自然さを見る。
        どちらも、途中の層の出力(特徴)も返す(生成器の feature matching の損失に使う)。

shape の軸の記号:
    B: バッチ内の発話数、F: Mel の帯域数、T: Mel のフレーム数、N: 波形のサンプル数(= 256T)、C: チャンネル数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

LRELU = 0.1  # 生成器・判別器の LeakyReLU の負の傾き(論文の設定)

CONFIGS = {  # 論文の V1・V2(upsample は 8・8・2・2 で合計 256 倍。この Mel の HOP = 256 と一致する)
    "hifigan_v1": {"channels": 512, "upsample_rates": (8, 8, 2, 2), "upsample_kernels": (16, 16, 4, 4),
                   "res_kernels": (3, 7, 11), "res_dilations": ((1, 3, 5), (1, 3, 5), (1, 3, 5))},
    "hifigan_v2": {"channels": 128, "upsample_rates": (8, 8, 2, 2), "upsample_kernels": (16, 16, 4, 4),
                   "res_kernels": (3, 7, 11), "res_dilations": ((1, 3, 5), (1, 3, 5), (1, 3, 5))},
}


def _pad(kernel: int, dilation: int = 1) -> int:
    """長さを変えない畳み込みの詰め物の数。"""
    return dilation * (kernel - 1) // 2


class ResBlock(nn.Module):
    """dilation の違う畳み込みを 3 回重ねる残差ブロック(論文の ResBlock1)。

    各段: x ← x + conv(LeakyReLU(dilated conv(LeakyReLU(x))))。dilation を広げることで、少ない層で広い範囲を見る。
    """

    def __init__(self, ch: int, kernel: int, dilations: tuple):
        super().__init__()
        self.convs1 = nn.ModuleList([weight_norm(nn.Conv1d(ch, ch, kernel, padding=_pad(kernel, d), dilation=d))
                                     for d in dilations])
        self.convs2 = nn.ModuleList([weight_norm(nn.Conv1d(ch, ch, kernel, padding=_pad(kernel))) for _ in dilations])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            x = x + c2(F.leaky_relu(c1(F.leaky_relu(x, LRELU)), LRELU))
        return x


class HiFiGANGenerator(nn.Module):
    """log-Mel → 波形。

    Args:
        n_mels: Mel の帯域数 F。
        channels: 最初のチャンネル数(V1 512、V2 128)。段ごとに半分にする。
        upsample_rates, upsample_kernels: 各段の時間の引き伸ばしの倍率と、転置畳み込みのカーネルの大きさ。
        res_kernels, res_dilations: MRF の残差ブロックのカーネルの大きさと dilation。
    """

    def __init__(self, n_mels: int, channels: int, upsample_rates, upsample_kernels, res_kernels, res_dilations):
        super().__init__()
        self.conv_pre = weight_norm(nn.Conv1d(n_mels, channels, 7, padding=3))
        self.ups, self.mrfs = nn.ModuleList(), nn.ModuleList()
        ch = channels
        for r, k in zip(upsample_rates, upsample_kernels):
            self.ups.append(weight_norm(nn.ConvTranspose1d(ch, ch // 2, k, stride=r, padding=(k - r) // 2)))
            ch //= 2
            self.mrfs.append(nn.ModuleList([ResBlock(ch, rk, rd) for rk, rd in zip(res_kernels, res_dilations)]))
        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, padding=3))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: log-Mel [B, F, T]。

        Returns:
            波形 [B, 1, N](N = 256T)。
        """
        x = self.conv_pre(mel)                                                       # [B, C, T]
        for up, mrf in zip(self.ups, self.mrfs):
            x = up(F.leaky_relu(x, LRELU))                                           # 時間を r 倍、チャンネルを半分に
            x = sum(block(x) for block in mrf) / len(mrf)                            # MRF: 残差ブロックの平均
        return torch.tanh(self.conv_post(F.leaky_relu(x)))                           # [B, C', N] → [B, 1, N]


class PeriodDiscriminator(nn.Module):
    """周期 p の判別器: 波形 [B, 1, N] を [B, 1, N/p, p] に折りたたみ、時間方向だけの 2 次元の畳み込みをかける。"""

    def __init__(self, period: int):
        super().__init__()
        self.period = period
        chs = [1, 32, 128, 512, 1024]
        self.convs = nn.ModuleList([weight_norm(nn.Conv2d(i, o, (5, 1), (3, 1), padding=(2, 0)))
                                    for i, o in zip(chs[:-1], chs[1:])])
        self.convs.append(weight_norm(nn.Conv2d(1024, 1024, (5, 1), padding=(2, 0))))
        self.post = weight_norm(nn.Conv2d(1024, 1, (3, 1), padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = F.pad(x, (0, -x.shape[-1] % self.period), mode="reflect")                # 長さを p の倍数に
        x = rearrange(x, "b c (t p) -> b c t p", p=self.period)                      # [B, 1, N] → [B, 1, N/p, p]
        feats = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU)
            feats.append(x)
        x = self.post(x)
        feats.append(x)
        return rearrange(x, "b c t p -> b (c t p)"), feats


class ScaleDiscriminator(nn.Module):
    """1 つの時間解像度の判別器(グループ畳み込みで時間を縮めていく 1 次元の畳み込み)。"""

    def __init__(self, norm=weight_norm):
        super().__init__()
        spec = [(1, 128, 15, 1, 1), (128, 128, 41, 2, 4), (128, 256, 41, 2, 16), (256, 512, 41, 4, 16),
                (512, 1024, 41, 4, 16), (1024, 1024, 41, 1, 16), (1024, 1024, 5, 1, 1)]  # (入力, 出力, カーネル, stride, groups)
        self.convs = nn.ModuleList([norm(nn.Conv1d(i, o, k, s, padding=_pad(k), groups=g)) for i, o, k, s, g in spec])
        self.post = norm(nn.Conv1d(1024, 1, 3, padding=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        feats = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU)
            feats.append(x)
        x = self.post(x)
        feats.append(x)
        return rearrange(x, "b c t -> b (c t)"), feats


class MultiPeriodDiscriminator(nn.Module):
    """周期 2・3・5・7・11 の判別器(互いに素な周期で、いろいろな周期の構造を重ならずに見る)。"""

    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.discs = nn.ModuleList([PeriodDiscriminator(p) for p in periods])

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        outs = [d(x) for d in self.discs]
        return [o for o, _ in outs], [f for _, f in outs]


class MultiScaleDiscriminator(nn.Module):
    """元の波形・1/2・1/4 に縮めた波形の 3 つの判別器(元の解像度だけ spectral norm。論文の設定)。"""

    def __init__(self):
        super().__init__()
        self.discs = nn.ModuleList([ScaleDiscriminator(spectral_norm), ScaleDiscriminator(), ScaleDiscriminator()])

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        outs, feats = [], []
        for i, d in enumerate(self.discs):
            if i > 0:
                x = F.avg_pool1d(x, 4, 2, padding=2)                                 # 時間を 1/2 に
            o, f = d(x)
            outs.append(o)
            feats.append(f)
        return outs, feats
