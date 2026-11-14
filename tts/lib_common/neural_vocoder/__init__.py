"""NN ボコーダ(log-Mel → 波形)。HiFi-GAN V1・V2、Vocos、BigVGAN-v2 を、論文・公式の設定に合わせてここで実装したもの。

ボコーダは GAN で学習する: 生成器が Mel から波形を作り、判別器が本物の録音と生成した波形を見分けようとする。生成器の損失は
    敵対的な損失(判別器をだます)+ feature matching(判別器の途中の層の出力が本物と近くなる)+ Mel の L1(作った波形の Mel が
    元の Mel と近くなる。lib_common.audio と同じ Mel で、論文と同じ重み 45)
入力の Mel は lib_common.audio の log-Mel(24kHz、80 帯域、HOP 256)そのまま(S2M の出力と同じもの)。

使い方:
    gen = build_generator("vocos")              # 学習用
    voc = load_vocoder(path, "cuda")            # 推論用(train_vocoder.py が保存した checkpoint)
    wave = voc.generate(mel)                    # [B, T, F] → [B, N]
"""
from pathlib import Path

import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..audio import HOP, N_MELS, SR
from .bigvgan import CONFIGS as BIGVGAN_CONFIGS
from .bigvgan import BigVGANGenerator, MultiScaleSubbandCQTDiscriminator
from .hifigan import CONFIGS, HiFiGANGenerator, MultiPeriodDiscriminator, MultiScaleDiscriminator
from .vocos import MultiResolutionDiscriminator, VocosGenerator

# 学習の設定(論文・公式の設定に合わせた値)。segment は 1 つの学習データの切り出しの長さ(サンプル数)
RECIPES = {
    "hifigan_v1": {"segment": 8192, "batch": 16, "lr": 2e-4, "betas": (0.8, 0.99), "schedule": "exponential",
                   "discs": ("mpd", "msd"), "gan": "lsgan", "disc_weights": (1.0, 1.0), "fm_weight": 2.0, "mean_over_discs": False},
    "hifigan_v2": {"segment": 8192, "batch": 16, "lr": 2e-4, "betas": (0.8, 0.99), "schedule": "exponential",
                   "discs": ("mpd", "msd"), "gan": "lsgan", "disc_weights": (1.0, 1.0), "fm_weight": 2.0, "mean_over_discs": False},
    "vocos": {"segment": 16384, "batch": 16, "lr": 5e-4, "betas": (0.8, 0.9), "schedule": "cosine",
              "discs": ("mpd", "mrd"), "gan": "hinge", "disc_weights": (1.0, 0.1), "fm_weight": 1.0, "mean_over_discs": True},
    # BigVGAN-v2(公式の 24kHz 版の設定)。バッチだけは GPU に収まるように変える(公式は 4 × 65,536 サンプル)。
    # 学習率は step ごとに 0.9999996 倍(公式の lr_decay)、勾配の大きさの上限 500、Mel の損失は複数の解像度(重み 15)
    "bigvgan_v2": {"segment": 32768, "batch": 2, "lr": 1e-4, "betas": (0.8, 0.99), "schedule": "exponential_step",
                   "gamma": 0.9999996, "discs": ("mpd", "cqtd"), "gan": "lsgan", "disc_weights": (1.0, 1.0), "fm_weight": 2.0,
                   "mean_over_discs": False, "mel_loss": "multiscale", "mel_weight": 15.0, "clip_grad": 500.0},
}
MEL_WEIGHT = 45.0  # 1 つの Mel の L1 の重み(HiFi-GAN・Vocos)。BigVGAN-v2 は RECIPES の mel_weight
DISCRIMINATORS = {"mpd": MultiPeriodDiscriminator, "msd": MultiScaleDiscriminator, "mrd": MultiResolutionDiscriminator,
                  "cqtd": MultiScaleSubbandCQTDiscriminator}


class MultiScaleMelLoss(nn.Module):
    """複数の解像度の Mel の L1(BigVGAN-v2・DAC と同じ設定): 窓の長さ 32〜2048(間隔は 1/4)と Mel の帯域数 5〜320 の 7 通りで、
    振幅の Mel の log10 の差の絶対値の平均を足す。短い窓は時間の細かさを、長い窓は周波数の細かさを見る。"""

    def __init__(self, sr: int = SR, windows=(32, 64, 128, 256, 512, 1024, 2048), n_mels=(5, 10, 20, 40, 80, 160, 320)):
        super().__init__()
        self.windows = windows
        for w, m in zip(windows, n_mels):
            self.register_buffer(f"mel{w}", torch.from_numpy(librosa.filters.mel(sr=sr, n_fft=w, n_mels=m)).float(),
                                 persistent=False)
            self.register_buffer(f"win{w}", torch.hann_window(w), persistent=False)

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """y_hat・y: 波形 [B, N]。"""
        loss = 0.0
        for w in self.windows:
            spec = lambda x: torch.stft(x, w, w // 4, window=getattr(self, f"win{w}"), center=True,  # noqa: E731
                                        return_complex=True).abs()
            mel = lambda x: torch.log10(torch.clamp(getattr(self, f"mel{w}") @ spec(x), min=1e-5))  # noqa: E731
            loss = loss + F.l1_loss(mel(y_hat), mel(y))
        return loss


def build_generator(name: str) -> nn.Module:
    """生成器を作る。"""
    if name == "vocos":
        return VocosGenerator(N_MELS, hop=HOP)
    if name.startswith("bigvgan"):
        return BigVGANGenerator(N_MELS, **BIGVGAN_CONFIGS[name])
    return HiFiGANGenerator(N_MELS, **CONFIGS[name])


def build_discriminators(name: str) -> nn.ModuleList:
    """判別器を作る(RECIPES の discs の順)。"""
    return nn.ModuleList([DISCRIMINATORS[d]() for d in RECIPES[name]["discs"]])


def adversarial_d(real: list[torch.Tensor], fake: list[torch.Tensor], gan: str) -> torch.Tensor:
    """判別器の損失(1 つの判別器の中の、全ての小さな判別器の和)。lsgan: 本物 → 1・生成 → 0 の二乗誤差 / hinge: 余裕 1 の hinge。"""
    if gan == "lsgan":
        return sum(((1 - r) ** 2).mean() + (f ** 2).mean() for r, f in zip(real, fake))
    return sum(F.relu(1 - r).mean() + F.relu(1 + f).mean() for r, f in zip(real, fake))


def adversarial_g(fake: list[torch.Tensor], gan: str) -> torch.Tensor:
    """生成器の敵対的な損失(判別器に本物と判定させる)。"""
    if gan == "lsgan":
        return sum(((1 - f) ** 2).mean() for f in fake)
    return sum(F.relu(1 - f).mean() for f in fake)


def feature_matching(feats_real: list[list[torch.Tensor]], feats_fake: list[list[torch.Tensor]]) -> torch.Tensor:
    """判別器の途中の層の出力の L1(全ての小さな判別器・全ての層の和)。本物の側は勾配を流さない。"""
    return sum(F.l1_loss(f, r.detach()) for fr, ff in zip(feats_real, feats_fake) for r, f in zip(fr, ff))


class NeuralVocoder(nn.Module):
    """推論用: log-Mel [B, T, F] → 波形 [B, N](N = HOP × (T - 1)。元の録音の長さにそろえる)。"""

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.generator = build_generator(name)

    @torch.no_grad()
    def generate(self, mel: torch.Tensor) -> torch.Tensor:
        n = HOP * (mel.shape[1] - 1)
        wave = self.generator(rearrange(mel, "b t f -> b f t"))                     # [B, 1, N']
        wave = F.pad(wave, (0, max(0, n - wave.shape[-1])))
        return rearrange(wave[..., :n], "b 1 n -> b n")


def load_vocoder(path: str | Path, device: str) -> NeuralVocoder:
    """train_vocoder.py が保存した checkpoint({"name": 名前, "generator": 生成器の重み})を読み込む。"""
    ckpt = torch.load(path, map_location="cpu")
    voc = NeuralVocoder(ckpt["name"])
    voc.generator.load_state_dict(ckpt["generator"])
    return voc.to(device).eval()
