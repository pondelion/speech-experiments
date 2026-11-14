"""semantic codec: w2v-BERT 2.0 の特徴 ⇔ 音声トークン(IndexTTS-2.5 の `indextts/codec/models.py` の EnhancedCodec と同じ構成)。

公式の構成(この exp では層の数・次元だけ小さくする):
    特徴 [B, T, 1024](w2v-BERT 2.0 の 17 層目を全データの平均・標準偏差で正規化、50 フレーム/秒)
    → stride 2 の Conv + GELU で時間を 1/2 に(25 フレーム/秒 = トークン 1 個 40ms)
    → encoder(ConvNeXt の積み重ね。公式のクラス名は VocosBackbone)→ Linear で 1,024 次元に戻す
    → 量子化(factorized VQ: 1×1 Conv で 8 次元に射影し、L2 正規化したベクトルどうしの距離で codebook の一番近い番号を選ぶ。
      選んだ codebook の値(正規化前)を 1×1 Conv で 1,024 次元に戻す)
    → decoder(encoder と同じ形)→ 時間を 2 倍に(最近傍で引き伸ばして Conv)→ 復元した特徴 [B, T, 1024]
損失(公式の forward と同じ): 特徴の二乗誤差 + 0.15 × commit(encoder の出力を codebook に寄せる)+ 1.0 × codebook(codebook を寄せる)

S2M に渡すのは decoder の出力を 2 倍に引き伸ばしたもの(公式の推論の `semantic_codec.decode(codes)`。50 フレーム/秒)。

shape の軸の記号:
    B: バッチ内の発話数、T: 特徴のフレーム数(50 フレーム/秒、偶数にそろえる)、U: トークン数(= T/2、25 個/秒)、
    D: 特徴の次元(1,024)、H: ConvNeXt の次元、C: codebook のベクトルの次元(8)、K: トークンの種類数(8,192)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils.parametrizations import weight_norm


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block(公式の kmeans/vocos.py と同じ): depthwise Conv(窓 7)→ LayerNorm → Linear → GELU → Linear → γ 倍して残差に足す。"""

    def __init__(self, dim: int, intermediate_dim: int, layer_scale: float):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(layer_scale * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, H, T] → [B, H, T]。"""
        h = rearrange(self.dwconv(x), "b h t -> b t h")
        h = self.gamma * self.pwconv2(F.gelu(self.pwconv1(self.norm(h))))
        return x + rearrange(h, "b t h -> b h t")


class ConvNeXtBackbone(nn.Module):
    """公式の VocosBackbone: Conv(窓 7)→ LayerNorm → ConvNeXt × num_layers → LayerNorm。γ の初期値は 1 / num_layers。"""

    def __init__(self, in_dim: int, dim: int, intermediate_dim: int, num_layers: int):
        super().__init__()
        self.embed = nn.Conv1d(in_dim, dim, 7, padding=3)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.blocks = nn.ModuleList([ConvNeXtBlock(dim, intermediate_dim, 1 / num_layers) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, D_in, T] → [B, T, H]。"""
        x = rearrange(self.norm(rearrange(self.embed(x), "b h t -> b t h")), "b t h -> b h t")
        for blk in self.blocks:
            x = blk(x)
        return self.final_norm(rearrange(x, "b h t -> b t h"))


class FactorizedVQ(nn.Module):
    """factorized VQ(公式の amphion_codec の FactorizedVectorQuantize、L2 正規化あり)。"""

    def __init__(self, dim: int, codebook_size: int, code_dim: int = 8, commitment: float = 0.15, codebook_weight: float = 1.0):
        super().__init__()
        self.in_project = weight_norm(nn.Conv1d(dim, code_dim, 1))
        self.out_project = weight_norm(nn.Conv1d(code_dim, dim, 1))
        self.codebook = nn.Embedding(codebook_size, code_dim)
        self.commitment, self.codebook_weight = commitment, codebook_weight

    def codes(self, z_e: torch.Tensor) -> torch.Tensor:
        """射影後のベクトル [B, C, U] → 一番近い codebook の番号 [B, U](どちらも L2 正規化して比べる)。"""
        e = F.normalize(rearrange(z_e, "b c u -> b u c"), dim=-1)
        return torch.einsum("buc,kc->buk", e, F.normalize(self.codebook.weight, dim=-1)).argmax(-1)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """番号 [B, U] → codebook の値を 1,024 次元に戻したもの [B, D, U]。"""
        return self.out_project(rearrange(self.codebook(ids), "b u c -> b c u"))

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """z [B, D, U]、mask [B, U](有効な位置が True)→ (量子化したもの [B, D, U], 番号 [B, U], commit の損失, codebook の損失)。"""
        z_e = self.in_project(z)                                                # [B, C, U]
        ids = self.codes(z_e)
        z_q = rearrange(self.codebook(ids), "b u c -> b c u")                   # 正規化前の codebook の値 [B, C, U]
        m = mask[:, None].float()
        denom = m.sum() * z_e.shape[1]
        commit = (((z_e - z_q.detach()) ** 2) * m).sum() / denom * self.commitment
        cb = (((z_q - z_e.detach()) ** 2) * m).sum() / denom * self.codebook_weight
        z_q = z_e + (z_q - z_e).detach()                                        # straight-through
        return self.out_project(z_q), ids, commit, cb


class SemanticCodec(nn.Module):
    """公式の EnhancedCodec(downsample_scale 2、量子化 1 段)。"""

    def __init__(self, in_dim: int = 1024, dim: int = 384, intermediate_dim: int = 2048, num_layers: int = 12,
                 codebook_size: int = 8192, code_dim: int = 8):
        super().__init__()
        self.codebook_size, self.hidden_dim = codebook_size, in_dim
        self.down = nn.Conv1d(in_dim, in_dim, 3, stride=2, padding=1)
        self.up = nn.Conv1d(in_dim, in_dim, 3, padding=1)
        self.encoder = ConvNeXtBackbone(in_dim, dim, intermediate_dim, num_layers)
        self.enc_out = nn.Linear(dim, in_dim)
        self.decoder = ConvNeXtBackbone(in_dim, dim, intermediate_dim, num_layers)
        self.dec_out = nn.Linear(dim, in_dim)
        self.quantizer = FactorizedVQ(in_dim, codebook_size, code_dim)
        for m in self.modules():  # 公式の init_weights(重みの正規化をしていない Conv・Linear)
            if isinstance(m, (nn.Conv1d, nn.Linear)) and not hasattr(m, "parametrizations"):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.constant_(m.bias, 0)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """特徴 [B, T, D] → encoder の出力 [B, D, U]。"""
        h = F.gelu(self.down(rearrange(x, "b t d -> b d t")))                   # [B, D, T] → [B, D, U]
        return rearrange(self.enc_out(self.encoder(h)), "b u d -> b d u")

    def _decode(self, q: torch.Tensor) -> torch.Tensor:
        """量子化したもの [B, D, U] → 復元した特徴 [B, T, D](T = 2U)。"""
        h = rearrange(self.dec_out(self.decoder(q)), "b u d -> b d u")
        h = self.up(F.interpolate(h, scale_factor=2, mode="nearest"))           # [B, D, U] → [B, D, 2U]
        return rearrange(h, "b d t -> b t d")

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, dict, torch.Tensor]:
        """学習: x [B, T, D](T は偶数)、mask [B, T] → (損失, 内訳, 番号 [B, U])。"""
        q, ids, commit, cb = self.quantizer(self._encode(x), mask[:, ::2])
        rec = self._decode(q)
        m = mask[..., None].float()
        recon = (((rec - x) ** 2) * m).sum() / (m.sum() * x.shape[-1])
        return recon + commit + cb, {"recon": recon.item(), "commit": commit.item(), "codebook": cb.item()}, ids

    @torch.no_grad()
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """特徴 [B, T, D] → 番号 [B, U]。"""
        return self.quantizer.codes(self.quantizer.in_project(self._encode(x)))

    @torch.no_grad()
    def decode(self, ids: torch.Tensor) -> torch.Tensor:
        """番号 [B, U] → S2M に渡す特徴 [B, 2U, D](公式の semantic_codec.decode)。"""
        return self._decode(self.quantizer.embed(ids))
