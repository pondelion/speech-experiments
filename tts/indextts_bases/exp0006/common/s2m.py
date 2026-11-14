"""S2M: codec の特徴 → Mel(IndexTTS-2.5 の `indextts/s2mel/modules` の length regulator・DiT・Flow Matching と同じ構成)。

公式の構成(この exp では次元だけ小さくする。参照の Mel(プロンプト)・話者ベクトル(style)はフェーズ A では使わないので入れない):
    codec の特徴 [B, S, 1024](50 フレーム/秒)
    → length regulator: Linear → 最近傍で Mel の長さに引き伸ばす → (Conv 窓 3 → GroupNorm → Mish) × 4 → 1×1 Conv
    → DiT(Flow Matching の速度を予測):
        入力 = Linear([ノイズの Mel, プロンプトの Mel(ここでは 0), Linear(条件)]) → Transformer(各層の正規化を時刻で調整する
        AdaptiveLayerNorm(RMSNorm)、RoPE、SwiGLU、前半の層の出力を後半の層に渡す U 字の skip 接続)
        → 長い skip 接続(Transformer の出力とノイズの Mel をつなげて Linear)→ WaveNet(時刻を条件に入れる)+ 残差
        → 最後の層(LayerNorm を時刻で調整)→ 1×1 Conv で Mel の帯域数に
    学習: 時刻 t ~ 一様分布、x_t = (1 - (1 - σ) t) ノイズ + t Mel、正解の速度 = Mel - (1 - σ) ノイズ、損失は L1(σ = 1e-6)。
          条件を落とす学習(CFG 用)はバッチ全体で 10% の確率(公式と同じ)
    生成: Euler 法 25 step、CFG 0.7(条件ありの速度 × 1.7 - 条件なしの速度 × 0.7)
Mel は正規化しない(公式と同じく log-Mel をそのまま扱う)。

shape の軸の記号:
    B: バッチ内の発話数、S: codec の特徴のフレーム数(50 フレーム/秒)、T: Mel のフレーム数(86.13 フレーム/秒)、
    F: Mel の帯域数(80)、Dc: codec の特徴の次元(1,024)、H: DiT の次元、Hh: ヘッドの次元(64)、Nh: ヘッドの数
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils.parametrizations import weight_norm

from lib_common.audio import N_MELS

MEL_PER_CODEC_FRAME = 1.72  # 推論で Mel の長さ = codec の特徴のフレーム数 × 1.72(公式の推論と同じ。86.13 / 50 ≒ 1.72)


def _find_multiple(n: int, k: int) -> int:
    return n if n % k == 0 else n + k - n % k


class LengthRegulator(nn.Module):
    """公式の InterpolateRegulator(sampling_ratios が 4 つ、連続値の入力)。"""

    def __init__(self, in_dim: int, channels: int, n_blocks: int = 4):
        super().__init__()
        self.content_in_proj = nn.Linear(in_dim, channels)
        layers = []
        for _ in range(n_blocks):
            layers += [nn.Conv1d(channels, channels, 3, padding=1), nn.GroupNorm(1, channels), nn.Mish()]
        self.model = nn.Sequential(*layers, nn.Conv1d(channels, channels, 1))

    def forward(self, x: torch.Tensor, n_frames: int) -> torch.Tensor:
        """x [B, S, Dc] → [B, T, C](T = n_frames)。"""
        x = F.interpolate(rearrange(self.content_in_proj(x), "b s c -> b c s"), size=n_frames, mode="nearest")
        return rearrange(self.model(x), "b c t -> b t c")


class TimestepEmbedder(nn.Module):
    """時刻 t(0〜1)→ ベクトル。t × 1000 の正弦波(256 次元)を MLP に通す(公式と同じ)。"""

    def __init__(self, dim: int, freq_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        half = freq_dim // 2
        self.register_buffer("freqs", torch.exp(-math.log(10000) * torch.arange(half) / half), persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t [B] → [B, H]。"""
        args = 1000 * t[:, None].float() * self.freqs[None]
        return self.mlp(torch.cat([args.cos(), args.sin()], dim=-1))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


class AdaptiveRMSNorm(nn.Module):
    """RMSNorm の出力に、時刻のベクトルから作った倍率・ずれを掛ける・足す(公式の AdaptiveLayerNorm)。"""

    def __init__(self, dim: int):
        super().__init__()
        self.norm, self.proj = RMSNorm(dim), nn.Linear(dim, 2 * dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x [B, T, H]、c [B, 1, H] → [B, T, H]。"""
        weight, bias = self.proj(c).chunk(2, dim=-1)
        return weight * self.norm(x) + bias


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """回転位置埋め込み(隣り合う 2 次元の組を回す。公式の apply_rotary_emb と同じ)。x [B, T, Nh, Hh]、cos・sin [T, Hh/2]。"""
    x1, x2 = rearrange(x.float(), "b t n (d two) -> two b t n d", two=2)
    c, s = cos[None, :, None], sin[None, :, None]
    return rearrange(torch.stack([x1 * c - x2 * s, x2 * c + x1 * s]), "two b t n d -> b t n (d two)").type_as(x)


class Block(nn.Module):
    """Transformer の 1 層(公式の gpt_fast の TransformerBlock): 注意 + SwiGLU。U 字の skip を受ける層は、入力と skip をつなげて Linear。"""

    def __init__(self, dim: int, heads: int, receive_skip: bool):
        super().__init__()
        self.heads = heads
        self.attention_norm, self.ffn_norm = AdaptiveRMSNorm(dim), AdaptiveRMSNorm(dim)
        self.wqkv = nn.Linear(dim, 3 * dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        hidden = _find_multiple(int(2 * 4 * dim / 3), 256)                     # 公式の ModelArgs と同じ中間の幅
        self.w1, self.w3 = nn.Linear(dim, hidden, bias=False), nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.skip_in_linear = nn.Linear(2 * dim, dim) if receive_skip else None

    def forward(self, x, c, cos, sin, attn_mask, skip=None):
        """x [B, T, H]、c [B, 1, H]、attn_mask [B, 1, 1, T](有効なフレームが True)→ [B, T, H]。"""
        if self.skip_in_linear is not None:
            x = self.skip_in_linear(torch.cat([x, skip], dim=-1))
        q, k, v = rearrange(self.wqkv(self.attention_norm(x, c)), "b t (three n d) -> three b t n d", three=3, n=self.heads)
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        q, k, v = (rearrange(a, "b t n d -> b n t d") for a in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)       # [B, Nh, T, Hh]
        h = x + self.wo(rearrange(y, "b n t d -> b t (n d)"))
        z = self.ffn_norm(h, c)
        return h + self.w2(F.silu(self.w1(z)) * self.w3(z))


class WaveNet(nn.Module):
    """公式の WN(VITS 系): 拡張なしの Conv(窓 5)→ 時刻の条件を足す → tanh × sigmoid → 残差と skip に分ける、を n_layers 回。"""

    def __init__(self, dim: int, n_layers: int = 8, kernel: int = 5, p_dropout: float = 0.2):
        super().__init__()
        self.dim, self.n_layers = dim, n_layers
        self.cond_layer = weight_norm(nn.Conv1d(dim, 2 * dim * n_layers, 1))
        self.in_layers = nn.ModuleList([weight_norm(nn.Conv1d(dim, 2 * dim, kernel, padding=kernel // 2)) for _ in range(n_layers)])
        self.res_skip = nn.ModuleList([weight_norm(nn.Conv1d(dim, 2 * dim if i < n_layers - 1 else dim, 1)) for i in range(n_layers)])
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """x [B, H, T]、mask [B, 1, T]、g(時刻)[B, H, 1] → [B, H, T]。"""
        out = torch.zeros_like(x)
        g = self.cond_layer(g)                                                  # [B, 2H × n_layers, 1]
        for i in range(self.n_layers):
            h = self.in_layers[i](x) + g[:, 2 * self.dim * i: 2 * self.dim * (i + 1)]
            a, b = h.chunk(2, dim=1)
            rs = self.res_skip[i](self.drop(torch.tanh(a) * torch.sigmoid(b)))
            if i < self.n_layers - 1:
                x = (x + rs[:, : self.dim]) * mask
                out = out + rs[:, self.dim:]
            else:
                out = out + rs
        return out * mask


class DiT(nn.Module):
    """公式の DiT(final_layer_type wavenet、long_skip_connection、uvit_skip_connection、style・time を token にしない)。"""

    def __init__(self, dim: int = 512, depth: int = 13, heads: int = 8, content_dim: int = 512, wavenet_layers: int = 8,
                 class_dropout: float = 0.1, max_len: int = 16384):
        super().__init__()
        self.class_dropout = class_dropout
        self.t_embedder, self.t_embedder2 = TimestepEmbedder(dim), TimestepEmbedder(dim)
        self.cond_projection = nn.Linear(content_dim, dim)
        self.cond_x_merge_linear = nn.Linear(dim + 2 * N_MELS, dim)
        self.emit = list(range(depth // 2))                                     # 前半: 出力を skip として残す層
        self.receive = [i for i in range(depth) if i > depth // 2]              # 後半: skip を受け取る層(後ろから順に対応)
        self.blocks = nn.ModuleList([Block(dim, heads, i in self.receive) for i in range(depth)])
        self.norm = AdaptiveRMSNorm(dim)
        self.skip_linear = nn.Linear(dim + N_MELS, dim)
        self.conv1 = nn.Linear(dim, dim)
        self.wavenet = WaveNet(dim, wavenet_layers)
        self.res_projection = nn.Linear(dim, dim)
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.final_linear = weight_norm(nn.Linear(dim, dim))
        self.conv2 = nn.Conv1d(dim, N_MELS, 1)
        hh = dim // heads
        freqs = 1.0 / 10000 ** (torch.arange(0, hh, 2).float() / hh)
        ang = torch.outer(torch.arange(max_len).float(), freqs)                 # [max_len, Hh/2]
        self.register_buffer("cos", ang.cos(), persistent=False)
        self.register_buffer("sin", ang.sin(), persistent=False)

    def forward(self, x: torch.Tensor, prompt_x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor, mask: torch.Tensor,
                drop_cond: bool = False) -> torch.Tensor:
        """速度を予測する。x・prompt_x [B, T, F]、cond [B, T, C]、t [B]、mask [B, T](有効なフレームが True)→ [B, T, F]。"""
        T = x.shape[1]
        t1 = self.t_embedder(t)                                                 # [B, H]
        h = torch.cat([x, prompt_x, self.cond_projection(cond)], dim=-1)        # [B, T, 2F + H]
        if drop_cond:                                                           # 条件なし: Mel 以外を 0 に(公式と同じ)
            h = torch.cat([h[..., :N_MELS], torch.zeros_like(h[..., N_MELS:])], dim=-1)
        h = self.cond_x_merge_linear(h)                                         # [B, T, H]
        c = t1[:, None]                                                         # [B, 1, H]
        attn_mask = mask[:, None, None, :]                                      # [B, 1, 1, T]
        skips = []
        for i, blk in enumerate(self.blocks):
            h = blk(h, c, self.cos[:T], self.sin[:T], attn_mask, skips.pop() if i in self.receive else None)
            if i in self.emit:
                skips.append(h)
        x_res = self.skip_linear(torch.cat([self.norm(h, c), x], dim=-1))       # 長い skip 接続 [B, T, H]
        m = mask[:, None].to(x.dtype)                                           # [B, 1, T]
        y = self.wavenet(rearrange(self.conv1(x_res), "b t h -> b h t"), m, self.t_embedder2(t)[..., None])
        y = rearrange(y, "b h t -> b t h") + self.res_projection(x_res)
        shift, scale = self.final_ada(t1).chunk(2, dim=-1)
        y = self.final_linear(self.final_norm(y) * (1 + scale[:, None]) + shift[:, None])
        return rearrange(self.conv2(rearrange(y, "b t h -> b h t")), "b f t -> b t f")


class S2M(nn.Module):
    """length regulator + DiT による Flow Matching(公式の MyModel の length_regulator と cfm)。"""

    SIGMA_MIN = 1e-6

    def __init__(self, codec_dim: int = 1024, dim: int = 512, depth: int = 13, heads: int = 8, wavenet_layers: int = 8):
        super().__init__()
        self.regulator = LengthRegulator(codec_dim, dim)
        self.dit = DiT(dim, depth, heads, content_dim=dim, wavenet_layers=wavenet_layers)

    def forward(self, mel: torch.Tensor, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """学習の損失(L1)。mel [B, T, F](log-Mel、正規化しない)、feat [B, S, Dc](codec の特徴)、mask [B, T]。"""
        cond = self.regulator(feat, mel.shape[1])
        B = mel.shape[0]
        t = torch.rand(B, device=mel.device)
        tb = t[:, None, None]
        z = torch.randn_like(mel)
        y = (1 - (1 - self.SIGMA_MIN) * tb) * z + tb * mel
        u = mel - (1 - self.SIGMA_MIN) * z
        drop = self.training and bool(torch.rand(1) < self.dit.class_dropout)  # 検証の損失では条件を落とさない
        v = self.dit(y, torch.zeros_like(mel), cond, t, mask, drop_cond=drop)
        err = (v - u).abs().mean(-1)                                            # [B, T]
        return ((err * mask).sum(1) / mask.sum(1)).mean()                       # 発話ごとの平均の、バッチの平均(公式と同じ)

    @torch.no_grad()
    def generate(self, feat: torch.Tensor, n_frames: torch.Tensor, n_steps: int = 25, cfg: float = 0.7) -> torch.Tensor:
        """feat [B, S, Dc]、n_frames [B](各発話の Mel のフレーム数)→ Mel [B, T, F](T = n_frames の最大。超えた部分は無意味)。"""
        T = int(n_frames.max())
        mask = torch.arange(T, device=feat.device)[None] < n_frames[:, None]
        cond = self.regulator(feat, T)
        x = torch.randn(feat.shape[0], T, N_MELS, device=feat.device)
        zero = torch.zeros_like(x)
        for i in range(n_steps):
            t = torch.full((x.shape[0],), i / n_steps, device=x.device)
            v = self.dit(x, zero, cond, t, mask)
            if cfg > 0:  # 条件なしの速度: 公式の推論と同じく、length regulator の出力を 0 にして DiT に入れる
                v = (1 + cfg) * v - cfg * self.dit(x, zero, torch.zeros_like(cond), t, mask)
            x = x + v / n_steps
        return x
