"""S2M(Semantic-to-Mel): 音声トークン列から Mel を作る Flow Matching。IndexTTS2.5 の S2M に当たる部分。

なぜ codec の decoder だけでは足りないのか:
    codec の decoder も Mel を復元できるが、二乗誤差で学習した復元は「ありそうな Mel の平均」になり、
    細かい部分がぼやけやすい。S2M は Flow Matching(ノイズから少しずつ Mel へ流す生成モデル)なので、
    平均ではなく「本物らしい 1 つの Mel」を作れる。

Flow Matching のおさらい(flow_matching ハンズオンの段階1〜3と同じ):
    学習: 正解 Mel x1 とノイズ x0 を混ぜた x_t = (1-t)·x0 + t·x1 を入力に、速度 x1 - x0 を当てる。
    生成: ノイズから始めて、予測した速度で t=0→1 まで少しずつ動かす(Euler 法)。

構成(IndexTTS2.5 の公開 DiT 版に合わせた簡易版):
    条件: codec の decode_hidden() の特徴 [B,U,Dc](約 23.4Hz)
          → Mel の長さ(4 倍)に補間 → Conv で整える(length regulator)→ [B,T,D]
    速度を出す NN: フレームごとに [今の状態 x_t, 条件] をつないで Transformer に入れる。
          - 時刻 t は adaptive LayerNorm(t によって正規化の倍率・ずれ・残差の強さを変える)で入れる
          - 位置は RoPE(attention の中で、フレーム同士の相対的な距離を表す方法)で入れる
    CFG: 学習時に一定確率で条件を 0 にし、推論時に (1+w)·v(条件あり) - w·v(条件なし) で強調する。
IndexTTS2.5 と違う点:
    - 参照 Mel・話者ベクトルは入れない(今回は 1 話者の読み上げだけなので)
    - Transformer は U 字型の skip 接続などを省いた、素直な積み重ね

shape の軸の記号:
    B: バッチ内の発話数
    U: 音声トークン数(約 23.4 個/秒)
    T: Mel のフレーム数(= 4U、93.75 フレーム/秒)
    F: Mel の帯域数(80)
    Dc: codec の特徴の次元(cond_in_dim)
    D: Transformer の次元(dim)
    H: attention のヘッド数、Dh: 1 ヘッドの次元(D / H)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from lib_common.audio import N_MELS

from .codec import DOWNSAMPLE


class LengthRegulator(nn.Module):
    """トークンの時間解像度の特徴を、Mel のフレーム数に引き伸ばして整える。

    Args:
        in_dim: 入力特徴の次元 Dc(codec の hidden_dim)。
        dim: 出力の次元 D。
    """

    def __init__(self, in_dim: int, dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, dim)
        self.convs = nn.Sequential(*[nn.Sequential(nn.Conv1d(dim, dim, 3, padding=1), nn.GroupNorm(1, dim),
                                                   nn.Mish()) for _ in range(2)], nn.Conv1d(dim, dim, 1))

    def forward(self, h: torch.Tensor, n_frames: int) -> torch.Tensor:
        """
        Args:
            h: トークンの特徴 [B(発話数), U(トークン数), Dc(codec の特徴の次元)]。
            n_frames: 引き伸ばした後のフレーム数 T(= 4U)。

        Returns:
            条件 [B, T, D(出力の次元)]。
        """
        x = rearrange(self.proj(h), "b u d -> b d u")        # [B, U, Dc] → [B, U, D] → [B, D, U](Conv 用)
        x = F.interpolate(x, size=n_frames, mode="nearest")  # 1 トークン → 4 フレーム [B, D, U] → [B, D, T]
        return rearrange(self.convs(x), "b d t -> b t d")    # [B, D, T] → [B, T, D]


def rope(x: torch.Tensor) -> torch.Tensor:
    """RoPE(rotary position embedding)をかける。

    特徴を 2 次元ずつのペアに分け、フレーム位置に比例した角度だけ回転させる。
    回転したベクトル同士の内積は「位置の差」だけで決まるので、attention が相対的な距離を扱える。

    Args:
        x: attention の query か key [B(発話数), H(ヘッド数), T(フレーム数), Dh(1 ヘッドの次元)]。

    Returns:
        同じ形のテンソル [B, H, T, Dh]。
    """
    T, d = x.shape[2], x.shape[3]
    freq = 1.0 / (10000 ** (torch.arange(0, d, 2, device=x.device).float() / d))  # ペアごとの回転の速さ [Dh/2]
    # (x の軸: B 発話, H ヘッド, T フレーム, Dh 1 ヘッドの次元)
    ang = torch.outer(torch.arange(T, device=x.device).float(), freq)             # 位置 × 回転の速さ = 角度 [T, Dh/2]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]                                            # ペアの 1 つ目・2 つ目 [B, H, T, Dh/2]
    # 2 次元の回転 (x1, x2) → (x1 cos − x2 sin, x1 sin + x2 cos) を計算し、ペアを元の並びに戻す
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)     # [B, H, T, Dh/2, 2]
    return rearrange(rotated, "b h t d two -> b h t (d two)")                      # [B, H, T, Dh]


class DiTBlock(nn.Module):
    """時刻 t で調整する Transformer block(DiT の adaLN-Zero 形式)。

    LayerNorm の後の倍率(scale)・ずれ(shift)と、残差に足す強さ(gate)を、時刻の埋め込みから作る。
    gate を出す層の重みを 0 で初期化するので、学習の最初は各 block が恒等写像になり、学習が安定する。

    text_cond=True のときは、self-attention の後に、テキストの隠れ状態への cross-attention を足す
    (各フレームが、テキストのどの位置を参考にするかを自分で選ぶ)。その出力の層の重みを 0 で初期化するので、
    学習済みの S2M に足して追加学習するとき、学習の最初は元の S2M とまったく同じ動きから始まる。

    Args:
        dim: 隠れ層の次元 D。
        heads: attention のヘッド数 H。
        text_cond: テキストへの cross-attention を持つか。
    """

    def __init__(self, dim: int, heads: int, text_cond: bool = False):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.ada = nn.Linear(dim, dim * 6)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)
        if text_cond:
            self.norm_c = nn.LayerNorm(dim)
            self.q_c = nn.Linear(dim, dim)
            self.kv_c = nn.Linear(dim, dim * 2)
            self.out_c = nn.Linear(dim, dim)
            nn.init.zeros_(self.out_c.weight)
            nn.init.zeros_(self.out_c.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, pad_mask: torch.Tensor,
                text: torch.Tensor | None = None, text_pad: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: 特徴 [B(発話数), T(フレーム数), D(隠れ層の次元)]。
            t_emb: 時刻の埋め込み [B, D]。
            pad_mask: 詰め物のフレームが True [B, T](attention の対象から外す)。
            text: テキストの隠れ状態(D 次元に射影済み)[B, L(テキストのトークン数), D](text_cond のときだけ)。
            text_pad: テキストの詰め物の位置が True [B, L]。

        Returns:
            [B, T, D]。
            (以下のコメントの記号: H ヘッド数、Dh 1 ヘッドの次元 = D / H)
        """
        # 時刻から 6 種類の調整値を作る: [B, D] → [B, 6D] → [B, 1, 6D] → 6 個の [B, 1, D](T 方向に放送)
        ada = rearrange(self.ada(t_emb), "b d -> b 1 d")
        shift1, scale1, gate1, shift2, scale2, gate2 = ada.chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale1) + shift1                                 # [B, T, D]
        # Q・K・V をまとめて作り、ヘッドに分ける: [B, T, 3D] → 3 × [B, H, T, Dh]
        q, k, v = rearrange(self.qkv(h), "b t (three h dh) -> three b h t dh", three=3, h=self.heads)
        # attention: 各フレームが全フレームを参照する(詰め物のフレームは参照しない)→ [B, H, T, Dh]
        key_mask = rearrange(~pad_mask, "b t -> b 1 1 t")                         # 参照してよいフレームが True
        attn = F.scaled_dot_product_attention(rope(q), rope(k), v, attn_mask=key_mask)
        x = x + gate1 * self.out(rearrange(attn, "b h t dh -> b t (h dh)"))      # ヘッドをつなげて [B, T, D]
        if text is not None:
            # cross-attention: フレームの特徴から query、テキストの隠れ状態から key・value を作る
            qc = rearrange(self.q_c(self.norm_c(x)), "b t (h dh) -> b h t dh", h=self.heads)        # [B, H, T, Dh]
            kc, vc = rearrange(self.kv_c(text), "b l (two h dh) -> two b h l dh", two=2, h=self.heads)  # 2 × [B, H, L, Dh]
            text_mask = rearrange(~text_pad, "b l -> b 1 1 l")                    # 参照してよいテキストの位置が True
            attn_c = F.scaled_dot_product_attention(qc, kc, vc, attn_mask=text_mask)  # [B, H, T, Dh]
            x = x + self.out_c(rearrange(attn_c, "b h t dh -> b t (h dh)"))        # [B, T, D]
        h = self.norm2(x) * (1 + scale2) + shift2
        return x + gate2 * self.mlp(h)                                            # [B, T, D] → [B, T, 4D] → [B, T, D]


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """時刻 t(0〜1)を、いろいろな周波数の sin・cos を並べたベクトルにする。

    Args:
        t: 発話ごとの時刻 [B(発話数)]。
        dim: 出力の次元 D。

    Returns:
        [B, D]。
    """
    half = dim // 2
    freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)  # [D/2]
    ang = torch.outer(t * 1000, freq)                                                         # 時刻 × 周波数 = 角度 [B, D/2]
    return torch.cat([ang.cos(), ang.sin()], dim=-1)                                          # [B, D]


class SemanticToMel(nn.Module):
    """音声トークンの特徴を条件に、Mel を Flow Matching で作る。

    Args:
        cond_in_dim: 条件として受け取る codec 特徴の次元 Dc。
        dim: Transformer の次元 D。
        layers: block の数。
        heads: attention のヘッド数 H。
        p_uncond: 学習時に条件を 0 にする確率(CFG 用)。
        text_dim: テキストの隠れ状態(T2S の GPT のテキスト部分)を条件に加えるときの、その次元 Dt(0 なら加えない)。
            加えるときは、各 block にテキストへの cross-attention を持つ。
    """

    def __init__(self, cond_in_dim: int, dim: int = 512, layers: int = 8, heads: int = 8, p_uncond: float = 0.1,
                 text_dim: int = 0):
        super().__init__()
        self.dim, self.p_uncond, self.text_dim = dim, p_uncond, text_dim
        self.regulator = LengthRegulator(cond_in_dim, dim)
        self.in_proj = nn.Linear(N_MELS + dim, dim)
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.text_proj = nn.Linear(text_dim, dim) if text_dim > 0 else None
        self.blocks = nn.ModuleList([DiTBlock(dim, heads, text_cond=text_dim > 0) for _ in range(layers)])
        self.norm_out = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, N_MELS)

    def velocity(self, xt: torch.Tensor, cond: torch.Tensor, t: torch.Tensor, pad_mask: torch.Tensor,
                 text: torch.Tensor | None = None, text_pad: torch.Tensor | None = None) -> torch.Tensor:
        """速度 v(x_t, 条件, t) を出す。

        Args:
            xt: 今の状態 [B(発話数), T(フレーム数), F(Mel の帯域数 = 80)]。
            cond: length regulator を通した条件 [B, T, D(Transformer の次元)](条件なしなら 0)。
            t: 時刻 [B]。
            pad_mask: 詰め物のフレームが True [B, T]。
            text: テキストの隠れ状態 [B, L(テキストのトークン数), Dt](text_dim > 0 のときだけ。条件なしなら 0)。
            text_pad: テキストの詰め物の位置が True [B, L]。

        Returns:
            速度 [B, T, F]。
        """
        x = self.in_proj(torch.cat([xt, cond], dim=-1))       # [B, T, F + D] → [B, T, D]
        t_emb = self.t_mlp(timestep_embedding(t, self.dim))   # [B] → [B, D]
        text_h = self.text_proj(text) if self.text_proj is not None else None  # [B, L, Dt] → [B, L, D]
        for block in self.blocks:
            x = block(x, t_emb, pad_mask, text_h, text_pad)   # [B, T, D]
        return self.out(self.norm_out(x))                     # [B, T, D] → [B, T, F]

    def forward(self, mel: torch.Tensor, hidden: torch.Tensor, mask: torch.Tensor,
                text: torch.Tensor | None = None, text_pad: torch.Tensor | None = None) -> torch.Tensor:
        """学習用の損失(Flow Matching の速度の二乗誤差)。

        Args:
            mel: 正規化した正解 Mel [B(発話数), T(フレーム数 = 4U), F(Mel の帯域数 = 80)]。
            hidden: codec.decode_hidden() の特徴 [B, U(トークン数), Dc(codec の特徴の次元)]。
            mask: 有効なフレームが True [B, T]。
            text: テキストの隠れ状態 [B, L(テキストのトークン数), Dt](text_dim > 0 のときだけ)。
            text_pad: テキストの詰め物の位置が True [B, L]。

        Returns:
            損失(スカラー)。
        """
        B = mel.shape[0]
        cond = self.regulator(hidden, mel.shape[1])                                     # [B, U, Dc] → [B, T, D]
        keep = (torch.rand(B, device=mel.device) >= self.p_uncond).float()              # 条件を残す発話が 1 [B]
        keep = rearrange(keep, "b -> b 1 1")                                            # [B, T, D] に掛けられる形に
        x0 = torch.randn_like(mel)                                                      # ノイズ [B, T, F]
        t = torch.rand(B, device=mel.device)                                            # 発話ごとの時刻 [B]
        tb = rearrange(t, "b -> b 1 1")                                                 # [B, T, F] に掛けられる形に
        xt = (1 - tb) * x0 + tb * mel                                                   # [B, T, F]
        text = text * keep if text is not None else None                                # 条件なしの発話はテキストも 0 に
        v = self.velocity(xt, cond * keep, t, ~mask, text, text_pad)                    # [B, T, F]
        err = ((v - (mel - x0)) ** 2).mean(-1)                                          # フレームごとの誤差 [B, T]
        return (err * mask).sum() / mask.sum()

    @torch.no_grad()
    def generate(self, hidden: torch.Tensor, mask: torch.Tensor, n_steps: int = 25, cfg: float = 0.7,
                 temperature: float = 1.0, text: torch.Tensor | None = None,
                 text_pad: torch.Tensor | None = None) -> torch.Tensor:
        """ノイズから Euler 法 + CFG で Mel を作る。

        Args:
            hidden: codec.decode_hidden() の特徴 [B(発話数), U(トークン数), Dc(codec の特徴の次元)]。
            mask: 有効なフレームが True [B, T(フレーム数 = 4U)]。
            n_steps: Euler 法のステップ数(IndexTTS2.5 の公開推論は 25)。
            cfg: CFG の強さ w(IndexTTS2.5 の公開推論は 0.7)。
            temperature: 初期ノイズの標準偏差。
            text: テキストの隠れ状態 [B, L(テキストのトークン数), Dt](text_dim > 0 のときだけ)。
            text_pad: テキストの詰め物の位置が True [B, L]。

        Returns:
            正規化した Mel [B, T, F(Mel の帯域数 = 80)](詰め物の部分は 0)。
            (以下のコメントの記号: D Transformer の次元)
        """
        T = hidden.shape[1] * DOWNSAMPLE
        cond = self.regulator(hidden, T)                                                 # [B, U, Dc] → [B, T, D]
        pad = ~mask
        x = torch.randn(hidden.shape[0], T, N_MELS, device=hidden.device) * temperature  # 初期ノイズ [B, T, F]
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device)                       # [B]
            if cfg > 0:
                # 条件あり・なしを 1 つのバッチ [2B, ...] にまとめて計算し、[B, T, F] 2 つに分ける
                # (条件なしは、トークンの条件もテキストの条件も 0 にする)
                text2 = torch.cat([text, torch.zeros_like(text)]) if text is not None else None
                text_pad2 = torch.cat([text_pad, text_pad]) if text_pad is not None else None
                v_c, v_u = self.velocity(torch.cat([x, x]), torch.cat([cond, torch.zeros_like(cond)]),
                                         torch.cat([t, t]), torch.cat([pad, pad]), text2, text_pad2).chunk(2)
                v = (1 + cfg) * v_c - cfg * v_u
            else:
                v = self.velocity(x, cond, t, pad, text, text_pad)
            x = x + v * dt
        return x * rearrange(mask, "b t -> b t 1")                                       # 詰め物を 0 に [B, T, F]
