"""話者の条件を入れた S2M: exp0001 の S2M(exp0005 の F0 の条件つき)に、参照音声の Mel(プロンプト)と話者ベクトルを足す。

公式(IndexTTS-2.5)の S2M と同じ 2 つの条件を、同じ意味で入れる:
    プロンプト: 正解の Mel と同じ形の入力 prompt_x を足す(参照の区間は本物の Mel、残りは 0)。公式は DiT の入力につなげるので、
                ここでは入力の Linear に「prompt_x からの Linear(重みは 0 から)」を足す(つなげて 1 つの Linear に通すのと同じ)。
                参照の区間の x_t は 0 にし、損失からも除く(公式の CFM と同じ)
    style:      話者ベクトル(CAM++、192 次元)を Linear(重みは 0 から)で次元をそろえ、時刻の埋め込みに足す(全フレーム共通の条件)
学習のときの参照: 同じ発話の先頭の一部(長さは 0〜半分のフレームからランダム。公式の学習コードは非公開なので、ここで決めた値)。
    参照の区間のトークン・F0 の条件は、そのまま入れる(推論で参照音声のトークンを前につなげるのと同じ形)
CFG の「条件なし」(学習で p_uncond の確率、推論の CFG): トークン・F0 の条件、プロンプト、話者ベクトルを全部 0 にする(公式と同じ)。
重みを 0 から始める部品しか足さないので、フェーズ A の S2M から学習を始めると、最初はフェーズ A の S2M と同じ出力になる。

shape の軸の記号: B 発話数、T Mel のフレーム数、F Mel の帯域数(80)、D Transformer の次元、S 話者ベクトルの次元(192)
"""
import torch
import torch.nn as nn
from einops import rearrange

from exp0001.common.s2m import SemanticToMel, timestep_embedding
from lib_common.audio import N_MELS

SPK_DIM = 192


def _zero_linear(i: int, o: int) -> nn.Linear:
    m = nn.Linear(i, o)
    nn.init.zeros_(m.weight)
    nn.init.zeros_(m.bias)
    return m


class SpeakerS2M(SemanticToMel):
    """use_prompt=False(exp0102 の no_s2m_prompt): 学習で参照の区間を作らない(推論でも参照音声を前につなげない)。
    話者ベクトルを外す条件(no_s2m_style)は、呼ぶ側で話者ベクトルを 0 にして渡す。"""

    def __init__(self, cond_in_dim: int, use_prompt: bool = True, **kwargs):
        super().__init__(cond_in_dim, **kwargs)
        self.use_prompt = use_prompt
        self.prompt_proj = _zero_linear(N_MELS, self.dim)
        self.spk_proj = _zero_linear(SPK_DIM, self.dim)

    def velocity(self, xt, cond, t, pad_mask, prompt_x, spk):
        """xt・prompt_x [B, T, F]、cond [B, T, D]、t [B]、pad_mask [B, T](詰め物が True)、spk [B, S] → 速度 [B, T, F]。"""
        x = self.in_proj(torch.cat([xt, cond], dim=-1)) + self.prompt_proj(prompt_x)
        t_emb = self.t_mlp(timestep_embedding(t, self.dim)) + self.spk_proj(spk)            # [B, D]
        for block in self.blocks:
            x = block(x, t_emb, pad_mask)
        return self.out(self.norm_out(x))

    def forward(self, mel: torch.Tensor, hidden: torch.Tensor, mask: torch.Tensor, spk: torch.Tensor) -> torch.Tensor:
        """学習の損失(exp0001 と同じ速度の二乗誤差。参照の区間と詰め物は除く)。mel は正規化した Mel [B, T, F]。"""
        B, T = mel.shape[:2]
        cond = self.regulator(hidden, T)                                                    # [B, T, D]
        keep = (torch.rand(B, device=mel.device) >= self.p_uncond).float()                  # 条件を残す発話が 1 [B]
        n_frames = mask.sum(1)                                                              # [B]
        plen = (torch.rand(B, device=mel.device) * n_frames / 2).long()                     # 参照の長さ [B]
        if not self.use_prompt:
            plen = torch.zeros_like(plen)
        frames = rearrange(torch.arange(T, device=mel.device), "t -> 1 t")
        in_prompt = frames < rearrange(plen, "b -> b 1")                                    # [B, T]
        prompt_x = mel * rearrange(in_prompt.float() * rearrange(keep, "b -> b 1"), "b t -> b t 1")
        x0 = torch.randn_like(mel)
        t = torch.rand(B, device=mel.device)
        tb = rearrange(t, "b -> b 1 1")
        xt = ((1 - tb) * x0 + tb * mel) * rearrange((~in_prompt).float(), "b t -> b t 1")  # 参照の区間の x_t は 0
        v = self.velocity(xt, cond * rearrange(keep, "b -> b 1 1"), t, ~mask, prompt_x, spk * rearrange(keep, "b -> b 1"))
        err = ((v - (mel - x0)) ** 2).mean(-1)                                              # [B, T]
        w = mask & ~in_prompt
        return (err * w).sum() / w.sum()

    @torch.no_grad()
    def generate(self, hidden: torch.Tensor, mask: torch.Tensor, prompt_mel: torch.Tensor, prompt_len: torch.Tensor,
                 spk: torch.Tensor, n_steps: int = 25, cfg: float = 0.7) -> torch.Tensor:
        """hidden は「参照のトークン + 読むトークン」の codec の特徴(F0 の条件を足したもの)[B, U, Dc]、mask [B, T]、
        prompt_mel は参照の区間に本物の Mel(正規化)を入れたもの [B, T, F](残りは 0)、prompt_len [B]、spk [B, S]。
        → 正規化した Mel [B, T, F](参照の区間も含む。呼ぶ側で取り除く)。"""
        T = mask.shape[1]
        cond = self.regulator(hidden, T)
        pad = ~mask
        frames = rearrange(torch.arange(T, device=hidden.device), "t -> 1 t")
        gen = rearrange((frames >= rearrange(prompt_len, "b -> b 1")).float(), "b t -> b t 1")  # 作る区間が 1
        x = torch.randn(hidden.shape[0], T, N_MELS, device=hidden.device) * gen
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device)
            if cfg > 0:
                v_c, v_u = self.velocity(torch.cat([x, x]), torch.cat([cond, torch.zeros_like(cond)]), torch.cat([t, t]),
                                         torch.cat([pad, pad]), torch.cat([prompt_mel, torch.zeros_like(prompt_mel)]),
                                         torch.cat([spk, torch.zeros_like(spk)])).chunk(2)
                v = (1 + cfg) * v_c - cfg * v_u
            else:
                v = self.velocity(x, cond, t, pad, prompt_mel, spk)
            x = (x + v * dt) * gen                                                          # 参照の区間は 0 のまま(公式と同じ)
        return x * rearrange(mask, "b t -> b t 1")
