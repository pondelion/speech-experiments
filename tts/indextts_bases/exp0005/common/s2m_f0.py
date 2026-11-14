"""S2M に F0 を条件として与える(exp0005 の対策 A)。

S2M の条件(codec の decoder の特徴 [B, U, Dc])に、各音声トークンの区間の (有声か, 正規化した log F0) を線形の層で Dc 次元にして
足す。線形の層の重みは 0 から始めるので、学習済みの S2M に足して追加で学習すると、学習の最初は元の S2M と全く同じ動きになる。
学習では正解の F0(extract_f0.py の runs/f0.pt)、推論では T2S が予測した F0(HybridT2S.predict_f0)を使う。

shape の軸の記号: B 発話数、U 音声トークン数、T Mel のフレーム数(= 4U)、F Mel の帯域数、Dc codec の特徴の次元
"""
import torch
import torch.nn as nn
from einops import rearrange

FPT = 4  # 音声トークン 1 個 = Mel 4 フレーム


class F0Condition(nn.Module):
    """(有声か, 正規化した log F0) [B, U, 2] → S2M の条件に足す特徴 [B, U, Dc]。"""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(2, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        return self.proj(f0)


@torch.no_grad()
def codes_to_mel_f0(pipe, f0cond: F0Condition, codes: list[torch.Tensor], f0s: list[torch.Tensor], n_steps: int = 25,
                    cfg: float = 0.7, batch: int = 16) -> list[torch.Tensor]:
    """exp0001 の TTSPipeline.codes_to_mel と同じ手順で、条件に F0 を足して Mel を作る。

    Args:
        pipe: exp0001 の TTSPipeline(codec・S2M・Mel の正規化を使う)。
        f0cond: 学習した F0Condition。
        codes: 発話ごとの音声トークン列 [U_i]。
        f0s: 発話ごとの (有声か, 正規化した log F0) [U_i, 2]。

    Returns:
        発話ごとの正規化前の log-Mel [T_i, F](CPU)。
    """
    out, dev = [], pipe.device
    for s in range(0, len(codes), batch):
        cs, fs = codes[s: s + batch], f0s[s: s + batch]
        ids = nn.utils.rnn.pad_sequence(cs, batch_first=True).to(dev)                         # [B, U]
        f0 = nn.utils.rnn.pad_sequence(fs, batch_first=True).to(dev).float()                  # [B, U, 2]
        hidden = pipe.codec.decode_hidden(ids) + f0cond(f0)                                   # [B, U, Dc]
        frames = rearrange(torch.arange(ids.shape[1] * FPT, device=dev), "t -> 1 t")
        mask = frames < rearrange(torch.tensor([len(c) * FPT for c in cs], device=dev), "b -> b 1")  # [B, T]
        mel = pipe.s2m.generate(hidden, mask, n_steps=n_steps, cfg=cfg)                       # [B, T, F]
        out += [pipe.normalizer.denormalize(m[: len(c) * FPT]).cpu() for m, c in zip(mel, cs)]
    return out
