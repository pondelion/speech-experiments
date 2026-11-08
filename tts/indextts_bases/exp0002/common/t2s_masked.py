"""MaskGCT 系の非自己回帰の T2S: 長さを先に決め、全部隠した音声トークン列を、数回に分けて埋めていく。

MaskGCT(Amphion の models/tts/maskgct/maskgct_t2s.py)の T2S と同じ手順にする。
    学習: 音声トークン列の一部を [MASK] に置き換え、[テキスト][音声トークン] を双方向の Transformer に通して、
          隠した位置の正解を当てる。隠す割合は時刻 t(0〜1 の一様乱数)から sin(πt/2) で決める(下限 0.2)。
          モデルは t も条件として受け取る(何割くらい隠れているか)
    生成: 全位置を隠した状態(t = 1)から始め、n_steps 回に分けて t を 0 まで下げる。毎回、隠れている位置を全て予測して
          サンプルし、「確信度の低いもの」を次の t に応じた数だけ隠し直す。最後の回は一番確率の高いトークンを選ぶ
    長さ: 公式は「参照音声の 1 音素あたりの長さ × 読む文の音素数」で決める。参照音声が無いので、
          学習データ全体の 1 音素あたりの長さ(tokens_per_phone)× 音素数で決める
    CFG: テキストありとテキストなしの出力(Transformer の最後の隠れ状態)の差を強める(公式の既定値 cfg 2.5、rescale 0.75)。
         テキストなしの出力を学ぶため、学習では text_drop の確率でテキストを除く(この確率はここで決めた値)

公式との違い:
    - Transformer は exp0001 の S2M と同じ DiTBlock(双方向、RoPE、時刻で LayerNorm を調整する adaLN-Zero)。
      公式は Llama 型(RMSNorm を時刻で調整)で、ほぼ同じ形
    - 参照音声(prompt)を使わない(このタスクはテキストのみ)ので、学習でも参照音声にあたる部分は作らない

shape の軸の記号:
    B: バッチ内の発話数、L: テキストのトークン数、U: 音声トークン数、S: つないだ系列の長さ(L + U)、
    D: 隠れ層の次元、V: 音声トークンの種類数
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from exp0001.common.s2m import DiTBlock, timestep_embedding


def mask_ratio(t: torch.Tensor) -> torch.Tensor:
    """時刻 t(1 = 全部隠す、0 = 隠さない)のときに隠す割合。"""
    return torch.sin(t * math.pi / 2)


def gumbel_noise(x: torch.Tensor) -> torch.Tensor:
    """Gumbel 分布の乱数(logits に足して argmax を取ると、softmax の確率どおりにサンプルしたことになる)。"""
    return -torch.log(-torch.log(torch.rand_like(x).clamp(1e-10, 1.0)) + 1e-10)


class MaskedT2S(nn.Module):
    """テキスト → 音声トークン列(隠したトークンを数回に分けて埋める)。

    Args:
        n_text: テキストの語彙数。
        n_codes: 音声トークンの種類数 V(番号 V は [MASK] に使う)。
        dim: 隠れ層の次元 D。
        layers: Transformer の層数。
        heads: attention のヘッド数。
        dropout: 入力の埋め込みにかける dropout の確率。
        text_drop: 学習でテキストを除く確率(CFG 用)。
        tokens_per_phone: 1 音素あたりの音声トークン数(学習データ全体の平均。長さを決めるのに使う)。
        fixed_code_dim: 0 より大きければ、音声トークンの入力の埋め込みを、codec の固定の代表ベクトル(この次元)+
            学習する線形の射影にする(exp0003 の ablation)。0 なら、トークンごとに学習する埋め込み(既定)。
    """

    def __init__(self, n_text: int, n_codes: int, dim: int = 512, layers: int = 8, heads: int = 8,
                 dropout: float = 0.1, text_drop: float = 0.1, tokens_per_phone: float = 1.0, fixed_code_dim: int = 0):
        super().__init__()
        self.n_codes, self.text_drop, self.tokens_per_phone = n_codes, text_drop, tokens_per_phone
        self.text_emb = nn.Embedding(n_text, dim)
        if fixed_code_dim > 0:
            # 音声トークンの入力に、codec の固定の代表ベクトル [V, C] を使い、線形の射影 C → D だけを学習する
            # (exp0003 の ablation。代表ベクトルは load_code_vectors() で入れる。[MASK] のベクトルだけは学習する)
            self.register_buffer("code_vecs", torch.zeros(n_codes, fixed_code_dim))
            self.code_proj = nn.Linear(fixed_code_dim, dim)
            self.mask_emb = nn.Parameter(torch.randn(dim) * 0.02)
        else:
            self.code_emb = nn.Embedding(n_codes + 1, dim)  # 最後の番号が [MASK]
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([DiTBlock(dim, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_codes)

    @property
    def device(self) -> torch.device:
        return self.head.weight.device

    def embed_codes(self, x: torch.Tensor) -> torch.Tensor:
        """音声トークン(番号 V は [MASK])の埋め込み: [U] → [U, D]。"""
        if not hasattr(self, "code_vecs"):
            return self.code_emb(x)
        is_mask = x == self.n_codes
        e = self.code_proj(self.code_vecs[x.clamp(max=self.n_codes - 1)])                     # [U, C] → [U, D]
        return torch.where(is_mask[:, None], self.mask_emb.to(e.dtype), e)

    def load_code_vectors(self, vecs: torch.Tensor):
        """固定の代表ベクトル [V, C] を入れる(fixed_code_dim > 0 のときだけ)。"""
        self.code_vecs.copy_(vecs.to(self.code_vecs.device))

    def hidden(self, text_ids: list[torch.Tensor], xt: list[torch.Tensor], t: torch.Tensor,
               drop_text: torch.Tensor, audio_cond: list[torch.Tensor] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """[テキスト][音声トークン(一部 [MASK])] を Transformer に通し、音声部分の最後の隠れ状態を返す。

        Args:
            text_ids: 発話ごとのテキスト ID 列 [L_i] のリスト。
            xt: 発話ごとの、一部を [MASK](番号 V)にした音声トークン列 [U_i] のリスト。
            t: 発話ごとの時刻 [B]。
            drop_text: テキストを除く発話が True [B](CFG のテキストなしの出力用)。
            audio_cond: 音声トークンの各位置に足す条件 [U_i, D] のリスト(ハイブリッド型の「その時刻の音素」)。
                テキストを除く発話には足さない。

        Returns:
            音声部分の隠れ状態 [B, U, D](U は一番長い発話に合わせる)と、有効な位置が True のマスク [B, U]。
        """
        # 発話ごとに [テキスト][音声] の埋め込みをつなぐ(テキストを除く発話は音声だけ): 各 [L_i + U_i, D]
        drops = drop_text.tolist()
        texts = [tx.new_zeros(0) if d else tx for tx, d in zip(text_ids, drops)]
        audios = [self.embed_codes(x.to(self.device)) for x in xt]                               # 各 [U_i, D]
        if audio_cond is not None:
            audios = [a if d else a + c for a, c, d in zip(audios, audio_cond, drops)]
        seqs = [torch.cat([self.text_emb(tx.to(self.device)), a]) for tx, a in zip(texts, audios)]
        x = self.dropout(nn.utils.rnn.pad_sequence(seqs, batch_first=True))                    # [B, S, D]
        pad = nn.utils.rnn.pad_sequence([torch.zeros(len(s), dtype=torch.bool) for s in seqs], batch_first=True,
                                        padding_value=True).to(self.device)                    # 詰め物が True [B, S]
        t_emb = self.time_mlp(timestep_embedding(t.to(self.device), x.shape[-1]))              # [B, D]
        for blk in self.blocks:
            x = blk(x, t_emb, pad)
        x = self.norm(x)
        # 音声部分だけを取り出して並べ直す: 各 [U_i, D] → [B, U, D]
        audio = [x[b, len(tx): len(tx) + len(a)] for b, (tx, a) in enumerate(zip(texts, xt))]
        mask = nn.utils.rnn.pad_sequence([torch.ones(len(a), dtype=torch.bool) for a in audio],
                                         batch_first=True).to(self.device)
        return nn.utils.rnn.pad_sequence(audio, batch_first=True), mask

    def forward(self, text_ids: list[torch.Tensor], codes: list[torch.Tensor],
                audio_cond: list[torch.Tensor] | None = None) -> tuple[torch.Tensor, dict]:
        """学習の損失: 隠した位置の音声トークンの分類の損失。

        Args:
            text_ids: 発話ごとのテキスト ID 列 [L_i] のリスト。
            codes: 発話ごとの正解の音声トークン列 [U_i] のリスト。
            audio_cond: 音声トークンの各位置に足す条件 [U_i, D] のリスト(hidden() を参照)。

        Returns:
            損失と、記録用の値 {"acc": 隠した位置のトークンの正解率}。
        """
        B = len(codes)
        t = torch.rand(B).clamp(1e-5, 1.0)
        ratio = mask_ratio(t).clamp(min=0.2)                                                   # 隠す割合(下限 0.2)[B]
        masks = []
        for c, r in zip(codes, ratio):
            m = torch.rand(len(c)) < r                                                         # 各位置を確率 r で隠す [U_i]
            if not m.any():
                m[torch.randint(len(c), ())] = True                                            # 最低 1 個は隠す
            masks.append(m)
        xt = [torch.where(m, self.n_codes, c) for c, m in zip(codes, masks)]
        drop = torch.rand(B) < self.text_drop
        h, _ = self.hidden(text_ids, xt, t, drop, audio_cond)                                  # [B, U, D]
        sel = nn.utils.rnn.pad_sequence(masks, batch_first=True).to(self.device)               # 隠した位置 [B, U]
        logits = self.head(h[sel]).float()                                                     # [隠した数, V]
        y = torch.cat([c[m] for c, m in zip(codes, masks)]).to(self.device)
        loss = F.cross_entropy(logits, y)
        return loss, {"acc": (logits.argmax(-1) == y).float().mean().item()}

    def rule_lengths(self, n_phones: list[int]) -> list[int]:
        """音素数から音声トークン数を決める(1 音素あたりの長さ × 音素数)。"""
        return [max(1, round(n * self.tokens_per_phone)) for n in n_phones]

    @torch.no_grad()
    def generate(self, text_ids: list[torch.Tensor], lengths: list[int], n_steps: int = 50, temperature: float = 0.9,
                 filter_thres: float = 0.98, cfg: float = 2.5, rescale_cfg: float = 0.75,
                 audio_cond: list[torch.Tensor] | None = None) -> list[torch.Tensor]:
        """テキストと長さから音声トークン列を作る(全部隠した状態から n_steps 回で埋める。既定値は公式と同じ)。

        Args:
            text_ids: 発話ごとのテキスト ID 列 [L_i] のリスト。
            lengths: 発話ごとの音声トークン数 U_i(rule_lengths() か、分析用に正解の長さ)。
            audio_cond: 音声トークンの各位置に足す条件 [U_i, D] のリスト(hidden() を参照)。
            n_steps: 埋める回数。
            temperature: サンプルの温度(t に比例して下げる)。
            filter_thres: 確率の上位 (1 - filter_thres) の割合のトークンだけからサンプルする。
            cfg: CFG の強さ(0 ならテキストなしの出力を使わない)。
            rescale_cfg: CFG で大きくなった隠れ状態の大きさを、元の標準偏差に戻す割合。

        Returns:
            発話ごとの音声トークン列 [U_i] のリスト(CPU)。
        """
        B, Um = len(text_ids), max(lengths)
        lens = torch.tensor(lengths, device=self.device)
        valid = rearrange(torch.arange(Um, device=self.device), "u -> 1 u") < rearrange(lens, "b -> b 1")  # [B, U]
        seq = torch.zeros(B, Um, dtype=torch.long, device=self.device)   # 今のトークン [B, U]
        masked = valid.clone()                                            # 隠れている位置 [B, U]
        k = math.ceil((1 - filter_thres) * self.n_codes)                  # サンプルの候補にするトークン数
        t_list = [1.0 - i / n_steps for i in range(n_steps)] + [0.0]
        for i in range(n_steps):
            t = torch.full((B,), t_list[i])
            xt = [torch.where(masked[b, :n], self.n_codes, seq[b, :n]) for b, n in enumerate(lengths)]
            h, _ = self.hidden(text_ids, xt, t, torch.zeros(B, dtype=torch.bool), audio_cond)  # [B, U, D]
            if cfg > 0:  # テキストなしの出力との差を強め、大きさを元の標準偏差に近づける(公式と同じ式)
                h_u, _ = self.hidden(text_ids, xt, t, torch.ones(B, dtype=torch.bool))
                std = h[valid].std()
                h = h + cfg * (h - h_u)
                h = rescale_cfg * (h * std / h[valid].std()) + (1 - rescale_cfg) * h
            logits = self.head(h).float()                                                      # [B, U, V]
            logits[logits < logits.topk(k, dim=-1).values[..., -1:]] = float("-inf")         # 上位 k 個だけ残す
            if i == n_steps - 1:
                sampled = logits.argmax(-1)                                                    # 最後は一番確率の高いもの
            else:
                sampled = (logits / max(temperature * t_list[i], 1e-3) + gumbel_noise(logits)).argmax(-1)
            seq = torch.where(masked, sampled, seq)                                            # 隠れていた位置だけ埋める
            # 確信度(選んだトークンの確率 + 時刻に比例した乱数)が低い順に、次の時刻の割合の数だけ隠し直す
            n_next = (mask_ratio(torch.tensor(t_list[i + 1])) * lens).long()                   # 発話ごとの数 [B]
            if int(n_next.max()) == 0:
                break
            prob = logits.softmax(-1).gather(-1, rearrange(sampled, "b u -> b u 1"))
            unsure = 1 - (rearrange(prob, "b u 1 -> b u") + t_list[i] * gumbel_noise(prob[..., 0]))
            unsure = unsure.masked_fill(~masked, float("-inf"))                                # 埋め終えた位置は隠し直さない
            rank = unsure.argsort(dim=-1, descending=True).argsort(dim=-1)                     # 確信度の低い順の順位 [B, U]
            masked = (rank < rearrange(n_next, "b -> b 1")) & valid
        return [seq[b, :n].cpu() for b, n in enumerate(lengths)]
