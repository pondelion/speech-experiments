"""ハイブリッド型の T2S: 音素ごとの長さを先に予測し、その時間の対応を条件にして、MaskGCT 系の方法で音声トークンを埋める。

exp0002 の結果で、MaskGCT 系(時間の対応を attention で自分で見つける)はテスト文では一番良いが長い文で崩れ、
FastSpeech2 系(時間の対応を長さとして外から与える)は長い文でも崩れなかった。両方の長所を合わせる。

    テキスト → encoder(Transformer)→ 各トークンの隠れ状態 [L, D]
      ├→ 長さの予測器 → 各トークンが音声トークン何個分か(学習時は jsut-lab の正解の長さを使う)
      └→ 長さの数だけ繰り返して並べる → 各音声トークンの位置の「その時刻の音素」[U, D]
         → [MASK] を含む音声トークンの埋め込みに足して、MaskGCT 系と同じく数回に分けて埋める

長さの予測器は 2 通り(--duration_model):
    nar: FastSpeech2 と同じ(畳み込み 2 層で、全トークンの長さを一度に回帰する)
    ar:  自己回帰(前のトークンの長さを入力にして、次のトークンの長さを 0〜MAX_DUR の分類で 1 個ずつ予測する)。
         ロードマップの「長さのような粗い系列だけ自己回帰にする」ハイブリッド
         (生成は一番確率の高い長さを選ぶ。nar の丸めと同じく、乱数を使わない)

shape の軸の記号:
    B: バッチ内の発話数、L: テキストのトークン数(<bos>・<eos> 込み)、U: 音声トークン数、D: 隠れ層の次元
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model

from .t2s_duration import DurationPredictor, make_transformer
from .t2s_masked import MaskedT2S

MAX_DUR = 63  # 自己回帰の長さの予測器が扱う最大の長さ(音声トークン 63 個 = 2.7 秒。これより長いものは 63 にする)


class ARDurationPredictor(nn.Module):
    """自己回帰の長さの予測器: テキストの隠れ状態 + 1 つ前のトークンの長さ → 今のトークンの長さ(分類)。

    Args:
        dim: 隠れ層の次元 D。
        layers: Transformer(GPT-2 と同じ、前だけを見る)の層数。
        heads: attention のヘッド数。
        dropout: dropout の確率。
    """

    def __init__(self, dim: int, layers: int = 2, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.prev_emb = nn.Embedding(MAX_DUR + 2, dim)  # 1 つ前の長さ 0〜MAX_DUR と、先頭用の番号 MAX_DUR + 1
        config = GPT2Config(vocab_size=1, n_positions=1024, n_embd=dim, n_layer=layers, n_head=heads,
                            resid_pdrop=dropout, embd_pdrop=dropout, attn_pdrop=dropout)
        self.gpt = GPT2Model(config)
        self.head = nn.Linear(dim, MAX_DUR + 1)

    def logits(self, h: torch.Tensor, mask: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: テキストの隠れ状態 [B, L, D]。
            mask: 有効な位置が True [B, L]。
            prev: 各位置の 1 つ前のトークンの長さ(先頭は MAX_DUR + 1)[B, L]。

        Returns:
            各位置の長さの logits [B, L, MAX_DUR + 1]。
        """
        x = self.gpt(inputs_embeds=h + self.prev_emb(prev), attention_mask=mask.long()).last_hidden_state
        return self.head(x)

    def loss(self, h: torch.Tensor, mask: torch.Tensor, dur: torch.Tensor) -> torch.Tensor:
        """正解の長さ dur [B, L] で teacher forcing して、分類の損失を返す。"""
        dur = dur.clamp(max=MAX_DUR)
        prev = torch.cat([torch.full_like(dur[:, :1], MAX_DUR + 1), dur[:, :-1]], dim=1)     # 1 つ右にずらす [B, L]
        return F.cross_entropy(self.logits(h, mask, prev)[mask].float(), dur[mask])

    @torch.no_grad()
    def generate(self, h: torch.Tensor, mask: torch.Tensor, decode: str = "argmax") -> torch.Tensor:
        """前から 1 個ずつ長さを決める。[B, L] の長さを返す。

        Args:
            h: テキストの隠れ状態 [B, L, D]。
            mask: 有効な位置が True [B, L]。
            decode: 長さの決め方。"argmax" 一番確率の高い長さ / "mean" 確率で重み付けした平均(期待値)を丸める /
                "sample" 確率どおりにサンプルする。
        """
        B, L, _ = h.shape
        dur = torch.full((B, L), MAX_DUR + 1, dtype=torch.long, device=h.device)
        values = torch.arange(MAX_DUR + 1, device=h.device, dtype=torch.float)                 # 長さの候補 0〜MAX_DUR
        for i in range(L):
            prev = torch.cat([torch.full_like(dur[:, :1], MAX_DUR + 1), dur[:, :-1]], dim=1)
            logits = self.logits(h[:, : i + 1], mask[:, : i + 1], prev[:, : i + 1])[:, i].float()  # [B, MAX_DUR + 1]
            if decode == "argmax":
                dur[:, i] = logits.argmax(-1)
            elif decode == "mean":
                dur[:, i] = (logits.softmax(-1) @ values).round().long()
            else:
                dur[:, i] = torch.distributions.Categorical(logits=logits).sample()
        return dur


class HybridT2S(MaskedT2S):
    """長さの予測 + MaskGCT 系(MaskedT2S を拡張する)。

    Args:
        n_text: テキストの語彙数。
        n_codes: 音声トークンの種類数。
        duration_model: 長さの予測器("nar" か "ar")。
        enc_layers: テキストの encoder の層数。
        dur_weight: 長さの損失の重み。
        **kwargs: MaskedT2S の引数(dim, layers, heads, dropout, text_drop)。
    """

    def __init__(self, n_text: int, n_codes: int, duration_model: str = "nar", enc_layers: int = 4,
                 dur_weight: float = 1.0, **kwargs):
        super().__init__(n_text, n_codes, **kwargs)
        dim, dropout = self.head.in_features, self.dropout.p
        self.duration_model, self.dur_weight = duration_model, dur_weight
        self.enc_emb = nn.Embedding(n_text, dim)
        self.encoder = make_transformer(dim, enc_layers, kwargs.get("heads", 8), dropout)
        self.duration = DurationPredictor(dim, dropout) if duration_model == "nar" else ARDurationPredictor(dim, dropout=dropout)
        self.align_proj = nn.Linear(dim, dim)

    def encode(self, text_ids: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """テキストを encoder に通す。隠れ状態 [B, L, D] と有効な位置のマスク [B, L] を返す。"""
        text = nn.utils.rnn.pad_sequence(text_ids, batch_first=True).to(self.device)
        mask = nn.utils.rnn.pad_sequence([torch.ones(len(t), dtype=torch.bool) for t in text_ids],
                                         batch_first=True).to(self.device)
        return self.encoder(inputs_embeds=self.enc_emb(text), attention_mask=mask.long()).last_hidden_state, mask

    def align(self, h: torch.Tensor, durations: list[torch.Tensor]) -> list[torch.Tensor]:
        """テキストの i 番目の隠れ状態を durations[i] 回繰り返して並べる: 各 [L_i, D] → [U_i, D]。"""
        a = self.align_proj(h)
        return [a[b, : len(d)].repeat_interleave(d.to(self.device), dim=0) for b, d in enumerate(durations)]

    def forward(self, text_ids: list[torch.Tensor], codes: list[torch.Tensor],
                durations: list[torch.Tensor], f0_targets: list[torch.Tensor] | None = None) -> tuple[torch.Tensor, dict]:
        """学習の損失: 隠した位置のトークンの分類の損失 + dur_weight × 長さの損失(正解の長さで並べる)。"""
        h, mask = self.encode(text_ids)
        dur = nn.utils.rnn.pad_sequence(durations, batch_first=True).to(self.device)           # [B, L]
        if self.duration_model == "nar":
            dur_loss = F.mse_loss(self.duration(h)[mask].float(), torch.log1p(dur[mask].float()))
        else:
            dur_loss = self.duration.loss(h, mask, dur)
        loss, parts = super().forward(text_ids, codes, self.align(h, durations), f0_targets)
        return loss + self.dur_weight * dur_loss, {**parts, "dur_loss": dur_loss.item()}

    @torch.no_grad()
    def predict_f0(self, text_ids: list[torch.Tensor], codes: list[torch.Tensor],
                   durations: list[torch.Tensor]) -> list[torch.Tensor]:
        """生成したトークン列から、各トークンの区間の (有声か 0/1, 正規化した log F0) を予測する(F0 の補助の損失で学習したとき)。

        全てのトークンを埋めた状態(時刻 t ≒ 0)で T2S に通し、F0 の補助の出力を読む。exp0005 で S2M の F0 の条件に使う。

        Returns:
            発話ごとの [U_i, 2](無声のトークンの log F0 は 0)。
        """
        h, _ = self.encode(text_ids)
        t = torch.full((len(codes),), 1e-5)
        hid, valid = self.hidden(text_ids, codes, t, torch.zeros(len(codes), dtype=torch.bool), self.align(h, durations))
        p = self.f0_head(hid).float()                                                           # [B, U, 2]
        voiced = (p[..., 0] > 0).float()                                                        # sigmoid > 0.5
        out = torch.stack([voiced, p[..., 1] * voiced], -1)
        return [out[b, : len(c)].cpu() for b, c in enumerate(codes)]

    @torch.no_grad()
    def predict_durations(self, text_ids: list[torch.Tensor], ar_decode: str = "argmax") -> list[torch.Tensor]:
        """各トークンの長さを予測する(合計が 1 以上になるようにする)。

        Args:
            text_ids: 発話ごとのテキスト ID 列 [L_i] のリスト。
            ar_decode: 自己回帰の長さの予測器での長さの決め方(ARDurationPredictor.generate() を参照)。
        """
        h, mask = self.encode(text_ids)
        if self.duration_model == "nar":
            dur = torch.expm1(self.duration(h).float()).round().clamp(min=0).long()
        else:
            dur = self.duration.generate(h, mask, ar_decode)
        out = [dur[b, : len(t)] for b, t in enumerate(text_ids)]
        for d in out:
            if d.sum() == 0:
                d[0] = 1
        return out

    @torch.no_grad()
    def generate(self, text_ids: list[torch.Tensor], durations: list[torch.Tensor] | None = None,
                 **kwargs) -> list[torch.Tensor]:
        """テキストから音声トークン列を作る(長さを予測し、その時間の対応を条件にして埋める)。

        Args:
            text_ids: 発話ごとのテキスト ID 列 [L_i] のリスト。
            durations: 長さを外から与えるとき(分析用に正解の長さを使うなど)。None なら予測する。
            **kwargs: MaskedT2S.generate() の生成の設定(n_steps, temperature, cfg など)。

        Returns:
            発話ごとの音声トークン列 [U_i] のリスト(CPU)。
        """
        if durations is None:
            durations = self.predict_durations(text_ids)
        h, _ = self.encode(text_ids)
        lengths = [int(d.sum()) for d in durations]
        return super().generate(text_ids, lengths, audio_cond=self.align(h, durations), **kwargs)
