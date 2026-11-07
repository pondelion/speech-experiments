"""FastSpeech2 系の非自己回帰の T2S: テキストの各トークンの長さを予測し、全ての音声トークンを一度に並列で予測する。

exp0001 の自己回帰の T2S(GPT)は、音声トークンを 1 個ずつ作り、<EA> を出したところで長さが決まる。
「今テキストのどこを読んでいるか」もモデルが自分で覚えておく必要があり、途中で流れを外れると戻れなかった。
ここでは、テキストと音声の時間の対応(どの音素が音声トークン何個分か)を外から与える。

    テキスト ID 列 → encoder(Transformer)→ テキストの隠れ状態 [L, D]
      ├→ 長さの予測器 → 各トークンの長さ(学習時は jsut-lab の正解の長さを使う)
      └→ 長さの数だけ繰り返して並べる(length regulator)→ [U, D]
         → decoder(Transformer)→ 各位置の音声トークンを分類(全位置を同時に)

FastSpeech2 の音の高さ(F0)・大きさ(energy)の予測は入れない。出力が Mel ではなく音声トークンで、
高さ・大きさの情報もトークンに含まれるため(exp0002 では、まず長さだけの最小の形で比べる)。
encoder・decoder は transformers の BertModel(双方向の Transformer、学習する位置埋め込み)を使う。

shape の軸の記号:
    B: バッチ内の発話数、L: テキストのトークン数(<bos>・<eos> 込み)、U: 音声トークン数、
    D: 隠れ層の次元、V: 音声トークンの種類数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import BertConfig, BertModel


def make_transformer(dim: int, layers: int, heads: int, dropout: float, max_len: int = 2048) -> BertModel:
    """双方向の Transformer(BERT と同じ形。単語の埋め込みは使わず、inputs_embeds で入力する)。

    Args:
        dim: 隠れ層の次元 D。
        layers: 層数。
        heads: attention のヘッド数。
        dropout: dropout の確率。
        max_len: 位置埋め込みの最大の長さ。

    Returns:
        BertModel。
    """
    config = BertConfig(vocab_size=1, hidden_size=dim, num_hidden_layers=layers, num_attention_heads=heads,
                        intermediate_size=dim * 4, hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout,
                        max_position_embeddings=max_len, type_vocab_size=1)
    return BertModel(config, add_pooling_layer=False)


class DurationPredictor(nn.Module):
    """FastSpeech2 の長さの予測器: 1 次元の畳み込み 2 層 → 各トークンの log(長さ + 1)。

    Args:
        dim: 入力の次元 D。
        dropout: dropout の確率。
    """

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(dim, dim, 3, padding=1) for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(2)])
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: テキストの隠れ状態 [B, L, D]。

        Returns:
            log(長さ + 1) の予測 [B, L]。
        """
        for conv, norm in zip(self.convs, self.norms):
            h = rearrange(conv(rearrange(h, "b l d -> b d l")), "b d l -> b l d")  # 畳み込みは [B, D, L] で行う
            h = self.dropout(norm(F.relu(h)))
        return rearrange(self.out(h), "b l 1 -> b l")


class DurationT2S(nn.Module):
    """テキスト → 音声トークン列(長さを予測してから並列に予測する)。

    Args:
        n_text: テキストの語彙数。
        n_codes: 音声トークンの種類数 V。
        dim: 隠れ層の次元 D。
        enc_layers: encoder の層数。
        dec_layers: decoder の層数。
        heads: attention のヘッド数。
        dropout: dropout の確率。
        dur_weight: 長さの損失の重み。
    """

    def __init__(self, n_text: int, n_codes: int, dim: int = 512, enc_layers: int = 4, dec_layers: int = 4,
                 heads: int = 8, dropout: float = 0.1, dur_weight: float = 1.0):
        super().__init__()
        self.dur_weight = dur_weight
        self.text_emb = nn.Embedding(n_text, dim)
        self.encoder = make_transformer(dim, enc_layers, heads, dropout)
        self.duration = DurationPredictor(dim, dropout)
        self.decoder = make_transformer(dim, dec_layers, heads, dropout)
        self.head = nn.Linear(dim, n_codes)

    @property
    def device(self) -> torch.device:
        return self.head.weight.device

    def encode(self, text_ids: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """テキストを encoder に通す。

        Args:
            text_ids: 発話ごとのテキスト ID 列 [L_i] のリスト。

        Returns:
            テキストの隠れ状態 [B, L, D] と、有効な位置が True のマスク [B, L]。
        """
        text = nn.utils.rnn.pad_sequence(text_ids, batch_first=True).to(self.device)            # [B, L](詰め物は 0)
        mask = nn.utils.rnn.pad_sequence([torch.ones(len(t), dtype=torch.bool) for t in text_ids],
                                         batch_first=True).to(self.device)                       # [B, L]
        h = self.encoder(inputs_embeds=self.text_emb(text), attention_mask=mask.long()).last_hidden_state
        return h, mask

    def decode(self, h: torch.Tensor, durations: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """テキストの隠れ状態を長さの数だけ繰り返して並べ(length regulator)、decoder で各位置のトークンを分類する。

        Args:
            h: テキストの隠れ状態 [B, L, D]。
            durations: 発話ごとの各トークンの長さ [L_i] のリスト(合計がその発話の音声トークン数 U_i)。

        Returns:
            各位置の音声トークンの logits [B, U, V] と、有効な位置が True のマスク [B, U]。
        """
        # 発話ごとに、テキストの i 番目の隠れ状態を durations[i] 回繰り返す: [L_i, D] → [U_i, D]
        x = [h[b, : len(d)].repeat_interleave(d.to(self.device), dim=0) for b, d in enumerate(durations)]
        mask = nn.utils.rnn.pad_sequence([torch.ones(len(e), dtype=torch.bool) for e in x],
                                         batch_first=True).to(self.device)                       # [B, U]
        x = nn.utils.rnn.pad_sequence(x, batch_first=True)                                       # [B, U, D]
        y = self.decoder(inputs_embeds=x, attention_mask=mask.long()).last_hidden_state          # [B, U, D]
        return self.head(y), mask

    def forward(self, text_ids: list[torch.Tensor], codes: list[torch.Tensor],
                durations: list[torch.Tensor]) -> tuple[torch.Tensor, dict]:
        """学習の損失: 音声トークンの分類の損失 + dur_weight × 長さの損失(log(長さ + 1) の二乗誤差)。

        Args:
            text_ids: 発話ごとのテキスト ID 列 [L_i] のリスト。
            codes: 発話ごとの正解の音声トークン列 [U_i] のリスト。
            durations: 発話ごとの正解の長さ [L_i] のリスト(jsut-lab から作ったもの)。

        Returns:
            損失と、記録用の値 {"acc": トークンの正解率, "dur_loss": 長さの損失}。
        """
        h, text_mask = self.encode(text_ids)
        dur_gt = nn.utils.rnn.pad_sequence(durations, batch_first=True).to(self.device)        # [B, L]
        dur_loss = F.mse_loss(self.duration(h)[text_mask].float(), torch.log1p(dur_gt[text_mask].float()))
        logits, mask = self.decode(h, durations)                                               # 正解の長さで並べる
        y = nn.utils.rnn.pad_sequence(codes, batch_first=True).to(self.device)                 # [B, U]
        ce = F.cross_entropy(logits[mask].float(), y[mask])
        acc = (logits[mask].argmax(-1) == y[mask]).float().mean()
        return ce + self.dur_weight * dur_loss, {"acc": acc.item(), "dur_loss": dur_loss.item()}

    @torch.no_grad()
    def predict_durations(self, text_ids: list[torch.Tensor], scale: float = 1.0) -> list[torch.Tensor]:
        """各トークンの長さを予測する(整数に丸める。scale で全体の話す速さを変えられる)。

        Args:
            text_ids: 発話ごとのテキスト ID 列 [L_i] のリスト。
            scale: 長さに掛ける倍率。

        Returns:
            発話ごとの長さ [L_i] のリスト(合計が 1 以上になるようにする)。
        """
        h, _ = self.encode(text_ids)
        dur = (torch.expm1(self.duration(h).float()) * scale).round().clamp(min=0).long()      # [B, L]
        out = [dur[b, : len(t)] for b, t in enumerate(text_ids)]
        for d in out:
            if d.sum() == 0:
                d[0] = 1
        return out

    @torch.no_grad()
    def generate(self, text_ids: list[torch.Tensor], durations: list[torch.Tensor] | None = None,
                 sample: bool = False, temperature: float = 1.0, top_k: int = 30) -> list[torch.Tensor]:
        """テキストから音声トークン列を作る(長さを予測し、全位置を一度に予測する)。

        Args:
            text_ids: 発話ごとのテキスト ID 列 [L_i] のリスト。
            durations: 長さを外から与えるとき(分析用に正解の長さを使うなど)。None なら予測する。
            sample: True なら各位置でサンプルする(False なら一番確率の高いトークン)。
            temperature: サンプルの温度。
            top_k: サンプルする候補の数。

        Returns:
            発話ごとの音声トークン列 [U_i] のリスト(CPU)。
        """
        if durations is None:
            durations = self.predict_durations(text_ids)
        h, _ = self.encode(text_ids)
        logits, mask = self.decode(h, durations)                                               # [B, U, V]
        if sample:
            logits = logits.float() / temperature
            logits[logits < logits.topk(top_k, dim=-1).values[..., -1:]] = float("-inf")       # 上位 top_k 個だけ残す
            ids = torch.distributions.Categorical(logits=logits).sample()                     # [B, U]
        else:
            ids = logits.argmax(-1)
        return [ids[b, : int(m.sum())].cpu() for b, m in enumerate(mask)]
