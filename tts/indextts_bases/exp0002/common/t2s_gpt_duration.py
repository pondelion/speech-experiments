"""exp0002 の追加検証: exp0001 の自己回帰の T2S(GPT-2)に、「長さの予測」だけを足したもの。

ハイブリッド型は、exp0001 の自己回帰から「長さの予測」と「T2S の形(自己回帰 → MaskGCT 系)」の 2 つを同時に変えたので、
それぞれの効果を分けるため、2 × 2 の足りない条件として作る:
    T2S の形は exp0001 の自己回帰(TextToSemantic)のまま。大きさ・学習の設定・生成の設定(温度 0.8、top_p 0.8、top_k 30、
    繰り返しのペナルティ 10、KV キャッシュ)も同じ
    足すもの(ハイブリッド型と同じ「長さの予測」): テキストの encoder(4 層)+ 長さの予測器(畳み込み 2 層で log(長さ + 1) を回帰)
    + 各音声トークンの位置に「その時刻の音素」の隠れ状態(encoder の出力を長さの数だけ繰り返したもの)を足す。
    生成するトークンの数は予測した長さの合計(長さで時間の対応を与えるので、終わりの記号では止めない)
入力の位置 j(<BA> が 0)は、次のトークン q_{j+1} を当てるので、q_{j+1} の時刻の音素を足す。

shape の軸の記号: B 発話数、L テキストのトークン数、U 音声トークン数、D 隠れ層の次元
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from exp0001.common.t2s import TextToSemantic, _sample

from .t2s_duration import DurationPredictor, make_transformer


class GPTDurationT2S(TextToSemantic):
    def __init__(self, n_text: int, n_codes: int, enc_layers: int = 4, dur_weight: float = 1.0, **kwargs):
        super().__init__(n_text, n_codes, **kwargs)
        dim = self.text_emb.embedding_dim
        dropout = kwargs.get("dropout", 0.1)
        self.dur_weight = dur_weight
        self.enc_emb = nn.Embedding(n_text, dim)
        self.encoder = make_transformer(dim, enc_layers, kwargs.get("heads", 8), dropout)
        self.duration = DurationPredictor(dim, dropout)
        self.align_proj = nn.Linear(dim, dim)

    def encode(self, text_ids: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """テキストを encoder に通す(ハイブリッド型と同じ): 隠れ状態 [B, L, D] と有効な位置のマスク [B, L]。"""
        text = nn.utils.rnn.pad_sequence(text_ids, batch_first=True).to(self.device)
        mask = nn.utils.rnn.pad_sequence([torch.ones(len(t), dtype=torch.bool) for t in text_ids],
                                         batch_first=True).to(self.device)
        return self.encoder(inputs_embeds=self.enc_emb(text), attention_mask=mask.long()).last_hidden_state, mask

    def align(self, h: torch.Tensor, durations: list[torch.Tensor]) -> list[torch.Tensor]:
        """テキストの i 番目の隠れ状態を durations[i] 回繰り返して並べる: 各 [L_i, D] → [U_i, D](ハイブリッド型と同じ)。"""
        a = self.align_proj(h)
        return [a[b, : len(d)].repeat_interleave(d.to(self.device), dim=0) for b, d in enumerate(durations)]

    @torch.no_grad()
    def predict_durations(self, text_ids: list[torch.Tensor]) -> list[torch.Tensor]:
        """各トークンの長さを予測する(ハイブリッド型の nar と同じ。合計が 1 以上になるようにする)。"""
        h, _ = self.encode(text_ids)
        dur = torch.expm1(self.duration(h).float()).round().clamp(min=0).long()
        out = [dur[b, : len(t)] for b, t in enumerate(text_ids)]
        for d in out:
            if d.sum() == 0:
                d[0] = 1
        return out

    def forward(self, text_ids: list[torch.Tensor], codes: list[torch.Tensor],
                durations: list[torch.Tensor]) -> tuple[torch.Tensor, dict]:
        """学習の損失: exp0001 の自己回帰と同じ次のトークンの交差エントロピー + dur_weight × 長さの損失(正解の長さで並べる)。"""
        h, mask = self.encode(text_ids)
        dur = nn.utils.rnn.pad_sequence(durations, batch_first=True).to(self.device)            # [B, L]
        dur_loss = F.mse_loss(self.duration(h)[mask].float(), torch.log1p(dur[mask].float()))
        hid, y, _ = self._teacher_forced(text_ids, codes, audio_add=self.align(h, durations))
        sel = y != -100
        logits = self.head(hid[sel]).float()
        loss = F.cross_entropy(logits, y[sel])
        return loss + self.dur_weight * dur_loss, {"acc": (logits.argmax(-1) == y[sel]).float().mean().item(),
                                                   "dur_loss": dur_loss.item()}

    @torch.no_grad()
    def generate(self, text_ids: list[torch.Tensor], durations: list[torch.Tensor] | None = None, temperature: float = 0.8,
                 top_p: float = 0.8, top_k: int = 30, repetition_penalty: float = 10.0) -> list[torch.Tensor]:
        """exp0001 の TextToSemantic.generate と同じ手順(左詰めのバッチ、KV キャッシュ、同じサンプリング)で、
        各入力に「次のトークンの時刻の音素」を足し、予測した長さの合計の数だけ作る(<BA>・<EA> は選ばない)。"""
        if durations is None:
            durations = self.predict_durations(text_ids)
        device = self.device
        h_enc, _ = self.encode(text_ids)
        cond = self.align(h_enc, durations)                                                    # 各 [U_i, D]
        lengths = [len(c) for c in cond]
        Um = max(lengths)
        cond = nn.utils.rnn.pad_sequence(cond, batch_first=True)                               # [B, Um, D]
        cond = F.pad(cond, (0, 0, 0, 1))                                                       # 最後の次は 0 [B, Um + 1, D]
        ba = self._audio_embeds(torch.tensor([self.BA], device=device))                        # [1, D]
        prefixes = [torch.cat([self._text_embeds(t.to(device)), ba + cond[b, :1]]) for b, t in enumerate(text_ids)]
        n = max(len(p) for p in prefixes)
        x = torch.stack([F.pad(p, (0, 0, n - len(p), 0)) for p in prefixes])                   # 左詰め [B, n, D]
        attn = torch.stack([F.pad(torch.ones(len(p), device=device), (n - len(p), 0)) for p in prefixes]).long()
        out = self.gpt(inputs_embeds=x, attention_mask=attn, use_cache=True)
        past, h = out.past_key_values, out.last_hidden_state[:, -1]
        generated = torch.zeros(len(text_ids), 0, dtype=torch.long, device=device)
        for i in range(Um):
            logits = self.head(h).float()
            if repetition_penalty != 1.0 and generated.shape[1] > 0:
                prev = logits.gather(1, generated)
                logits.scatter_(1, generated, torch.where(prev > 0, prev / repetition_penalty, prev * repetition_penalty))
            logits = logits / temperature
            logits[:, self.BA] = -float("inf")
            logits[:, self.EA] = -float("inf")  # 長さは予測した長さで決まるので、終わりの記号は選ばない
            nxt = _sample(logits, top_p, top_k)                                                # [B]
            generated = torch.cat([generated, rearrange(nxt, "b -> b 1")], dim=1)
            if i == Um - 1:
                break
            attn = torch.cat([attn, torch.ones_like(attn[:, :1])], dim=1)
            emb = self._audio_embeds(rearrange(nxt, "b -> b 1"), start=i + 1) + cond[:, i + 1: i + 2]
            out = self.gpt(inputs_embeds=emb, attention_mask=attn, past_key_values=past, use_cache=True)
            past, h = out.past_key_values, out.last_hidden_state[:, -1]
        return [generated[b, :n].cpu() for b, n in enumerate(lengths)]
