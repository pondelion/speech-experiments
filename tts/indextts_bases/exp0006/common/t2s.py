"""T2S: テキスト → 音声トークン(IndexTTS-2.5 の `indextts/gpt/model_v2.py` の UnifiedVoice と同じ構成の、自己回帰の GPT-2)。

公式の構成(この exp では層の数・次元だけ小さくする。話者・感情の条件はフェーズ A では使わないので入れない):
    入力 = [テキストの埋め込み + テキストの位置の埋め込み][音声トークンの埋め込み + 音声トークンの位置の埋め込み]
           テキスト = 始まり(0)・テキスト・終わり(1)、音声 = 始まり(8192)・トークン・終わり(8193)
    GPT-2(transformers の GPT2Model。GPT-2 自身の位置の埋め込みは使わず、テキストと音声で別々に学習する位置の埋め込みを足す)
    → LayerNorm → テキストの出力層・音声の出力層
    学習: 次のトークンを当てる交差エントロピー。バッチの詰め物は終わりの記号で埋め、そこも当てる(公式の set_*_padding)。
          テキストの損失の重みは 0.01、音声の損失は 1(公式の学習コードは非公開。同じ UnifiedVoice の元の tortoise の学習の設定)
    生成: transformers の generate で、サンプリング(温度 0.8、top_p 0.8、top_k 30)+ ビームサーチ 3 + 繰り返しのペナルティ 10
          (公式の推論の既定)。最初の入力の ID は、テキストの部分を 1 で埋めた仮の ID + 音声の始まり(公式と同じ。繰り返しの
          ペナルティもこの仮の ID に掛かる)。生成した音声トークンの位置の埋め込みの番号は、公式の推論と同じく
          「attention mask の長さ - テキストの部分の長さ」(学習のときの番号より 1 大きい)

shape の軸の記号: B 発話数、L テキストの ID 数(始まり・終わりを含む)、U 音声トークン数(始まり・終わりを含む)、H 次元
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, GPT2Config, GPT2Model, GPT2PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

N_CODES = 8192
START_MEL, STOP_MEL = N_CODES, N_CODES + 1
START_TEXT, STOP_TEXT = 0, 1


class _NoPosition(nn.Module):
    """GPT-2 自身の位置の埋め込みを 0 にする(公式は wpe を null_position_embeddings に差し替える)。"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(*position_ids.shape, self.dim, device=position_ids.device)


class LearnedPosition(nn.Module):
    def __init__(self, n: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(n, dim)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, length: int, device) -> torch.Tensor:
        """位置 0〜length-1 の埋め込み [length, H]。"""
        return self.emb(torch.arange(length, device=device))


def _gpt(dim: int, layers: int, heads: int, n_positions: int) -> GPT2Model:
    gpt = GPT2Model(GPT2Config(vocab_size=256, n_positions=n_positions, n_embd=dim, n_layer=layers, n_head=heads))
    gpt.wpe = _NoPosition(dim)
    del gpt.wte  # 埋め込みは外で作って inputs_embeds で渡す
    return gpt


class UnifiedVoice(nn.Module):
    def __init__(self, n_text: int, dim: int = 1280, layers: int = 24, heads: int = 20, max_text: int = 600,
                 max_mel: int = 1815, text_weight: float = 0.01):
        super().__init__()
        self.max_mel, self.text_weight = max_mel, text_weight
        self.gpt = _gpt(dim, layers, heads, max_mel + max_text + 2)
        self.text_embedding = nn.Embedding(n_text, dim)
        self.mel_embedding = nn.Embedding(N_CODES + 2, dim)
        for e in (self.text_embedding, self.mel_embedding):  # GPT-2 の初期化(公式と同じ)
            nn.init.normal_(e.weight, std=0.02)
        self.text_pos = LearnedPosition(max_text + 2, dim)
        self.mel_pos = LearnedPosition(max_mel + 3, dim)
        self.final_norm = nn.LayerNorm(dim)
        self.text_head = nn.Linear(dim, n_text)
        self.mel_head = nn.Linear(dim, N_CODES + 2)

    def forward(self, text: list[torch.Tensor], codes: list[torch.Tensor]) -> tuple[torch.Tensor, dict]:
        """学習の損失。text: 各発話のテキストの ID(CharVocab.encode の出力。終わりの 1 を含む)、codes: 各発話の音声トークン。"""
        dev = self.text_embedding.weight.device
        t = F.pad(nn.utils.rnn.pad_sequence(text, batch_first=True, padding_value=STOP_TEXT), (0, 1), value=STOP_TEXT).to(dev)
        m = F.pad(nn.utils.rnn.pad_sequence(codes, batch_first=True, padding_value=STOP_MEL), (0, 1), value=STOP_MEL).to(dev)
        t_in, t_tar = F.pad(t, (1, 0), value=START_TEXT), F.pad(t, (0, 1), value=STOP_TEXT)    # [B, L+1]
        m_in, m_tar = F.pad(m, (1, 0), value=START_MEL), F.pad(m, (0, 1), value=STOP_MEL)      # [B, U+1]
        t_emb = self.text_embedding(t_in) + self.text_pos(t_in.shape[1], dev)
        m_emb = self.mel_embedding(m_in) + self.mel_pos(m_in.shape[1], dev)
        h = self.final_norm(self.gpt(inputs_embeds=torch.cat([t_emb, m_emb], dim=1)).last_hidden_state)
        text_logits = self.text_head(h[:, : t_in.shape[1]])                                   # [B, L+1, V_text]
        mel_logits = self.mel_head(h[:, t_in.shape[1]:])                                      # [B, U+1, V_mel]
        loss_text = F.cross_entropy(text_logits.transpose(1, 2), t_tar)
        loss_mel = F.cross_entropy(mel_logits.transpose(1, 2), m_tar)
        acc = (mel_logits.argmax(-1) == m_tar).float().mean()
        return self.text_weight * loss_text + loss_mel, {"text_ce": loss_text.item(), "mel_ce": loss_mel.item(), "acc": acc.item()}

    @torch.no_grad()
    def sample(self, text: torch.Tensor, max_tokens: int = 1500, **kwargs) -> list[tuple[torch.Tensor, bool]]:
        """1 文のテキストの ID [L](終わりの 1 を含む)→ num_return_sequences 個の (音声トークン [U](終わりの記号の手前まで),
        終わりの記号で止まったか)。kwargs は transformers の generate の引数。既定は公式の推論の値(下の GENERATE_DEFAULTS)。"""
        dev = self.text_embedding.weight.device
        t = F.pad(text.to(dev)[None], (1, 0), value=START_TEXT)                              # [1, L+1]
        prefix = self.text_embedding(t) + self.text_pos(t.shape[1], dev)                      # [1, P, H]
        model = _InferenceModel(self, prefix)
        fake = torch.ones(1, prefix.shape[1] + 1, dtype=torch.long, device=dev)
        fake[:, -1] = START_MEL
        out = model.generate(fake, attention_mask=torch.ones_like(fake), bos_token_id=START_MEL, pad_token_id=STOP_MEL,
                             eos_token_id=STOP_MEL, max_length=fake.shape[1] + max_tokens, **{**GENERATE_DEFAULTS, **kwargs})
        res = []
        for codes in out[:, fake.shape[1]:]:
            stop = (codes == STOP_MEL).nonzero()
            res.append((codes[: stop[0, 0]], True) if len(stop) else (codes, False))
        return res

    def generate(self, text: torch.Tensor, max_tokens: int = 1500, **kwargs) -> torch.Tensor:
        """1 文のテキストの ID [L] → 音声トークン [U](公式の推論と同じく、1 つだけ作る)。"""
        return self.sample(text, max_tokens, **kwargs)[0][0]

    def token_logps(self, text: list[torch.Tensor], codes: list[torch.Tensor], ended: list[bool],
                    temperature: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """各発話の音声トークン(と止まった発話は終わりの記号)の対数確率 [B, U+1] と、数える位置の mask [B, U+1](GRPO 用)。

        logits は温度で割ってから log_softmax する(TRL の GRPO と同じ)。位置の埋め込みは学習のときと同じ番号。
        """
        dev = self.text_embedding.weight.device
        t = F.pad(nn.utils.rnn.pad_sequence(text, batch_first=True, padding_value=STOP_TEXT), (0, 1), value=STOP_TEXT).to(dev)
        m = F.pad(nn.utils.rnn.pad_sequence(codes, batch_first=True, padding_value=STOP_MEL), (0, 1), value=STOP_MEL).to(dev)
        t_in = F.pad(t, (1, 0), value=START_TEXT)
        m_in, m_tar = F.pad(m, (1, 0), value=START_MEL), F.pad(m, (0, 1), value=STOP_MEL)      # [B, U+2]
        t_emb = self.text_embedding(t_in) + self.text_pos(t_in.shape[1], dev)
        m_emb = self.mel_embedding(m_in) + self.mel_pos(m_in.shape[1], dev)
        h = self.final_norm(self.gpt(inputs_embeds=torch.cat([t_emb, m_emb], dim=1)).last_hidden_state)
        logits = self.mel_head(h[:, t_in.shape[1]:]).float() / temperature                 # [B, U+2, V]
        logp = torch.gather(logits.log_softmax(-1), 2, m_tar[..., None])[..., 0]            # [B, U+2]
        n = torch.tensor([len(c) + int(e) for c, e in zip(codes, ended)], device=dev)
        mask = torch.arange(logp.shape[1], device=dev)[None] < n[:, None]
        return logp, mask


GENERATE_DEFAULTS = {"do_sample": True, "top_p": 0.8, "top_k": 30, "temperature": 0.8, "num_beams": 3,
                     "repetition_penalty": 10.0, "length_penalty": 0.0, "num_return_sequences": 1}


class _InferenceModel(GPT2PreTrainedModel, GenerationMixin):
    """transformers の generate に渡すための包み(公式の GPT2InferenceModel にあたる)。テキストの部分の埋め込みを持っておく。"""

    def __init__(self, uv: UnifiedVoice, prefix: torch.Tensor):
        config = copy.deepcopy(uv.gpt.config)
        config.vocab_size = N_CODES + 2  # generate(ビームサーチ)は語彙の大きさをここから読む
        super().__init__(config)
        self.uv, self.prefix = uv, prefix

    def forward(self, input_ids=None, past_key_values=None, attention_mask=None, use_cache=None, **kwargs):
        P = self.prefix.shape[1]
        if input_ids.shape[1] != 1:  # 最初: [テキストの部分][音声の始まり](位置 0)
            mel = input_ids[:, P:]
            emb = self.uv.mel_embedding(mel) + self.uv.mel_pos(mel.shape[1], mel.device)
            emb = torch.cat([self.prefix.expand(emb.shape[0], -1, -1), emb], dim=1)
        else:  # 2 回目から: 直前に生成したトークン。位置の番号は公式と同じく attention mask の長さ - P
            emb = self.uv.mel_embedding(input_ids) + self.uv.mel_pos.emb.weight[attention_mask.shape[1] - P][None, None]
        out = self.uv.gpt(inputs_embeds=emb, past_key_values=past_key_values, attention_mask=attention_mask,
                          use_cache=use_cache, return_dict=True)
        logits = self.uv.mel_head(self.uv.final_norm(out.last_hidden_state))
        return CausalLMOutputWithPast(logits=logits, past_key_values=out.past_key_values)
