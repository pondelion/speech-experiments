"""T2S(Text-to-Semantic): テキストから音声トークン列を作る GPT。IndexTTS2.5 の T2S に当たる部分。

仕組み:
    言語モデル(ChatGPT など)と同じ「これまでの系列から、次のトークンを当てる」モデル。
    入力の系列は次の形で、前半のテキストを読んだうえで、後半の音声トークンを 1 個ずつ予測する。

        [テキスト] <bos> t1 t2 ... tL <eos>  [音声] <BA> q1 q2 ... qU <EA>

    <BA>/<EA> は音声の開始・終了を表す特別なトークン。<EA> を出したら読み上げ終わり、なので
    音声の長さ(何秒話すか)もモデルが自分で決める。

    学習: 正解の音声トークン列を入力し、各位置で「次のトークン」を当てる(teacher forcing)。
    推論: <BA> から始めて、予測からトークンを 1 個選び、それを入力に足して次を予測…を繰り返す。

IndexTTS2.5 と同じ点:
    - 本体は GPT-2(transformers の GPT2Model をランダム初期化)
    - GPT-2 組み込みの位置埋め込みは使わず(0 にする)、テキスト用と音声用の位置埋め込みを別々に足す
      (テキストの何文字目か / 音声の何トークン目かを、別々に数えるため)
    - 損失は音声トークンの予測だけ(テキスト部分の予測は学習しない)
IndexTTS2.5 と違う点:
    - 話者・感情の条件ベクトルは入れない(今回は 1 話者の読み上げだけなので)

exp0001 で比べる学習の仕方(objective などの引数で切り替える。既定値は IndexTTS2.5 と同じ分類の損失):
    "ce":         次のトークンを K + 2 クラスの分類として当てる(クロスエントロピー)
    "soft_ce":    正解だけを 1 にする代わりに、代表ベクトルが正解に近いトークンにも確率を配った目標で学習する
                  (正解とほぼ同じ音のトークンを選んでも、完全な外れとして罰しない)
    "regression": トークンの分類ではなく、正解のトークンの代表ベクトル(codec の 8 次元)そのものを予測する
                  (cosine 距離)。推論では、予測したベクトルに一番近い代表ベクトルのトークンに変換する。
                  終わり(<EA>)は分類できないので、「ここで終わるか」を当てる頭を別に持つ
    aux_weight > 0: 音声側の各位置で「今どの音素か」も当てる補助の損失を足す
                  (テキストのどこを読んでいるかを、モデルの中に持たせる)
    from_fine / to_fine: codec の 1024 種類のトークンを、k-means でまとめた粗いトークンに置き換えるときの対応表

shape の軸の記号:
    B: バッチ内の発話数
    L: テキストのトークン数(<bos>・<eos> 込み)
    U: 音声トークン数(<BA>・<EA> は含まない)
    S: GPT に入れる系列の長さ(1 発話なら L + 1 + U。バッチでは一番長い発話に合わせて詰める)
    D: 隠れ層の次元(dim)
    K: 音声トークンの種類数(n_codes)、V: 音声側の語彙数(K + <BA> + <EA> = K + 2)
    C: codec の代表ベクトルの次元(8)
    N: 損失をかける位置の数(バッチ内の全発話の、音声部分の位置の合計)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import GPT2Config, GPT2Model

OBJECTIVES = ["ce", "soft_ce", "regression"]


def _null_position_embeddings(position_ids: torch.Tensor, dim: int) -> torch.Tensor:
    """GPT-2 組み込みの位置埋め込みの代わりに 0 を返す(IndexTTS 公式と同じ方法)。

    Args:
        position_ids: [B(発話数), S(系列の長さ)]。
        dim: 埋め込みの次元 D。

    Returns:
        0 のテンソル [B, S, D]。
    """
    return torch.zeros(position_ids.shape[0], position_ids.shape[1], dim, device=position_ids.device)


class TextToSemantic(nn.Module):
    """テキスト ID 列 → 音声トークン列の GPT。

    Args:
        n_text_vocab: テキストの語彙数(特殊トークンを含む)。
        n_codes: 音声トークンの種類数 K(codec の codebook_size、粗いトークンならそのグループ数)。
        dim: 隠れ層の次元 D。
        layers: Transformer の層数。
        heads: attention のヘッド数。
        max_text: テキストの最大長(位置埋め込みの数)。
        max_audio: 音声トークンの最大長(位置埋め込みの数)。
        dropout: dropout 率。
        objective: 学習の仕方("ce" / "soft_ce" / "regression")。
        code_vecs: 各音声トークンの代表ベクトル(L2 正規化)[K(トークンの種類数), C(8)]。"soft_ce"・"regression" で使う。
        soft_tau: "soft_ce" で確率を配るときの温度(小さいほど正解の近くだけに配る)。
        n_phonemes: 補助の損失で当てる音素の種類数(0 なら補助の損失なし)。
        aux_weight: 補助の損失の重み。
        from_fine: codec のトークン → このモデルのトークンの対応表 [1024](粗いトークンを使うとき。None なら同じ)。
        to_fine: このモデルのトークン → S2M に渡す codec のトークンの対応表 [K](粗いトークンを使うとき)。
    """

    def __init__(self, n_text_vocab: int, n_codes: int, dim: int = 512, layers: int = 8, heads: int = 8,
                 max_text: int = 512, max_audio: int = 1024, dropout: float = 0.1, objective: str = "ce",
                 code_vecs: torch.Tensor | None = None, soft_tau: float = 0.1, n_phonemes: int = 0,
                 aux_weight: float = 0.0, from_fine: torch.Tensor | None = None, to_fine: torch.Tensor | None = None):
        super().__init__()
        assert objective in OBJECTIVES
        assert objective == "ce" or code_vecs is not None, f"{objective} には code_vecs が必要"
        self.n_codes, self.objective, self.soft_tau, self.aux_weight = n_codes, objective, soft_tau, aux_weight
        self.BA, self.EA = n_codes, n_codes + 1  # 音声トークンの語彙: 0..K-1 + <BA> + <EA>
        self.text_emb = nn.Embedding(n_text_vocab, dim)
        self.audio_emb = nn.Embedding(n_codes + 2, dim)
        self.text_pos = nn.Embedding(max_text, dim)
        self.audio_pos = nn.Embedding(max_audio, dim)
        config = GPT2Config(n_embd=dim, n_layer=layers, n_head=heads, n_positions=max_text + max_audio,
                            vocab_size=1, bos_token_id=None, eos_token_id=None,
                            resid_pdrop=dropout, embd_pdrop=dropout, attn_pdrop=dropout)
        self.gpt = GPT2Model(config)
        del self.gpt.wpe
        self.gpt.wpe = lambda position_ids: _null_position_embeddings(position_ids, dim)
        if objective == "regression":
            self.reg_head = nn.Linear(dim, code_vecs.shape[1])  # 代表ベクトルの予測 [.., D] → [.., C]
            self.stop_head = nn.Linear(dim, 1)                  # ここで終わるか(<EA> か)[.., D] → [.., 1]
        else:
            self.head = nn.Linear(dim, n_codes + 2)             # 次の音声トークン(<EA> を含む)の分類 [.., D] → [.., V]
        self.phone_head = nn.Linear(dim, n_phonemes) if n_phonemes > 0 else None
        self.register_buffer("code_vecs", None if code_vecs is None else F.normalize(code_vecs.float(), dim=-1))
        self.register_buffer("from_fine", from_fine)
        self.register_buffer("to_fine", to_fine)

    @property
    def device(self) -> torch.device:
        return self.audio_emb.weight.device

    def _text_embeds(self, text_ids: torch.Tensor) -> torch.Tensor:
        """テキスト ID → 埋め込み + テキスト用の位置埋め込み。

        Args:
            text_ids: [L(テキストのトークン数)] または [B(発話数), L]。最後の軸が系列。

        Returns:
            [L, D(隠れ層の次元)] または [B, L, D]。
        """
        pos = torch.arange(text_ids.shape[-1], device=text_ids.device)  # [L]
        return self.text_emb(text_ids) + self.text_pos(pos)             # [.., L, D] + [L, D](先頭の軸方向に放送)

    def _audio_embeds(self, audio_ids: torch.Tensor, start: int = 0) -> torch.Tensor:
        """音声トークン → 埋め込み + 音声用の位置埋め込み(start 番目から数える)。

        Args:
            audio_ids: [U'(入れる音声トークン数)] または [B(発話数), U']。最後の軸が系列。
            start: 最初のトークンの位置番号。

        Returns:
            [U', D(隠れ層の次元)] または [B, U', D]。
        """
        pos = torch.arange(start, start + audio_ids.shape[-1], device=audio_ids.device)  # [U']
        return self.audio_emb(audio_ids) + self.audio_pos(pos)                           # [.., U', D]

    def _teacher_forced(self, text_ids: list[torch.Tensor], codes: list[torch.Tensor],
                        phonemes: list[torch.Tensor] | None = None):
        """正解の音声トークン列を入力にして(teacher forcing)GPT を通し、各位置の隠れ状態と正解を返す。

        発話ごとに [テキスト][<BA> 音声トークン] をつなげて 1 本の系列にし、右側を詰め物で埋めてバッチにする。
        発話ごとに Python のループで組み立てると、小さな GPU 演算が大量に出て遅い(1 step の大半を占めた)ので、
        テキスト部分・音声部分をそれぞれバッチでまとめて埋め込み、発話ごとの「つなぎ目」は番号の対応表(gather)で作る。

        Args:
            text_ids: 発話ごとのテキスト ID 列 [L_i(i 番目の発話のテキストのトークン数。<bos>・<eos> 込み)] のリスト。
            codes: 発話ごとの正解の音声トークン列 [U_i(i 番目の発話の音声トークン数)] のリスト。
            phonemes: 発話ごとの、各音声トークンの音素 ID 列 [U_i](補助の損失を使うときだけ)。

        Returns:
            隠れ状態 h [B, S, D]、正解の次トークン y [B, S](損失をかけない位置は -100)、
            正解の音素 ph [B, S](phonemes が None なら None。損失をかけない位置は -100)。
            (以下のコメントの記号: Lm テキストの最大長、Am 音声側の最大長(= 最大の U + 1))
        """
        device = self.device
        pad = nn.utils.rnn.pad_sequence
        text_len = torch.tensor([len(t) for t in text_ids], device=device)                    # [B]
        audio_len = torch.tensor([len(c) + 1 for c in codes], device=device)                  # <BA>(または <EA>)込み [B]
        text = pad(text_ids, batch_first=True).to(device)                                     # [B, Lm](詰め物は ID 0)
        audio_in = pad([torch.cat([c.new_tensor([self.BA]), c]) for c in codes], batch_first=True).to(device)  # <BA> q1 ... qU [B, Am]
        audio_gt = pad([torch.cat([c, c.new_tensor([self.EA])]) for c in codes], batch_first=True,
                       padding_value=-100).to(device)                                         # q1 ... qU <EA>(1 個先が正解)[B, Am]
        # テキスト部分と音声部分を、それぞれまとめて埋め込んで横に並べる: [B, Lm, D] + [B, Am, D] → [B, Lm + Am, D]
        src = torch.cat([self._text_embeds(text), self._audio_embeds(audio_in)], dim=1)
        Lm = text.shape[1]
        S = int((text_len + audio_len).max())
        pos = rearrange(torch.arange(S, device=device), "s -> 1 s")                           # 系列内の位置 [1, S]
        tl = rearrange(text_len, "b -> b 1")                                                  # [B, 1]
        in_text = pos < tl                                                                    # テキスト部分の位置 [B, S]
        valid = pos < tl + rearrange(audio_len, "b -> b 1")                                   # 詰め物以外の位置 [B, S]
        # つないだ系列の位置 s が src のどこから来るか: テキスト部分は s 番目、音声部分は Lm + (s - L_i) 番目
        audio_pos = (pos - tl).clamp(0, audio_in.shape[1] - 1)                                # 音声部分の中での位置 [B, S]
        src_idx = torch.where(in_text, pos, Lm + audio_pos)                                   # [B, S]
        x = torch.gather(src, 1, rearrange(src_idx, "b s -> b s 1").expand(-1, -1, src.shape[-1]))  # [B, S, D]
        # 正解: 音声部分だけ audio_gt から取り、テキスト部分と詰め物は損失に入れない(-100)。[B, S]
        is_audio = ~in_text & valid
        y = torch.where(is_audio, torch.gather(audio_gt, 1, audio_pos), torch.full_like(audio_pos, -100))
        ph = None
        if phonemes is not None:  # 次に来るトークンの音素(<EA> の位置は損失に入れない)
            ph_gt = pad([torch.cat([p, p.new_tensor([-100])]) for p in phonemes], batch_first=True,
                        padding_value=-100).to(device)                                        # [B, Am]
            ph = torch.where(is_audio, torch.gather(ph_gt, 1, audio_pos), torch.full_like(audio_pos, -100))
        h = self.gpt(inputs_embeds=x, attention_mask=valid.long()).last_hidden_state          # [B, S, D]
        return h, y, ph

    def _nearest_code(self, vec: torch.Tensor) -> torch.Tensor:
        """ベクトルを、一番近い(cosine 類似度が一番高い)代表ベクトルのトークン ID にする。

        Args:
            vec: [..., C(代表ベクトルの次元)]。

        Returns:
            トークン ID [...]。
        """
        return torch.einsum("...c,kc->...k", F.normalize(vec.float(), dim=-1), self.code_vecs).argmax(-1)

    def forward(self, text_ids: list[torch.Tensor], codes: list[torch.Tensor],
                phonemes: list[torch.Tensor] | None = None) -> tuple[torch.Tensor, dict]:
        """学習用の順伝播と損失の計算。

        Args:
            text_ids: 発話ごとのテキスト ID 列 [L_i(i 番目の発話のテキストのトークン数)] のリスト。
            codes: 発話ごとの正解の音声トークン列 [U_i(i 番目の発話の音声トークン数)] のリスト。
            phonemes: 発話ごとの、各音声トークンの音素 ID 列 [U_i](補助の損失を使うときだけ)。

        Returns:
            損失と、正解率などの dict("acc": 次のトークンの正解率(regression は一番近いトークンで判定)、
            "aux_acc": 補助の損失を使うときの音素の正解率)。
        """
        h, y, ph = self._teacher_forced(text_ids, codes, phonemes if self.phone_head is not None else None)
        sel = y != -100                                                                       # 損失をかける位置 [B, S]
        h_sel, y_sel = h[sel], y[sel]                                                         # [N, D]、[N]
        is_code = y_sel < self.n_codes                                                        # <EA> 以外の位置 [N]
        if self.objective == "regression":
            vec = self.reg_head(h_sel).float()                                                # [N, C]
            target = self.code_vecs[y_sel[is_code]]                                           # 正解の代表ベクトル [Nc, C]
            reg = (1 - F.cosine_similarity(vec[is_code], target, dim=-1)).mean()              # cosine 距離
            stop = F.binary_cross_entropy_with_logits(self.stop_head(h_sel)[:, 0].float(), (~is_code).float())
            loss = reg + stop
            acc = (self._nearest_code(vec[is_code]) == y_sel[is_code]).float().mean()
        else:
            logits = self.head(h_sel).float()                                                 # [N, V]
            if self.objective == "ce":
                loss = F.cross_entropy(logits, y_sel)
            else:  # soft_ce: 代表ベクトルが正解に近いトークンにも確率を配る
                q = F.one_hot(y_sel, logits.shape[-1]).float()                                # <EA> の位置はこのまま [N, V]
                sim = self.code_vecs[y_sel[is_code]] @ self.code_vecs.T                       # 正解と各トークンの類似度 [Nc, K]
                q[is_code] = F.pad((sim / self.soft_tau).softmax(-1), (0, 2))                 # <BA>・<EA> の列は 0 [Nc, V]
                loss = -(q * logits.log_softmax(-1)).sum(-1).mean()
            acc = (logits.argmax(-1) == y_sel).float().mean()
        parts = {"acc": acc.item()}
        if self.phone_head is not None:
            ph_sel = ph[sel]                                                                  # [N]
            ok = ph_sel != -100
            ph_logits = self.phone_head(h_sel[ok]).float()                                    # [Np, 音素の種類数]
            loss = loss + self.aux_weight * F.cross_entropy(ph_logits, ph_sel[ok])
            parts["aux_acc"] = (ph_logits.argmax(-1) == ph_sel[ok]).float().mean().item()
        return loss, parts

    @torch.no_grad()
    def text_hidden(self, text_ids: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """テキスト部分の隠れ状態(S2M の条件に加える用)。

        GPT は前の位置しか見ない(causal)ので、テキストだけを入れたときのテキスト部分の隠れ状態は、
        後ろに音声トークンが続く学習時・推論時と同じ値になる。

        Args:
            text_ids: 発話ごとのテキスト ID 列 [L_i(i 番目の発話のテキストのトークン数)] のリスト。

        Returns:
            隠れ状態 [B(発話数), Lm(テキストの最大長), D(隠れ層の次元)] と、詰め物の位置が True のマスク [B, Lm]。
        """
        text = nn.utils.rnn.pad_sequence(text_ids, batch_first=True).to(self.device)      # [B, Lm]
        lens = torch.tensor([len(t) for t in text_ids], device=self.device)                # [B]
        pad = rearrange(torch.arange(text.shape[1], device=self.device), "l -> 1 l") >= rearrange(lens, "b -> b 1")
        h = self.gpt(inputs_embeds=self._text_embeds(text), attention_mask=(~pad).long()).last_hidden_state
        return h, pad

    @torch.no_grad()
    def predict_teacher_forced(self, text_ids: list[torch.Tensor], codes: list[torch.Tensor], sample: bool = False,
                               temperature: float = 0.8, top_p: float = 0.8, top_k: int = 30) -> list[torch.Tensor]:
        """分析用: 正解のトークン列を入力にして、各位置で予測したトークン列を返す(正解と同じ長さ)。

        Args:
            text_ids: 発話ごとのテキスト ID 列 [L_i] のリスト。
            codes: 発話ごとの正解の音声トークン列 [U_i(このモデルのトークン)] のリスト。
            sample: True なら推論と同じ設定でサンプルする(regression は常に一番近いトークン)。
            temperature: サンプリングの温度。
            top_p: nucleus sampling のしきい値。
            top_k: 上位 k 個に絞る。

        Returns:
            発話ごとの予測トークン列 [U_i](CPU)のリスト。
        """
        h, y, _ = self._teacher_forced(text_ids, codes)
        out = []
        for b, c in enumerate(codes):
            pos = (y[b] != -100).nonzero()[:, 0][: len(c)]                                    # q1 ... qU を予測する位置 [U]
            if self.objective == "regression":
                pred = self._nearest_code(self.reg_head(h[b, pos]))
            else:
                logits = self.head(h[b, pos]).float()[:, : self.n_codes]                     # <BA>・<EA> は除く [U, K]
                pred = _sample(logits / temperature, top_p, top_k) if sample else logits.argmax(-1)
            out.append(pred.cpu())
        return out

    @torch.no_grad()
    def generate(self, text_ids: list[torch.Tensor], max_len: int = 600, temperature: float = 0.8,
                 top_p: float = 0.8, top_k: int = 30, repetition_penalty: float = 10.0) -> list[torch.Tensor]:
        """テキストから音声トークン列を 1 個ずつ作る(自己回帰生成)。

        既定値は IndexTTS2 公式推論(infer_v2.py)の既定値に合わせている。ただし公式はビームサーチ
        (num_beams=3)も併用するが、ここでは単純なサンプリングだけにしている。
        "regression" はサンプリングせず、予測したベクトルに一番近いトークンを選び、終わりの頭が 0.5 を超えたら止める。
        バッチで生成するため、テキスト部分は左側を詰め物で埋める(<BA> の位置を全発話でそろえる)。
        過去の計算結果(KV キャッシュ)を使い回すので、1 ステップの計算は新しい 1 トークン分だけで済む。

        Args:
            text_ids: 発話ごとのテキスト ID 列 [L_i(i 番目の発話のテキストのトークン数。<bos>・<eos> 込み)] のリスト。
            max_len: 生成する音声トークンの上限(<EA> が出なかったときの打ち切り)。
            temperature: サンプリングの温度(小さいほど確率の高いトークンに偏る)。
            top_p: 確率の高い順に、累積確率が top_p になるまでの候補からだけ選ぶ(nucleus sampling)。
            top_k: 0 より大きければ、確率の高い上位 top_k 個からだけ選ぶ。
            repetition_penalty: すでに生成したトークンのスコアを下げる強さ(1.0 なら何もしない)。
                自己回帰生成で同じ音のトークンを延々と繰り返す失敗を抑える。

        Returns:
            発話ごとの音声トークン列 [U_i(i 番目の発話で生成したトークン数。<EA> は含まない)] のリスト。
            (以下のコメントの記号: B 発話数、n テキスト + <BA> の長さの最大、D 隠れ層の次元、V 音声側の語彙数、
            i それまでに生成したトークン数)
        """
        device = self.device
        # 発話ごとの先頭部分: テキスト + <BA> → [L_i + 1, D]
        ba = self._audio_embeds(torch.tensor([self.BA], device=device))                # [1, D]
        prefixes = [torch.cat([self._text_embeds(t.to(device)), ba]) for t in text_ids]
        n = max(len(p) for p in prefixes)
        x = torch.stack([F.pad(p, (0, 0, n - len(p), 0)) for p in prefixes])          # 左詰めでそろえる [B, n, D]
        attn = torch.stack([F.pad(torch.ones(len(p), device=device), (n - len(p), 0)) for p in prefixes]).long()  # [B, n]
        out = self.gpt(inputs_embeds=x, attention_mask=attn, use_cache=True)
        past, h = out.past_key_values, out.last_hidden_state[:, -1]                   # 最後の位置(<BA>)の出力 [B, D]
        generated = torch.zeros(len(text_ids), 0, dtype=torch.long, device=device)    # 生成済み [B, i]
        done = torch.zeros(len(text_ids), dtype=torch.bool, device=device)            # <EA> を出した発話 [B]
        for i in range(max_len):
            if self.objective == "regression":
                nxt = self._nearest_code(self.reg_head(h))                            # [B, D] → [B, C] → [B]
                stop = torch.sigmoid(self.stop_head(h)[:, 0]) > 0.5                   # [B]
                nxt = torch.where(stop, torch.full_like(nxt, self.EA), nxt)
            else:
                logits = self.head(h).float()                                         # [B, D] → [B, V]
                if repetition_penalty != 1.0 and generated.shape[1] > 0:
                    # HF transformers と同じ方式: 生成済みトークンのスコアが正なら割り、負なら掛けて下げる
                    prev = logits.gather(1, generated)                                # [B, i]
                    logits.scatter_(1, generated, torch.where(prev > 0, prev / repetition_penalty,
                                                              prev * repetition_penalty))
                logits = logits / temperature
                logits[:, self.BA] = -float("inf")  # <BA> は生成しない
                nxt = _sample(logits, top_p, top_k)                                   # [B]
            nxt = torch.where(done, torch.full_like(nxt, self.EA), nxt)               # 終わった発話は <EA> を出し続ける
            nxt_col = rearrange(nxt, "b -> b 1")                                      # [B] → [B, 1]
            generated = torch.cat([generated, nxt_col], dim=1)                        # [B, i] → [B, i+1]
            done |= nxt == self.EA
            if done.all():
                break
            attn = torch.cat([attn, torch.ones_like(attn[:, :1])], dim=1)             # [B, n+i] → [B, n+i+1]
            # 新しい 1 トークンだけを入れる [B, 1, D](過去の分は KV キャッシュ past にある)
            out = self.gpt(inputs_embeds=self._audio_embeds(nxt_col, start=i + 1), attention_mask=attn,
                           past_key_values=past, use_cache=True)
            past, h = out.past_key_values, out.last_hidden_state[:, -1]               # [B, D]
        result = []
        for g in generated:                                                           # g: [生成した長さ]
            end = (g == self.EA).nonzero()
            result.append(g[: end[0, 0]] if len(end) else g)                          # <EA> の手前まで [U_i]
        return result


def _sample(logits: torch.Tensor, top_p: float, top_k: int) -> torch.Tensor:
    """logits から 1 個サンプルする(top-k / top-p で候補を絞る)。

    Args:
        logits: 各トークンのスコア [B(発話数), V(語彙数)]。
        top_p: nucleus sampling の累積確率のしきい値(1.0 なら絞らない)。
        top_k: 上位 k 個に絞る(0 なら絞らない)。

    Returns:
        選んだ ID [B]。
    """
    if top_k > 0:
        kth = logits.topk(top_k, dim=-1).values[:, -1:]                              # k 番目に大きい値 [B, 1]
        logits = logits.masked_fill(logits < kth, -float("inf"))
    if top_p < 1.0:
        sorted_logits, idx = logits.sort(dim=-1, descending=True)                    # [B, V]
        probs = sorted_logits.softmax(-1)
        remove = probs.cumsum(-1) - probs > top_p  # 自分より前までの累積が top_p を超えたら除く [B, V]
        sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
        logits = torch.full_like(logits, -float("inf")).scatter(-1, idx, sorted_logits)  # 元の並びに戻す [B, V]
    return rearrange(torch.multinomial(logits.softmax(-1), 1), "b 1 -> b")           # [B, 1] → [B]
