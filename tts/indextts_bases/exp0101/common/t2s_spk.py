"""話者の条件を入れた T2S: exp0005 のハイブリッド型(F0 の補助の損失つき)に、話者ベクトル(CAM++、192 次元)を足す。

公式(IndexTTS-2.5)は話者ベクトルの射影を T2S の先頭に置く(全体にかかる条件)。ハイブリッド型には先頭にあたる場所が無いので、
全体にかかる条件として 2 か所に足す(どちらも Linear、重みは 0 から):
    1. テキストの encoder の入力(各テキストのトークンの埋め込み)に足す: 長さの予測が話者ごとの話す速さに合わせられるように
    2. 音声トークンを埋める Transformer の時刻の埋め込みに足す(adaLN を通して全層にかかる)。CFG の「テキストなし」の出力にも残る
重みを 0 から始めるので、フェーズ A の T2S から学習を始めると、最初はフェーズ A の T2S と同じ出力になる。

使い方: 話者ベクトルは `with model.speaker(spk):` の中で渡す(ハイブリッド型の各メソッドの引数を変えずに使うため)。
    with model.speaker(spk):            # spk [B, 192]
        loss, parts = model(text_ids, codes, durations, f0_targets)
"""
from contextlib import contextmanager

import torch
import torch.nn as nn

from exp0002.common.t2s_hybrid import HybridT2S

from .s2m_spk import SPK_DIM, _zero_linear


class _TimeWithSpeaker(nn.Module):
    """ハイブリッド型の time_mlp の出力に、話者ベクトルの射影を足す。"""

    def __init__(self, time_mlp: nn.Module, owner: "SpeakerHybridT2S"):
        super().__init__()
        self.time_mlp, self.spk_dec = time_mlp, _zero_linear(SPK_DIM, time_mlp[-1].out_features)
        self._owner = [owner]  # 循環参照を state_dict に入れないため、リストに入れて持つ

    def forward(self, x):
        return self.time_mlp(x) + self.spk_dec(self._owner[0]._spk)


class SpeakerHybridT2S(HybridT2S):
    def __init__(self, n_text: int, n_codes: int, **kwargs):
        super().__init__(n_text, n_codes, **kwargs)
        dim = self.head.in_features
        self.spk_enc = _zero_linear(SPK_DIM, dim)
        self.time_mlp = _TimeWithSpeaker(self.time_mlp, self)
        self._spk = None

    @contextmanager
    def speaker(self, spk: torch.Tensor):
        """この中の計算に、話者ベクトル spk [B, 192] を使う。"""
        self._spk = spk.to(self.device)
        try:
            yield
        finally:
            self._spk = None

    def encode(self, text_ids: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """ハイブリッド型の encode と同じ。テキストの埋め込みに話者ベクトルの射影を足してから encoder に通す。"""
        assert self._spk is not None, "with model.speaker(spk): の中で呼ぶ"
        text = nn.utils.rnn.pad_sequence(text_ids, batch_first=True).to(self.device)
        mask = nn.utils.rnn.pad_sequence([torch.ones(len(t), dtype=torch.bool) for t in text_ids],
                                         batch_first=True).to(self.device)
        emb = self.enc_emb(text) + self.spk_enc(self._spk)[:, None]                        # [B, L, D]
        return self.encoder(inputs_embeds=emb, attention_mask=mask.long()).last_hidden_state, mask


def load_phase_a_state(model: nn.Module, state: dict) -> list[str]:
    """フェーズ A の重み(話者の条件の部品が無いもの)を読み込む。time_mlp は包んだので名前を合わせる。読み込まなかった名前を返す。"""
    renamed = {(k.replace("time_mlp.", "time_mlp.time_mlp.", 1) if k.startswith("time_mlp.") else k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(renamed, strict=False)
    assert not unexpected, unexpected
    return missing
