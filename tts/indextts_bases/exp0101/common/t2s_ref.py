"""exp0102 の t2s_ref_encoder: T2S の話者の条件を、CAM++ の 1 本から「参照音声の Mel の系列を読む encoder」の出力に替えた T2S。

encoder: IndexTTS2 の T2S の話者の条件と同じ形(Conformer + Perceiver)。ネットワークの定義は公式のリポジトリ(vendor/index-tts)のものを使い、
    大きさも公式の設定(vendor/indextts2.5-config の condition_module)と同じ: Conformer は 512 次元・6 層・FFN 2048・8 ヘッド・
    入力の時間方向を 1/2 にする畳み込み、Perceiver は 32 本・FFN 倍率 2・8 ヘッド。512 次元は T2S の次元と同じ。
    入力は公式の w2v-BERT の特徴の代わりに、参照音声の Mel(80 帯域、S2M と同じ正規化)。重みはスクラッチ(ランダムな初期値)から学習する
入れる場所: 基準(SpeakerHybridT2S)と同じ 2 か所。32 本の系列になるので、それぞれ次のように入れる:
    1. テキストの encoder の入力: 32 本をテキストのトークンの埋め込みの前につなげる(テキストのトークンが attention で参照する)。
       encoder の出力では、前につなげた 32 本の位置を捨てる
    2. 時刻の埋め込み: 32 本の平均を Linear(重みは 0 から)で射影して足す(基準の CAM++ の射影と同じ場所・同じ形)

使い方:
    with model.speaker(ref_mels):       # ref_mels: 発話ごとの参照音声の Mel [M_i, 80] のリスト(正規化済み)
        loss, parts = model(text_ids, codes, durations, f0_targets)

shape の記号: B = 発話数、M = 参照音声の Mel のフレーム数、M' = Conformer の出力の長さ(≒ M / 2)、N = 32(Perceiver の出力の本数)、
    L = テキストのトークン数、D = T2S の次元(512)
"""
import sys
import types
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from . import EXP_DIR
from .s2m_spk import _zero_linear
from .t2s_spk import SpeakerHybridT2S

N_LATENTS = 32
REF_MAX_FRAMES = 1000  # 参照音声の Mel は先頭の 1,000 フレームまで使う(メモリの上限のため。全発話の 94% はこれより短い)


def _official_modules():
    """公式の Conformer・Perceiver を読み込む。公式の indextts.utils.common が先頭で torchaudio を import するが、
    使う make_pad_mask は torchaudio を使わないので、torchaudio が無い環境では空のモジュールで置き換えて読み込む。"""
    sys.path.insert(0, str(EXP_DIR.parents[2] / "vendor" / "index-tts"))
    sys.modules.setdefault("torchaudio", types.ModuleType("torchaudio"))
    from indextts.gpt.conformer_encoder import ConformerEncoder
    from indextts.gpt.perceiver import PerceiverResampler
    return ConformerEncoder, PerceiverResampler


class _Checkpointed(nn.Module):
    """学習中は勾配 checkpointing で層を通す(逆伝播のときに前向きの計算をやり直し、途中の値を持たない)。

    1 バッチに短い発話が多く入ると、参照音声の Conformer の途中の値で GPU のメモリが約 20GiB になるため(計算の結果は同じ)。
    """

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

    def forward(self, *args):
        if self.training and torch.is_grad_enabled():
            return checkpoint(self.layer, *args, use_reentrant=False)
        return self.layer(*args)


class RefEncoderT2S(SpeakerHybridT2S):
    def __init__(self, n_text: int, n_codes: int, n_mels: int = 80, **kwargs):
        super().__init__(n_text, n_codes, **kwargs)
        dim = self.head.in_features
        ConformerEncoder, PerceiverResampler = _official_modules()
        self.ref_conformer = ConformerEncoder(input_size=n_mels, output_size=512, linear_units=2048, attention_heads=8,
                                              num_blocks=6, input_layer="conv2d2")
        self.ref_conformer.encoders = nn.ModuleList(_Checkpointed(m) for m in self.ref_conformer.encoders)
        self.ref_perceiver = PerceiverResampler(dim, dim_context=512, ff_mult=2, heads=8, num_latents=N_LATENTS)
        del self.spk_enc                                                                     # CAM++ の射影は使わない
        self.time_mlp.spk_dec = _zero_linear(dim, self.time_mlp.time_mlp[-1].out_features)
        self._conds = None

    @contextmanager
    def speaker(self, ref_mels: list[torch.Tensor]):
        """この中の計算に、参照音声の Mel(発話ごとの [M_i, 80])から作った話者の条件を使う。"""
        mels = [m[:REF_MAX_FRAMES].to(self.device) for m in ref_mels]
        x = nn.utils.rnn.pad_sequence(mels, batch_first=True)                                 # [B, M, 80]
        lens = torch.tensor([len(m) for m in mels], device=self.device)
        h, mask = self.ref_conformer(x, lens)                                                 # [B, M', 512], [B, 1, M']
        self._conds = self.ref_perceiver(h, F.pad(mask.squeeze(1), (N_LATENTS, 0), value=True))  # [B, N, D]
        self._spk = self._conds.mean(1)                                                       # [B, D](時刻の埋め込みに足す)
        try:
            yield
        finally:
            self._conds, self._spk = None, None

    def encode(self, text_ids: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """テキストの埋め込みの前に話者の条件 32 本をつなげて encoder に通し、テキストの位置の出力だけを返す。"""
        assert self._conds is not None, "with model.speaker(ref_mels): の中で呼ぶ"
        text = nn.utils.rnn.pad_sequence(text_ids, batch_first=True).to(self.device)
        mask = nn.utils.rnn.pad_sequence([torch.ones(len(t), dtype=torch.bool) for t in text_ids],
                                         batch_first=True).to(self.device)              # [B, L]
        emb = torch.cat([self._conds, self.enc_emb(text)], 1)                              # [B, N + L, D]
        full = F.pad(mask, (N_LATENTS, 0), value=True)                                     # [B, N + L]
        h = self.encoder(inputs_embeds=emb, attention_mask=full.long()).last_hidden_state  # [B, N + L, D]
        return h[:, N_LATENTS:], mask
