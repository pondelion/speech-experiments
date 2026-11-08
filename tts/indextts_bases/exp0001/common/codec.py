"""Mel を離散トークン(整数 ID の列)にする codec。IndexTTS2.5 の semantic codec に当たる部分。

なぜ離散トークンにするのか:
    IndexTTS は、テキストから音声を作る仕事を 2 段に分ける。
      1. T2S(GPT): テキスト → 音声トークン列を、言語モデルと同じ「次のトークン予測」で 1 個ずつ作る
      2. S2M(Flow Matching): 音声トークン列 → Mel を作る
    GPT は離散の記号列を扱うのが得意なので、音声を「約 23 個/秒 の整数 ID」に変換する仕組みが要る。
    それがこの codec。音声の長さも、T2S が何個トークンを出したかで自然に決まる。

IndexTTS2.5 との違い:
    IndexTTS2.5 は、学習済みの音声モデル w2v-BERT 2.0 の特徴(発音・内容の情報が多い)を量子化する。
    ここでは外部モデルを使わず、Mel そのものを量子化する(声質など音響の情報も多く含むトークンになる)。

構成(IndexTTS2.5 の codec と同じ形):
    Mel [B,T,80] → Conv → ConvNeXt ×n → 時間を 1/4 に(93.75Hz → 約 23.4Hz)→ ConvNeXt ×n
    → 8 次元に射影 → L2 正規化して、codebook(K 個の代表ベクトル)の中で一番近いものの番号を選ぶ
    → その代表ベクトル → 射影 → ConvNeXt ×n → 時間を ×4 に → ConvNeXt ×n → Mel を復元
    8 次元という低い次元で近さを測る(factorized VQ)と、codebook の多くの番号が使われやすくなる。

損失(IndexTTS2.5 と同じ):
    再構成(復元 Mel と元の Mel の二乗誤差)+ 0.15 × commit + codebook
    「一番近い番号を選ぶ」操作は微分できないので、straight-through(順伝播では選んだベクトルを使い、
    逆伝播では選ぶ前のベクトルに勾配をそのまま流す)で encoder を学習する。
    IndexTTS2.5 は codebook の値そのものを decoder に渡すが、ここでは正規化したベクトルを渡す(簡略化)。

shape の軸の記号:
    B: バッチ内の発話数
    T: Mel のフレーム数(93.75 フレーム/秒。4 の倍数にそろえて渡す)
    U: 音声トークン数(= T/4、約 23.4 個/秒)
    F: Mel の帯域数(80)
    D: 内部のチャンネル数(dim)
    C: トークン 1 個を表すベクトルの次元(code_dim = 8)。codebook の代表ベクトルもこの次元で、近さもここで測る。
       周波数の軸(80 帯域)を、学習された特徴の軸(チャンネル)に変換して 8 次元まで圧縮したもの。
       その途中で時間方向の周辺も混ぜている: 最初の Conv と ConvNeXt が近くのフレームを混ぜ、stride 2 の Conv
       2 回で 4 フレームを 1 時刻(トークン)にまとめる。畳み込みの窓を重ねると、1 トークンのベクトルは
       前後合わせて約 136 フレーム(約 1.45 秒)の範囲の Mel から作られる(勾配で実測した値)。
    K: トークンの種類数(codebook_size)
    PyTorch の Conv1d は [B, チャンネル, 時間] の順、Linear・LayerNorm は最後の軸に作用するので、途中で軸を入れ替える。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from lib_common.audio import N_MELS

DOWNSAMPLE = 4  # Mel 4 フレーム(42.7ms)= トークン 1 個


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block(1 次元版)。

    depthwise Conv(各チャンネルごとに時間方向の近傍を混ぜる)→ LayerNorm → Linear → GELU → Linear
    (チャンネル方向を混ぜる)→ 小さい係数を掛けて残差に足す。時間方向と特徴方向の処理を分けた軽い構造。

    Args:
        dim: チャンネル数 D。
        mult: 中間層の幅の倍率。
    """

    def __init__(self, dim: int, mult: int = 3):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.ff = nn.Sequential(nn.Linear(dim, dim * mult), nn.GELU(), nn.Linear(dim * mult, dim))
        self.scale = nn.Parameter(torch.full((dim,), 1e-6))  # 学習初期は残差がほぼ恒等写像になる

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 特徴 [B(発話数), D(チャンネル数), 時間]。

        Returns:
            同じ形の特徴 [B, D, 時間]。
        """
        h = rearrange(self.dwconv(x), "b d t -> b t d")  # LayerNorm・Linear 用に D を最後へ
        h = self.ff(self.norm(h)) * self.scale           # [B, 時間, D] → [B, 時間, D×mult] → [B, 時間, D]
        return x + rearrange(h, "b t d -> b d t")        # 元の並びに戻して残差に足す


class MelCodec(nn.Module):
    """Mel ⇔ 離散トークンの codec。

    Args:
        dim: 内部のチャンネル数 D。
        n_blocks: ConvNeXt block の数(encoder・decoder の各段)。
        codebook_size: トークンの種類数 K(IndexTTS2.5 は 8192。ここはデータが少ないので小さくする)。
        code_dim: トークン 1 個を表すベクトルの次元 C(IndexTTS2.5 と同じ 8)。
        commit_weight: commit 損失の重み(IndexTTS2.5 と同じ 0.15)。
        in_dim: 入力(= 復元する)特徴の次元。既定は Mel の 80。exp0003 では SSL の特徴の次元にする。
    """

    def __init__(self, dim: int = 384, n_blocks: int = 4, codebook_size: int = 1024, code_dim: int = 8,
                 commit_weight: float = 0.15, in_dim: int = N_MELS):
        super().__init__()
        self.codebook_size, self.commit_weight = codebook_size, commit_weight
        self.hidden_dim = dim
        # encoder: Mel → 時間 1/4 → 8 次元(in_dim を変えると、Mel 以外の特徴(exp0003 の SSL の特徴)も同じ形で量子化できる)
        self.enc_in = nn.Conv1d(in_dim, dim, 7, padding=3)
        self.enc_blocks1 = nn.Sequential(*[ConvNeXtBlock(dim) for _ in range(n_blocks)])
        self.down = nn.Sequential(nn.Conv1d(dim, dim, 4, stride=2, padding=1), nn.GELU(),  # 時間 1/2
                                  nn.Conv1d(dim, dim, 4, stride=2, padding=1))             # さらに 1/2
        self.enc_blocks2 = nn.Sequential(*[ConvNeXtBlock(dim) for _ in range(n_blocks)])
        self.to_code = nn.Conv1d(dim, code_dim, 1)
        # codebook: K 個の代表ベクトル [K, C]
        self.codebook = nn.Embedding(codebook_size, code_dim)
        # decoder: 8 次元 → 時間 ×4 → Mel
        self.from_code = nn.Conv1d(code_dim, dim, 1)
        self.dec_blocks1 = nn.Sequential(*[ConvNeXtBlock(dim) for _ in range(n_blocks)])
        self.up = nn.Sequential(nn.Upsample(scale_factor=DOWNSAMPLE, mode="nearest"),  # 各時刻を 4 回繰り返す
                                nn.Conv1d(dim, dim, 3, padding=1))
        self.dec_blocks2 = nn.Sequential(*[ConvNeXtBlock(dim) for _ in range(n_blocks)])
        self.dec_out = nn.Conv1d(dim, in_dim, 7, padding=3)

    def encode_vectors(self, mel: torch.Tensor) -> torch.Tensor:
        """Mel を、量子化する前のベクトル列にする。

        Args:
            mel: 正規化した Mel [B(発話数), T(Mel のフレーム数。4 の倍数), F(Mel の帯域数 = 80)]。

        Returns:
            L2 正規化したベクトル [B, U(トークン数 = T/4), C(トークン 1 個を表すベクトルの次元 = 8)]。
        """
        h = self.enc_blocks1(self.enc_in(rearrange(mel, "b t f -> b f t")))  # [B, F, T] → [B, D, T]
        h = self.enc_blocks2(self.down(h))                                   # [B, D, T] → [B, D, U](U = T/4)
        e = rearrange(self.to_code(h), "b c u -> b u c")                     # [B, D, U] → [B, C, U] → [B, U, C]
        return F.normalize(e, dim=-1)                                        # 各ベクトルを長さ 1 に

    def quantize(self, e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """各時刻で、codebook の中で一番近い(cosine 類似度が一番高い)ベクトルを選ぶ。

        Args:
            e: L2 正規化したベクトル [B(発話数), U(トークン数), C(トークン 1 個を表すベクトルの次元 = 8)]。

        Returns:
            選んだ番号(トークン ID)[B, U] と、そのベクトル(L2 正規化済み)[B, U, C]。
        """
        codes = F.normalize(self.codebook.weight, dim=-1)  # [K(トークンの種類数), C]
        sim = torch.einsum("buc,kc->buk", e, codes)        # 各時刻と各代表ベクトルの cosine 類似度 [B, U, K]
        ids = sim.argmax(-1)                               # 一番近い番号 [B, U]
        return ids, codes[ids]                             # codes[ids]: [B, U, C]

    def decode_hidden(self, ids: torch.Tensor) -> torch.Tensor:
        """トークン ID 列を、トークンの時間解像度の特徴にする。S2M の条件としても使う。

        Args:
            ids: トークン ID [B(発話数), U(トークン数)]。

        Returns:
            特徴 [B, U, D(内部のチャンネル数)]。
        """
        q = F.normalize(self.codebook(ids), dim=-1)  # [B, U] → 代表ベクトル [B, U, C(トークン 1 個を表すベクトルの次元 = 8)]
        return self._hidden_from_vectors(q)

    def _hidden_from_vectors(self, q: torch.Tensor) -> torch.Tensor:
        """代表ベクトル [B(発話数), U(トークン数), C(トークン 1 個を表すベクトルの次元 = 8)] → 特徴 [B, U, D(内部のチャンネル数)]。"""
        h = self.from_code(rearrange(q, "b u c -> b c u"))      # [B, C, U] → [B, D, U]
        return rearrange(self.dec_blocks1(h), "b d u -> b u d")  # [B, D, U] → [B, U, D]

    def decode_mel(self, hidden: torch.Tensor) -> torch.Tensor:
        """decode_hidden() の特徴から Mel を復元する。

        Args:
            hidden: 特徴 [B(発話数), U(トークン数), D(内部のチャンネル数)]。

        Returns:
            正規化した Mel [B, T(Mel のフレーム数 = 4U), F(Mel の帯域数 = 80)]。
        """
        h = self.dec_blocks2(self.up(rearrange(hidden, "b u d -> b d u")))  # [B, D, U] → 時間 ×4 → [B, D, T]
        return rearrange(self.dec_out(h), "b f t -> b t f")                 # [B, D, T] → [B, F, T] → [B, T, F]

    @torch.no_grad()
    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        """Mel をトークン ID 列にする(T2S・S2M の学習データを作るときに使う)。

        Args:
            mel: 正規化した Mel [B(発話数), T(Mel のフレーム数。4 の倍数), F(Mel の帯域数 = 80)]。

        Returns:
            トークン ID [B, U(トークン数 = T/4)]。
        """
        return self.quantize(self.encode_vectors(mel))[0]

    def forward(self, mel: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, dict, torch.Tensor, torch.Tensor]:
        """学習用の順伝播と損失の計算。

        Args:
            mel: 正規化した Mel [B(発話数), T(Mel のフレーム数。4 の倍数), F(Mel の帯域数 = 80)]。
                発話ごとに長さが違うので、足りない分は 0 で詰めてある。
            mask: 有効なフレームが True [B, T](0 で詰めた部分は損失に入れない)。

        Returns:
            損失(スカラー)、損失の内訳(dict)、トークン ID [B, U(トークン数 = T/4)]、復元した Mel [B, T, F]。
        """
        e = self.encode_vectors(mel)                         # [B, T, F] → [B, U, C]
        ids, q = self.quantize(e)                            # ids [B, U]、q [B, U, C]
        q_st = e + (q - e).detach()                          # straight-through: 値は q、勾配は e へそのまま流れる
        rec = self.decode_mel(self._hidden_from_vectors(q_st))  # [B, U, C] → [B, U, D] → [B, T, F]
        tmask = mask[:, ::DOWNSAMPLE].float()                # フレームの mask [B, T] → トークンの mask [B, U]
        mask = mask.float()
        # 再構成: フレームごとに 80 帯域の二乗誤差を平均 [B, T] → 有効なフレームで平均
        recon = (((rec - mel) ** 2).mean(-1) * mask).sum() / mask.sum()
        # commit: encoder の出力を、選ばれた代表ベクトルに近づける(encoder 側だけ更新)。[B, U, C] → [B, U] → 平均
        commit = (((e - q.detach()) ** 2).sum(-1) * tmask).sum() / tmask.sum()
        # codebook: 代表ベクトルを、encoder の出力に近づける(codebook 側だけ更新)
        codebook = (((q - e.detach()) ** 2).sum(-1) * tmask).sum() / tmask.sum()
        loss = recon + self.commit_weight * commit + codebook
        return loss, {"recon": recon.item(), "commit": commit.item(), "codebook": codebook.item()}, ids, rec
