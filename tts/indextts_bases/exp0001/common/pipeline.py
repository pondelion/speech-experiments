"""学習済みの codec・T2S・S2M とボコーダを組み立てて、テキストから波形まで合成する推論用の pipeline。

処理の流れ(IndexTTS2.5 の推論と同じ順):
    テキスト → 前処理(文字 / 音素 + アクセント)→ テキスト ID 列
    → ① T2S が音声トークンを 1 個ずつ生成(<EA> が出たら終わり。ここで音声の長さが決まる)
    → ② codec decoder の特徴 → S2M が Flow Matching で Mel を生成(トークン 1 個 = 4 フレーム)
    → ③ Griffin-Lim で波形にする

T2S の生成が全部終わってから S2M を動かす(トークンを 1 個作るたびに S2M を呼ぶわけではない)。
途中の段階を個別に呼べるようにしてあるので、評価で「正解トークンから S2M を通す」なども同じ部品で行える。

使い方:
    pipe = TTSPipeline.from_checkpoints(codec_path, t2s_path, s2m_path, device="cuda")
    out = pipe(["今日はいい天気ですね。"])
    out[0]["wave"]  # 24kHz の波形

shape の軸の記号:
    B: 一度に処理する発話数
    L: テキストのトークン数(<bos>・<eos> 込み)
    U: 音声トークン数(約 23.4 個/秒)
    T: Mel のフレーム数(= 4U、93.75 フレーム/秒)
    F: Mel の帯域数(80)
    Dc: codec の特徴の次元
    N: 波形のサンプル数(24,000 点/秒)
"""
from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange

from lib_common.audio import MelNormalizer
from lib_common.text import TextVocab
from lib_common.vocoder import GriffinLim

from .codec import DOWNSAMPLE, MelCodec
from .s2m import SemanticToMel
from .t2s import TextToSemantic
from .utils import RUNS_DIR, load_codec


def load_t2s(path: Path, device: str) -> tuple[TextToSemantic, TextVocab]:
    """train_t2s.py が保存した T2S と、テキストの語彙を読み込む。

    Args:
        path: t2s.pt のパス。
        device: 置き先のデバイス。

    Returns:
        評価モードの T2S と、学習時と同じ語彙(テキストの前処理の種類も含む)。
    """
    ckpt = torch.load(path, map_location="cpu")
    vocab = TextVocab.from_state_dict(ckpt["vocab"])
    sd = ckpt["model"]
    # 代表ベクトルや粗いトークンの対応表は、モデルの buffer として state_dict に入っている
    model = TextToSemantic(len(vocab), ckpt["n_codes"], **ckpt["config"], code_vecs=sd.get("code_vecs"),
                           from_fine=sd.get("from_fine"), to_fine=sd.get("to_fine"))
    model.load_state_dict(sd)
    return model.to(device).eval(), vocab


def load_s2m(path: Path, device: str) -> SemanticToMel:
    """train_s2m.py が保存した S2M を読み込む。

    Args:
        path: s2m.pt のパス。
        device: 置き先のデバイス。

    Returns:
        評価モードの S2M。
    """
    ckpt = torch.load(path, map_location="cpu")
    model = SemanticToMel(ckpt["cond_in_dim"], **ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.text_t2s = ckpt.get("text_t2s")  # テキストの隠れ状態を条件にする S2M なら、それを作る T2S の runs/ の名前
    return model.to(device).eval()


class TTSPipeline(nn.Module):
    """テキスト → 音声トークン → Mel → 波形 の推論 pipeline。

    Args:
        codec: 学習済みの codec。
        normalizer: codec の学習時に求めた Mel の正規化の値(S2M の出力を log-Mel に戻すのに使う)。
        s2m: 学習済みの S2M。
        t2s: 学習済みの T2S(None なら、テキストからの合成はできず、トークンからの合成だけになる)。
        vocab: T2S の学習時と同じテキストの語彙。
        device: 計算に使うデバイス。
        gl_iters: Griffin-Lim の反復回数。
    """

    def __init__(self, codec: MelCodec, normalizer: MelNormalizer, s2m: SemanticToMel,
                 t2s: TextToSemantic | None = None, vocab: TextVocab | None = None, device: str = "cuda",
                 gl_iters: int = 64):
        super().__init__()
        self.codec, self.normalizer, self.s2m, self.t2s, self.vocab = codec, normalizer, s2m, t2s, vocab
        self.device = device
        self.vocoder = GriffinLim(device, n_iter=gl_iters)

    @classmethod
    def from_checkpoints(cls, codec_path: Path, s2m_path: Path, t2s_path: Path | None = None,
                         device: str = "cuda") -> "TTSPipeline":
        """各スクリプトが保存した checkpoint から組み立てる。

        Args:
            codec_path: train_codec.py の codec.pt。
            s2m_path: train_s2m.py の s2m.pt。
            t2s_path: train_t2s.py の t2s.pt(テキストの前処理の種類は checkpoint に入っている)。
            device: 計算に使うデバイス。

        Returns:
            組み立てた TTSPipeline。
        """
        codec, normalizer = load_codec(codec_path, device)
        s2m = load_s2m(s2m_path, device)
        if s2m.text_t2s:  # テキストの隠れ状態を作る T2S(S2M の学習時と同じもの)
            s2m.text_encoder = load_t2s(RUNS_DIR / s2m.text_t2s / "t2s.pt", device)
        t2s, vocab = load_t2s(t2s_path, device) if t2s_path else (None, None)
        return cls(codec, normalizer, s2m, t2s, vocab, device)

    @torch.no_grad()
    def text_to_codes(self, texts: list[str], batch: int = 16, **sampling) -> list[torch.Tensor]:
        """① T2S: テキストから音声トークン列を生成する。

        Args:
            texts: 読み上げる文章のリスト。
            batch: 一度に生成する発話数。
            **sampling: TextToSemantic.generate() のサンプリングの設定(temperature、top_p など)。

        Returns:
            発話ごとの音声トークン列 [U_i(i 番目の発話で生成したトークン数)](codec のトークン、CPU)のリスト。
        """
        assert self.t2s is not None, "T2S の checkpoint を読み込んでいない"
        codes = []
        for s in range(0, len(texts), batch):
            text_ids = [torch.tensor(self.vocab.encode(t)) for t in texts[s:s + batch]]  # 各 [L_i]
            codes += [self.t2s_to_codec(c) for c in self.t2s.generate(text_ids, **sampling)]
        return codes

    def t2s_to_codec(self, ids: torch.Tensor) -> torch.Tensor:
        """T2S のトークンを、S2M に渡す codec のトークンにする(粗いトークンの T2S なら、グループの代表に戻す)。

        Args:
            ids: T2S のトークン ID 列 [U(トークン数)]。

        Returns:
            codec のトークン ID 列 [U](CPU)。
        """
        if self.t2s.to_fine is not None:
            ids = self.t2s.to_fine[ids.to(self.t2s.to_fine.device)]
        return ids.cpu()

    @torch.no_grad()
    def codes_to_mel(self, codes: list[torch.Tensor], n_steps: int = 25, cfg: float = 0.7,
                     batch: int = 16, texts: list[str] | None = None) -> list[torch.Tensor]:
        """② S2M: 音声トークン列から log-Mel を生成する。

        Args:
            codes: 発話ごとの音声トークン列 [U_i(i 番目の発話のトークン数)] のリスト。
            n_steps: Euler 法のステップ数(IndexTTS2.5 の公開推論は 25)。
            cfg: CFG の強さ(IndexTTS2.5 の公開推論は 0.7)。
            batch: 一度に生成する発話数。
            texts: 読み上げる文章(テキストの隠れ状態を条件にする S2M のときだけ必要)。

        Returns:
            発話ごとの正規化前の log-Mel [T_i(フレーム数 = 4U_i), F(Mel の帯域数 = 80)](CPU)のリスト。
            (以下のコメントの記号: B 発話数、U・T 一番長い発話のトークン数・フレーム数、Dc codec の特徴の次元)
        """
        out = []
        for s in range(0, len(codes), batch):
            cs = [c if len(c) else torch.zeros(1, dtype=torch.long) for c in codes[s:s + batch]]  # 空の列は 1 個で代用
            ids = nn.utils.rnn.pad_sequence(cs, batch_first=True).to(self.device)            # [B, U]
            frames = rearrange(torch.arange(ids.shape[1] * DOWNSAMPLE, device=self.device), "t -> 1 t")  # [1, T]
            n_valid = rearrange(torch.tensor([len(c) * DOWNSAMPLE for c in cs], device=self.device), "b -> b 1")  # [B, 1]
            mask = frames < n_valid                                                           # 有効なフレームが True [B, T]
            text = text_pad = None
            if self.s2m.text_dim > 0:  # テキストの隠れ状態 [B, L, Dt] と詰め物のマスク [B, L]
                enc, enc_vocab = self.s2m.text_encoder
                text, text_pad = enc.text_hidden([torch.tensor(enc_vocab.encode(t)) for t in texts[s:s + batch]])
            mel = self.s2m.generate(self.codec.decode_hidden(ids), mask, n_steps=n_steps, cfg=cfg,
                                    text=text, text_pad=text_pad)  # [B, U, Dc] → [B, T, F]
            out += [self.normalizer.denormalize(m[: len(c) * DOWNSAMPLE]).cpu() for m, c in zip(mel, cs)]  # 各 [T_i, F]
        return out

    @torch.no_grad()
    def codes_to_mel_codec(self, codes: list[torch.Tensor]) -> list[torch.Tensor]:
        """S2M の代わりに codec の decoder で、音声トークン列から log-Mel を復元する(評価での比較用)。

        Args:
            codes: 発話ごとの音声トークン列 [U_i(i 番目の発話のトークン数)] のリスト。

        Returns:
            発話ごとの正規化前の log-Mel [T_i(フレーム数 = 4U_i), F(Mel の帯域数 = 80)](CPU)のリスト。
        """
        out = []
        for c in codes:
            ids = rearrange(c, "u -> 1 u").to(self.device)                                   # 1 発話のバッチ [1, U]
            mel = self.codec.decode_mel(self.codec.decode_hidden(ids))                        # [1, U, Dc] → [1, T, F]
            out.append(self.normalizer.denormalize(rearrange(mel, "1 t f -> t f")).cpu())
        return out

    def mel_to_wave(self, logmel: torch.Tensor) -> torch.Tensor:
        """③ ボコーダ: log-Mel から波形を作る。

        Args:
            logmel: 正規化前の log-Mel [T(フレーム数), F(Mel の帯域数 = 80)]。

        Returns:
            24kHz の波形 [N(サンプル数)](CPU)。
        """
        return self.vocoder(logmel)

    @torch.no_grad()
    def forward(self, texts: list[str], n_steps: int = 25, cfg: float = 0.7, **sampling) -> list[dict]:
        """テキストから波形まで合成する。

        Args:
            texts: 読み上げる文章のリスト。
            n_steps: S2M の Euler 法のステップ数。
            cfg: S2M の CFG の強さ。
            **sampling: T2S のサンプリングの設定(temperature、top_p、top_k、repetition_penalty、max_len)。

        Returns:
            発話ごとの dict のリスト。各 dict の中身:
              "codes": 音声トークン列 [U(トークン数)]
              "mel":   log-Mel [T(フレーム数 = 4U), F(Mel の帯域数 = 80)]
              "wave":  24kHz の波形 [N(サンプル数)]
        """
        codes = self.text_to_codes(texts, **sampling)
        mels = self.codes_to_mel(codes, n_steps=n_steps, cfg=cfg, texts=texts)
        return [{"codes": c, "mel": m, "wave": self.mel_to_wave(m)} for c, m in zip(codes, mels)]
