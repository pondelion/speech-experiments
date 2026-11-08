"""exp0003 固有の小物(保存先、SSL の特徴の取り出しと正規化、SSL の codec の読み込み、トークン化)。

SSL の特徴は保存せず、各スクリプトの起動時に取り出す(w2v-BERT 2.0 で JSUT 全体に数分)。

shape の軸の記号:
    S: SSL のフレーム数、T: Mel のフレーム数、D: SSL の特徴の次元、U: 音声トークン数(= T/4)
"""
from pathlib import Path

import torch
import torch.nn as nn

from exp0001.common.codec import DOWNSAMPLE, MelCodec
from exp0001.common.utils import encode_all
from exp0002.common.utils import TOKEN_SEC
from lib_common.audio import MelNormalizer
from lib_common.utils import Logger

from .ext_codec import EXT_CODECS
from .ssl import SSLFeatures, to_mel_rate

EXP_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = EXP_DIR / "runs"  # checkpoint・ログ・全テスト文の音声の置き場(git の対象外)


def extract_all(ssl: SSLFeatures, utts: list, log: Logger) -> list[torch.Tensor]:
    """全発話の SSL の特徴 [S_i, D](float16、CPU)を取り出す。"""
    out = []
    for i, u in enumerate(utts):
        out.append(ssl(u.path))
        if (i + 1) % 1000 == 0:
            log(f"  extracted {i + 1}/{len(utts)}")
    return out


def fit_normalizer(feats: list[torch.Tensor]) -> MelNormalizer:
    """次元ごとの平均・標準偏差を求める(全フレームを 1 つにつなぐとメモリが大きいので、和を足していく)。"""
    n, s, s2 = 0, 0.0, 0.0
    for f in feats:
        f = f.float()
        n, s, s2 = n + len(f), s + f.sum(0), s2 + (f ** 2).sum(0)
    mean = s / n
    return MelNormalizer(mean, (s2 / n - mean ** 2).clamp(min=1e-8).sqrt())


def codec_inputs(feats: list[torch.Tensor], utts: list, norm: MelNormalizer) -> list[torch.Tensor]:
    """codec に入れる形にする: Mel のフレーム数(4 の倍数に切り捨て)にそろえて正規化する。各 [T_i, D]。"""
    return [norm.normalize(to_mel_rate(f, len(u.mel) // DOWNSAMPLE * DOWNSAMPLE)) for f, u in zip(feats, utts)]


def load_ssl_codec(path: Path, device: str) -> tuple[MelCodec, MelNormalizer, MelNormalizer, dict]:
    """train_codec.py が保存した SSL の codec を読み込む。

    Returns:
        評価モードの codec、Mel の正規化(S2M の出力用)、SSL の特徴の正規化、{"ssl": SSL の名前, "layer": 層}。
    """
    ckpt = torch.load(path, map_location="cpu")
    codec = MelCodec(**ckpt["config"])
    codec.load_state_dict(ckpt["model"])
    return (codec.to(device).eval(), MelNormalizer.from_state_dict(ckpt["normalizer"]),
            MelNormalizer.from_state_dict(ckpt["feat_normalizer"]), {"ssl": ckpt["ssl"], "layer": ckpt["layer"]})


class TokenEmbedding(nn.Module):
    """外部のコーデックのトークン → S2M の条件の特徴(S2M と一緒に学習する埋め込み)。

    自作の codec では decoder の特徴(decode_hidden)を S2M の条件にするが、外部のコーデックでは同じものが無いので、
    トークンごとの埋め込みを S2M と一緒に学習する。自作の codec(MelCodec)と同じ属性を持たせて、同じ処理で使えるようにする。

    Args:
        codebook_size: トークンの種類数。
        token_sec: トークン 1 個の長さ(秒)。
        dim: 埋め込みの次元(自作の codec の decoder の特徴と同じ 384)。
    """

    def __init__(self, codebook_size: int, token_sec: float, dim: int = 384, fixed_code_dim: int = 0):
        super().__init__()
        self.codebook_size, self.token_sec, self.hidden_dim = codebook_size, token_sec, dim
        if fixed_code_dim > 0:  # ablation: コーデックの固定の代表ベクトル [K, C] + 学習する線形の射影 C → D
            self.register_buffer("code_vecs", torch.zeros(codebook_size, fixed_code_dim))
            self.code_proj = nn.Linear(fixed_code_dim, dim)
        else:  # 既定: トークンごとに学習する埋め込み
            self.codebook = nn.Embedding(codebook_size, dim)

    def decode_hidden(self, ids: torch.Tensor) -> torch.Tensor:
        """[B, U] → [B, U, D]。"""
        return self.code_proj(self.code_vecs[ids]) if hasattr(self, "code_vecs") else self.codebook(ids)


def code_vectors(codec: str | Path, device: str) -> torch.Tensor:
    """トークンの番号 → コーデックの固定の代表ベクトル [K, C](ablation 用)。

    SSL の codec: L2 正規化した codebook(decoder に渡すベクトルと同じ、8 次元)。
    XCodec2: FSQ の全ての組み合わせの表(8 次元のそれぞれが 4 段階、[-1, 1] に正規化した値)。
    """
    if str(codec) == "ext:xcodec2":
        return EXT_CODECS["xcodec2"]("cpu").model.quantizer.quantizer.codebook.float().clone()
    if str(codec).startswith("ext:"):
        raise NotImplementedError(f"{codec} の代表ベクトルの取り出しは未実装")
    model = load_ssl_codec(codec, "cpu")[0]
    return torch.nn.functional.normalize(model.codebook.weight.detach(), dim=-1)


def tokenize(codec: str | Path, utts: list, device: str, log: Logger) -> tuple[list[torch.Tensor], nn.Module, MelNormalizer | None]:
    """発話をトークン列にする。codec は SSL の codec の codec.pt か、外部のコーデックの "ext:{名前}"(EXT_CODECS)。

    SSL の codec: SSL の特徴 → codec でトークンにする(SSL モデルは使い終わったら GPU から外す)。
    外部のコーデック: コーデック自身の encoder でトークンにする。
    w2v-BERT 2.0 では特徴の取り出しに JSUT 全体で約 10 分かかるので、作ったトークン列は tokens.pt(発話 ID → トークン列、
    合計数 MB。SSL の codec は codec と同じ場所、外部のコーデックは runs/codec_{名前}/)に保存し、次からはそこから読む
    (S2M・T2S・評価の各スクリプトで使い回す)。codec を学習し直したときは tokens.pt を消すこと。

    Returns:
        発話ごとのトークン列 [U_i] のリスト、codec(SSL の codec か、未学習の TokenEmbedding)、
        Mel の正規化(SSL の codec のみ。外部のコーデックは None で、S2M の学習で求める)。
    """
    ext = str(codec).startswith("ext:")
    if ext:
        name = str(codec)[4:]
        cls = EXT_CODECS[name]
        model, mel_norm, cache_path = TokenEmbedding(cls.codebook_size, 1 / cls.token_rate), None, RUNS_DIR / f"codec_{name}" / "tokens.pt"
    else:
        model, mel_norm, feat_norm, info = load_ssl_codec(codec, device)
        model.token_sec, name, cache_path = TOKEN_SEC, f"{info['ssl']} layer {info['layer']}", Path(codec).parent / "tokens.pt"
    cache = torch.load(cache_path) if cache_path.exists() else {}
    todo = [u for u in utts if u.id not in cache]
    if todo:
        if ext:
            enc = cls(device)
            for u in todo:
                cache[u.id] = enc.encode(u.path)
        else:
            enc = SSLFeatures(info["ssl"], device, info["layer"])
            for s in range(0, len(todo), 256):  # 特徴をまとめて持たないように、256 発話ずつトークンにする
                us = todo[s: s + 256]
                cache.update(zip([u.id for u in us], encode_all(model, codec_inputs([enc(u.path) for u in us], us, feat_norm), device)))
        del enc
        torch.cuda.empty_cache()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)
    log(f"tokenized {len(utts)} utterances with {name} ({len(utts) - len(todo)} from {cache_path.name})")
    # トークンの番号は int64 にそろえる(XCodec2 は int32 で返し、cross entropy の正解に使えない)
    return [cache[u.id].long() for u in utts], model.to(device), mel_norm
