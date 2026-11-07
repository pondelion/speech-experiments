"""exp0002 固有の小物(保存先、使い回す exp0001 の部品の場所、音素ごとの長さ、学習データの準備、T2S の読み込み)。

shape の軸の記号:
    L: テキストのトークン数(<bos>・<eos> 込み)、U: 音声トークン数
"""
from pathlib import Path

import torch

from exp0001.common.codec import DOWNSAMPLE
from exp0001.common.utils import encode_all
from lib_common.audio import HOP, SR
from lib_common.data import load_jsut_alignment
from lib_common.text import TextVocab

EXP_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = EXP_DIR / "runs"  # checkpoint・ログ・全テスト文の音声の置き場(git の対象外)
EXP1_RUNS = EXP_DIR.parent / "exp0001" / "runs"
# T2S 以外は exp0001 で一番良かった codebook 256 の codec と、その codec で学習した S2M を使い回す
BASE_CODEC = EXP1_RUNS / "codec_k256" / "codec.pt"
BASE_S2M = EXP1_RUNS / "s2m_k256" / "s2m.pt"
TOKEN_SEC = DOWNSAMPLE * HOP / SR  # 音声トークン 1 個が受け持つ時間(42.7ms)
PROSODY_MARKS = {"#", "[", "]"}     # 音を持たないアクセントの記号(長さ 0)。^ $ ? _ は無音・ポーズなので長さを持つ


def _same(token: str, n: int, phoneme: str) -> bool:
    """pyopenjtalk のトークンと、jsut-lab の n 行目の音素が同じ音を指すか。

    無音・ポーズは表記が違う: 文頭の無音 sil = ^、文末の無音 sil = $(疑問文は ?)、文中のポーズ pau = _。
    """
    if phoneme == "sil":
        return token == "^" if n == 0 else token in ("$", "?")
    return token == ("_" if phoneme == "pau" else phoneme)


def phoneme_durations(utt, tokens: list[str], n_codes: int) -> list[int] | None:
    """テキストの各トークンが、音声トークン何個分の長さかを、jsut-lab の音素の時刻から作る(FastSpeech2 系の正解)。

    pyopenjtalk の音素(lib_common.text.phoneme_tokens)は、ラベル 1 個につき音素か無音の記号 1 個を出し、
    その後にアクセントの記号を挟むことがある。jsut-lab の行も同じラベルの並びなので、アクセントの記号を除けば
    1 対 1 に対応する(並びが違う発話は使わない。JSUT 全体の約 5%)。
    時刻は音声トークンの境目に丸め、合計がちょうど n_codes になるように最後の音素で合わせる。

    Args:
        utt: JSUT の発話。
        tokens: テキストのトークン(TextVocab.tokenize の出力。<bos>・<eos> は含まない)。
        n_codes: その発話の音声トークン数 U。

    Returns:
        <bos> + 各トークン + <eos> の長さ(音声トークンの個数)のリスト [L]。並びが合わなければ None。
    """
    align = load_jsut_alignment(utt)
    sounding = [t for t in tokens if t not in PROSODY_MARKS]
    if len(sounding) != len(align) or not all(_same(t, n, p) for t, (n, (_, _, p)) in zip(sounding, enumerate(align))):
        return None
    # 各音素の終わりの時刻を音声トークンの境目に丸める(単調に増えるようにし、最後は U にそろえる)
    ends, prev = [], 0
    for _, e, _ in align:
        prev = max(prev, min(round(e / TOKEN_SEC), n_codes))
        ends.append(prev)
    ends[-1] = n_codes
    durs = iter([b - a for a, b in zip([0] + ends[:-1], ends)])
    return [0] + [0 if t in PROSODY_MARKS else next(durs) for t in tokens] + [0]


def prepare(utts: list, vocab: TextVocab, codec, norm, device: str, with_durations: bool) -> dict:
    """学習・検証用に、テキスト ID 列・音声トークン列(・音素ごとの長さ)を作る。

    Args:
        utts: 発話のリスト。
        vocab: テキストの語彙。
        codec: 学習済みの codec。
        norm: Mel の正規化。
        device: codec を動かすデバイス。
        with_durations: 音素ごとの長さも作るか(作れない発話は除く)。

    Returns:
        {"texts": 各 [L_i], "codes": 各 [U_i], "durations": 各 [L_i] または None, "n_dropped": 除いた発話数}。
    """
    texts = [torch.tensor(vocab.encode(u.text)) for u in utts]
    codes = encode_all(codec, [norm.normalize(u.mel) for u in utts], device)
    if not with_durations:
        return {"texts": texts, "codes": codes, "durations": None, "n_dropped": 0}
    durs = [phoneme_durations(u, vocab.tokenize(u.text), len(c)) for u, c in zip(utts, codes)]
    keep = [i for i, d in enumerate(durs) if d is not None]
    return {"texts": [texts[i] for i in keep], "codes": [codes[i] for i in keep],
            "durations": [torch.tensor(durs[i]) for i in keep], "n_dropped": len(utts) - len(keep)}


def load_t2s(path: Path, device: str):
    """train_t2s.py が保存した非自己回帰の T2S と、テキストの語彙を読み込む。

    Args:
        path: t2s.pt のパス。
        device: 置き先のデバイス。

    Returns:
        (評価モードの T2S、語彙、方式の名前 "duration" / "masked" / "hybrid")。
    """
    from .t2s_duration import DurationT2S
    from .t2s_hybrid import HybridT2S
    from .t2s_masked import MaskedT2S

    ckpt = torch.load(path, map_location="cpu")
    vocab = TextVocab.from_state_dict(ckpt["vocab"])
    cls = {"duration": DurationT2S, "masked": MaskedT2S, "hybrid": HybridT2S}[ckpt["method"]]
    model = cls(len(vocab), ckpt["n_codes"], **ckpt["config"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), vocab, ckpt["method"]
