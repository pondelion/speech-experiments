"""読み上げの正確さの評価。学習済みの日本語 ASR(音声認識)で書き起こし、文字誤り率(CER)を測る。

ASR には NVIDIA の parakeet-tdt_ctc-0.6b-ja を使う(以前の比較で、JSUT の読み CER が 2.40% と最も正確だった)。
ASR はモデル本体ではなく評価にだけ使う。

CER = (置換 + 削除 + 挿入された文字数) / 正解の文字数。次の 3 通りで測る。
  raw:        書き起こしをそのまま比べる
  normalized: 句読点・記号・空白を除いて比べる
  reading:    さらに読み(カタカナ)に変換して比べる。「歪めた」と「ゆがめた」のような表記の違いを
              誤りに数えないので、「正しく読めているか」に一番近い。これを主な指標にする
"""
import logging
import re
import unicodedata

import jiwer
from janome.tokenizer import Tokenizer

ASR_MODEL = "nvidia/parakeet-tdt_ctc-0.6b-ja"
_PUNCT_RE = re.compile(r"[、。!!??,.・「」『』()()【】\[\]…\s]")
_janome = Tokenizer()


def normalize(text: str) -> str:
    """NFKC 正規化して、句読点・記号・空白を除く。"""
    return _PUNCT_RE.sub("", unicodedata.normalize("NFKC", text))


def to_reading(text: str) -> str:
    """読み(カタカナ)に変換する。読みが分からない語はそのまま残す。"""
    return "".join(t.reading if t.reading != "*" else t.surface for t in _janome.tokenize(text))


def cer_scores(refs: list[str], hyps: list[str]) -> dict:
    """3 通りの CER をまとめて計算する(全発話の誤り文字数の合計 / 正解の文字数の合計)。

    Args:
        refs: 正解の文章のリスト。
        hyps: ASR の書き起こしのリスト。

    Returns:
        {"raw": ..., "normalized": ..., "reading": ...}。
    """
    n_refs, n_hyps = [normalize(r) for r in refs], [normalize(h) for h in hyps]
    return {
        "raw": jiwer.cer(refs, hyps),
        "normalized": jiwer.cer(n_refs, n_hyps),
        "reading": jiwer.cer([normalize(to_reading(r)) for r in n_refs], [normalize(to_reading(h)) for h in n_hyps]),
    }


def utterance_cer(ref: str, hyp: str) -> float:
    """1 発話の読み CER。"""
    return jiwer.cer(normalize(to_reading(normalize(ref))), normalize(to_reading(normalize(hyp))))


class ASR:
    """parakeet で wav ファイルを書き起こす。

    Args:
        device: 計算に使うデバイス。
    """

    def __init__(self, device: str = "cuda"):
        import nemo.collections.asr as nemo_asr  # 読み込みが重いので、使うときだけ import する
        logging.getLogger("nemo_logger").addFilter(lambda r: r.levelno >= logging.ERROR)
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=ASR_MODEL).to(device).eval()

    def transcribe(self, paths: list[str], batch_size: int = 16) -> list[str]:
        """
        Args:
            paths: wav ファイルのパスのリスト。
            batch_size: 一度に書き起こす数。

        Returns:
            書き起こしのリスト。
        """
        outs = self.model.transcribe(paths, batch_size=batch_size, verbose=False)
        return [o.text if hasattr(o, "text") else str(o) for o in outs]
