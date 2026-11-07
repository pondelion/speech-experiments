"""テキストの前処理。読み上げる文章を、モデルに入れる ID 列にする。

2 種類の入力形式を、同じインターフェース(TextVocab)で扱う。

- "char": NFKC 正規化した文字を、そのまま 1 文字 = 1 トークンにする。
  IndexTTS2.5 も日本語は G2P(文字 → 発音記号の変換)をせずに文字のまま入れている。
  漢字の読み(「今日」を「きょう」と読むか「こんにち」と読むか)は、モデルがデータから覚える必要がある。

- "phoneme": pyopenjtalk(日本語の読み・アクセント辞書を持つツール)で音素に変換し、
  アクセントを表す記号を挟む。読みは辞書で決まるので、少ないデータでも読み間違いが起きにくい。
  記号の意味:
    ^ 文頭 / $ 文末 / ? 疑問文の文末 / _ ポーズ(息継ぎ)/ # アクセント句の区切り
    [ ここで声の高さが上がる / ] ここで声の高さが下がる(アクセント核)
  例: 「東京都に住んでいる。」→ ^ t o [ o ky o o t o ] n i # s u ...
  (日本語はアクセントを「高低」で表す言語なので、どこで上がり・どこで下がるかが読み方の自然さを決める)
"""
import re
import unicodedata

import pyopenjtalk

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIALS = [PAD, UNK, BOS, EOS]  # 詰め物 / 未知のトークン / 文の始まり / 文の終わり


def char_tokens(text: str) -> list[str]:
    """文字単位に分ける。

    Args:
        text: 読み上げる文章。

    Returns:
        文字のリスト。NFKC 正規化(全角英数字を半角にする等)をして、空白は除く。
    """
    return list(unicodedata.normalize("NFKC", text).replace(" ", ""))


def _num(pattern: str, label: str) -> int:
    """ラベル文字列から数値を 1 つ取り出す。値が無い("xx")場合は -50 を返す。"""
    m = re.search(pattern, label)
    return -50 if m is None or m.group(1) == "xx" else int(m.group(1))


def phoneme_tokens(text: str) -> list[str]:
    """pyopenjtalk で音素に変換し、アクセントの記号を挟む。

    pyopenjtalk のフルコンテキストラベル(音素 1 個ごとに、前後の音素・アクセント句の情報などを
    まとめた文字列)から、次の値を読んで記号を決める。
      a1: アクセント核(声が下がる位置)から見た、今のモーラの位置。0 なら今がアクセント核
      a2: アクセント句の中で、前から数えて何モーラ目か
      a3: アクセント句の中で、後ろから数えて何モーラ目か
      f1: アクセント句のモーラ数
    (モーラ = 日本語の拍。「きょう」は「きょ」「う」の 2 モーラ)

    Args:
        text: 読み上げる文章。

    Returns:
        音素とアクセント記号のリスト。
    """
    labels = pyopenjtalk.extract_fullcontext(text)
    out = []
    for n, lab in enumerate(labels):
        p = re.search(r"-(.*?)\+", lab).group(1)  # 今の音素
        if p == "sil":  # 文頭・文末の無音
            if n == 0:
                out.append("^")
            elif n == len(labels) - 1:
                out.append("?" if _num(r"!(\d+)_", lab) == 1 else "$")  # E フィールドの疑問文フラグ
            continue
        if p == "pau":  # 文中のポーズ
            out.append("_")
            continue
        out.append(p)
        a1 = _num(r"/A:([0-9\-]+)\+", lab)
        a2 = _num(r"\+(\d+)\+", lab)
        a3 = _num(r"\+(\d+)/", lab)
        f1 = _num(r"/F:(\d+)_", lab)
        a2_next = _num(r"\+(\d+)\+", labels[n + 1])
        if a3 == 1 and a2_next == 1 and p in "aeiouAEIOUNcl":
            out.append("#")  # アクセント句の最後のモーラで、次から新しい句が始まる
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            out.append("]")  # アクセント核:この後で声が下がる
        elif a2 == 1 and a2_next == 2:
            out.append("[")  # 句の 1 モーラ目の後で声が上がる
    return out


TOKENIZERS = {"char": char_tokens, "phoneme": phoneme_tokens}


class TextVocab:
    """トークン ⇔ ID の対応表。語彙は学習データから作り、学習データに無いトークンは <unk> にする。

    Args:
        kind: "char" か "phoneme"。
        tokens: ID 順のトークンのリスト(先頭は特殊トークン)。
    """

    def __init__(self, kind: str, tokens: list[str]):
        self.kind = kind
        self.itos = tokens
        self.stoi = {t: i for i, t in enumerate(tokens)}

    @classmethod
    def build(cls, kind: str, texts: list[str]) -> "TextVocab":
        """学習データの文章から語彙を作る。

        Args:
            kind: "char" か "phoneme"。
            texts: 学習データの文章のリスト。

        Returns:
            作った TextVocab。
        """
        seen = sorted({t for text in texts for t in TOKENIZERS[kind](text)})
        return cls(kind, SPECIALS + seen)

    def tokenize(self, text: str) -> list[str]:
        """文章をトークンのリストにする。"""
        return TOKENIZERS[self.kind](text)

    def encode(self, text: str) -> list[int]:
        """文章を ID 列にする。

        Args:
            text: 読み上げる文章。

        Returns:
            <bos> + トークンの ID + <eos> の ID 列。
        """
        unk = self.stoi[UNK]
        return [self.stoi[BOS]] + [self.stoi.get(t, unk) for t in self.tokenize(text)] + [self.stoi[EOS]]

    def __len__(self) -> int:
        return len(self.itos)

    def state_dict(self) -> dict:
        """checkpoint に保存する値。"""
        return {"kind": self.kind, "tokens": self.itos}

    @classmethod
    def from_state_dict(cls, d: dict) -> "TextVocab":
        """state_dict() で保存した値から復元する。"""
        return cls(d["kind"], d["tokens"])
