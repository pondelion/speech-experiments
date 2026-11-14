"""テキストの前処理(IndexTTS-2.5 の日本語の処理に合わせたもの)と、文字単位の語彙。

公式(`indextts/infer_v2_5.py` の推論、`utils/ja_g2p.py`、`utils/front.py`)の日本語の流れ:
    1. 記号の置き換え(「、」→「,」、「。」→「.」、括弧 → 「'」など。front.py の char_rep_map)
    2. 小文字にする(日本語では英字だけが変わる)。日本語の数字などの読み方への変換(NeMo)は、日本語の文法が無いので何もしない
    3. MeCab(fugashi + unidic-lite)で単語に区切り、単語の間に空白を入れる。漢字はそのまま(読みには変えない)
       学習では、漢字を含む単語の一部を、その読み(平仮名)に置き換える(論文: 漢字をランダムにかなの読みに置き換えて学習。
       コードの JapaneseG2PProcessor の既定の割合 g2p_ratio = 0.2。推論は 0)
    4. 先頭に言語の記号「<|ja|> 」を付けて tokenizer で ID にし、最後に終わりの記号(ID 1)を付ける

tokenizer: 公式は多言語の文字単位の tokenizer(60,509 種類)。ここでは大きさ(パラメータの数)を自作の部品とそろえるため、
JSUT の学習データに出てくる文字と平仮名だけの文字単位の語彙にする(1 文字 = 1 ID。日本語の部分は公式と同じ文字単位)。
"""
import random
import re

import fugashi

CHAR_REP_MAP = {  # 公式の front.py の char_rep_map(日本語のテキストにも、この置き換えをしてから処理する)
    "：": ",", "；": ",", ";": ",", "，": ",", "。": ".", "！": "!", "？": "?", "\n": " ", "·": "-", "、": ",",
    "...": "…", ",,,": "…", "，，，": "…", "……": "…", "“": "'", "”": "'", '"': "'", "‘": "'", "’": "'",
    "（": "'", "）": "'", "(": "'", ")": "'", "《": "'", "》": "'", "【": "'", "】": "'", "[": "'", "]": "'",
    "—": "-", "～": "-", "~": "-", "「": "'", "」": "'", ":": ",",
}
_CLEAN = re.compile("|".join(re.escape(p) for p in CHAR_REP_MAP))
LANG_TOKEN = "<|ja|>"
HIRAGANA = [chr(c) for c in range(0x3041, 0x3097)]  # ぁ〜ゖ(読みに置き換えたときに出てくる文字)
_TAGGER = fugashi.Tagger()


def _kata2hira(text: str) -> str:
    return "".join(chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c for c in text)


def _has_kanji(text: str) -> bool:
    return any("一" <= c <= "鿿" for c in text)


def _segment(text: str, g2p_ratio: float, rng: random.Random) -> str:
    """空白を含まない部分を単語に区切って空白でつなぐ。漢字を含む単語のうち g2p_ratio の割合を、読み(平仮名)に置き換える。"""
    words = []
    for w in _TAGGER(text):
        reading = w.feature.kana
        words.append((w.surface, reading if reading and reading != "*" else w.surface))
    kanji = [i for i, (s, _) in enumerate(words) if _has_kanji(s)]
    n = int(len(kanji) * g2p_ratio)
    if n == 0 and kanji and rng.random() < g2p_ratio:  # 公式と同じ: 割合が小さくても、その確率で 1 語は置き換える
        n = 1
    replace = set(rng.sample(kanji, min(n, len(kanji))))
    return " ".join(_kata2hira(r) if i in replace else s for i, (s, r) in enumerate(words))


def frontend(text: str, g2p_ratio: float = 0.0, rng: random.Random | None = None) -> str:
    """書き起こし → T2S に入れる文字列(先頭に「<|ja|> 」)。

    例: 「東京都に住んでいる。」→「<|ja|> 東京 都 に 住ん で いる .」
    """
    text = _CLEAN.sub(lambda m: CHAR_REP_MAP[m.group()], text).lower()
    parts = re.split(r"( +)", text)  # 元の空白は残し、空白の無い部分ごとに区切る(公式の process_ja_text と同じ)
    rng = rng or random.Random(0)
    text = "".join(_segment(p, g2p_ratio, rng) if p.strip() else p for p in parts)
    return f"{LANG_TOKEN} {text}"


class CharVocab:
    """文字単位の語彙。ID 0 = テキストの始まり、1 = 終わり(公式の start_text_token・stop_text_token)、2 = <|ja|>、3 = 未知の文字。"""

    SPECIALS = ["<start>", "<stop>", LANG_TOKEN, "<unk>"]
    START, STOP, UNK = 0, 1, 3

    def __init__(self, chars: list[str]):
        self.itos = self.SPECIALS + sorted(set(chars) - set(self.SPECIALS))
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @classmethod
    def build(cls, texts: list[str]) -> "CharVocab":
        """学習データのテキスト(frontend を通す前)の文字と、平仮名・空白・置き換え後の記号から作る。"""
        chars = set(HIRAGANA) | {" "}
        for t in texts:
            chars |= set(frontend(t)[len(LANG_TOKEN):])
        return cls(sorted(chars))

    def encode(self, processed: str) -> list[int]:
        """frontend の出力 → ID の列(最後に終わりの記号 1 を付ける。公式の推論と同じ)。"""
        assert processed.startswith(LANG_TOKEN)
        return [2] + [self.stoi.get(c, self.UNK) for c in processed[len(LANG_TOKEN):]] + [self.STOP]

    def __len__(self) -> int:
        return len(self.itos)
