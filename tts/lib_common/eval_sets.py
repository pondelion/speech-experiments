"""全実験で共通に使う評価用の文章を読み込む。

どれも参照音声は無く、合成した音声を ASR で書き起こして、入力した文章との文字誤り率(CER)で評価する。
学習データ(JSUT)と違う種類の文章で、読み上げが崩れないかを見るためのもの。

文章のセット(ファイルはリポジトリ直下の datasets/eval_texts/ にある。詳しくは datasets/eval_texts/README.md):
    ita_emotion:    ITA コーパスの感情 100 文(パブリックドメイン)。感情を指定する実験でも使う
    ita_recitation: ITA コーパスの朗読 324 文(パブリックドメイン)。普段の単語にあまり出ない音まで含む
    practical:      自作の実用文 55 文。ニュース風・会話・数字・英字・固有名詞・読みが文脈で変わる語・
                    長文・短文・疑問/感嘆・オノマトペのカテゴリに分けてある
"""
import csv
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # lib_common → tts → リポジトリ
EVAL_TEXTS_DIR = REPO_ROOT / "datasets" / "eval_texts"
SETS = ["ita_emotion", "ita_recitation", "practical"]


@dataclass
class EvalText:
    """評価用の文章 1 つ。

    Attributes:
        id: 文章の ID(例 EMOTION100_001、num_01)。
        set: セットの名前(SETS のどれか)。
        category: カテゴリ(practical のみ。ITA コーパスはセットの名前と同じ)。
        text: 読み上げる文章。
        expected_reading: 期待する読み。ITA コーパスは文全体のカタカナの読み、practical は読みに注意が要る
            部分だけの注記(無い文は空)。
    """
    id: str
    set: str
    category: str
    text: str
    expected_reading: str


def _load_ita(name: str) -> list[EvalText]:
    """ITA コーパスの書き起こし(1 行 = `ID:文章,カタカナの読み`)を読む。"""
    out = []
    path = EVAL_TEXTS_DIR / "ita" / f"{name.removeprefix('ita_')}_transcript_utf8.txt"
    for line in path.read_text(encoding="utf-8").splitlines():
        uid, rest = line.split(":", 1)
        text, reading = rest.rsplit(",", 1)
        out.append(EvalText(uid, name, name, text, reading))
    return out


def _load_practical() -> list[EvalText]:
    """自作の実用文(practical.tsv)を読む。"""
    with open(EVAL_TEXTS_DIR / "practical.tsv", encoding="utf-8") as f:
        return [EvalText(r["id"], "practical", r["category"], r["text"], r["expected_reading"])
                for r in csv.DictReader(f, delimiter="\t")]


def load_eval_texts(names: list[str] | None = None) -> list[EvalText]:
    """評価用の文章を読み込む。

    Args:
        names: 読み込むセットの名前のリスト(SETS のどれか)。None なら全部。

    Returns:
        EvalText のリスト(セットの順、ファイルの順)。
    """
    out = []
    for name in names or SETS:
        out += _load_practical() if name == "practical" else _load_ita(name)
    return out
