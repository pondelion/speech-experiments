"""話者と文の分け方を決めて、results/split.json に保存する(README の「データ」のとおり。乱数は使わず、決まった規則で選ぶ)。

- 学習に使わない話者(10 人): 男女それぞれ、声の高さ(gender_f0range.txt の最低と最高の F0 の真ん中)で並べ、等間隔に 5 人ずつ選ぶ
- 学習に使った話者から評価する 10 人: 残りの話者から、同じ規則で男女 5 人ずつ選ぶ
- テスト文(25 文): parallel100(= JSUT の voiceactress100)の 100 文を文字数で並べ、等間隔に 25 文選ぶ(4 文ごとに 1 文)

使い方:
    python make_split.py
出力: results/split.json
"""
import json

import common

JVS = common.DATASETS / "jvs" / "jvs_ver1"
JSUT_VA = common.DATASETS / "jsut" / "jsut_ver1.1" / "voiceactress100"


def evenly(items: list, n: int) -> list:
    """並べた items から、両端を含めて等間隔に n 個選ぶ。"""
    return [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)]


def main():
    rows = [line.split() for line in (JVS / "gender_f0range.txt").read_text().splitlines()[1:]]
    spk = {r[0]: (r[1], (float(r[2]) + float(r[3])) / 2) for r in rows}           # 話者 → (性別, F0 の範囲の真ん中)
    unseen, seen_eval = [], []
    for g in ("M", "F"):
        ranked = sorted((s for s in spk if spk[s][0] == g), key=lambda s: spk[s][1])
        picked = evenly(ranked, 5)
        unseen += picked
        rest = [s for s in ranked if s not in picked]
        seen_eval += evenly(rest, 5)
    texts = dict(line.split(":", 1) for line in (JSUT_VA / "transcript_utf8.txt").read_text(encoding="utf-8").splitlines())
    by_len = sorted(texts, key=lambda k: (len(texts[k]), k))
    test_ids = sorted(by_len[i] for i in range(0, 100, 4))
    out = {
        "unseen_speakers": sorted(unseen),
        "seen_eval_speakers": sorted(seen_eval),
        "train_speakers": sorted(s for s in spk if s not in unseen),
        "test_sentences": test_ids,
        "train_sentences": sorted(k for k in texts if k not in test_ids),
        "speaker_info": {s: {"gender": spk[s][0], "f0_mid": spk[s][1]} for s in sorted(unseen + seen_eval)},
        "test_texts": {k: texts[k] for k in test_ids},
    }
    path = common.EXP_DIR / "results" / "split.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"unseen {out['unseen_speakers']}\nseen_eval {out['seen_eval_speakers']}\ntrain speakers {len(out['train_speakers'])}, "
          f"test sentences {len(test_ids)}, train sentences {len(out['train_sentences'])}")
    print("test sentence lengths:", sorted(len(texts[k]) for k in test_ids))
    print("all lengths min/median/max:", len(texts[by_len[0]]), len(texts[by_len[50]]), len(texts[by_len[-1]]))


if __name__ == "__main__":
    main()
