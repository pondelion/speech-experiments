"""試聴用の音声: 評価(evaluate.py・evaluate_exp0005.py)で保存した音声(乱数の種 0)を MP3 にする。生成し直しはしない。

テスト文の先頭 3 文と、実用文の各カテゴリの先頭 1 文。results/samples/ に MP3 を保存し、index.json を作り直す。
評価の音声が無い条件(まだ評価していない条件)は飛ばす。

使い方:
    python make_samples.py
"""
import json

import common
from lib_common.audio import save_preview
from lib_common.data import list_jsut, split
from lib_common.eval_sets import load_eval_texts

EVAL = common.RUNS_DIR / "eval"
CONDITIONS = {  # 条件の名前 → 評価の音声のディレクトリの名前(evaluate.py の --cond と --tag)
    "gt_mel": "gt_mel",
    "tts_k8192": "tts",
    "tts_large": "tts_large",
    "tts_k1024": "tts_k1024",
    "tts_k256": "tts_k256",
    "exp0005_best": "exp0005_best",
}


def preview(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    save_preview(src, dst)


def main():
    test = split(list_jsut(), "test")[:3]
    firsts = {}
    for t in load_eval_texts(["practical"]):
        firsts.setdefault(t.category, t)
    out = common.EXP_DIR / "results" / "samples"
    index = []
    for u in test:  # 元の録音
        preview(u.path, out / "test" / "real" / f"{u.id}.mp3")
        index.append({"group": "test", "condition": "real", "id": u.id, "text": u.text, "category": "jsut_test",
                      "file": f"test/real/{u.id}.mp3"})
    for cond, src in CONDITIONS.items():
        for group, items, sub in [("test", test, "seed0"), ("eval_texts", list(firsts.values()), "practical")]:
            d = EVAL / src / sub
            for x in items:
                mp3 = out / group / cond / f"{x.id}.mp3"
                if (d / f"{x.id}.wav").exists():
                    preview(d / f"{x.id}.wav", mp3)
                if not mp3.exists():  # 評価の音声が無く、前に作った MP3 も無い条件は飛ばす(評価の wav は消してあることがある)
                    continue
                index.append({"group": group, "condition": cond, "id": x.id, "text": x.text,
                              "category": getattr(x, "category", "jsut_test"), "file": f"{group}/{cond}/{x.id}.mp3"})
    (out / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False))
    print(f"{len(index)} samples")


if __name__ == "__main__":
    main()
