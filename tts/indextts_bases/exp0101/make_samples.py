"""試聴用の音声: 評価(synth.py)で保存した音声(乱数の種 0)を MP3 にする(生成し直しはしない)。

行は「話者 × 文」、列は条件(参照音声・本物の録音・フェーズ A のモデル・scratch・finetune)。行の名前に話者の区分を書く。
話者: 学習に使わない 4 人(男女 2 人ずつ)、学習に使った 2 人(男女 1 人ずつ)、ずんだもん(VOICEVOX で作った参照音声。学習には使っていない)。
文: テスト文 2 文と実用文 2 文(ニュースと数字)。results/samples/ に MP3 と index.json を作る。

使い方:
    python synth.py ... --seeds 1 --speakers jvs006 jvs045 jvs004 jvs061 jvs021 jvs090   # 評価の wav を消したときは作り直す
    python synth.py ... --seeds 1 --ref_wav runs/zundamon/ref.wav --ref_name zundamon --name zundamon_{scratch,finetune}
    python make_samples.py
"""
import json

import common
from common.data import list_all, load_split
from lib_common.audio import save_preview
from lib_common.eval_sets import load_eval_texts

SPEAKERS = {"jvs006": "学習に使わない・男性", "jvs045": "学習に使わない・男性", "jvs004": "学習に使わない・女性",
            "jvs061": "学習に使わない・女性", "jvs021": "学習に使った・男性", "jvs090": "学習に使った・女性",
            "zundamon": "ずんだもん、VOICEVOX で作った参照音声、学習に使わない"}
PRACTICAL = ["news_01", "num_01"]
CONDS = ["phase_a", "scratch", "finetune"]


def main():
    sp = load_split()
    utts = {u.id: u for u in list_all()}
    ev = common.RUNS_DIR / "eval"
    refs = json.loads((ev / "finetune" / "refs.json").read_text())
    ref_paths = {s: utts[refs[s]].path for s in refs} | {"zundamon": str(common.RUNS_DIR / "zundamon" / "ref.wav")}
    test_ids = sorted(sp["test_texts"])[:2]
    practical = [t for t in load_eval_texts(["practical"]) if t.id in PRACTICAL]
    out = common.EXP_DIR / "results" / "samples"
    index = []

    def add(group, row_id, text, cond, src):
        name = row_id.split("(")[0] + "_" + row_id.split("/")[-1]                           # 例 jvs006_VOICEACTRESS100_011
        dst = out / group / cond / f"{name}.mp3"
        dst.parent.mkdir(parents=True, exist_ok=True)
        save_preview(src, dst)
        index.append({"group": group, "condition": cond, "id": row_id, "text": text, "file": f"{group}/{cond}/{dst.name}"})

    for s, desc in SPEAKERS.items():
        for group, items, sub in [("test", [(t, sp["test_texts"][t]) for t in test_ids], "test"),
                                  ("eval_texts", [(t.id, t.text) for t in practical], "practical")]:
            for tid, text in items:
                row = f"{s}({desc})/{tid}"
                add(group, row, text, "ref", ref_paths[s])
                if group == "test" and s != "zundamon":
                    add(group, row, text, "real", utts[f"{s}/{tid}"].path)
                for c in CONDS:
                    src = "phase_a" if c == "phase_a" else (f"zundamon_{c}" if s == "zundamon" else c)
                    add(group, row, text, c, ev / src / "seed0" / sub / ("jsut" if c == "phase_a" else s) / f"{tid}.wav")
    (out / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False))
    print(f"{len(index)} samples")


if __name__ == "__main__":
    main()
