"""試聴用の音声: 評価(exp0101 の synth.py)で保存した音声(乱数の種 0)を MP3 にする(生成し直しはしない)。

行・文・話者は exp0101 の試聴用の音声(exp0101/make_samples.py)と同じ。列は条件(参照音声・本物の録音・基準 B0・exp0102 の 5 条件)。
参照音声・本物の録音・B0(exp0101 の scratch)は、exp0101 の results/samples/ の MP3 をそのまま写す(exp0101 の評価の wav は残していないため)。
ずんだもんの行は、条件ごとに synth.py --ref_wav で作った音声(runs/eval/zundamon_{条件})。results/samples/ に MP3 と index.json を作る。

使い方(exp0101 のディレクトリで synth.py を動かした後):
    python synth.py ... --seeds 1 --ref_wav runs/zundamon/ref.wav --ref_name zundamon --out_root ../exp0102/runs --name zundamon_{条件}
    python ../exp0102/make_samples.py
"""
import json
import shutil
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parent / "exp0101"))

import common  # noqa: E402  (exp0101 の common)
from lib_common.audio import save_preview  # noqa: E402

CONDS = ["no_t2s_spk", "no_s2m_prompt", "no_s2m_style", "f0_per_speaker", "t2s_ref_encoder"]
COPIED = {"ref": "ref", "real": "real", "scratch": "b0"}  # exp0101 の条件 → この exp の条件の名前


def main():
    src_index = json.loads((common.EXP_DIR / "results" / "samples" / "index.json").read_text())
    ev, out = EXP_DIR / "runs" / "eval", EXP_DIR / "results" / "samples"
    index = []
    for e in src_index:
        if e["condition"] not in COPIED:
            continue
        cond = COPIED[e["condition"]]
        file = f"{e['group']}/{cond}/{Path(e['file']).name}"
        (out / file).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(common.EXP_DIR / "results" / "samples" / e["file"], out / file)
        index.append({**e, "condition": cond, "file": file})
        if e["condition"] != "scratch":
            continue
        speaker, tid = e["id"].split("(")[0], e["id"].split("/")[-1]
        sub = "test" if e["group"] == "test" else "practical"
        for c in CONDS:
            run = f"zundamon_{c}" if speaker == "zundamon" else c
            file = f"{e['group']}/{c}/{Path(e['file']).name}"
            (out / file).parent.mkdir(parents=True, exist_ok=True)
            save_preview(ev / run / "seed0" / sub / speaker / f"{tid}.wav", out / file)
            index.append({**e, "condition": c, "file": file})
    (out / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False))
    print(f"{len(index)} samples")


if __name__ == "__main__":
    main()
