"""結果 7(2 × 2)の追加の条件 `t2s_gpt_duration` の試聴用の音声を、既存のサンプルと同じ文で作り、results/samples/index.json に足す。

evaluate_2x2.py が保存した音声(乱数の種 0)を MP3 にする(生成し直しはしない)。既存の条件のサンプルはそのまま残す。

使い方:
    python make_samples_2x2.py
"""
import json

import common  # noqa: F401
from common.utils import EXP_DIR, RUNS_DIR
from lib_common.audio import save_preview

NAME = "t2s_gpt_duration"


def main():
    out = EXP_DIR / "results" / "samples"
    path = out / "index.json"
    index = [r for r in json.loads(path.read_text()) if r["condition"] != NAME]
    ref = {(r["group"], r["id"]): r for r in index if r["condition"] == "t2s_hybrid_nar"}  # 同じ文を使う
    for (group, uid), r in ref.items():
        dst = out / group / NAME / f"{uid}.mp3"
        dst.parent.mkdir(parents=True, exist_ok=True)
        save_preview(RUNS_DIR / "eval" / "2x2" / NAME / "seed0" / group / f"{uid}.wav", dst)
        index.append({**r, "condition": NAME, "file": f"{group}/{NAME}/{uid}.mp3"})
    path.write_text(json.dumps(index, indent=2, ensure_ascii=False))
    print(f"added {len(ref)} samples")


if __name__ == "__main__":
    main()
