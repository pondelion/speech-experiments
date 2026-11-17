"""ボコーダの追加学習に使う wav の一覧(学習データ・検証データ。common/data.py の train・val)を runs/vocoder_wavs.json に書く。

ボコーダは、比べる 2 つの条件(scratch・finetune)で共通の部品。exp0004 の Vocos(JSUT の 1 話者)から、学習に使う話者の音声で追加学習する:
    python make_vocoder_list.py
    cd ../exp0004 && python train_vocoder.py --model vocos --steps 30000 --name vocos_multi --runs_dir ../exp0101/runs \
        --wav_list ../exp0101/runs/vocoder_wavs.json --init runs/vocos/last.pt
"""
import json

import common
from common.data import list_all


def main():
    utts = list_all()
    out = {s: [u.path for u in utts if u.split == s] for s in ("train", "val")}
    (common.RUNS_DIR / "vocoder_wavs.json").write_text(json.dumps(out, indent=1))
    print({k: len(v) for k, v in out.items()})


if __name__ == "__main__":
    main()
