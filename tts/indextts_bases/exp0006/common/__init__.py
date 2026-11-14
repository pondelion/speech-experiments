"""exp0006 固有の処理。

- 音声の設定を IndexTTS-2.5 と同じ 22.05kHz(lib_common.audio の indextts22k)にする。lib_common を import する前に決める必要があるので、
  各スクリプトは最初に `import common` する
- tts/ と indextts_bases/ を import の探索パスに追加する(`from lib_common...`、`from exp0005.common...` と書ける)
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TTS_AUDIO_PROFILE", "indextts22k")  # exp0005 の構成を測り直す evaluate_exp0005.py だけは jsut24k にする

_BASES_DIR = Path(__file__).resolve().parents[2]  # common → exp0006 → indextts_bases
for _p in [_BASES_DIR.parent, _BASES_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

EXP_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = EXP_DIR / "runs"  # checkpoint・ログ・評価の音声(git の対象外)
