"""exp0004 固有の処理。tts/ と indextts_bases/ を import の探索パスに追加するので、各スクリプトは `import common` の後で
`from lib_common...`、`from exp0001.common...`、`from exp0003.common...` と書ける。"""
import sys
from pathlib import Path

_BASES_DIR = Path(__file__).resolve().parents[2]  # common → exp0004 → indextts_bases
for _p in [_BASES_DIR.parent, _BASES_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

EXP_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = EXP_DIR / "runs"  # checkpoint・ログ・評価の音声(git の対象外)
