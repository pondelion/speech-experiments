"""exp0002 固有の処理(非自己回帰の T2S のモデルと、それに依存する小物)。

codec・S2M・pipeline は exp0001 のものを使い回す。ここで tts/ と indextts_bases/ を import の探索パスに追加するので、
各スクリプトは `import common` の後で `from lib_common.xxx import ...`、`from exp0001.common.xxx import ...` と書ける。
"""
import sys
from pathlib import Path

_BASES_DIR = Path(__file__).resolve().parents[2]  # common → exp0002 → indextts_bases
for _p in [_BASES_DIR.parent, _BASES_DIR]:          # tts/(lib_common 用)、indextts_bases/(exp0001 用)
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
