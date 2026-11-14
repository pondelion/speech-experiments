"""exp0005 固有の処理。tts/ と indextts_bases/ を import の探索パスに追加するので、各スクリプトは `import common` の後で
`from lib_common...`、`from exp0001.common...` などと書ける。"""
import sys
from pathlib import Path

_BASES_DIR = Path(__file__).resolve().parents[2]  # common → exp0005 → indextts_bases
for _p in [_BASES_DIR.parent, _BASES_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
