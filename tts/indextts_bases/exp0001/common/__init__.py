"""exp0001 固有の処理(codec・T2S・S2M のモデルと、それに依存する小物)。

実験に依存しない共通処理は tts/lib_common/ にある。ここで tts/ を import の探索パスに追加するので、
各スクリプトは `import common` の後で `from lib_common.xxx import ...` と書ける。
"""
import sys
from pathlib import Path

_TTS_DIR = Path(__file__).resolve().parents[3]  # common → exp0001 → indextts_bases → tts
if str(_TTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TTS_DIR))
