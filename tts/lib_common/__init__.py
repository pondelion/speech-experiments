"""TTS の実験全体で共通に使う処理(音声・Mel、テキストの前処理、データセット、ボコーダ、評価、学習の小物)。

特定の実験のモデル設計に依存するもの(codec・T2S・S2M など)は、各実験の common/ に置く。
各実験からは、tts/ を import の探索パスに入れたうえで `from lib_common.audio import ...` のように使う
(各実験の common/__init__.py がパスを追加する)。
"""
