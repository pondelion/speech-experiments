"""torchaudio の代わり(外部のコードが使う一部の機能だけ)。

この環境の torch に合う torchaudio のビルドが無いので、vendor/ の外部のコード(WavTokenizer・Spark-TTS)が
import・使用する機能だけを、torch・librosa で同じ計算になるように書く。使うときだけ sys.path の先頭に
tts/lib_common/torchaudio_shim を足す(exp0003 の common/ext_codec.py)。
使える機能: transforms.MelSpectrogram(Slaney 型の Mel のみ)、functional.resample、functional.functional._hz_to_mel・_mel_to_hz。
それ以外を呼ぶと NotImplementedError になる。
"""
from . import functional, transforms  # noqa: F401
