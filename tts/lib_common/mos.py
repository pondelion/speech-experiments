"""音声の自然さの評価。人が付ける 5 段階の自然さの点数(MOS)を、学習済みのモデル UTMOS22 で推定する。

CER は「言っている内容が正しく聞き取れるか」しか測らないので、音の細部(倍音のくっきりさ、雑音、歪み)の違いは
こちらで見る。正解の音声は要らない。TTS の論文(F5-TTS、CosyVoice など)でよく使われる。
モデルは SpeechMOS(https://github.com/tarepan/SpeechMOS)の utmos22_strong を torch.hub で読み込む。
UTMOS22 は主に英語の音声で学習されているので、日本語での絶対値はあてにせず、条件どうしの相対比較に使う。

SpeechMOS は 16kHz への変換にだけ torchaudio を使うが、この環境の torch に合う torchaudio のビルドが無い。
そこで 16kHz への変換は librosa で先に済ませ、torchaudio の import だけを、変換しない代わりの関数で満たす。
"""
import importlib.machinery
import sys
import types

import librosa
import numpy as np
import torch

UTMOS_SR = 16000  # UTMOS22 の入力のサンプリング周波数


def _torchaudio_stub():
    """SpeechMOS が使う torchaudio.functional.resample だけを持つ代わりのモジュールを登録する。

    呼ばれるのは 16kHz → 16kHz(変換なし)の場合だけにするので、入力をそのまま返す。
    """
    if "torchaudio" in sys.modules:
        return

    def resample(wave, orig_freq, new_freq):
        assert orig_freq == new_freq, "16kHz への変換は呼び出し側で済ませる"
        return wave

    ta = types.ModuleType("torchaudio")
    ta.__spec__ = importlib.machinery.ModuleSpec("torchaudio", None)  # torch.hub が依存の有無をこれで確かめる
    ta.functional = types.SimpleNamespace(resample=resample)
    sys.modules["torchaudio"] = ta


class UTMOS:
    """UTMOS22 で音声ファイルの MOS(1〜5、高いほど自然)を推定する。

    Args:
        device: 推論に使うデバイス。
    """

    def __init__(self, device: str = "cuda"):
        _torchaudio_stub()
        self.model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device).eval()
        self.device = device

    @torch.no_grad()
    def score(self, paths: list[str]) -> list[float]:
        """音声ファイルごとの MOS を推定する(長さが発話ごとに違うので 1 本ずつ通す)。

        Args:
            paths: 音声ファイルのパスのリスト。

        Returns:
            ファイルごとの MOS のリスト。
        """
        scores = []
        for p in paths:
            y, _ = librosa.load(p, sr=UTMOS_SR)  # 16kHz に変換して読む。[T]
            wave = torch.from_numpy(y.astype(np.float32))[None].to(self.device)  # [1, T](1 = バッチ)
            scores.append(self.model(wave, UTMOS_SR).item())
        return scores
