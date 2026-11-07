"""音声の読み込みと log-Mel スペクトログラムの計算。

TTS では波形を直接作らず、まず「Mel スペクトログラム」(以下 Mel)を作ってから波形に戻すのが一般的。
Mel は、短い区間(フレーム)ごとに周波数成分の強さを求め、人の聴覚に合わせて低音を細かく・高音を粗く
80 個の帯域(bin)にまとめたもの。波形(24,000 点/秒)よりずっと少ない数値(93.75 フレーム/秒 × 80)で
音の中身(何をどんな声で言っているか)を表せるので、モデルが扱いやすい。

Mel の設定は codec・S2M・Griffin-Lim の全部で共通にする必要がある(設定が違うと別物の Mel になる)。
そのため Mel の定義はこのファイルだけに置く。

shape の軸の記号:
    N: 波形のサンプル数(24,000 点/秒)
    T: フレーム数(93.75 フレーム/秒。N / HOP + 1)
    Fq: STFT の周波数の数(N_FFT/2 + 1 = 513)
    F: Mel の帯域数(80)
"""
import librosa
import numpy as np
import soundfile as sf
import torch
from einops import rearrange

SR = 24000          # サンプリング周波数(1 秒あたりの波形の点数)
N_FFT = 1024        # 1 フレームの窓の長さ(42.7ms)
HOP = 256           # フレームをずらす幅(10.7ms)。1 秒 = 93.75 フレーム
N_MELS = 80         # Mel の帯域の数
FMIN, FMAX = 0, 12000  # Mel で扱う周波数の範囲(12kHz = 24kHz サンプリングで表せる上限)
LOG_FLOOR = 1e-5    # log を取るときの下限(無音で log(0) にならないように)


def load_wave(path: str) -> np.ndarray:
    """wav を読み、モノラル・24kHz の波形にする。

    Args:
        path: wav ファイルのパス。

    Returns:
        波形 [サンプル数](float32、値の範囲はおおよそ -1〜1)。
    """
    y, sr = sf.read(path, dtype="float32", always_2d=True)  # [N, チャンネル数]
    y = y.mean(axis=1)  # ステレオなら左右の平均でモノラルにする [N, チャンネル数] → [N]
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y


def mel_basis() -> np.ndarray:
    """STFT の周波数(513 個)を Mel の 80 帯域にまとめる重み行列を返す。

    Returns:
        Mel フィルタ [80, N_FFT/2+1]。
    """
    return librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)


_MEL_BASIS = torch.from_numpy(mel_basis())
_WINDOW = torch.hann_window(N_FFT)


def wave_to_logmel(y: np.ndarray) -> torch.Tensor:
    """波形から log-Mel を作る。

    手順: STFT(フレームごとの周波数分解)→ 振幅だけ取る(位相は捨てる)→ Mel フィルタで 80 帯域に
    まとめる → log を取る(音の大きさの感じ方は対数に近いので、log にすると値の範囲も扱いやすくなる)。
    librosa.stft と同じ設定(Hann 窓・center・端は 0 詰め)を torch で計算する(librosa より約 13 倍速い)。

    Args:
        y: 24kHz の波形 [N(サンプル数)]。

    Returns:
        log-Mel [T(フレーム数), F(Mel の帯域数 = 80)]。
        (以下のコメントの記号: Fq STFT の周波数の数 = 513)
    """
    spec = torch.stft(torch.from_numpy(y), N_FFT, HOP, window=_WINDOW, center=True,
                      pad_mode="constant", return_complex=True).abs()        # 波形 [N] → 振幅 [Fq, T]
    mel = _MEL_BASIS @ spec                                                  # [F, Fq] @ [Fq, T] → [F, T]
    return rearrange(torch.log(torch.clamp(mel, min=LOG_FLOOR)), "f t -> t f").contiguous()  # log をとって [T, F] に


class MelNormalizer:
    """Mel を帯域ごとに平均 0・標準偏差 1 に正規化する。

    Flow Matching はノイズ N(0, I) から Mel へ流す。Mel の値の範囲をノイズと同程度にそろえると学習しやすい。
    平均・標準偏差は学習データから求め、推論でも同じ値を使う(checkpoint に保存する)。

    Args:
        mean: 帯域ごとの平均 [80]。
        std: 帯域ごとの標準偏差 [80]。
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean, self.std = mean, std

    @classmethod
    def fit(cls, mels: list[torch.Tensor]) -> "MelNormalizer":
        """学習データの Mel 全フレームから平均・標準偏差を求める。

        Args:
            mels: log-Mel [T(フレーム数), 80(Mel の帯域)] のリスト。

        Returns:
            求めた値を持つ MelNormalizer。
        """
        frames = torch.cat(mels, dim=0)            # 全発話のフレームを縦につなぐ [T_1 + T_2 + ..., F]
        return cls(frames.mean(0), frames.std(0))  # 帯域ごとの平均・標準偏差 [F]

    def normalize(self, logmel: torch.Tensor) -> torch.Tensor:
        """log-Mel [..., 80(Mel の帯域)] → 正規化した Mel [..., 80]。... は任意の先頭の軸(フレームなど)。"""
        return (logmel - self.mean.to(logmel.device)) / self.std.to(logmel.device)

    def denormalize(self, mel: torch.Tensor) -> torch.Tensor:
        """正規化した Mel [..., 80(Mel の帯域)] → log-Mel [..., 80]。... は任意の先頭の軸(フレームなど)。"""
        return mel * self.std.to(mel.device) + self.mean.to(mel.device)

    def state_dict(self) -> dict:
        """checkpoint に保存する値。"""
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, d: dict) -> "MelNormalizer":
        """state_dict() で保存した値から復元する。"""
        return cls(d["mean"], d["std"])
