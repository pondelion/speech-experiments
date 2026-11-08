"""学習済みの自己教師ありモデル(SSL)で、音声から特徴を取り出す。

SSL モデルは、大量の音声だけ(書き起こしなし)で「音声の一部を隠して当てる」などの学習をしたモデル。
途中の層の特徴は、発音(どの音素か)の情報を多く含み、話者・録音環境の情報は少ないことが知られている。
IndexTTS2.5 は w2v-BERT 2.0 の 17 層目の特徴を量子化して音声トークンにする(exp0001・exp0002 は Mel を量子化していた)。

取り出した特徴は SSL モデルの時間解像度(どちらも 50 フレーム/秒)のまま CPU に置き、使うときに Mel のフレーム数
(93.75 フレーム/秒)に線形補間でそろえる(to_mel_rate)。こうすると exp0001 の codec(Mel 4 フレーム = トークン 1 個)・
S2M・exp0002 の T2S(音素ごとの長さ)を、そのまま使える。

モデルは Hugging Face から transformers が既定の場所にダウンロードする。

shape の軸の記号:
    N: 16kHz の波形のサンプル数、S: SSL のフレーム数(50 フレーム/秒)、T: Mel のフレーム数、D: 特徴の次元
"""
import librosa
import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoFeatureExtractor, AutoModel

from lib_common.audio import SR, load_wave

SSL_SR = 16000  # SSL モデルの入力のサンプリング周波数
# 名前 → (Hugging Face のモデル名, 既定で使う層)
SSL_MODELS = {
    "w2vbert2": ("facebook/w2v-bert-2.0", 17),         # IndexTTS2・2.5 と同じモデル・層(多言語、24 層、1024 次元)
    "hubert_ja": ("reazon-research/japanese-hubert-base-k2", 9),  # ReazonSpeech v2.0(日本語)で事前学習した HuBERT
                                                                  # (12 層、768 次元、Apache-2.0)。9 層目は HuBERT の論文で
                                                                  # 離散単位を作るのに使った層
}


class SSLFeatures:
    """SSL モデルの指定した層の特徴を取り出す。

    Args:
        name: SSL_MODELS の名前。
        device: モデルを動かすデバイス。
        layer: 使う層(None なら SSL_MODELS の既定)。hidden_states の番号で、0 は Transformer に入る前の埋め込み。
    """

    def __init__(self, name: str, device: str, layer: int | None = None):
        hf_id, default_layer = SSL_MODELS[name]
        self.layer = default_layer if layer is None else layer
        self.device = device
        self.extractor = AutoFeatureExtractor.from_pretrained(hf_id)
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(hf_id, torch_dtype=dtype).to(device).eval()
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def __call__(self, path: str) -> torch.Tensor:
        """音声ファイル 1 つの特徴 [S, D](float16、CPU)を返す。"""
        y = librosa.resample(load_wave(path), orig_sr=SR, target_sr=SSL_SR)                   # 24kHz → 16kHz [N]
        inputs = self.extractor(y, sampling_rate=SSL_SR, return_tensors="pt")
        inputs = {k: v.to(self.device, self.model.dtype if v.is_floating_point() else v.dtype) for k, v in inputs.items()}
        h = self.model(**inputs, output_hidden_states=True).hidden_states[self.layer]         # [1, S, D]
        return h[0].half().cpu()


def to_mel_rate(feat: torch.Tensor, n_frames: int) -> torch.Tensor:
    """SSL の特徴を、線形補間で Mel のフレーム数にそろえる: [S, D] → [T, D](float32)。"""
    x = rearrange(feat.float(), "s d -> 1 d s")
    return rearrange(F.interpolate(x, size=n_frames, mode="linear", align_corners=False), "1 d t -> t d")
