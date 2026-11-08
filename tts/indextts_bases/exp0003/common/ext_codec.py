"""学習済みの外部の離散音声コーデック(XCodec2・BiCodec・WavTokenizer)を、同じ形で使えるようにする。

どれもトークン列が 1 本(1 時刻に 1 個のトークン)なので、exp0001・exp0002 の T2S でそのまま予測できる。
自作の codec と違い、トークンの時間解像度はコーデックごとに決まっている(token_rate)。

    XCodec2(HKUST、transformers の Xcodec2Model): 16kHz、50 個/秒、65,536 種類(FSQ)。w2v-BERT 2.0 の意味の特徴と音の特徴を
        合わせて量子化する。重みのライセンスは CC-BY-NC-4.0
    BiCodec(Spark-TTS、vendor/Spark-TTS): 16kHz、50 個/秒、8,192 種類の「意味のトークン」+ 発話ごとに 32 個の「global トークン」
        (話者・録音の特徴)。T2S が予測するのは意味のトークンだけで、global トークンは学習データの 1 発話(set_reference)から
        取ったものを全ての合成で使う(JSUT は話者が 1 人なので)。重みのライセンスは CC-BY-NC-SA-4.0
    WavTokenizer(vendor/WavTokenizer、large-speech-75token): 24kHz、75 個/秒、4,096 種類。意味の特徴を使わず、
        音を直接量子化する。MIT ライセンス

コーデックごとのモデルは Hugging Face の既定の場所にダウンロードする。外部のコードは vendor/ に置く(README を参照)。
vendor/ のコードが使う torchaudio は、tts/lib_common/torchaudio_shim の代わりのモジュールで満たす。
"""
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

from lib_common.audio import SR, load_wave

_REPO = Path(__file__).resolve().parents[4]                 # common → exp0003 → indextts_bases → tts → リポジトリ
VENDOR = _REPO / "vendor"
_SHIM = _REPO / "tts" / "lib_common" / "torchaudio_shim"


def _add_path(p: Path):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


class ExtCodec:
    """外部のコーデックの共通の形。

    Attributes:
        sr: コーデックの入力・出力のサンプリング周波数。
        token_rate: 1 秒あたりのトークン数。
        codebook_size: トークンの種類数。
    """
    sr: int
    token_rate: float
    codebook_size: int

    def encode(self, path: str) -> torch.Tensor:
        """音声ファイル → トークン列 [U](CPU)。"""
        raise NotImplementedError

    def decode(self, ids: torch.Tensor) -> np.ndarray:
        """トークン列 [U] → 24kHz の波形 [N]。"""
        raise NotImplementedError

    def _wave(self, path: str) -> np.ndarray:
        """音声ファイルを、コーデックのサンプリング周波数の波形にする。"""
        y = load_wave(path)  # 24kHz
        return y if self.sr == SR else librosa.resample(y, orig_sr=SR, target_sr=self.sr)

    def _to_24k(self, y: np.ndarray) -> np.ndarray:
        return y if self.sr == SR else librosa.resample(y, orig_sr=self.sr, target_sr=SR)


class XCodec2(ExtCodec):
    sr, token_rate, codebook_size = 16000, 50.0, 65536

    HOP = 320  # 16kHz で 50 個/秒

    def __init__(self, device: str):
        from transformers import SeamlessM4TFeatureExtractor, Xcodec2Model
        self.device = device
        # transformers の Xcodec2FeatureExtractor は torchaudio が要るので、元の XCodec2 の実装と同じく
        # w2v-BERT 2.0 の前処理(SeamlessM4TFeatureExtractor、numpy で計算)で意味の特徴の入力を作る
        self.extractor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        # 重みは Hugging Face(HKUSTAudio/xcodec2-hf)から vendor/xcodec2-hf に置く(この環境では Python からのダウンロードが
        # 失敗したので curl で取得した。README を参照)
        self.model = Xcodec2Model.from_pretrained(VENDOR / "xcodec2-hf").to(device).eval()

    @torch.no_grad()
    def encode(self, path: str) -> torch.Tensor:
        y = np.pad(self._wave(path), (0, 1))
        y = np.pad(y, (0, -len(y) % self.HOP))                              # HOP の倍数の長さにする
        feats = self.extractor(np.pad(y, (self.HOP // 2, self.HOP // 2)), sampling_rate=self.sr,
                               return_tensors="pt")["input_features"]       # [1, フレーム数 / 2, 160]
        wave = torch.from_numpy(y).float().view(1, 1, -1)
        out = self.model.encode(input_values=wave.to(self.device), input_features=feats.to(self.device))
        return out.audio_codes[0, 0].cpu()                                  # [B, 1, U] → [U]

    @torch.no_grad()
    def decode(self, ids: torch.Tensor) -> np.ndarray:
        y = self.model.decode(audio_codes=ids.view(1, 1, -1).to(self.device)).audio_values
        return self._to_24k(y.reshape(-1).float().cpu().numpy())


class BiCodec(ExtCodec):
    sr, token_rate, codebook_size = 16000, 50.0, 8192

    def __init__(self, device: str):
        from huggingface_hub import snapshot_download
        _add_path(_SHIM)
        _add_path(VENDOR / "Spark-TTS")
        from sparktts.models.audio_tokenizer import BiCodecTokenizer
        model_dir = snapshot_download("SparkAudio/Spark-TTS-0.5B",
                                      allow_patterns=["config.yaml", "BiCodec/*", "wav2vec2-large-xlsr-53/*"])
        self.tok = BiCodecTokenizer(Path(model_dir), device=torch.device(device))
        self.global_tokens = None

    def set_reference(self, path: str):
        """合成に使う global トークン(話者・録音の特徴)を、この音声から取る。"""
        g, _ = self.tok.tokenize(path)
        self.global_tokens = g.reshape(1, -1)  # detokenize は [発話数, 32] を受け取る

    @torch.no_grad()
    def encode(self, path: str) -> torch.Tensor:
        _, semantic = self.tok.tokenize(path)                                # [1, U]
        return semantic.reshape(-1).cpu()

    @torch.no_grad()
    def decode(self, ids: torch.Tensor) -> np.ndarray:
        assert self.global_tokens is not None, "先に set_reference() で global トークンを決める"
        return self._to_24k(self.tok.detokenize(self.global_tokens, ids.view(1, -1).to(self.tok.device)))


class WavTokenizer(ExtCodec):
    sr, token_rate, codebook_size = 24000, 75.0, 4096

    def __init__(self, device: str):
        from huggingface_hub import hf_hub_download
        _add_path(_SHIM)
        _add_path(VENDOR / "WavTokenizer")
        from decoder.pretrained import WavTokenizer as _WT
        ckpt = hf_hub_download("novateur/WavTokenizer-large-speech-75token", "wavtokenizer_large_speech_320_v2.ckpt")
        config = VENDOR / "WavTokenizer" / "configs" / "wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
        self.device = device
        self.model = _WT.from_pretrained0802(str(config), ckpt).to(device).eval()
        self.bandwidth = torch.tensor([0], device=device)

    @torch.no_grad()
    def encode(self, path: str) -> torch.Tensor:
        wav = torch.from_numpy(self._wave(path)).float().view(1, -1).to(self.device)
        _, codes = self.model.encode_infer(wav, bandwidth_id=self.bandwidth)   # [nq = 1, 1, U]
        return codes.reshape(-1).cpu()

    @torch.no_grad()
    def decode(self, ids: torch.Tensor) -> np.ndarray:
        feats = self.model.codes_to_features(ids.view(1, 1, -1).to(self.device))
        return self.model.decode(feats, bandwidth_id=self.bandwidth).reshape(-1).float().cpu().numpy()


EXT_CODECS = {"xcodec2": XCodec2, "bicodec": BiCodec, "wavtokenizer": WavTokenizer}
