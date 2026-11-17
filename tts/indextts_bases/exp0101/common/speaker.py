"""話者ベクトルを作る 2 つの学習済みモデル(どちらも学習はしない)。

CAM++(モデルの部品): 公式(IndexTTS-2.5)と同じ。重みは HuggingFace の funasr/campplus(vendor/campplus)、ネットワークの定義は公式の
    リポジトリのもの(vendor/index-tts)。入力は 16kHz の Kaldi 形式の 80 帯域の fbank から時間方向の平均を引いたもの(公式と同じ)
WavLM(評価): microsoft/wavlm-base-plus-sv(transformers の WavLMForXVector)。モデルの部品とは別のモデルで話者類似度を測るため
"""
import sys

import kaldi_native_fbank as knf
import librosa
import numpy as np
import torch

from . import EXP_DIR

VENDOR = EXP_DIR.parents[2] / "vendor"


class CAMPlus:
    def __init__(self, device: str = "cpu"):
        sys.path.insert(0, str(VENDOR / "index-tts"))
        from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
        self.model = CAMPPlus(feat_dim=80, embedding_size=192)
        self.model.load_state_dict(torch.load(VENDOR / "campplus" / "campplus_cn_common.bin", map_location="cpu"))
        self.model.to(device).eval()
        self.device = device
        self.opts = knf.FbankOptions()
        self.opts.frame_opts.dither = 0
        self.opts.frame_opts.samp_freq = 16000
        self.opts.mel_opts.num_bins = 80

    @torch.no_grad()
    def __call__(self, path: str) -> torch.Tensor:
        """音声ファイル → 話者ベクトル [192](CPU)。"""
        y, _ = librosa.load(path, sr=16000)
        fb = knf.OnlineFbank(self.opts)
        fb.accept_waveform(16000, (y * 32768).tolist())
        fb.input_finished()
        x = torch.tensor(np.stack([fb.get_frame(k) for k in range(fb.num_frames_ready)]))
        return self.model((x - x.mean(0, keepdim=True))[None].to(self.device))[0].cpu()


class WavLMSV:
    def __init__(self, device: str = "cuda"):
        from transformers import AutoFeatureExtractor, WavLMForXVector
        self.fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        self.model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, path: str) -> torch.Tensor:
        """音声ファイル → 話者ベクトル [512](CPU)。"""
        y, _ = librosa.load(path, sr=16000)
        x = {k: v.to(self.device) for k, v in self.fe(y, sampling_rate=16000, return_tensors="pt").items()}
        return self.model(**x).embeddings[0].cpu()
