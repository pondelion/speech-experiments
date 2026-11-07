"""学習済みモデルで、好きな文章を読み上げて wav に保存する。

使い方:
    python synthesize.py --text "今日はいい天気ですね。"
    python synthesize.py --text "今日はいい天気ですね。" --text_kind char --out runs/synth/char.wav
出力: --out の wav(24kHz、モノラル)
"""
import argparse
from pathlib import Path

import soundfile as sf
import torch

import common  # noqa: F401  tts/lib_common を import できるようにする
from common.pipeline import TTSPipeline
from common.utils import RUNS_DIR
from lib_common.audio import SR
from lib_common.utils import seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="読み上げる文章")
    parser.add_argument("--text_kind", choices=["char", "phoneme"], default="phoneme",
                        help="どちらの入力形式で学習した T2S を使うか")
    parser.add_argument("--out", default=str(RUNS_DIR / "synth" / "out.wav"))
    parser.add_argument("--n_steps", type=int, default=25, help="S2M の Euler 法のステップ数")
    parser.add_argument("--cfg", type=float, default=0.7, help="S2M の CFG の強さ")
    parser.add_argument("--temperature", type=float, default=0.8, help="T2S のサンプリングの温度")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    pipe = TTSPipeline.from_checkpoints(RUNS_DIR / "codec" / "codec.pt", RUNS_DIR / "s2m" / "s2m.pt",
                                        RUNS_DIR / f"t2s_{args.text_kind}" / "t2s.pt", device=args.device)
    out = pipe([args.text], n_steps=args.n_steps, cfg=args.cfg, temperature=args.temperature)[0]
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, out["wave"].numpy(), SR)
    print(f"tokens {len(out['codes'])}, {len(out['wave']) / SR:.2f} s → {path}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
