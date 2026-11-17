"""比べる基準: 評価する話者の本物の録音(テスト文 25 文)の CER・UTMOS と、ボコーダの上限(本物の Mel → 多話者で追加学習した Vocos)の
CER・UTMOS・話者類似度(参照音声との類似度。metrics.py と同じ)。

使い方:
    python real_metrics.py
出力: results/metrics_real.json({"unseen"・"seen": {"real": {...}, "vocoder": {...}}})
"""
import json

import numpy as np
import soundfile as sf
import torch

import common
from common.data import list_all, load_split
from common.speaker import CAMPlus, WavLMSV
from lib_common.asr_eval import ASR, cer_scores
from lib_common.audio import SR
from lib_common.mos import UTMOS
from lib_common.neural_vocoder import load_vocoder


def main():
    sp = load_split()
    utts = {u.id: u for u in list_all()}
    refs = json.loads((common.RUNS_DIR / "eval" / "finetune" / "refs.json").read_text())
    mels = torch.load(common.RUNS_DIR / "mels.pt")
    voc = load_vocoder(common.RUNS_DIR / "vocos_multi" / "vocoder.pt", "cuda")
    asr, mos, wavlm, cam = ASR("cuda"), UTMOS("cuda"), WavLMSV("cuda"), CAMPlus("cuda")
    out_dir = common.RUNS_DIR / "eval" / "vocoder_copy"
    out = {}
    for g, speakers in [("unseen", sp["unseen_speakers"]), ("seen", sp["seen_eval_speakers"])]:
        items = [utts[f"{s}/{t}"] for s in speakers for t in sorted(sp["test_texts"])]
        texts = [u.text for u in items]
        copies = []
        for u in items:
            p = out_dir / f"{u.id}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(p, voc.generate(mels[u.id].float().cuda()[None])[0].cpu().numpy(), SR)
            copies.append(str(p))
        res = {}
        for name, paths in [("real", [u.path for u in items]), ("vocoder", copies)]:
            sims = []
            for p, u in zip(paths, items):
                r = utts[refs[u.speaker]].path
                sims.append((float(torch.nn.functional.normalize(wavlm(p), dim=0) @ torch.nn.functional.normalize(wavlm(r), dim=0)),
                             float(torch.nn.functional.normalize(cam(p), dim=0) @ torch.nn.functional.normalize(cam(r), dim=0))))
            res[name] = {"cer": cer_scores(texts, asr.transcribe(paths))["reading"], "utmos": float(np.mean(mos.score(paths))),
                         "sim_wavlm": float(np.mean([s[0] for s in sims])), "sim_campp": float(np.mean([s[1] for s in sims]))}
        out[g] = res
        print(g, res, flush=True)
    (common.EXP_DIR / "results" / "metrics_real.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
