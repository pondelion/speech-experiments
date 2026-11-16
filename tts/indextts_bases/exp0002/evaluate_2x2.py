"""追加検証: 「長さの予測」×「T2S の形」の 2 × 2 の条件を、analyze_inference.py の (1) と同じ手順で測る。

    |                   | 長さの予測なし                    | 長さの予測あり                             |
    | 自己回帰(GPT-2) | exp0001 の `ar/t2s_ph_k256`        | `t2s_gpt_duration`(この追加検証で学習)    |
    | MaskGCT 系         | `t2s_masked`                       | `t2s_hybrid_nar`(ハイブリッド型)          |
手順: 同じ codec・S2M(exp0001 の codebook 256)・Griffin-Lim、乱数の種 0・1・2、テスト 100 文と実用文 55 文の読み CER、
テスト文の長さの比(正解のトークン数との比)。どの T2S も、それぞれの既定の生成の設定で作る。
MaskGCT 系の 2 つは analyze_inference.py で同じ手順の値がある(results/inference_compare.json)ので、自己回帰の 2 つだけを測る。

使い方:
    python evaluate_2x2.py --name ar/t2s_ph_k256
    python evaluate_2x2.py --name t2s_gpt_duration
出力: results/2x2.json に追記。音声は runs/eval/2x2/(git の対象外)
"""
import argparse
import json

import soundfile as sf
import torch

import common  # noqa: F401  tts/lib_common と exp0001 を import できるようにする
from common.utils import BASE_CODEC, BASE_S2M, EXP1_RUNS, EXP_DIR, RUNS_DIR, load_t2s
from exp0001.common.pipeline import TTSPipeline
from exp0001.common.pipeline import load_t2s as load_ar_t2s
from exp0001.common.utils import encode_all
from lib_common.asr_eval import ASR, cer_scores
from lib_common.audio import SR
from lib_common.data import load_dataset, split
from lib_common.eval_sets import load_eval_texts
from lib_common.utils import Logger, seed_everything

BATCH = 16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="ar/{exp0001 の T2S の名前} か、runs/{name}/t2s.pt")
    name = parser.parse_args().name
    torch.set_num_threads(4)
    log = Logger(RUNS_DIR / "eval" / "log_2x2.txt")
    test = split(load_dataset(None, log), "test")
    eval_texts = load_eval_texts(["practical"])
    pipe = TTSPipeline.from_checkpoints(BASE_CODEC, BASE_S2M, device="cuda")
    gt = encode_all(pipe.codec, [pipe.normalizer.normalize(u.mel) for u in test], "cuda")
    if name.startswith("ar/"):
        pipe.t2s, pipe.vocab = load_ar_t2s(EXP1_RUNS / name[3:] / "t2s.pt", "cuda")
    else:
        model, vocab, _ = load_t2s(RUNS_DIR / name / "t2s.pt", "cuda")
    asr = ASR("cuda")

    def codes_for(texts: list[str]) -> list[torch.Tensor]:
        if name.startswith("ar/"):
            return pipe.text_to_codes(texts)
        out = []
        for s in range(0, len(texts), BATCH):
            ids = [torch.tensor(vocab.encode(t)) for t in texts[s: s + BATCH]]
            out += model.generate(ids, durations=model.predict_durations(ids))
        return out

    res = []
    for seed in range(3):
        r = {}
        for group, items in [("test", test), ("eval_texts", eval_texts)]:
            seed_everything(seed)
            cs = codes_for([x.text for x in items])
            d = RUNS_DIR / "eval" / "2x2" / name.replace("/", "_") / f"seed{seed}" / group
            d.mkdir(parents=True, exist_ok=True)
            for x, m in zip(items, pipe.codes_to_mel(cs)):
                sf.write(d / f"{x.id}.wav", pipe.mel_to_wave(m).numpy(), SR)
            r[group] = cer_scores([x.text for x in items], asr.transcribe([str(d / f"{x.id}.wav") for x in items]))["reading"]
            if group == "test":
                r["length_ratio"] = sum(len(c) / len(g) for c, g in zip(cs, gt)) / len(cs)
        log(f"{name} seed {seed} " + " ".join(f"{k} {v:.4f}" for k, v in r.items()))
        res.append(r)
    path = EXP_DIR / "results" / "2x2.json"
    allres = json.loads(path.read_text()) if path.exists() else {}
    allres[name] = res
    path.write_text(json.dumps(allres, indent=2))


if __name__ == "__main__":
    main()
