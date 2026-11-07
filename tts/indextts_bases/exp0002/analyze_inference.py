"""分析: 学習し直さずに、(1) 測り直しのばらつき、(2) ハイブリッド型の推論の設定、(3) 自己回帰の長さの決め方、を比べる。

    (1) seeds:     主な条件を、乱数の種 0・1・2 で 3 回測る(T2S のサンプリングと S2M のノイズが変わる)。
                   条件どうしの差が、測り直しのばらつきより大きいかを見る
    (2) steps_cfg: ハイブリッド型(t2s_hybrid_nar)の埋める回数 n_steps(1・5・10・25)と CFG の強さ(0・1)を変える
                   (既定は n_steps 50・CFG 2.5 で、(1) の値と比べる)。n_steps 1 は全位置を 1 回で決める
    (3) ar_decode: ハイブリッド型(t2s_hybrid_ar)の長さの決め方を、一番確率の高い値(既定)から
                   期待値(確率で重み付けした平均)・サンプリングに変える
各条件は乱数の種を決めてから合成するので、どの条件も単独で同じ結果を再現できる(evaluate.py の値とは乱数の使われ方が違う)。

使い方:
    python analyze_inference.py
出力: results/inference_compare.json(条件ごとの読み CER と長さの比)。音声は runs/eval/inference/(git の対象外)
"""
import json

import soundfile as sf
import torch

import common  # noqa: F401  tts/lib_common と exp0001 を import できるようにする
from common.utils import BASE_CODEC, BASE_S2M, EXP1_RUNS, EXP_DIR, PROSODY_MARKS, RUNS_DIR, load_t2s
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
    torch.set_num_threads(4)
    log = Logger(RUNS_DIR / "eval" / "log_inference.txt")
    test = split(load_dataset(None, log), "test")
    eval_texts = load_eval_texts(["practical"])
    pipe = TTSPipeline.from_checkpoints(BASE_CODEC, BASE_S2M, device="cuda")
    gt = encode_all(pipe.codec, [pipe.normalizer.normalize(u.mel) for u in test], "cuda")
    models = {name: load_t2s(RUNS_DIR / name / "t2s.pt", "cuda") for name in ["t2s_masked", "t2s_hybrid_nar", "t2s_hybrid_ar"]}
    asr = ASR("cuda")

    def codes_for(name: str, texts: list[str], gen: dict, ar_decode: str) -> list[torch.Tensor]:
        """条件 name の T2S で音声トークン列を作る(gen は MaskedT2S.generate() の設定)。"""
        if name == "s2m_gt_tokens":
            return gt
        if name.startswith("ar/"):
            pipe.t2s, pipe.vocab = load_ar_t2s(EXP1_RUNS / name[3:] / "t2s.pt", "cuda")
            return pipe.text_to_codes(texts)
        model, vocab, method = models[name]
        out = []
        for s in range(0, len(texts), BATCH):
            ids = [torch.tensor(vocab.encode(t)) for t in texts[s: s + BATCH]]
            if method == "hybrid":
                out += model.generate(ids, durations=model.predict_durations(ids, ar_decode), **gen)
            else:
                n_phones = [len([t for t in vocab.tokenize(x) if t not in PROSODY_MARKS]) for x in texts[s: s + BATCH]]
                out += model.generate(ids, model.rule_lengths(n_phones), **gen)
        return out

    def measure(key: str, name: str, seed: int, with_eval_texts: bool, gen: dict | None = None,
                ar_decode: str = "argmax") -> dict:
        """乱数の種を決めて合成し、読み CER(とテスト文の長さの比)を測る。"""
        res = {}
        groups = [("test", test)] + ([("eval_texts", eval_texts)] if with_eval_texts else [])
        for group, items in groups:
            if name == "s2m_gt_tokens" and group == "eval_texts":
                continue
            seed_everything(seed)
            cs = codes_for(name, [x.text for x in items], gen or {}, ar_decode)
            d = RUNS_DIR / "eval" / "inference" / key / group
            d.mkdir(parents=True, exist_ok=True)
            for x, m in zip(items, pipe.codes_to_mel(cs)):
                sf.write(d / f"{x.id}.wav", pipe.mel_to_wave(m).numpy(), SR)
            r = cer_scores([x.text for x in items], asr.transcribe([str(d / f"{x.id}.wav") for x in items]))
            res[group] = r["reading"]
            if group == "test":
                res["length_ratio"] = sum(len(c) / len(g) for c, g in zip(cs, gt)) / len(cs)
        log(f"{key:40s} " + " ".join(f"{k} {v:.4f}" for k, v in res.items()))
        return res

    out_path = EXP_DIR / "results" / "inference_compare.json"
    out = {"seeds": {}, "steps_cfg": {}, "ar_decode": {}}
    # (1) 測り直しのばらつき
    for name in ["s2m_gt_tokens", "ar/t2s_ph_k256_aux", "t2s_masked", "t2s_hybrid_nar", "t2s_hybrid_ar"]:
        out["seeds"][name] = [measure(f"seeds/{name}/{s}", name, s, with_eval_texts=True) for s in range(3)]
        out_path.write_text(json.dumps(out, indent=2))
    # (2) ハイブリッド型の埋める回数と CFG の強さ(テスト文のみ)
    for n_steps, cfg in [(1, 2.5), (5, 2.5), (10, 2.5), (25, 2.5), (50, 0.0), (50, 1.0)]:
        key = f"n_steps{n_steps}_cfg{cfg}"
        out["steps_cfg"][key] = measure(f"steps_cfg/{key}", "t2s_hybrid_nar", 0, False, {"n_steps": n_steps, "cfg": cfg})
        out_path.write_text(json.dumps(out, indent=2))
    # (3) 自己回帰の長さの決め方
    for decode in ["mean", "sample"]:
        out["ar_decode"][decode] = measure(f"ar_decode/{decode}", "t2s_hybrid_ar", 0, True, ar_decode=decode)
        out_path.write_text(json.dumps(out, indent=2))
    log(f"saved {out_path.name}")


if __name__ == "__main__":
    main()
