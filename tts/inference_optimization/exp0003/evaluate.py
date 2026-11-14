"""CPU S2M動的INT8の対比較。T2S出力と生成ノイズを固定する。"""
import copy
import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "exp0001"))
from benchmark import build_models
from quantize import dynamic_int8


def errors(a, b):
    delta = (a - b).float()
    return {"max_abs": delta.abs().max().item(), "rmse": delta.square().mean().sqrt().item(),
            "relative_l2": (delta.norm() / a.float().norm().clamp_min(1e-12)).item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t2s-int8", action="store_true", help="B段階:T2S INT8と両方INT8を比較")
    args = parser.parse_args()
    import soundfile as sf
    torch, pipe, t2s, vocab, seed = build_models("cpu")
    torch.set_num_threads(8)
    labels = ["cpu_fp32", "cpu_s2m_int8"]
    output_name = "quality"
    if args.t2s_int8:
        t2s = dynamic_int8(torch, t2s, exclude={"head"})
        labels = ["cpu_t2s_int8", "cpu_both_int8"]
        output_name = "quality_b"
    from lib_common.data import list_jsut, split
    from lib_common.eval_sets import load_eval_texts
    from lib_common.asr_eval import ASR, cer_scores
    from lib_common.audio import SR

    fp32 = pipe.s2m
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        int8 = torch.ao.quantization.quantize_dynamic(copy.deepcopy(fp32), {torch.nn.Linear},
                                                     dtype=torch.qint8, inplace=True)
    names = [name for name, m in int8.named_modules() if isinstance(m, torch.ao.nn.quantized.dynamic.Linear)]
    weights = [m.weight() for m in int8.modules() if isinstance(m, torch.ao.nn.quantized.dynamic.Linear)]
    result = {"seed": 0, "threads": 8, "t2s_batch": 16, "s2m_batch": 4,
              "t2s_int8": args.t2s_int8,
              "t2s_excluded_layers": ["head"] if args.t2s_int8 else [],
              "backend": torch.backends.quantized.engine, "quantized_layers": names,
              "weight_qschemes": sorted({str(w.qscheme()) for w in weights}),
              "weight_dtype": "qint8", "activation": "dynamic quantization inside Linear",
              "groups": {}}
    # 部品単体は固定乱数入力。許容誤差の合否ではなく誤差量を記録する。
    seed(0)
    x = torch.randn(2, 508, 80)
    cond = torch.randn(2, 508, fp32.dim)
    cond[1].zero_()
    pad = torch.zeros(2, 508, dtype=torch.bool)
    t = torch.full((2,), 0.5)
    with torch.inference_mode():
        result["synthetic_velocity_error"] = errors(fp32.velocity(x, cond, t, pad), int8.velocity(x, cond, t, pad))

    groups = {"test": split(list_jsut(), "test"), "eval_texts": load_eval_texts(["practical"])}
    audio = {}
    for group, items in groups.items():
        seed(0)
        codes = []
        with torch.inference_mode():
            for start in range(0, len(items), 16):
                ids = [torch.tensor(vocab.encode(u.text)) for u in items[start:start + 16]]
                codes += t2s.generate(ids, durations=t2s.predict_durations(ids))
                print(f"T2S {group}: {min(start + 16, len(items))}/{len(items)}", flush=True)
        digest = hashlib.sha256(b"".join(c.numpy().tobytes() for c in codes)).hexdigest()
        group_result = {"utterances": len(items), "shared_codes_sha256": digest, "conditions": {}}
        audio[group] = {}
        reference = []
        for label, model in zip(labels, [fp32, int8]):
            pipe.s2m = model
            seed(0)
            mels = []
            with torch.inference_mode():
                for start in range(0, len(codes), 4):
                    mels += pipe.codes_to_mel(codes[start:start + 4], n_steps=25, cfg=0.7, batch=4)
                    print(f"S2M {label}/{group}: {min(start + 4, len(codes))}/{len(codes)}", flush=True)
            if label == labels[0]:
                reference = mels
            metrics = {"mel_errors": [errors(a, b) for a, b in zip(reference, mels)],
                       "finite": all(bool(torch.isfinite(m).all()) for m in mels)}
            paths = []
            for item, mel in zip(items, mels):
                path = ROOT / "runs" / "quality" / label / group / f"{item.id}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                wave = pipe.mel_to_wave(mel.float())
                metrics["finite"] = metrics["finite"] and bool(torch.isfinite(wave).all())
                sf.write(path, wave.numpy(), SR)
                paths.append(str(path))
            audio[group][label] = paths
            group_result["conditions"][label] = metrics
        result["groups"][group] = group_result
        (ROOT / "results" / f"{output_name}_progress.json").write_text(json.dumps(result, indent=2) + "\n")

    # ASRは共通のGPU評価器。TTSの速度計測には含めない。
    asr = ASR("cuda")
    for group, items in groups.items():
        refs = [u.text for u in items]
        for label, paths in audio[group].items():
            score = cer_scores(refs, asr.transcribe(paths, batch_size=16))
            result["groups"][group]["conditions"][label]["cer"] = score
            print(f"CER {group}/{label}: {score['reading']:.5f}", flush=True)
    (ROOT / "results" / f"{output_name}.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
