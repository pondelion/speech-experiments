"""C: 同じ設定で重み量子化と基準のCERを測る。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "exp0001"))
from benchmark import build_models
from quantize import weight_only


def main():
    import soundfile as sf
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--bits", type=int, choices=[0, 4, 8], default=0)
    parser.add_argument("--group-size", type=int, choices=[32, 64, 128, 256], default=128)
    parser.add_argument("--calibration", choices=["awq", "gptq"])
    parser.add_argument("--packing", choices=["tile_packed_to_4d", "plain"], default="tile_packed_to_4d")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    args = parser.parse_args()
    if args.dtype == "bfloat16" and args.device != "cuda":
        parser.error("このrunnerのBF16条件はCUDAのみ")
    condition = f"{args.device}_both_wo{args.bits}_{args.dtype}" if args.bits else f"{args.device}_" + ("bf16" if args.dtype == "bfloat16" else "fp32")
    if args.bits == 4 and args.group_size != 128:
        condition += f"_g{args.group_size}"
    if args.bits == 4 and args.packing == "plain":
        condition += "_plain"
    if args.calibration:
        condition += f"_{args.calibration}"
    torch, pipe, t2s, vocab, seed = build_models(args.device)
    torch.set_num_threads(8)
    metadata = {}
    if args.dtype == "bfloat16":
        t2s = t2s.to(dtype=torch.bfloat16)
        pipe.s2m = pipe.s2m.to(dtype=torch.bfloat16)
    if args.calibration:
        from calibrate import calibrated_models
        metadata["calibration"] = calibrated_models(torch, pipe, t2s, vocab, seed,
            args.calibration, args.bits, args.group_size, ROOT, condition)
    elif args.bits:
        metadata["t2s"] = weight_only(torch, t2s, args.bits, exclude={"head"}, group_size=args.group_size, packing=args.packing)
        metadata["s2m"] = weight_only(torch, pipe.s2m, args.bits, group_size=args.group_size, packing=args.packing)
    from lib_common.data import list_jsut, split
    from lib_common.eval_sets import load_eval_texts
    from lib_common.asr_eval import ASR, cer_scores
    from lib_common.audio import SR
    from contextlib import nullcontext

    def amp():
        return torch.autocast("cuda", dtype=torch.bfloat16) if args.dtype == "bfloat16" else nullcontext()

    groups = {"test": split(list_jsut(), "test"), "eval_texts": load_eval_texts(["practical"])}
    result = {"condition": condition, "seed": 0, "dtype": args.dtype, "t2s_batch": 16,
              "s2m_batch": 4, "weight_only": metadata, "groups": {}}
    audio = {}
    for group, items in groups.items():
        seed(0)
        codes = []
        with torch.inference_mode(), amp():
            for start in range(0, len(items), 16):
                ids = [torch.tensor(vocab.encode(u.text)) for u in items[start:start + 16]]
                codes += t2s.generate(ids, durations=t2s.predict_durations(ids))
                print(f"{condition} T2S {group} {min(start+16,len(items))}/{len(items)}", flush=True)
            seed(0)
            mels = pipe.codes_to_mel(codes, batch=4, n_steps=25, cfg=0.7)
        paths = []
        finite = True
        for item, mel in zip(items, mels):
            wave = pipe.mel_to_wave(mel.float())
            finite = finite and bool(torch.isfinite(mel).all() and torch.isfinite(wave).all())
            path = ROOT / "runs" / "weight_only" / condition / group / f"{item.id}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, wave.numpy(), SR)
            paths.append(str(path))
        audio[group] = paths
        result["groups"][group] = {"utterances":len(items), "finite":finite}
    asr = ASR("cuda")
    for group, paths in audio.items():
        scores = cer_scores([u.text for u in groups[group]], asr.transcribe(paths, batch_size=16))
        result["groups"][group]["cer"] = scores
        print(f"{condition} CER {group}: {scores['reading']:.6f}", flush=True)
    output = ROOT / "results" / "weight_only" / f"quality_{condition}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
