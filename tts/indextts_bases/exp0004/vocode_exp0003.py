"""exp0003 の外部のコーデックの構成(XCodec2 + 固定の代表ベクトル・BiCodec)の TTS を、Griffin-Lim と Vocos で比べる。

exp0003 では、CER は日本語 HuBERT より良かったが、聞くと声がかすれる・震えるような違和感があった。それが Griffin-Lim の
せいだったかを切り分ける。テキスト → T2S(ハイブリッド型)→ S2M の Mel を 1 回だけ作り、同じ Mel を両方のボコーダで戻す。

使い方(exp0003 の部品を使うので、exp0003 のディレクトリで動かす):
    cd ../exp0003 && python ../exp0004/vocode_exp0003.py --variants xcodec2_fixed bicodec
出力: exp0004 の results/metrics.json の "exp0003_vocoder"(CER・UTMOS)と、results/samples/(試聴用、index.json に追記)
"""
import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path.cwd()))  # exp0003 の common を使う
import common  # noqa: E402,F401
from common.ext_codec import EXT_CODECS  # noqa: E402
from common.utils import RUNS_DIR  # noqa: E402
from evaluate import build_pipe  # noqa: E402
from exp0002.common.utils import load_t2s  # noqa: E402
from lib_common.asr_eval import ASR, cer_scores  # noqa: E402
from lib_common.audio import SR, save_preview  # noqa: E402
from lib_common.data import load_dataset, split  # noqa: E402
from lib_common.eval_sets import load_eval_texts  # noqa: E402
from lib_common.mos import UTMOS  # noqa: E402
from lib_common.neural_vocoder import load_vocoder  # noqa: E402
from lib_common.utils import seed_everything  # noqa: E402

E4 = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=["xcodec2_fixed", "bicodec"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(0)
    torch.set_num_threads(4)
    test = split(load_dataset(None, print), "test")
    ets = load_eval_texts(["practical"])
    firsts = {}
    for t in ets:
        firsts.setdefault(t.category, t)
    voc = load_vocoder(E4 / "runs" / "vocos" / "vocoder.pt", args.device)
    asr, mos = ASR(args.device), UTMOS(args.device)
    mpath, ipath = E4 / "results" / "metrics.json", E4 / "results" / "samples" / "index.json"
    metrics, index = json.loads(mpath.read_text()), json.loads(ipath.read_text())
    for v in args.variants:
        base = v.replace("_fixed", "")  # コーデックの名前(xcodec2_fixed は xcodec2 のトークンで、埋め込みだけ違う)
        assert base in EXT_CODECS
        pipe = build_pipe(base, RUNS_DIR / f"s2m_{v}" / "s2m.pt", None, args.device)
        t2s, vocab, _ = load_t2s(RUNS_DIR / f"t2s_{v}" / "t2s.pt", args.device)
        for group, items in [("test", test), ("eval_texts", ets)]:
            mels = []
            for s in range(0, len(items), 16):
                ids = [torch.tensor(vocab.encode(x.text)) for x in items[s: s + 16]]
                mels += pipe.codes_to_mel(t2s.generate(ids, durations=t2s.predict_durations(ids)))
            for name, fn in [("griffinlim", lambda m: pipe.mel_to_wave(m)),
                             ("vocos", lambda m: voc.generate(m.to(args.device)[None])[0].cpu())]:
                cond = f"exp0003_{v}/{name}"
                d = E4 / "runs" / "eval" / group / cond
                d.mkdir(parents=True, exist_ok=True)
                paths = []
                for x, m in zip(items, mels):
                    sf.write(d / f"{x.id}.wav", fn(m).numpy(), SR)
                    paths.append(str(d / f"{x.id}.wav"))
                res = {"cer": cer_scores([x.text for x in items], asr.transcribe(paths))["reading"],
                       "utmos": sum(mos.score(paths)) / len(paths)}
                metrics.setdefault("exp0003_vocoder", {}).setdefault(group, {})[cond] = res
                print(f"RESULT {group:10s} {cond:32s} cer {res['cer']:.4f} utmos {res['utmos']:.3f}", flush=True)
                index = [r for r in index if not (r["group"] == group and r["condition"] == cond)]
                sd = E4 / "results" / "samples" / group / cond
                sd.mkdir(parents=True, exist_ok=True)
                for x in (items[:3] if group == "test" else list(firsts.values())):
                    save_preview(d / f"{x.id}.wav", sd / f"{x.id}.mp3")
                    index.append({"group": group, "condition": cond, "id": x.id, "text": x.text,
                                  "category": getattr(x, "category", "jsut_test"), "file": f"{group}/{cond}/{x.id}.mp3"})
        del pipe, t2s
        torch.cuda.empty_cache()
        mpath.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        ipath.write_text(json.dumps(index, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
