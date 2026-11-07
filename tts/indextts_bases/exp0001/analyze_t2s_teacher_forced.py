"""分析: T2S の学習条件ごとに、「1 トークンずつの予測の悪さ」と「自分の出力の上で生成すると流れを外れること」を切り分けて測る。

推論(自由な生成)では、T2S は 1 つ前に自分が出したトークンを入力にして次を予測する。一度ずれると、その後は
正解の流れに戻れずにずれ続けることがある。そこで、正解のトークン列を入力にして(teacher forcing)各位置で
次のトークンを予測させ、その予測トークン列を S2M → Griffin-Lim で音にして、ASR の CER を測る。
このトークン列は正解と同じ長さで位置もそろっていて、「毎回の予測の誤り」だけを含み、「流れを外れる」誤りは含まない。

T2S の学習条件(runs/ の中の名前)ごとに、テスト 100 文で次の 3 つを測る:
    tf_argmax: teacher forcing で、各位置で一番良いトークンを選ぶ(regression は予測したベクトルに一番近いトークン)
    tf_sample: teacher forcing で、各位置で推論と同じ設定(temperature 0.8・top_k 30・top_p 0.8)でサンプルする
               (regression はサンプリングしないので無し。繰り返しペナルティは、入力が正解の列なので使わない)
    free:      自由な生成(推論と同じ。繰り返しペナルティ 10 も使う)
tf_sample と free の違いは「入力が正解か、自分の出力か」だけになる。
粗いトークンで学習した T2S は、予測をグループの代表の codec のトークンに戻してから S2M に渡す。

使い方:
    python analyze_t2s_teacher_forced.py --runs t2s_ph_drop03 t2s_char_drop03 t2s_ph_aux
codec・S2M は --codec・--s2m で選ぶ(既定は runs/codec・runs/s2m)。既定以外の S2M では、結果の名前を「T2S の名前+S2M の名前」にする。
--with_gt で正解トークン → S2M の CER(上限)も測る。
出力:
    runs/eval/t2s_compare/{名前}/{条件}/{発話ID}.wav   全テスト文の音声(git の対象外)
    results/t2s_compare.json                            名前ごと・条件ごとの CER とトークンの正解率
    results/samples/test/t2s_compare/{名前}/            試聴用に、自由な生成のテスト文の先頭 3 文(48kbps の MP3。index.json に追記)
"""
import argparse
import json

import soundfile as sf
import torch

import common  # noqa: F401  tts/lib_common を import できるようにする
from common.pipeline import TTSPipeline, load_t2s
from common.utils import EXP_DIR, RUNS_DIR, encode_all
from lib_common.asr_eval import ASR, cer_scores
from lib_common.audio import SR, save_preview
from lib_common.data import load_dataset, split
from lib_common.utils import Logger, seed_everything

N_SAMPLES = 3  # 試聴用にコピーするテスト文の数


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="比べる T2S の runs/ の中の名前")
    parser.add_argument("--codec", default="codec", help="codec の runs/ の名前")
    parser.add_argument("--s2m", default="s2m", help="S2M の runs/ の名前")
    parser.add_argument("--with_gt", action="store_true", help="正解トークン → S2M の CER(上限)も測る")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    torch.set_num_threads(4)
    log = Logger(RUNS_DIR / "eval" / "log_t2s_compare.txt")
    test = split(load_dataset(None, log), "test")
    pipe = TTSPipeline.from_checkpoints(RUNS_DIR / args.codec / "codec.pt", RUNS_DIR / args.s2m / "s2m.pt",
                                        device=args.device)
    gt_fine = encode_all(pipe.codec, [pipe.normalizer.normalize(u.mel) for u in test], args.device)  # codec のトークン、各 [U_i]
    asr = ASR(args.device)
    out_path = EXP_DIR / "results" / "t2s_compare.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}   # 前に測った名前の結果は残す
    index_path = EXP_DIR / "results" / "samples" / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else []

    for run in args.runs:
        name = run if args.s2m == "s2m" else f"{run}+{args.s2m}"  # 結果の名前(既定以外の S2M なら S2M の名前も付ける)
        pipe.t2s, pipe.vocab = load_t2s(RUNS_DIR / run / "t2s.pt", args.device)
        t2s = pipe.t2s
        texts = [torch.tensor(pipe.vocab.encode(u.text)) for u in test]
        gt = [t2s.from_fine[c.to(t2s.device)].cpu() if t2s.from_fine is not None else c for c in gt_fine]  # T2S のトークン
        preds = {"tf_argmax": t2s.predict_teacher_forced(texts, gt)}
        if t2s.objective != "regression":
            preds["tf_sample"] = t2s.predict_teacher_forced(texts, gt, sample=True)
        codes = {cond: [pipe.t2s_to_codec(p) for p in ps] for cond, ps in preds.items()}  # codec のトークンに戻す
        codes["free"] = pipe.text_to_codes([u.text for u in test])
        if args.with_gt:  # 正解トークン → S2M(この codec・S2M での上限)
            codes["gt"] = gt_fine
        out[name] = {"objective": t2s.objective}
        for cond, cs in codes.items():
            d = RUNS_DIR / "eval" / "t2s_compare" / name / cond
            d.mkdir(parents=True, exist_ok=True)
            for u, m in zip(test, pipe.codes_to_mel(cs, texts=[u.text for u in test])):  # 文章はテキスト条件つき S2M だけが使う
                sf.write(d / f"{u.id}.wav", pipe.mel_to_wave(m).numpy(), SR)
            hyps = asr.transcribe([str(d / f"{u.id}.wav") for u in test])
            res = cer_scores([u.text for u in test], hyps)
            if cond == "gt":
                pass
            elif cond in preds:  # teacher forcing の予測は正解と同じ長さなので、トークンの正解率を測れる
                res["token_acc"] = (torch.cat(preds[cond]) == torch.cat(gt)).float().mean().item()
            else:
                res["length_ratio"] = sum(len(c) / max(len(g), 1) for c, g in zip(cs, gt_fine)) / len(cs)
            out[name][cond] = res
            log(f"{name:24s} {cond:10s} reading CER {res['reading']:.4f} "
                + " ".join(f"{k} {res[k]:.3f}" for k in ["token_acc", "length_ratio"] if k in res))
        # 試聴用のコピー(自由な生成のみ)
        sd = EXP_DIR / "results" / "samples" / "test" / "t2s_compare" / name
        sd.mkdir(parents=True, exist_ok=True)
        index = [r for r in index if r["condition"] != f"t2s_compare/{name}"]
        for u in test[:N_SAMPLES]:
            save_preview(RUNS_DIR / "eval" / "t2s_compare" / name / "free" / f"{u.id}.wav", sd / f"{u.id}.mp3")
            index.append({"group": "test", "condition": f"t2s_compare/{name}", "id": u.id, "set": "jsut_test",
                          "category": "jsut_test", "text": u.text, "file": f"test/t2s_compare/{name}/{u.id}.mp3"})
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False))
    log(f"saved {out_path.name}")


if __name__ == "__main__":
    main()
