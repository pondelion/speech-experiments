"""synth.py で合成した音声の指標を計算する(README の「評価」のとおり)。

    テスト文 25 文(評価する話者の本物の録音がある。主に見るのは話者類似度・本物の録音と比べる指標):
        話者類似度(WavLM・CAM++。参照音声との類似度と、同じ文の本物の録音との類似度)、
        本物の録音との F0 の比較(DTW で時間を合わせた GPE・F0 の誤差。モーラの区間は本物の録音のラベルを DTW で写す)、
        声の高さの平均の差(セント)、長さの比、CER・UTMOS(参考)
    実用文 55 文(主に見るのは CER・UTMOS): CER、UTMOS、話者類似度(参照音声との類似度)
どれも「学習に使わない 10 人(unseen)」と「学習に使った 10 人(seen)」に分けて、乱数の種 0・1・2 の平均 ± 標準偏差。
phase_a(参照音声を使わない、JSUT の声)は、同じ音声を 20 人それぞれと比べる。
F0 は CPU の並列処理で先に求める(GPU を使う前に)。

使い方:
    python metrics.py --name finetune
出力: results/metrics_{name}.json
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import common


def f0_of(path: str):
    from lib_common.audio import load_wave
    from lib_common.prosody import f0
    return f0(load_wave(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="動作確認用に、文の数を絞る(synth.py の --limit と同じ値)")
    parser.add_argument("--exp_dir", default=None, help="評価の音声(runs/eval/)と結果(results/)の exp(既定は exp0101。exp0102 は ../exp0102)")
    args = parser.parse_args()
    exp_dir = Path(args.exp_dir) if args.exp_dir else common.EXP_DIR
    ev = exp_dir / "runs" / "eval" / args.name
    seeds = sorted(p.name for p in ev.glob("seed*"))
    refs = json.loads((ev / "refs.json").read_text())                              # 評価する話者 → 参照音声の ID
    sp = json.loads((common.EXP_DIR / "results" / "split.json").read_text())
    groups = {"unseen": sp["unseen_speakers"], "seen": sp["seen_eval_speakers"]}
    src = {s: (s if (ev / seeds[0] / "test" / s).exists() else "jsut") for s in refs}  # phase_a は全員 jsut の音声
    test_ids = sorted(sp["test_texts"])[: args.limit]
    uniq = sorted({str(ev / seed / "test" / src[s] / f"{t}.wav") for seed in seeds for s in refs for t in test_ids})
    with Pool(args.workers) as pool:                                                # GPU を使う前に F0 を求める
        f0s = dict(zip(uniq, pool.map(f0_of, uniq, chunksize=8)))

    import torch
    from common.data import list_all, phoneme_durations
    from common.speaker import CAMPlus, WavLMSV
    from lib_common.asr_eval import ASR, cer_scores
    from lib_common.audio import HOP, SR, load_wave, wave_to_logmel
    from lib_common.eval_sets import load_eval_texts
    from lib_common.mos import UTMOS
    from lib_common.prosody import compare_f0_dtw, mora_accent_scores, mora_phrases
    from lib_common.text import phoneme_tokens

    utts = {u.id: u for u in list_all()}
    gt_mels, gt_f0 = torch.load(common.RUNS_DIR / "mels.pt"), torch.load(common.RUNS_DIR / "f0.pt")
    practical = {t.id: t.text for t in load_eval_texts(["practical"])[: args.limit]}
    asr, mos, wavlm, cam = ASR("cuda"), UTMOS("cuda"), WavLMSV("cuda"), CAMPlus("cuda")
    emb_cache = {}

    def emb(path):
        if path not in emb_cache:
            emb_cache[path] = (torch.nn.functional.normalize(wavlm(path), dim=0), torch.nn.functional.normalize(cam(path), dim=0))
        return emb_cache[path]

    def sim(a, b):
        ea, eb = emb(a), emb(b)
        return float(ea[0] @ eb[0]), float(ea[1] @ eb[1])

    def dtw_map(ref_mel, hyp_mel):
        import librosa
        _, path = librosa.sequence.dtw(X=ref_mel.T, Y=hyp_mel.T, metric="euclidean")
        path = path[::-1]
        return np.array([int(np.median(path[path[:, 0] == i, 1])) for i in range(len(ref_mel))])

    out = {}
    for gname, speakers in groups.items():
        per_seed = []
        for seed in seeds:
            r = {}
            # --- テスト文 ---
            items = [(s, t) for s in speakers for t in test_ids]
            paths = [str(ev / seed / "test" / src[s] / f"{t}.wav") for s, t in items]
            gts = [utts[f"{s}/{t}"] for s, t in items]
            refp = [utts[refs[s]].path for s, _ in items]
            r["test_cer"] = cer_scores([sp["test_texts"][t] for _, t in items], asr.transcribe(paths))["reading"]
            r["test_utmos"] = float(np.mean(mos.score(paths)))
            sims_ref = [sim(p, q) for p, q in zip(paths, refp)]
            sims_gt = [sim(p, g.path) for p, g in zip(paths, gts)]
            r["test_sim_wavlm"], r["test_sim_campp"] = map(float, np.mean(sims_ref, 0))
            r["test_sim_gt_wavlm"], r["test_sim_gt_campp"] = map(float, np.mean(sims_gt, 0))
            hyp_mels = [wave_to_logmel(load_wave(p)).numpy() for p in paths]
            ref_f0 = [(gt_f0[g.id][0].float().numpy(), gt_f0[g.id][1].numpy().astype(bool)) for g in gts]
            hyp_f0 = [f0s[p] for p in paths]
            r.update({f"test_{k}": v for k, v in compare_f0_dtw(ref_f0, hyp_f0, [gt_mels[g.id].float().numpy() for g in gts], hyp_mels).items()})
            lvl = [1200 * (np.log2(np.nanmedian(hf[hv])) - np.log2(np.median(rf[rv]))) for (rf, rv), (hf, hv) in zip(ref_f0, hyp_f0)
                   if hv.any() and rv.any()]
            r["test_f0_level_diff_cent"] = float(np.mean(np.abs(lvl)))
            r["test_length_ratio"] = float(np.mean([len(h) / len(gt_mels[g.id]) for h, g in zip(hyp_mels, gts)]))
            ph_h, ph_r, f_h, f_r = [], [], [], []
            for g, hm, hf, rf in zip(gts, hyp_mels, hyp_f0, ref_f0):
                tk = phoneme_tokens(g.text)
                d = phoneme_durations(g, tk, len(gt_mels[g.id]), token_sec=HOP / SR)
                if d is None:
                    continue
                ph = mora_phrases(tk, d, 1.0)
                mp = dtw_map(gt_mels[g.id].float().numpy(), hm)
                ph_r.append(ph)
                ph_h.append([[(int(mp[a]), int(mp[min(b, len(mp)) - 1]) + 1, h) for a, b, h in p] for p in ph])
                f_h.append(hf)
                f_r.append(rf)
            r.update({f"test_{k}": v for k, v in mora_accent_scores(f_h, ph_h, f_r, ph_r).items()})
            # --- 実用文 ---
            pitems = [(s, t) for s in speakers for t in practical]
            ppaths = [str(ev / seed / "practical" / src[s] / f"{t}.wav") for s, t in pitems]
            r["practical_cer"] = cer_scores([practical[t] for _, t in pitems], asr.transcribe(ppaths))["reading"]
            r["practical_utmos"] = float(np.mean(mos.score(ppaths)))
            ps = [sim(p, utts[refs[s]].path) for p, (s, _) in zip(ppaths, pitems)]
            r["practical_sim_wavlm"], r["practical_sim_campp"] = map(float, np.mean(ps, 0))
            per_seed.append(r)
            print(gname, seed, {k: round(v, 4) for k, v in r.items() if not k.endswith("n_pairs")}, flush=True)
        out[gname] = {"seeds": per_seed, "mean": {k: float(np.mean([x[k] for x in per_seed])) for k in per_seed[0]},
                      "std": {k: float(np.std([x[k] for x in per_seed])) for k in per_seed[0]}}
    # 参考: 参照音声と同じ文の本物の録音の類似度(上限の目安)
    real = [sim(utts[f"{s}/{t}"].path, utts[refs[s]].path) for g in groups.values() for s in g for t in test_ids]
    out["real_vs_ref"] = {"wavlm": float(np.mean([x[0] for x in real])), "campp": float(np.mean([x[1] for x in real]))}
    (exp_dir / "results").mkdir(exist_ok=True)
    (exp_dir / "results" / f"metrics_{args.name}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
