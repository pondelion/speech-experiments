"""テスト 100 文と実用文 55 文を合成し、ASR の文字誤り率(CER)で読み上げの正確さを比べる。

codec・S2M は exp0001 のもの(codebook 256)で共通にし、T2S だけを変えて比べる。条件:
    s2m_gt_tokens        正解トークン → S2M(この codec・S2M での上限)
    ar/{名前}            exp0001 の自己回帰の T2S(--ar_runs で指定。比較の基準)
    {名前}               exp0002 の非自己回帰の T2S(--runs で指定)。長さは自分で決める(実際に使うときと同じ)
    {名前}/gt_length     分析用: 長さだけ正解を使う(duration・hybrid は正解の音素ごとの長さ、masked は正解の全体の長さ)。
                         長さの予測の誤りと、トークンの予測の誤りを切り分ける(テスト文のみ)
実用文(datasets/eval_texts/practical.tsv)は、テキストからの合成の条件だけ測る。

使い方:
    python evaluate.py --runs t2s_duration t2s_masked --ar_runs t2s_ph_k256 t2s_ph_k256_aux
    python evaluate.py --runs smoke --limit 4      # 動作確認(results/ は書き換えない)
出力:
    runs/eval/{条件}/{発話ID}.wav、runs/eval_texts/{条件}/{文章ID}.wav   全ての音声(git の対象外)
    results/metrics.json                    条件ごとの CER と、T2S が決めた長さの比(結果は前の値に追記する)
    results/samples/test/{条件}/             試聴用: テスト文の先頭 3 文(48kbps の MP3)
    results/samples/eval_texts/{条件}/       試聴用: 実用文の各カテゴリの先頭 1 文(48kbps の MP3)
    results/samples/index.json              試聴用の音声の一覧
"""
import argparse
import json

import soundfile as sf
import torch

import common  # noqa: F401  tts/lib_common と exp0001 を import できるようにする
from common.utils import BASE_CODEC, BASE_S2M, EXP1_RUNS, EXP_DIR, PROSODY_MARKS, RUNS_DIR, load_t2s, phoneme_durations
from exp0001.common.pipeline import TTSPipeline
from exp0001.common.pipeline import load_t2s as load_ar_t2s
from exp0001.common.utils import encode_all
from lib_common.asr_eval import ASR, cer_scores
from lib_common.audio import SR, save_preview
from lib_common.data import load_dataset, split
from lib_common.eval_sets import load_eval_texts
from lib_common.utils import Logger, seed_everything

RESULTS_DIR = EXP_DIR / "results"
N_SAMPLES = 3  # 試聴用にコピーするテスト文の数
BATCH = 16     # T2S の生成を一度に行う発話数


def nar_codes(model, vocab, method: str, texts: list[str], oracle: list | None = None) -> list[torch.Tensor]:
    """非自己回帰の T2S で音声トークン列を作る。

    Args:
        model: DurationT2S・MaskedT2S・HybridT2S のどれか。
        vocab: テキストの語彙。
        method: "duration"・"masked"・"hybrid" のどれか。
        texts: 読み上げる文章。
        oracle: 分析用に正解の長さを使うとき。duration・hybrid は音素ごとの長さ [L_i](None の文は予測する)、
                masked は全体の長さ(整数)のリスト。

    Returns:
        発話ごとの音声トークン列 [U_i] のリスト。
    """
    out = []
    for s in range(0, len(texts), BATCH):
        chunk = texts[s: s + BATCH]
        ids = [torch.tensor(vocab.encode(t)) for t in chunk]
        if method in ("duration", "hybrid"):  # 音素ごとの長さを予測してから合成する方式
            durs = model.predict_durations(ids)
            if oracle is not None:
                durs = [d if o is None else torch.tensor(o) for d, o in zip(durs, oracle[s: s + BATCH])]
            out += model.generate(ids, durations=durs)
        else:
            n_phones = [len([t for t in vocab.tokenize(x) if t not in PROSODY_MARKS]) for x in chunk]
            lengths = oracle[s: s + BATCH] if oracle is not None else model.rule_lengths(n_phones)
            out += model.generate(ids, lengths)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=[], help="exp0002 の非自己回帰の T2S の runs/ の名前")
    parser.add_argument("--ar_runs", nargs="*", default=[], help="比較する exp0001 の自己回帰の T2S の runs/ の名前")
    parser.add_argument("--limit", type=int, default=None, help="動作確認用に発話数を絞る")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    torch.set_num_threads(4)
    log = Logger(RUNS_DIR / "eval" / "log.txt")
    log(f"args: {vars(args)}")
    test = split(load_dataset(args.limit, log), "test")
    eval_texts = load_eval_texts(["practical"])[: args.limit]
    pipe = TTSPipeline.from_checkpoints(BASE_CODEC, BASE_S2M, device=args.device)
    gt = encode_all(pipe.codec, [pipe.normalizer.normalize(u.mel) for u in test], args.device)  # 各 [U_i]

    # 条件ごとの音声トークン列: テスト文 tokens[条件]、実用文 et_tokens[条件]
    tokens, et_tokens = {"s2m_gt_tokens": gt}, {}
    for name in args.ar_runs:
        pipe.t2s, pipe.vocab = load_ar_t2s(EXP1_RUNS / name / "t2s.pt", args.device)
        tokens[f"ar/{name}"] = pipe.text_to_codes([u.text for u in test])
        et_tokens[f"ar/{name}"] = pipe.text_to_codes([t.text for t in eval_texts])
    for name in args.runs:
        model, vocab, method = load_t2s(RUNS_DIR / name / "t2s.pt", args.device)
        tokens[name] = nar_codes(model, vocab, method, [u.text for u in test])
        et_tokens[name] = nar_codes(model, vocab, method, [t.text for t in eval_texts])
        if method in ("duration", "hybrid"):
            oracle = [phoneme_durations(u, vocab.tokenize(u.text), len(c)) for u, c in zip(test, gt)]
            log(f"{name}: 正解の音素ごとの長さを作れたテスト文 {sum(o is not None for o in oracle)}/{len(test)}")
        else:
            oracle = [len(c) for c in gt]
        tokens[f"{name}/gt_length"] = nar_codes(model, vocab, method, [u.text for u in test], oracle)
        log(f"{name}: generated")

    # 波形にして保存し、ASR で CER を測る
    asr = ASR(args.device)
    metrics_path = RESULTS_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    for group, items, toks in [("test", test, tokens), ("eval_texts", eval_texts, et_tokens)]:
        for cond, cs in toks.items():
            d = RUNS_DIR / ("eval" if group == "test" else "eval_texts") / cond
            d.mkdir(parents=True, exist_ok=True)
            for x, m in zip(items, pipe.codes_to_mel(cs)):
                sf.write(d / f"{x.id}.wav", pipe.mel_to_wave(m).numpy(), SR)
            res = cer_scores([x.text for x in items], asr.transcribe([str(d / f"{x.id}.wav") for x in items]))
            if group == "test":
                res["length_ratio"] = sum(len(c) / len(g) for c, g in zip(cs, gt)) / len(cs)
            metrics.setdefault(group, {})[cond] = res
            log(f"{group:10s} {cond:28s} reading CER {res['reading']:.4f}"
                + (f" length ratio {res['length_ratio']:.3f}" if "length_ratio" in res else ""))
    if args.limit:  # 動作確認では results/(GitHub に載せる結果)を書き換えない
        log("--limit のときは results/ に書かない")
        return
    RESULTS_DIR.mkdir(exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    # 試聴用のコピーと一覧(前に作った条件の分は残し、今回の条件の分を入れ替える)
    index_path = RESULTS_DIR / "samples" / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else []
    preview, seen = [], set()
    for t in eval_texts:  # 実用文はカテゴリごとの先頭 1 文
        if t.category not in seen:
            preview.append(t)
            seen.add(t.category)
    for group, items, toks in [("test", test[:N_SAMPLES], tokens), ("eval_texts", preview, et_tokens)]:
        src = RUNS_DIR / ("eval" if group == "test" else "eval_texts")
        for cond in toks:
            index = [r for r in index if not (r["group"] == group and r["condition"] == cond)]
            d = RESULTS_DIR / "samples" / group / cond
            d.mkdir(parents=True, exist_ok=True)
            for x in items:
                save_preview(src / cond / f"{x.id}.wav", d / f"{x.id}.mp3")
                index.append({"group": group, "condition": cond, "id": x.id, "text": x.text,
                              "category": getattr(x, "category", "jsut_test"), "file": f"{group}/{cond}/{x.id}.mp3"})
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False))
    log(f"saved {metrics_path.name}")


if __name__ == "__main__":
    main()
