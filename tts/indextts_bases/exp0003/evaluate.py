"""テスト 100 文と実用文 55 文を合成し、ASR の文字誤り率(CER)で読み上げの正確さを比べる。

音声トークンの種類(例: w2vbert2_k256、xcodec2)ごとに、runs/s2m_{種類}・t2s_{種類}(--ar なら t2s_ar_{種類}・s2m_f3_{種類}も)を使う。
SSL の codec は runs/codec_{種類}/codec.pt、外部のコーデック(common/ext_codec.py の EXT_CODECS の名前)はコーデック自身を使う。
条件:
    {種類}/s2m_gt_tokens     正解トークン → S2M(このトークンでの上限。テスト文のみ)
    {種類}/tts               テキスト → T2S(ハイブリッド型)→ S2M(実際に使うときと同じ)
    --ar のとき(exp0001 の構成: 自己回帰の T2S + F3 の S2M):
    {種類}/s2m_f3_gt_tokens  正解トークン → F3 の S2M
    {種類}/tts_ar            テキスト → 自己回帰の T2S → F3 の S2M
    外部のコーデックでは、S2M + Griffin-Lim の代わりにコーデック自身の decoder で波形に戻す条件も測る:
    {種類}/codec_rec         正解トークン → コーデックの decoder(コーデックの上限。テスト文のみ)
    {種類}/tts_dec           テキスト → T2S(ハイブリッド型)→ コーデックの decoder
    {種類}/tts_ar_dec        --ar のとき: テキスト → 自己回帰の T2S → コーデックの decoder
比較の基準(Mel を量子化したトークン)は exp0002 の `t2s_hybrid_nar`(exp0002 の README の結果 5)と、
exp0001 の一番良い構成(exp0001 の README の結果 9)の値を使う。

使い方:
    python evaluate.py --variants w2vbert2_k256 hubert_ja_k256 xcodec2 bicodec wavtokenizer --ar
    python evaluate.py --variants xcodec2 bicodec wavtokenizer --rec_only   # 学習の前のふるい分け(コーデックの上限だけ)
    python evaluate.py --variants smoke --limit 4 --device cpu   # 動作確認(results/ は書き換えない)
出力:
    runs/eval/{条件}/、runs/eval_texts/{条件}/   全ての音声(git の対象外)
    results/metrics.json                       条件ごとの CER と長さの比(前に測った条件の結果は残す)
    results/samples/{test,eval_texts}/{条件}/   試聴用(48kbps の MP3): テスト文の先頭 3 文・実用文の各カテゴリの先頭 1 文
    results/samples/index.json                 試聴用の音声の一覧
"""
import argparse
import json
from pathlib import Path

import soundfile as sf
import torch

import common  # noqa: F401  tts/lib_common・exp0001・exp0002 を import できるようにする
from common.ext_codec import EXT_CODECS
from common.utils import EXP_DIR, RUNS_DIR, TokenEmbedding, tokenize
from exp0001.common.pipeline import TTSPipeline, load_s2m
from exp0001.common.pipeline import load_t2s as load_ar_t2s
from exp0002.common.utils import TOKEN_SEC, load_t2s
from lib_common.asr_eval import ASR, cer_scores
from lib_common.audio import HOP, SR, MelNormalizer, save_preview
from lib_common.data import load_dataset, split
from lib_common.eval_sets import load_eval_texts
from lib_common.utils import Logger, seed_everything

RESULTS_DIR = EXP_DIR / "results"
N_SAMPLES = 3
BATCH = 16


def build_pipe(v: str, s2m_path: Path, t2s_path: Path | None, device: str) -> TTSPipeline:
    """S2M(+ 自己回帰の T2S)の合成の pipeline を組み立てる。外部のコーデックでは、S2M と一緒に学習したトークンの埋め込みを使う。"""
    if v not in EXT_CODECS:
        return TTSPipeline.from_checkpoints(RUNS_DIR / f"codec_{v}" / "codec.pt", s2m_path, t2s_path, device=device)
    ckpt = torch.load(s2m_path, map_location="cpu")
    cls = EXT_CODECS[v]
    emb = TokenEmbedding(cls.codebook_size, 1 / cls.token_rate, fixed_code_dim=ckpt.get("fixed_code_dim", 0))
    emb.load_state_dict(ckpt["token_emb"])
    s2m = load_s2m(s2m_path, device)
    if s2m.text_t2s:  # テキストの隠れ状態を作る T2S(F3 の S2M の学習時と同じもの)
        s2m.text_encoder = load_ar_t2s(Path(s2m.text_t2s) / "t2s.pt", device)
    t2s, vocab = load_ar_t2s(t2s_path, device) if t2s_path else (None, None)
    pipe = TTSPipeline(emb.to(device).eval(), MelNormalizer.from_state_dict(ckpt["normalizer"]), s2m, t2s, vocab, device)
    pipe.frames_per_token = SR / HOP * emb.token_sec  # トークン 1 個あたりの Mel のフレーム数(例: 50 個/秒なら 1.875)
    return pipe


def model_runs(v: str, gt: list, codec, test: list, eval_texts: list, ext: bool, use_ar: bool, device: str,
               suffix: str = "") -> list:
    """学習した S2M・T2S を使う条件(合成するトークン列と、波形への戻し方)を作る。

    suffix: ablation の学習結果の名前の後ろに付く文字(runs/t2s_{種類}{suffix}。S2M は runs/s2m_{種類}{suffix} が
        あればそれを、無ければ runs/s2m_{種類} を使う)。
    """
    s2m_dir = RUNS_DIR / f"s2m_{v}{suffix}"
    pipe = build_pipe(v, (s2m_dir if s2m_dir.exists() else RUNS_DIR / f"s2m_{v}") / "s2m.pt", None, device)
    t2s, vocab, _ = load_t2s(RUNS_DIR / f"t2s_{v}{suffix}" / "t2s.pt", device)

    def tts(texts: list[str]) -> list[torch.Tensor]:
        out = []
        for s in range(0, len(texts), BATCH):
            ids = [torch.tensor(vocab.encode(t)) for t in texts[s: s + BATCH]]
            out += t2s.generate(ids, durations=t2s.predict_durations(ids))
        return out

    tts_test, tts_et = tts([u.text for u in test]), tts([t.text for t in eval_texts])
    runs = [("test", "s2m_gt_tokens", test, gt, pipe), ("test", "tts", test, tts_test, pipe),
            ("eval_texts", "tts", eval_texts, tts_et, pipe)]
    if ext:
        runs += [("test", "tts_dec", test, tts_test, "dec"), ("eval_texts", "tts_dec", eval_texts, tts_et, "dec")]
    if use_ar:  # exp0001 の構成: 自己回帰の T2S + F3 の S2M(テキストの条件つき)
        pipe_ar = build_pipe(v, RUNS_DIR / f"s2m_f3_{v}" / "s2m.pt", RUNS_DIR / f"t2s_ar_{v}" / "t2s.pt", device)
        max_len = min(int(600 * TOKEN_SEC / codec.token_sec), pipe_ar.t2s.audio_pos.num_embeddings - 1)  # 約 25.6 秒まで
        ar_test = pipe_ar.text_to_codes([u.text for u in test], max_len=max_len)
        ar_et = pipe_ar.text_to_codes([t.text for t in eval_texts], max_len=max_len)
        runs += [("test", "s2m_f3_gt_tokens", test, gt, pipe_ar), ("test", "tts_ar", test, ar_test, pipe_ar),
                 ("eval_texts", "tts_ar", eval_texts, ar_et, pipe_ar)]
        if ext:
            runs += [("test", "tts_ar_dec", test, ar_test, "dec"), ("eval_texts", "tts_ar_dec", eval_texts, ar_et, "dec")]
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", required=True, help="音声トークンの種類")
    parser.add_argument("--ar", action="store_true", help="exp0001 の構成(runs/t2s_ar_{種類}・s2m_f3_{種類})も測る")
    parser.add_argument("--rec_only", action="store_true", help="外部のコーデックの decoder の上限(codec_rec)だけを測る")
    parser.add_argument("--run_suffix", default="", help="ablation の学習結果の名前の後ろの文字(例: _fixed)")
    parser.add_argument("--limit", type=int, default=None, help="動作確認用に発話数を絞る")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    torch.set_num_threads(4)
    log = Logger(RUNS_DIR / "eval" / "log.txt")
    log(f"args: {vars(args)}")
    utts = load_dataset(args.limit, log)
    test, ref_utt = split(utts, "test"), split(utts, "train")[0]  # ref_utt: BiCodec の global トークンを取る学習データの発話
    eval_texts = load_eval_texts(["practical"])[: args.limit]
    asr = ASR(args.device)
    metrics_path, index_path = RESULTS_DIR / "metrics.json", RESULTS_DIR / "samples" / "index.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    index = json.loads(index_path.read_text()) if index_path.exists() else []

    for v in args.variants:
        ext = v in EXT_CODECS
        gt, codec, _ = tokenize(f"ext:{v}" if ext else RUNS_DIR / f"codec_{v}" / "codec.pt", test, args.device, log)
        # 評価する条件: (群, 条件, 文, トークン列, 波形への戻し方)。戻し方は pipeline(S2M + Griffin-Lim)か "dec"(コーデックの decoder)
        runs, dec = [], None
        if ext:  # コーデック自身の decoder
            dec = EXT_CODECS[v](args.device)
            if hasattr(dec, "set_reference"):
                dec.set_reference(ref_utt.path)
            runs.append(("test", "codec_rec", test, gt, "dec"))
        if not args.rec_only:  # --rec_only(学習の前のふるい分け)では、コーデックの decoder の上限だけを測る
            runs += model_runs(v, gt, codec, test, eval_texts, ext, args.ar, args.device, args.run_suffix)

        for group, name, items, cs, p in runs:
            cond = f"{v}/{name}" if name == "codec_rec" else f"{v}{args.run_suffix}/{name}"
            d = RUNS_DIR / ("eval" if group == "test" else "eval_texts") / cond
            d.mkdir(parents=True, exist_ok=True)
            if p == "dec":
                waves = [dec.decode(c) if len(c) else torch.zeros(SR).numpy() for c in cs]
            else:  # 文章はテキスト条件つき S2M だけが使う
                waves = [p.mel_to_wave(m).numpy() for m in p.codes_to_mel(cs, texts=[x.text for x in items])]
            for x, w in zip(items, waves):
                sf.write(d / f"{x.id}.wav", w, SR)
            res = cer_scores([x.text for x in items], asr.transcribe([str(d / f"{x.id}.wav") for x in items]))
            if group == "test":
                res["length_ratio"] = sum(len(c) / len(g) for c, g in zip(cs, gt)) / len(cs)
            metrics.setdefault(group, {})[cond] = res
            log(f"{group:10s} {cond:30s} reading CER {res['reading']:.4f}"
                + (f" length ratio {res['length_ratio']:.3f}" if "length_ratio" in res else ""))
            # 試聴用: テスト文は先頭 N_SAMPLES 文、実用文はカテゴリごとの先頭 1 文
            if group == "test":
                preview = items[:N_SAMPLES]
            else:
                firsts = {}
                for x in items:
                    firsts.setdefault(x.category, x)
                preview = list(firsts.values())
            index = [r for r in index if not (r["group"] == group and r["condition"] == cond)]
            if not args.limit:
                sd = RESULTS_DIR / "samples" / group / cond
                sd.mkdir(parents=True, exist_ok=True)
                for x in preview:
                    save_preview(d / f"{x.id}.wav", sd / f"{x.id}.mp3")
                    index.append({"group": group, "condition": cond, "id": x.id, "text": x.text,
                                  "category": getattr(x, "category", "jsut_test"), "file": f"{group}/{cond}/{x.id}.mp3"})
        del runs, dec
        torch.cuda.empty_cache()
        # 種類ごとに書き出す(1 つのプロセスで複数の種類を続けて評価すると GPU のエラーで落ちることがあったため。
        # 種類ごとに別のプロセスで動かす方が安全)。動作確認では results/(GitHub に載せる結果)を書き換えない
        if not args.limit:
            RESULTS_DIR.mkdir(exist_ok=True)
            metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
            index_path.parent.mkdir(parents=True, exist_ok=True)
            index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False))
            log(f"saved {metrics_path.name} ({v})")
    if args.limit:
        log("--limit のときは results/ に書かない")


if __name__ == "__main__":
    main()
