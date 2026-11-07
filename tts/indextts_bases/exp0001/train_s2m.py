"""手順 3: 音声トークン列から Mel を作る S2M(Flow Matching)を学習する。

学習済みの codec(手順 1)を固定し、正解の音声から作ったトークン → codec decoder の特徴 → S2M → Mel、
という経路で学習する。既定ではテキストは使わないので、T2S(手順 2)とは独立に学習できる。

比較用の設定(どれも既定では使わない):
  --init PATH:       学習済みの S2M から追加学習する(新しく足した部分以外の重みを読み込む)
  --text_t2s NAME:   T2S(runs/NAME)の GPT のテキスト部分の隠れ状態を、cross-attention で条件に加える
  --pred_t2s NAME, --pred_mix P:
                     発話ごとに確率 P で、条件のトークンを正解から T2S(runs/NAME)の予測に置き換える
                     (teacher forcing で、推論と同じ設定でサンプルした予測。推論時に入る T2S の誤りに慣れさせる)

途中経過として、検証データで速度の二乗誤差を測る(正解のトークンを条件に、ノイズと時刻 t は固定)。

使い方:
    python train_s2m.py
    python train_s2m.py --init runs/s2m/s2m.pt --steps 10000 --lr 1e-4 --text_t2s t2s_ph_drop03 --name s2m_text
    python train_s2m.py --limit 50 --steps 20 --device cpu   # 動作確認
出力: runs/{name}/(既定は runs/s2m/)に s2m.pt(検証 loss が一番小さかった時点)と log.txt
"""
import argparse

import torch
from einops import rearrange

import common  # noqa: F401  tts/lib_common を import できるようにする
from common.pipeline import load_t2s
from common.s2m import SemanticToMel
from common.utils import RUNS_DIR, encode_all, load_codec, pad_mels
from lib_common.data import load_dataset, split
from lib_common.utils import Logger, length_batches, seed_everything, warmup_cosine

MODEL_CONFIG = {"dim": 512, "layers": 8, "heads": 8, "p_uncond": 0.1}


def batch_inputs(b, mels, codes, texts, codec, text_t2s, pred_t2s, pred_mix: float, device: str):
    """バッチの Mel・マスク・codec の特徴・テキストの隠れ状態を作る。

    Args:
        b: バッチの発話インデックスのリスト。
        mels, codes, texts: 発話ごとの Mel [T_i, F]・正解トークン [U_i]・T2S 用のテキスト ID [L_i] のリスト。
        codec: 固定した codec。
        text_t2s: テキストの隠れ状態を作る T2S(None なら使わない)。
        pred_t2s: 予測トークンを作る T2S(None なら使わない)。
        pred_mix: 予測トークンに置き換える確率。
        device: デバイス。

    Returns:
        mel [B, T, F]、mask [B, T]、hidden [B, U, Dc]、text [B, L, Dt] または None、text_pad [B, L] または None。
    """
    mel, mask = pad_mels([mels[i] for i in b], device)
    cs = [codes[i] for i in b]
    with torch.no_grad():  # codec・T2S は固定
        if pred_t2s is not None and pred_mix > 0:
            preds = pred_t2s.predict_teacher_forced([texts[i] for i in b], cs, sample=True)
            cs = [p if torch.rand(()) < pred_mix else c for p, c in zip(preds, cs)]
        ids = torch.nn.utils.rnn.pad_sequence(cs, batch_first=True).to(device)          # [B, U]
        hidden = codec.decode_hidden(ids)                                                # [B, U] → [B, U, Dc]
        text, text_pad = text_t2s.text_hidden([texts[i] for i in b]) if text_t2s is not None else (None, None)
    return mel, mask, hidden, text, text_pad


@torch.no_grad()
def evaluate(model, codec, mels, codes, texts, text_t2s, device: str, max_frames: int) -> float:
    """検証データでの速度の二乗誤差(正解のトークンを条件に、ノイズと t を固定)。"""
    model.eval()
    g = torch.Generator(device=device).manual_seed(0)
    total, n = 0.0, 0
    for b in length_batches([len(m) for m in mels], max_frames, shuffle=False):
        mel, mask, hidden, text, text_pad = batch_inputs(b, mels, codes, texts, codec, text_t2s, None, 0, device)
        x0 = torch.randn(mel.shape, device=device, generator=g)                  # 固定したノイズ [B, T, F]
        t = torch.rand(len(b), device=device, generator=g)                       # 固定した時刻 [B]
        tb = rearrange(t, "b -> b 1 1")
        xt = (1 - tb) * x0 + tb * mel
        v = model.velocity(xt, model.regulator(hidden, mel.shape[1]), t, ~mask, text, text_pad)  # [B, T, F]
        err = ((v - (mel - x0)) ** 2).mean(-1)                                  # フレームごとの誤差 [B, T]
        total += ((err * mask).sum() / mask.sum()).item() * len(b)
        n += len(b)
    model.train()
    return total / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec", default=str(RUNS_DIR / "codec" / "codec.pt"))
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--max_frames", type=int, default=16000, help="1 バッチの (最長フレーム数 × 発話数) の上限")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--init", default=None, help="追加学習の元にする s2m.pt")
    parser.add_argument("--text_t2s", default=None, help="テキストの隠れ状態を条件に加える T2S の runs/ の名前")
    parser.add_argument("--pred_t2s", default=None, help="予測トークンを作る T2S の runs/ の名前")
    parser.add_argument("--pred_mix", type=float, default=0.0, help="条件のトークンを予測に置き換える確率")
    parser.add_argument("--name", default="s2m", help="保存先 runs/{name}/")
    parser.add_argument("--limit", type=int, default=None, help="動作確認用に発話数を絞る")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    torch.set_num_threads(4)
    out = RUNS_DIR / args.name
    log = Logger(out / "log.txt")
    log(f"args: {vars(args)}")

    utts = load_dataset(args.limit, log)
    codec, norm = load_codec(args.codec, args.device)
    text_t2s = load_t2s(RUNS_DIR / args.text_t2s / "t2s.pt", args.device)[0] if args.text_t2s else None
    pred_t2s, vocab = load_t2s(RUNS_DIR / args.pred_t2s / "t2s.pt", args.device) if args.pred_t2s else (None, None)
    if vocab is None and args.text_t2s:
        vocab = load_t2s(RUNS_DIR / args.text_t2s / "t2s.pt", "cpu")[1]
    data = {}
    for name in ["train", "val"]:
        us = split(utts, name)
        mels = [norm.normalize(u.mel)[: len(u.mel) // 4 * 4] for u in us]  # トークン 1 個 = 4 フレームに合わせる
        texts = [torch.tensor(vocab.encode(u.text)) for u in us] if vocab else [None] * len(us)
        data[name] = (mels, encode_all(codec, mels, args.device), texts)
    tr_m, tr_c, tr_t = data["train"]

    config = {**MODEL_CONFIG, "text_dim": text_t2s.audio_emb.weight.shape[1] if text_t2s else 0}
    model = SemanticToMel(codec.hidden_dim, **config).to(args.device)
    if args.init:  # 学習済みの S2M の重みを読み込む(新しく足した cross-attention は 0 初期化のまま)
        missing, _ = model.load_state_dict(torch.load(args.init, map_location="cpu")["model"], strict=False)
        log(f"init from {args.init} (newly added parameters: {len(missing)})")
    log(f"s2m params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = warmup_cosine(opt, args.steps)

    step, epoch, best, running = 0, 0, float("inf"), 0.0
    while step < args.steps:
        for b in length_batches([len(m) for m in tr_m], args.max_frames, shuffle=True, seed=epoch):
            mel, mask, hidden, text, text_pad = batch_inputs(b, tr_m, tr_c, tr_t, codec, text_t2s, pred_t2s,
                                                             args.pred_mix, args.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                loss = model(mel, hidden, mask, text, text_pad)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            running += loss.item()
            if step % args.eval_every == 0 or step == args.steps:
                n = args.eval_every if step % args.eval_every == 0 else step % args.eval_every
                val = evaluate(model, codec, *data["val"], text_t2s, args.device, args.max_frames)
                mark = ""
                if val < best:
                    best, mark = val, " *saved"
                    torch.save({"model": model.state_dict(), "config": config, "cond_in_dim": codec.hidden_dim,
                                "text_t2s": args.text_t2s, "step": step}, out / "s2m.pt")
                log(f"step {step:6d} | train loss {running / n:.4f} | val loss {val:.4f}{mark}")
                running = 0.0
            if step >= args.steps:
                break
        epoch += 1
    log(f"best val loss {best:.4f}, checkpoint {out / 's2m.pt'}")


if __name__ == "__main__":
    main()
