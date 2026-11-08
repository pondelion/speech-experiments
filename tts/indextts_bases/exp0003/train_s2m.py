"""手順 2: SSL の codec のトークンから Mel を作る S2M を学習する(モデル・学習の仕方は exp0001 の S2M と同じ)。

条件は exp0001 と同じく、codec の decoder の特徴(codec.decode_hidden)。既定は exp0002 の基準(s2m_k256)と同じ設定。
exp0001 の構成で比べるときは、学習した S2M から F3(テキストの cross-attention + 予測トークンを混ぜる学習、
exp0001 の結果 8)と同じ設定で追加学習する(--init --text_t2s --pred_t2s --pred_mix)。
checkpoint の形は exp0001 と同じなので、exp0001 の pipeline で読める。
--codec ext:{名前} で外部のコーデック(common/ext_codec.py)のトークンにする。このときの条件は、トークンの埋め込み
(TokenEmbedding)を S2M と一緒に学習したもので、checkpoint の token_emb に保存する。

使い方:
    python train_s2m.py --codec runs/codec_w2vbert2_k256/codec.pt --name s2m_w2vbert2_k256
    python train_s2m.py --codec runs/codec_w2vbert2_k256/codec.pt --init runs/s2m_w2vbert2_k256/s2m.pt --steps 10000 --lr 1e-4 \
        --text_t2s t2s_ar_w2vbert2_k256 --pred_t2s t2s_ar_w2vbert2_k256 --pred_mix 0.5 --name s2m_f3_w2vbert2_k256
    python train_s2m.py --codec runs/smoke/codec.pt --limit 20 --steps 4 --eval_every 2 --device cpu --name smoke_s2m   # 動作確認
出力: runs/{name}/s2m.pt(検証 loss が一番小さかった時点)と log.txt
"""
import argparse

import torch
from einops import rearrange

import common  # noqa: F401  tts/lib_common と exp0001 を import できるようにする
from common.utils import RUNS_DIR, TokenEmbedding, code_vectors, tokenize
from lib_common.audio import MelNormalizer
from exp0001.common.codec import DOWNSAMPLE
from exp0001.common.pipeline import load_t2s
from exp0001.common.s2m import SemanticToMel
from exp0001.common.utils import pad_mels
from lib_common.data import load_dataset, split
from lib_common.utils import Logger, length_batches, seed_everything, warmup_cosine

MODEL_CONFIG = {"dim": 512, "layers": 8, "heads": 8, "p_uncond": 0.1, "text_dim": 0}  # exp0001 の S2M と同じ


def batch(b, mels, codes, texts, codec, device, text_t2s=None, pred_t2s=None, pred_mix: float = 0.0):
    """Mel [B, T, F]、mask [B, T]、条件の特徴 [B, U, Dc]、テキストの隠れ状態 [B, L, Dt]・詰め物のマスク [B, L](または None)。

    pred_t2s があれば、発話ごとに確率 pred_mix で条件のトークンを T2S の teacher forcing の予測(サンプル)に置き換える
    (exp0001 の F3 と同じ。codec・T2S は固定)。
    """
    mel, mask = pad_mels([mels[i] for i in b], device)
    cs = [codes[i] for i in b]
    with torch.no_grad():
        if pred_t2s is not None and pred_mix > 0:
            preds = pred_t2s.predict_teacher_forced([texts[i] for i in b], cs, sample=True)
            cs = [p if torch.rand(()) < pred_mix else c for p, c in zip(preds, cs)]
        text, text_pad = text_t2s.text_hidden([texts[i] for i in b]) if text_t2s is not None else (None, None)
    # 外部のコーデックのトークンの埋め込み(TokenEmbedding)は S2M と一緒に学習するので、勾配を流す
    with torch.set_grad_enabled(isinstance(codec, TokenEmbedding) and torch.is_grad_enabled()):
        hidden = codec.decode_hidden(torch.nn.utils.rnn.pad_sequence(cs, batch_first=True).to(device))
    return mel, mask, hidden, text, text_pad


@torch.no_grad()
def evaluate(model, codec, mels, codes, texts, device: str, max_frames: int, text_t2s=None) -> float:
    """検証 loss(ノイズと時刻を固定して、速度の二乗誤差を測る。exp0001 と同じ。予測トークンは混ぜない)。"""
    model.eval()
    g = torch.Generator(device=device).manual_seed(0)
    total, n = 0.0, 0
    for b in length_batches([len(m) for m in mels], max_frames, shuffle=False):
        mel, mask, hidden, text, text_pad = batch(b, mels, codes, texts, codec, device, text_t2s)
        x0 = torch.randn(mel.shape, device=device, generator=g)
        t = torch.rand(len(b), device=device, generator=g)
        xt = (1 - rearrange(t, "b -> b 1 1")) * x0 + rearrange(t, "b -> b 1 1") * mel
        v = model.velocity(xt, model.regulator(hidden, mel.shape[1]), t, ~mask, text, text_pad)
        err = ((v - (mel - x0)) ** 2).mean(-1)                                  # [B, T]
        total, n = total + ((err * mask).sum() / mask.sum()).item() * len(b), n + len(b)
    model.train()
    return total / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec", required=True, help="train_codec.py の codec.pt")
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--max_frames", type=int, default=16000, help="1 バッチの (最長フレーム数 × 発話数) の上限")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--init", default=None, help="追加学習の元にする s2m.pt(F3 用)")
    parser.add_argument("--text_t2s", default=None, help="テキストの隠れ状態を条件に加える自己回帰の T2S の runs/ の名前(F3 用)")
    parser.add_argument("--pred_t2s", default=None, help="予測トークンを作る自己回帰の T2S の runs/ の名前(F3 用)")
    parser.add_argument("--pred_mix", type=float, default=0.0, help="条件のトークンを予測に置き換える確率(F3 は 0.5)")
    parser.add_argument("--fixed_code_emb", action="store_true",
                        help="外部のコーデック: 条件をコーデックの固定の代表ベクトル + 学習する射影にする(ablation)")
    parser.add_argument("--name", required=True, help="保存先 runs/{name}/")
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
    train, val = split(utts, "train"), split(utts, "val")
    codes, codec, mel_norm = tokenize(args.codec, train + val, args.device, log)
    ext = isinstance(codec, TokenEmbedding)
    fixed_dim = 0
    if ext:  # 外部のコーデック: Mel の正規化を学習データから求める(S2M の checkpoint に保存し、合成でも使う)
        mel_norm = MelNormalizer.fit([u.mel for u in train])
        if args.fixed_code_emb:  # ablation: 条件をコーデックの固定の代表ベクトル + 学習する射影にする
            vecs = code_vectors(args.codec, args.device)
            fixed_dim = vecs.shape[1]
            codec = TokenEmbedding(codec.codebook_size, codec.token_sec, fixed_code_dim=fixed_dim).to(args.device)
            codec.code_vecs.copy_(vecs)
        if args.init:  # F3 の追加学習: 元の S2M と一緒に学習したトークンの埋め込みも読み込む
            codec.load_state_dict(torch.load(args.init, map_location="cpu")["token_emb"])
    text_t2s, vocab = load_t2s(RUNS_DIR / args.text_t2s / "t2s.pt", args.device) if args.text_t2s else (None, None)
    pred_t2s = load_t2s(RUNS_DIR / args.pred_t2s / "t2s.pt", args.device)[0] if args.pred_t2s else None
    if vocab is None and args.pred_t2s:
        vocab = load_t2s(RUNS_DIR / args.pred_t2s / "t2s.pt", "cpu")[1]
    data = {}
    for name, us, cs in [("train", train, codes[: len(train)]), ("val", val, codes[len(train):])]:
        texts = [torch.tensor(vocab.encode(u.text)) for u in us] if vocab else [None] * len(us)
        data[name] = ([mel_norm.normalize(u.mel)[: len(u.mel) // DOWNSAMPLE * DOWNSAMPLE] for u in us], cs, texts)

    config = {**MODEL_CONFIG, "text_dim": text_t2s.audio_emb.weight.shape[1] if text_t2s else 0}
    model = SemanticToMel(codec.hidden_dim, **config).to(args.device)
    if args.init:  # 学習済みの S2M の重みを読み込む(新しく足した cross-attention は 0 初期化のまま)
        missing, _ = model.load_state_dict(torch.load(args.init, map_location="cpu")["model"], strict=False)
        log(f"init from {args.init} (newly added parameters: {len(missing)})")
    log(f"s2m params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(list(model.parameters()) + (list(codec.parameters()) if ext else []), lr=args.lr)
    sched = warmup_cosine(opt, args.steps)
    tr_m, tr_c, tr_t = data["train"]
    step, epoch, best, running = 0, 0, float("inf"), 0.0
    while step < args.steps:
        for b in length_batches([len(m) for m in tr_m], args.max_frames, shuffle=True, seed=epoch):
            mel, mask, hidden, text, text_pad = batch(b, tr_m, tr_c, tr_t, codec, args.device, text_t2s, pred_t2s,
                                                      args.pred_mix)
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
                val_loss = evaluate(model, codec, *data["val"], args.device, args.max_frames, text_t2s)
                mark = ""
                if val_loss < best:
                    best, mark = val_loss, " *saved"
                    # text_t2s: exp0001 の pipeline が「exp0001 の runs/ / 名前」で読むので、exp0003 の T2S の場所を絶対パスで入れる
                    # (Path / 絶対パス は絶対パスになる。checkpoint は git の対象外)
                    torch.save({"model": model.state_dict(), "config": config, "cond_in_dim": codec.hidden_dim,
                                "text_t2s": str(RUNS_DIR / args.text_t2s) if args.text_t2s else None, "step": step,
                                "normalizer": mel_norm.state_dict(), "codec": str(args.codec),
                                "token_emb": codec.state_dict() if ext else None, "fixed_code_dim": fixed_dim},
                               out / "s2m.pt")
                log(f"step {step:6d} | train loss {running / n:.4f} | val loss {val_loss:.4f}{mark}")
                running = 0.0
            if step >= args.steps:
                break
        epoch += 1
    log(f"best val loss {best:.4f}, checkpoint {out / 's2m.pt'}")


if __name__ == "__main__":
    main()
