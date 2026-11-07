"""分析: T2S の予測が正解と違うとき、正解に「近い」トークンを選べているかを測る。

背景:
    codec の codebook は 1024 個あるが、日本語の音の種類はずっと少ないので、似た音に複数のトークンが
    割り当てられているはず。そうなら、T2S の正解率(1 位の予測がぴったり正解と一致する割合)が低くても、
    代表ベクトルが正解に近いトークンを選べていれば、音としてはそれほど崩れない可能性がある。

測り方(検証データ 100 文、teacher forcing = 正解のトークン列を入力して、各位置で次のトークンを予測させる):
    top1 / top5: 1 位の予測 / 上位 5 候補に正解が入る割合
    cosine: 予測したトークンと正解のトークンの代表ベクトル(L2 正規化した 8 次元)の cosine 類似度
            比較として、ランダムに選んだトークンと正解の類似度、正解の 1 つ前のトークンとの類似度も測る
    注意: この近さは codec が近さを測る 8 次元の空間でのもので、耳で聞いた近さと一致するとは限らない

使い方:
    python analyze_t2s_near_miss.py t2s_ph_drop03 t2s_ph_base t2s_char_drop03
出力: results/t2s_near_miss.json(runs/ の中の名前ごとの値)
"""
import argparse
import json

import torch
import torch.nn.functional as F
from einops import rearrange

import common  # noqa: F401  tts/lib_common を import できるようにする
from common.pipeline import load_t2s
from common.utils import EXP_DIR, RUNS_DIR, encode_all, load_codec
from lib_common.data import list_jsut, load_mels, split
from lib_common.utils import seed_everything


@torch.no_grad()
def teacher_forced_logits(t2s, text: torch.Tensor, codes: torch.Tensor, device: str) -> torch.Tensor:
    """1 発話について、正解のトークン列を入力したときの、各位置の次のトークンのスコアを返す。

    Args:
        t2s: 学習済みの T2S。
        text: テキスト ID 列 [L(テキストのトークン数)]。
        codes: 正解の音声トークン列 [U(音声トークン数)]。
        device: デバイス。

    Returns:
        U 個の正解トークンそれぞれを予測する位置のスコア [U, K(トークンの種類数。<BA>・<EA> は除く)]。
        (以下のコメントの記号: D 隠れ層の次元)
    """
    audio_in = torch.cat([codes.new_tensor([t2s.BA]), codes]).to(device)                      # <BA> q1 ... qU [U+1]
    x = torch.cat([t2s._text_embeds(text.to(device)), t2s._audio_embeds(audio_in)])          # [L + U + 1, D]
    h = t2s.gpt(inputs_embeds=rearrange(x, "s d -> 1 s d")).last_hidden_state                 # [1, L + U + 1, D]
    # <BA> の位置(L 番目)から q_{U-1} の位置までが、q1 ... qU を予測する位置
    h = rearrange(h, "1 s d -> s d")[len(text): len(text) + len(codes)]                       # [U, D]
    return t2s.head(h)[:, : t2s.n_codes]                                                      # [U, K]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="runs/ の中の T2S の名前")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    torch.set_num_threads(4)
    val = split(list_jsut(), "val")
    load_mels(val, log=lambda m: None)
    codec, norm = load_codec(RUNS_DIR / "codec" / "codec.pt", args.device)
    codes = encode_all(codec, [norm.normalize(u.mel) for u in val], args.device)  # 各 [U_i]
    book = F.normalize(codec.codebook.weight, dim=-1)                              # 代表ベクトル [K, C(8)]

    out = {}
    for name in args.runs:
        t2s, vocab = load_t2s(RUNS_DIR / name / "t2s.pt", args.device)
        preds, gts, in_top5 = [], [], []
        for u, c in zip(val, codes):
            logits = teacher_forced_logits(t2s, torch.tensor(vocab.encode(u.text)), c, args.device)  # [U, K]
            preds.append(logits.argmax(-1).cpu())
            gts.append(c)
            in_top5.append((logits.topk(5, -1).indices.cpu() == rearrange(c, "u -> u 1")).any(-1))
        p, g, top5 = torch.cat(preds), torch.cat(gts), torch.cat(in_top5)          # 全発話の全位置 [N]
        cos = lambda a, b: (book[a.to(args.device)] * book[b.to(args.device)]).sum(-1).cpu()  # 2 つの ID 列の類似度 [N]
        sim = cos(p, g)
        rnd = cos(torch.randint(0, codec.codebook_size, g.shape), g)               # ランダムなトークンとの類似度
        prev = cos(g[:-1], g[1:])                                                  # 正解の 1 つ前のトークンとの類似度
        wrong = p != g
        out[name] = {
            "top1_acc": (~wrong).float().mean().item(), "top5_acc": top5.float().mean().item(),
            "cos_all": sim.mean().item(), "cos_wrong": sim[wrong].mean().item(),
            "wrong_cos_gt_0.9": (sim[wrong] > 0.9).float().mean().item(),
            "wrong_cos_gt_0.7": (sim[wrong] > 0.7).float().mean().item(),
            "random_cos": rnd.mean().item(), "random_cos_gt_0.9": (rnd > 0.9).float().mean().item(),
            "random_cos_gt_0.7": (rnd > 0.7).float().mean().item(), "prev_gt_cos": prev.mean().item(),
            "n_positions": len(g),
        }
        print(name, {k: round(v, 3) for k, v in out[name].items()}, flush=True)
    (EXP_DIR / "results").mkdir(exist_ok=True)
    (EXP_DIR / "results" / "t2s_near_miss.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
