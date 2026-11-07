"""分析: codec の音声トークンが、音素ごとにまとまっているか(クラスターになっているか)を見る。

やること:
    1. jsut-lab の音素アライメント(どの時刻にどの音素があるか)を使い、各トークンが受け持つ区間
       (Mel 4 フレーム = 42.7ms)と一番長く重なる音素を、そのトークンの正解の音素にする
    2. 量子化する前の 8 次元ベクトル(encoder の出力。トークンの出現 1 つずつの連続値なので、同じトークン ID でも
       少しずつ違う点になる)を 2 次元に落とし、音素で色分けして描く。落とし方は --method で選ぶ:
         kpca: KernelPCA(RBF カーネル)。サンプル数の 2 乗のメモリを使うので、N_FIT 点で当てはめ、描く点はその軸に写す
         umap: UMAP。近いもの同士の関係を保つように落とすので、まとまり(島)が見えやすい。描く点全部で当てはめる
    3. codebook の代表ベクトル 1024 個(量子化した後に実際に使われる点)も同じ軸に写し、
       それぞれが一番多く当たる音素で色分けして描く
    4. 数値でも確かめる:
       純度: 各トークンについて、一番多い音素の割合(出現回数で重み付けして平均)
       NMI: トークンと音素の正規化相互情報量(0 なら無関係、1 なら一方から他方が完全に決まる)
       1 つの音素の出現の 90% を覆うのに必要なトークンの種類数
       比較として、トークンの並びをシャッフルした(音素と無関係にした)場合の値も出す

使い方:
    python analyze_codec_phoneme.py --method kpca
    python analyze_codec_phoneme.py --method umap
出力: results/fig_codec_phoneme_{method}.jpg、results/codec_phoneme_stats.json
"""
import argparse
import json
import random
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import KernelPCA
from sklearn.metrics import normalized_mutual_info_score

import common  # noqa: F401  tts/lib_common を import できるようにする
from common.codec import DOWNSAMPLE
from common.utils import EXP_DIR, RUNS_DIR, load_codec, pad_mels
from lib_common.audio import HOP, SR
from lib_common.data import align_to_segments, list_jsut, load_jsut_alignment, load_mels, split
from lib_common.utils import length_batches, seed_everything

N_FIT = 5000     # KernelPCA を当てはめる点の数
N_PLOT = 20000   # 図に描く点の数
TOKEN_SEC = DOWNSAMPLE * HOP / SR  # トークン 1 個が受け持つ時間(42.7ms)

# 音素の種類(pyopenjtalk / jsut-lab の表記)。無声化した母音は大文字
CATEGORIES = {
    "vowel": list("aiueo"), "devoiced vowel": list("AIUEO"), "moraic nasal (N)": ["N"], "geminate (cl)": ["cl"],
    "silence/pause": ["sil", "pau"], "plosive": ["p", "b", "t", "d", "k", "g", "py", "by", "ky", "gy"],
    "affricate": ["ch", "ts", "j"], "fricative": ["s", "sh", "z", "f", "h", "hy", "v"],
    "nasal": ["m", "n", "my", "ny"], "liquid": ["r", "ry"], "glide": ["w", "y"],
}
PHONEME_TO_CATEGORY = {p: c for c, ps in CATEGORIES.items() for p in ps}
FOCUS = ["a", "i", "u", "e", "o", "N", "k", "s", "t", "n", "r", "sil", "pau"]  # 個別に色を付ける音素


def stats(ids: np.ndarray, phonemes: np.ndarray) -> dict:
    """トークンと音素の対応の数値をまとめて出す。

    Args:
        ids: トークン ID [N(トークンの出現数)]。
        phonemes: 正解の音素 [N]。

    Returns:
        純度・NMI・1 音素あたりのトークンの種類数などの dict。
    """
    by_token = {}
    for i, p in zip(ids, phonemes):
        by_token.setdefault(i, Counter())[p] += 1
    purity = sum(c.most_common(1)[0][1] for c in by_token.values()) / len(ids)
    by_phoneme = {}
    for i, p in zip(ids, phonemes):
        by_phoneme.setdefault(p, Counter())[i] += 1
    n90 = {}
    for p, c in by_phoneme.items():  # 出現の 90% を覆うのに必要なトークンの種類数
        total, acc, k = sum(c.values()), 0, 0
        for _, n in c.most_common():
            acc, k = acc + n, k + 1
            if acc >= 0.9 * total:
                break
        n90[p] = k
    cats = np.array([PHONEME_TO_CATEGORY.get(p, "other") for p in phonemes])
    return {"purity_phoneme": purity, "nmi_phoneme": normalized_mutual_info_score(phonemes, ids),
            "nmi_category": normalized_mutual_info_score(cats, ids),
            "tokens_for_90pct": {p: n90[p] for p in sorted(n90, key=lambda p: -sum(by_phoneme[p].values()))},
            "n_tokens_used": len(by_token)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_utts", type=int, default=1500, help="使う発話数(検証データ 100 + 学習データから残り)")
    parser.add_argument("--method", choices=["kpca", "umap"], default="kpca", help="2 次元に落とす方法")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    torch.set_num_threads(4)
    utts = list_jsut()
    use = split(utts, "val") + random.Random(args.seed).sample(split(utts, "train"), args.n_utts - 100)
    load_mels(use, log=lambda m: None)
    codec, norm = load_codec(RUNS_DIR / "codec" / "codec.pt", args.device)

    # 各トークンの、量子化する前のベクトル・トークン ID・正解の音素を集める
    vecs, ids, phs, overlaps = [], [], [], []
    mels = [norm.normalize(u.mel) for u in use]
    with torch.no_grad():
        for b in length_batches([len(m) for m in mels], 32000, shuffle=False):
            mel, _ = pad_mels([mels[i] for i in b], args.device)               # [B(発話数), T(フレーム数), F(80)]
            e = codec.encode_vectors(mel)                                      # [B, U(トークン数), C(8)]
            tok = codec.quantize(e)[0]                                         # [B, U]
            for j, i in enumerate(b):
                n = len(mels[i]) // DOWNSAMPLE                                 # この発話のトークン数(詰め物を除く)
                for p, o in align_to_segments(load_jsut_alignment(use[i]), n, TOKEN_SEC):
                    phs.append(p)
                    overlaps.append(o)
                vecs.append(e[j, :n].cpu())
                ids.append(tok[j, :n].cpu())
    vecs = torch.cat(vecs).numpy()                                             # [N(全トークン), C]
    ids = torch.cat(ids).numpy()                                               # [N]
    phs, overlaps = np.array(phs), np.array(overlaps)
    print(f"{len(use)} utterances, {len(ids)} tokens, mean overlap with the assigned phoneme {overlaps.mean():.2f}")

    # 数値: 実際の対応と、トークンをシャッフルして音素と無関係にした場合
    result = {"n_utterances": len(use), "n_tokens": int(len(ids)), "mean_overlap": float(overlaps.mean()),
              "actual": stats(ids, phs), "shuffled": stats(np.random.default_rng(0).permutation(ids), phs)}
    for k in ["purity_phoneme", "nmi_phoneme", "nmi_category"]:
        print(f"{k}: actual {result['actual'][k]:.3f} / shuffled {result['shuffled'][k]:.3f}")
    print("tokens to cover 90% of each phoneme (top 12):", dict(list(result["actual"]["tokens_for_90pct"].items())[:12]))

    # 2 次元に落とす(量子化する前のベクトルで当てはめ、codebook の代表ベクトルも同じ軸に写す)
    rng = np.random.default_rng(args.seed)
    plot_idx = rng.choice(len(vecs), N_PLOT, replace=False)
    if args.method == "kpca":
        fit_idx = rng.choice(len(vecs), N_FIT, replace=False)
        reducer = KernelPCA(n_components=2, kernel="rbf", random_state=args.seed).fit(vecs[fit_idx])
        method_name = "KernelPCA (RBF)"
    else:
        import umap  # 使うときだけ import する(読み込みに時間がかかる)
        reducer = umap.UMAP(n_components=2, random_state=args.seed).fit(vecs[plot_idx])
        method_name = "UMAP"
    z = reducer.transform(vecs[plot_idx])                                       # [N_PLOT, 2]
    book = torch.nn.functional.normalize(codec.codebook.weight, dim=-1).detach().cpu().numpy()  # [K(1024), C]
    zb = reducer.transform(book)                                                # [K, 2]
    majority = {}
    for i, p in zip(ids, phs):
        majority.setdefault(i, Counter())[p] += 1
    book_cat = [PHONEME_TO_CATEGORY.get(majority[k].most_common(1)[0][0], "other") if k in majority else "unused"
                for k in range(len(book))]

    cmap = plt.get_cmap("tab20")
    cat_color = {c: cmap(i) for i, c in enumerate(list(CATEGORIES) + ["other"])}
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    cats = np.array([PHONEME_TO_CATEGORY.get(p, "other") for p in phs[plot_idx]])
    for c in cat_color:
        m = cats == c
        if m.any():
            axes[0].scatter(z[m, 0], z[m, 1], s=2, color=cat_color[c], label=f"{c} ({m.sum()})")
    axes[0].set_title(f"pre-quantization encoder outputs ({N_PLOT} token occurrences, continuous)\n"
                      "colored by phoneme category")
    focus_cmap = plt.get_cmap("tab20")
    p_plot = phs[plot_idx]
    other = ~np.isin(p_plot, FOCUS)
    axes[1].scatter(z[other, 0], z[other, 1], s=1, color="lightgray", label="others")
    for k, p in enumerate(FOCUS):
        m = p_plot == p
        axes[1].scatter(z[m, 0], z[m, 1], s=2, color=focus_cmap(k), label=f"{p} ({m.sum()})")
    axes[1].set_title(f"pre-quantization encoder outputs ({N_PLOT} token occurrences, continuous)\n"
                      "colored by main phonemes")
    for c in list(cat_color) + ["unused"]:
        m = np.array([bc == c for bc in book_cat])
        if m.any():
            axes[2].scatter(zb[m, 0], zb[m, 1], s=10, color=cat_color.get(c, "black"), label=f"{c} ({m.sum()})")
    axes[2].set_title("codebook vectors (the 1024 points tokens are quantized to)\n"
                      "colored by the category of their most frequent phoneme")
    for ax in axes:
        ax.legend(fontsize=7, markerscale=4, loc="best")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{method_name} of codec token vectors | purity {result['actual']['purity_phoneme']:.2f} "
                 f"(shuffled {result['shuffled']['purity_phoneme']:.2f}), NMI {result['actual']['nmi_phoneme']:.2f} "
                 f"(shuffled {result['shuffled']['nmi_phoneme']:.2f})")
    fig.tight_layout()
    (EXP_DIR / "results").mkdir(exist_ok=True)
    fig.savefig(EXP_DIR / "results" / f"fig_codec_phoneme_{args.method}.jpg", dpi=100)
    (EXP_DIR / "results" / "codec_phoneme_stats.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved results/fig_codec_phoneme_{args.method}.jpg and results/codec_phoneme_stats.json")


if __name__ == "__main__":
    main()
