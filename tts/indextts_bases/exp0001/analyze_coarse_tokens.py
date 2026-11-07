"""分析: codec の 1024 種類のトークンを、k-means で少ない種類(粗いトークン)にまとめても読めるかを測る。

T2S が当てにくい原因の 1 つは、トークンが細かすぎる(似た音に複数のトークンがある)ことだと考えられる。
そこで代表ベクトル 1024 個を k-means で K グループにまとめ、T2S にはグループの番号を当てさせる案を試す。
S2M は codec のトークンで学習してあるので、各グループの中で中心に一番近い代表ベクトルのトークンを「グループの代表」とし、
S2M にはそれを渡す。この方法で、正解のトークンを粗くしても読めるか(粗くしたときの CER の上限)を先に測る。

やること:
    1. K ごとに、代表ベクトル(L2 正規化した 8 次元)を k-means でまとめ、対応表を runs/coarse/k{K}.pt に保存する
       from_fine [1024]: codec のトークン → グループの番号
       to_fine [K]:      グループの番号 → S2M に渡す codec のトークン(グループの代表)
       centroids [K, C(8)]: グループの中心(L2 正規化)。T2S の soft_ce・regression で使う
    2. テスト 100 文の正解トークンを「グループの番号 → グループの代表」に置き換えて S2M → Griffin-Lim で音にし、CER を測る

使い方:
    python analyze_coarse_tokens.py --ks 64 128 256
出力: runs/coarse/k{K}.pt、runs/eval/coarse_k{K}/(音声)、results/coarse_tokens.json
"""
import argparse
import json

import soundfile as sf
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

import common  # noqa: F401  tts/lib_common を import できるようにする
from common.pipeline import TTSPipeline
from common.utils import EXP_DIR, RUNS_DIR, encode_all
from lib_common.asr_eval import ASR, cer_scores
from lib_common.audio import SR
from lib_common.data import load_dataset, split
from lib_common.utils import Logger, seed_everything


def make_coarse_map(code_vecs: torch.Tensor, k: int, seed: int) -> dict:
    """代表ベクトルを k-means で k グループにまとめ、対応表を作る。

    Args:
        code_vecs: L2 正規化した代表ベクトル [1024(codec のトークンの種類数), C(8)]。
        k: グループの数。
        seed: k-means の乱数の seed。

    Returns:
        {"from_fine": [1024], "to_fine": [k], "centroids": [k, C]} の dict。
    """
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(code_vecs.numpy())
    from_fine = torch.from_numpy(km.labels_).long()                                     # [1024]
    centroids = F.normalize(torch.from_numpy(km.cluster_centers_).float(), dim=-1)      # [k, C]
    sim = code_vecs @ centroids.T                                                       # 各トークンと各中心の類似度 [1024, k]
    to_fine = torch.stack([torch.where(from_fine == g, sim[:, g], torch.tensor(-2.0)).argmax() for g in range(k)])
    return {"from_fine": from_fine, "to_fine": to_fine, "centroids": centroids}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ks", type=int, nargs="+", default=[64, 128, 256], help="グループの数")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    torch.set_num_threads(4)
    log = Logger(RUNS_DIR / "eval" / "log_coarse.txt")
    test = split(load_dataset(None, log), "test")
    pipe = TTSPipeline.from_checkpoints(RUNS_DIR / "codec" / "codec.pt", RUNS_DIR / "s2m" / "s2m.pt", device=args.device)
    gt_codes = encode_all(pipe.codec, [pipe.normalizer.normalize(u.mel) for u in test], args.device)  # 各 [U_i]
    code_vecs = F.normalize(pipe.codec.codebook.weight.detach().cpu(), dim=-1)          # [1024, C]
    asr = ASR(args.device)
    out = {}
    (RUNS_DIR / "coarse").mkdir(parents=True, exist_ok=True)
    for k in args.ks:
        cmap = make_coarse_map(code_vecs, k, args.seed)
        torch.save(cmap, RUNS_DIR / "coarse" / f"k{k}.pt")
        coarse = [cmap["to_fine"][cmap["from_fine"][c]] for c in gt_codes]              # 正解 → グループの代表 [U_i]
        d = RUNS_DIR / "eval" / f"coarse_k{k}"
        d.mkdir(parents=True, exist_ok=True)
        for u, m in zip(test, pipe.codes_to_mel(coarse)):
            sf.write(d / f"{u.id}.wav", pipe.mel_to_wave(m).numpy(), SR)
        hyps = asr.transcribe([str(d / f"{u.id}.wav") for u in test])
        out[f"k{k}"] = cer_scores([u.text for u in test], hyps)
        log(f"K={k:4d}: oracle CER (GT tokens → coarse → representative → S2M) reading {out[f'k{k}']['reading']:.4f}")
    (EXP_DIR / "results").mkdir(exist_ok=True)
    (EXP_DIR / "results" / "coarse_tokens.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
