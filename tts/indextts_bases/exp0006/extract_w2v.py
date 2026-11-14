"""手順 1: 全発話の w2v-BERT 2.0 の 17 層目の特徴を取り出して保存する(codec の学習と、トークンへの変換で使う)。

公式の推論(`infer_v2_5.py` の get_emb)と同じく、16kHz の波形を SeamlessM4TFeatureExtractor で前処理し、
Wav2Vec2BertModel(float32)の hidden_states[17] を取る(50 フレーム/秒、1,024 次元)。
正規化の平均・分散は、公式は自分たちのデータで求めた値(wav2vec2bert_stats.pt)を使う。ここでは JSUT の学習データから求める。

使い方:
    python extract_w2v.py
    python extract_w2v.py --limit 20 --name w2v_smoke   # 動作確認
出力: runs/{name}.pt({"feats": {発話 ID: [S, 1024] float16}, "mean": [1024], "var": [1024]})。約 3.8GB(作り直せる。exp の終わりに消す)
"""
import argparse
import time

import librosa
import torch
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel

import common
from lib_common.data import list_jsut, split
from lib_common.utils import Logger

LAYER = 17


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="w2v_l17")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    torch.set_num_threads(4)
    log = Logger(common.RUNS_DIR / f"log_{args.name}.txt")
    utts = list_jsut()
    if args.limit:
        utts = split(utts, "train")[: args.limit] + split(utts, "val")[:2] + split(utts, "test")[:2]
    extractor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0").cuda().eval()
    feats, t0 = {}, time.time()
    with torch.no_grad():
        for i, u in enumerate(utts):
            y, _ = librosa.load(u.path, sr=16000)
            inputs = extractor(y, sampling_rate=16000, return_tensors="pt")
            out = model(input_features=inputs["input_features"].cuda(), attention_mask=inputs["attention_mask"].cuda(),
                        output_hidden_states=True)
            feats[u.id] = out.hidden_states[LAYER][0].half().cpu()                  # [S, 1024]
            if (i + 1) % 1000 == 0:
                log(f"  {i + 1}/{len(utts)} ({time.time() - t0:.0f}s)")
    train = [feats[u.id].float() for u in split(utts, "train")]
    n = sum(len(f) for f in train)
    mean = sum(f.sum(0) for f in train) / n
    var = sum(((f - mean) ** 2).sum(0) for f in train) / n
    torch.save({"feats": feats, "mean": mean, "var": var}, common.RUNS_DIR / f"{args.name}.pt")
    log(f"saved {len(feats)} utterances ({n} train frames) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
