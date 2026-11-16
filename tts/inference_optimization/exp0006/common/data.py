"""既存トークンcacheとテキスト・音素ラベルによる学習データ。"""

import hashlib
import random
import unicodedata

import torch

from .paths import CHECKPOINTS, RUNS, benchmark_module, digest, relative


def text_key(text):
    text = unicodedata.normalize("NFKC", text)
    return "".join(c for c in text if c.isalnum())


def load_data(vocab):
    """音声・画像を読み込まず、既存の離散トークンを利用する。"""
    benchmark_module().import_models()
    from exp0002.common.utils import TOKEN_SEC, prepare
    from lib_common.data import list_jsut, split
    from lib_common.eval_sets import load_eval_texts

    token_path = CHECKPOINTS["codec"].parent / "tokens.pt"
    cache_key = {
        "tokens_sha256": digest(token_path),
        "teacher_sha256": digest(CHECKPOINTS["t2s"]),
        "schema": 1,
    }
    data_path = RUNS / "data" / "prepared.pt"
    if data_path.exists():
        saved = torch.load(data_path, map_location="cpu", weights_only=False)
        if saved["identity"] == cache_key:
            return saved["data"], saved["audit"]
    tokens = torch.load(token_path, map_location="cpu", weights_only=False)
    utterances = list_jsut()
    practical = load_eval_texts(["practical"])
    held_out = split(utterances, "test") + practical
    val_keys = {text_key(u.text) for u in split(utterances, "val")}
    evaluation_keys = {text_key(u.text) for u in held_out}
    data, audit = (
        {},
        {
            "identity": cache_key,
            "tokens_path": relative(token_path),
            "evaluation_counts": {
                "test": len(split(utterances, "test")),
                "practical": len(practical),
            },
            "groups": {},
        },
    )
    for group in ("train", "val"):
        original = split(utterances, group)
        forbidden = evaluation_keys | (val_keys if group == "train" else set())
        excluded = [u.id for u in original if text_key(u.text) in forbidden]
        selected = [u for u in original if text_key(u.text) not in forbidden]
        missing = [u.id for u in selected if u.id not in tokens]
        if missing:
            raise ValueError(f"Missing cached tokens: {missing[:5]}")
        prepared = prepare(
            selected,
            vocab,
            None,
            None,
            "cpu",
            True,
            [tokens[u.id].long() for u in selected],
            TOKEN_SEC,
        )
        ids = [selected[i].id for i in prepared["keep"]]
        data[group] = prepared
        audit["groups"][group] = {
            "original": len(original),
            "excluded_text_overlap_ids": excluded,
            "alignment_dropped": prepared["n_dropped"],
            "used": len(ids),
            "ids": ids,
            "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        }
        if not ids:
            raise ValueError(f"Empty prepared split: {group}")
    assert not set(audit["groups"]["train"]["ids"]) & set(
        audit["groups"]["val"]["ids"]
    )
    data_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"identity": cache_key, "data": data, "audit": audit}, data_path
    )
    return data, audit


def select_batch(data, count=2):
    """動作確認には最短だけでなく長さの異なる発話を使う。"""
    ordered = sorted(
        range(len(data["codes"])), key=lambda i: len(data["codes"][i])
    )
    rng = random.Random(0)
    return sorted(rng.sample(ordered, min(count, len(ordered))))


def batches(data, max_tokens, epoch=0, shuffle=True):
    benchmark_module().import_models()
    from lib_common.utils import length_batches

    lengths = [len(t) + len(c) for t, c in zip(data["texts"], data["codes"])]
    return length_batches(lengths, max_tokens, shuffle=shuffle, seed=epoch)
