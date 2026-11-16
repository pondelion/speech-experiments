"""実験内の保存先と、公開結果の相対パス・JSON出力。"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
RESULTS = ROOT / "results"
RUNS = ROOT / "runs"
BASE = REPO / "tts" / "indextts_bases" / "exp0003" / "runs"
CHECKPOINTS = {
    name: BASE / f"{name}_hubert_ja_k256" / f"{name}.pt"
    for name in ("codec", "s2m", "t2s")
}


def relative(path):
    """公開ファイルにはリポジトリ相対パスだけを記録する。"""
    return str(Path(path).resolve().relative_to(REPO))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    """途中で中断しても完成済み結果を壊さない。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    )
    os.replace(temporary, path)


def benchmark_module():
    """既存ベンチマークを共通名の衝突なしで読み込む。"""
    path = ROOT.parent / "exp0001" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("speech_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
