"""Cの未実施CER・独立processメモリ計測を逐次実行する。依存追加はしない。"""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "weight_only"
VARIANTS = [
    ("cuda_fp32", ["--device", "cuda"]),
    ("cuda_bf16", ["--device", "cuda", "--dtype", "bfloat16"]),
    ("cuda_both_wo8_float32", ["--device", "cuda", "--bits", "8"]),
    ("cuda_both_wo4_bfloat16", ["--device", "cuda", "--bits", "4", "--dtype", "bfloat16"]),
    ("cpu_fp32", ["--device", "cpu"]),
    ("cpu_both_wo8_float32", ["--device", "cpu", "--bits", "8"]),
    ("cuda_both_wo4_bfloat16_g32", ["--device", "cuda", "--bits", "4", "--dtype", "bfloat16", "--group-size", "32"]),
    ("cuda_both_wo4_bfloat16_g64", ["--device", "cuda", "--bits", "4", "--dtype", "bfloat16", "--group-size", "64"]),
    ("cuda_both_wo4_bfloat16_awq", ["--device", "cuda", "--bits", "4", "--dtype", "bfloat16", "--calibration", "awq"]),
    ("cuda_both_wo8_float32_gptq", ["--device", "cuda", "--bits", "8", "--calibration", "gptq"]),
    ("cuda_both_wo4_bfloat16_gptq", ["--device", "cuda", "--bits", "4", "--dtype", "bfloat16", "--calibration", "gptq"]),
    ("cuda_both_wo4_bfloat16_plain", ["--device", "cuda", "--bits", "4", "--dtype", "bfloat16", "--packing", "plain"]),
]


def main():
    for script, prefix in [("evaluate_weight_only.py", "quality_"), ("measure_weight_memory.py", "memory_")]:
        for condition, args in VARIANTS:
            # CPU FP32のC内CERはAの同一seed・同一batch設定の基準を参照する。
            if prefix == "quality_" and condition == "cpu_fp32":
                continue
            path = OUT / f"{prefix}{condition}.json"
            if path.exists():
                json.loads(path.read_text())
                continue
            print(f"START {script} {condition}", flush=True)
            subprocess.run([sys.executable, str(ROOT / script), *args], check=True, cwd=ROOT)
            print(f"DONE {prefix}{condition}", flush=True)
    print("C CER and memory jobs complete", flush=True)


if __name__ == "__main__":
    main()
