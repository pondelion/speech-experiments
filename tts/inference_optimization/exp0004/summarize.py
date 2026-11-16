"""正式測定を集約し、READMEの比較表と試聴ページを更新する。"""

import argparse
import csv
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MEASURED = ROOT / "results" / "measured"
PREVIOUS = ROOT.parent / "exp0003" / "results"
FINAL_CONFIGS = {
    "cpu_torch_int8",
    "cpu_native8_nospin_8",
    "cpu_native7_nospin_8",
    "cuda_torch",
    "cuda_opt_nospin_8_front1_cg",
    "cuda_opt_nospin_8_front1_cg128",
}
CASES = [
    ("cpu", "torch", "PyTorch FP32・8 threads"),
    ("cpu", "torch_16", "PyTorch FP32・16 threads"),
    ("cpu", "torch_int8", "PyTorch 動的INT8"),
    ("cpu", "legacy", "ORT FP32・exp3設定"),
    ("cpu", "attn_shared_16_front1", "ORT Attention FP32"),
    ("cpu", "attn8_nospin_8", "ORT Attention INT8"),
    ("cpu", "native8_nospin_8", "ORT INT8・RoPE融合＋量子化条件調整"),
    ("cpu", "native7_nospin_8", "ORT INT8・入力量子化範囲をPyTorchへ合わせる"),
    ("cuda", "torch", "PyTorch FP32"),
    ("cuda", "torch_bf16", "PyTorch BF16"),
    ("cuda", "legacy", "ORT FP32・exp3設定"),
    ("cuda", "attn_nospin_8_front1", "ORT Attention FP32"),
    ("cuda", "attn16_nospin_8_front1", "ORT Attention FP16"),
    ("cuda", "opt_nospin_8_front1_cg", "ORT FP32・RoPE融合＋CUDA Graph"),
    ("cuda", "opt_nospin_8_front1_cg128", "ORT FP32・CUDA Graph＋文長bucket"),
]
REFERENCES = {
    "cpu_torch": "runtime/quality_cpu_g_torch.json",
    "cpu_torch_int8": "quality_b.json",
    "cpu_legacy": "runtime/quality_cpu_g_fp32.json",
    "cuda_torch": "runtime/quality_cuda_g_torch.json",
    "cuda_torch_bf16": "weight_only/quality_cuda_bf16.json",
    "cuda_legacy": "runtime/quality_cuda_g_fp32.json",
}


def read(path):
    return json.loads(path.read_text())


def quality(condition):
    path = MEASURED / f"quality_{condition}.json"
    if path.exists():
        return read(path)["groups"], path
    if condition not in REFERENCES:
        return None, None
    path = PREVIOUS / REFERENCES[condition]
    groups = read(path)["groups"]
    if condition == "cpu_torch_int8":
        groups = {
            name: {
                "utterances": group["utterances"],
                **group["conditions"]["cpu_both_int8"],
            }
            for name, group in groups.items()
        }
    return groups, path


def replace_section(text, name, body):
    begin = f"<!-- {name}_START -->"
    end = f"<!-- {name}_END -->"
    return (
        text[: text.index(begin) + len(begin)]
        + "\n"
        + body
        + "\n"
        + text[text.index(end) :]
    )


def table(rows, keys, header):
    lines = [header, "|" + "---|" * len(keys)]
    for row in rows:
        cells = []
        for key in keys:
            value = row.get(key, "—")
            cells.append(f"{value:.3f}" if isinstance(value, float) else value)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def publish(rows):
    manifest = ROOT.parents[2] / "site" / "manifest.json"
    text = manifest.read_text()
    start = text.index('"id": "inference_optimization"')
    start = text.index("[", text.index('"exps"', start))
    experiments, length = json.JSONDecoder().raw_decode(text[start:])
    groups = []
    for device in ["cpu", "cuda"]:
        conditions = []
        for row in rows:
            if row["device"] != device:
                continue
            conditions.append(
                {
                    "id": row["condition"],
                    "label": row["method"],
                    "cer": {
                        "test": row["test_cer_pct"] / 100,
                        "eval_texts": row["eval_texts_cer_pct"] / 100,
                    },
                    "default": row["config"]
                    in [
                        "torch_int8",
                        "native8_nospin_8",
                        "torch",
                        "opt_nospin_8_front1_cg",
                        "opt_nospin_8_front1_cg128",
                    ],
                }
            )
        groups.append({"title": device.upper(), "conditions": conditions})
    entry = {
        "id": "exp0004",
        "title": "exp0004: ONNX Runtimeの性能改善",
        "dir": "tts/inference_optimization/exp0004",
        "summary": (
            "thread管理とAttention融合を切り分け、"
            "CPU/GPUの速度・CER・メモリを比較。"
        ),
        "samples": "tts/inference_optimization/exp0004/results/samples",
        "condition_groups": groups,
    }
    experiments = [e for e in experiments if e["id"] != entry["id"]]
    experiments.append(entry)
    replacement = json.dumps(experiments, ensure_ascii=False, indent=2)
    replacement = replacement.replace("\n", "\n      ")
    manifest.write_text(text[:start] + replacement + text[start + length :])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    rows = []
    for device, config, label in CASES:
        condition = f"{device}_{config}"
        row = dict(
            condition=condition, device=device, config=config, method=label
        )
        bench = MEASURED / f"bench_{condition}.json"
        memory = MEASURED / f"memory_{condition}.json"
        groups, source = quality(condition)
        row["complete"] = bench.exists() and memory.exists() and bool(groups)
        if bench.exists():
            data = read(bench)
            if not all(c["finite"] for c in data["cases"].values()):
                raise RuntimeError(f"Nonfinite benchmark: {condition}")
            row["state_mib"] = sum(data["state_bytes"].values()) / 2**20
            row["load_prepare_s"] = data["load_prepare_s"]
            row["first_warmup_s"] = data["cases"]["short"]["warmup_s"]
            for name, case in data["cases"].items():
                row[f"{name}_s"] = case["stages"]["end_to_end_s"]["median_s"]
                row[f"{name}_rtf"] = case["rtf"]
        if groups:
            for name, group in groups.items():
                if not group["finite"]:
                    raise RuntimeError(f"Nonfinite quality: {condition}")
                row[f"{name}_cer_pct"] = group["cer"]["reading"] * 100
            row["quality_source"] = str(source.relative_to(ROOT.parent))
            row["quality_reused"] = source.parent != MEASURED
        if memory.exists():
            data = read(memory)
            row["rss_mib"] = data["process_peak_rss_bytes"] / 2**20
            if device == "cuda":
                row["vram_mib"] = data["device_used_peak_delta_bytes"] / 2**20
        rows.append(row)
    complete = all(r["complete"] for r in rows)
    for row in rows:
        row["measurement_round"] = (
            "final" if row["condition"] in FINAL_CONFIGS else "initial"
        )
        if row["condition"] not in FINAL_CONFIGS:
            continue
        for case in ["short", "medium", "long"]:
            key = f"{case}_s"
            native = [
                r
                for r in rows
                if r["device"] == row["device"]
                and r["config"].startswith("torch")
                and key in r
            ]
            if key not in row or not native:
                continue
            best = min(native, key=lambda r: r[key])
            row[f"{case}_torch_baseline"] = best["condition"]
            row[f"{case}_reduction_vs_torch_pct"] = 100 * (
                1 - row[key] / best[key]
            )
    if args.require_complete and not complete:
        missing = [r["condition"] for r in rows if not r["complete"]]
        raise RuntimeError(f"Incomplete: {missing}")
    output = ROOT / "results"
    (output / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    )
    buffer = io.StringIO()
    fields = list(dict.fromkeys(k for row in rows for k in row))
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    (output / "summary.csv").write_text(buffer.getvalue())
    keys = [
        "device",
        "method",
        "short_s",
        "medium_s",
        "long_s",
        "test_cer_pct",
        "eval_texts_cer_pct",
        "medium_reduction_vs_torch_pct",
    ]
    body = "### C.1: PyTorch最速条件との最終比較\n\n" + table(
        [r for r in rows if r["condition"] in FINAL_CONFIGS],
        keys,
        (
            "| Device | 条件 | 短文 秒 | 中文 秒 | 長文 秒 | "
            "テストCER % | 実用文CER % | PyTorch最速比・中文短縮 % |"
        ),
    )
    body += (
        "\n\n### C.2: 改善途中の条件（参考）\n\n"
        "以下は初回の測定値で、最終比較とは測定時期が異なる。"
        "PyTorch最速比の短縮率は最終比較の条件だけに示す。\n\n"
    )
    body += table(
        [r for r in rows if r["condition"] not in FINAL_CONFIGS],
        keys[:-1],
        "| Device | 条件 | 短文 秒 | 中文 秒 | 長文 秒 | "
        "テストCER % | 実用文CER % |",
    )
    body += "\n\n### C.3: 重みサイズと実行時メモリ\n\n" + table(
        rows,
        ["device", "method", "state_mib", "rss_mib", "vram_mib"],
        "| Device | 条件 | 重み・graph MiB | peak RSS MiB "
        "| device VRAM増分 MiB |",
    )
    path = ROOT / "README.md"
    path.write_text(replace_section(path.read_text(), "FORMAL_RESULTS", body))
    if complete:
        publish(rows)
    print(f"Complete: {sum(r['complete'] for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
