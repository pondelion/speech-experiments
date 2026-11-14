"""Cの全選定条件で速度・CER・メモリがそろっていることを検査し表用データを出す。"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "weight_only"
NAMES = [
    "cpu_fp32", "cpu_both_wo8_float32", "cuda_fp32", "cuda_bf16",
    "cuda_both_wo8_float32", "cuda_both_wo8_float32_gptq",
    "cuda_both_wo4_bfloat16", "cuda_both_wo4_bfloat16_g64", "cuda_both_wo4_bfloat16_g32",
    "cuda_both_wo4_bfloat16_awq", "cuda_both_wo4_bfloat16_plain", "cuda_both_wo4_bfloat16_gptq",
]


def read(path):
    return json.loads(path.read_text())


def main():
    rows = []
    base_t2s_bytes = read(ROOT / "results" / "cpu_both_int8.json")["t2s_state_bytes_fp32"]
    for name in NAMES:
        speed = read((ROOT / "results" / "cpu_fp32.json") if name == "cpu_fp32" else OUT / f"{name}.json")
        memory = read(OUT / f"memory_{name}.json")
        if name == "cpu_fp32":
            raw = read(ROOT / "results" / "quality.json")
            quality = {g: {**x["conditions"][name], "utterances": x["utterances"]} for g,x in raw["groups"].items()}
        else:
            raw = read(OUT / f"quality_{name}.json")
            quality = raw["groups"]
        if not all(v["finite"] for v in quality.values()) or not all(v["finite"] for v in speed["cases"].values()):
            raise RuntimeError(f"有限値検査失敗: {name}")
        assert quality["test"]["utterances"] == 100 and quality["eval_texts"]["utterances"] == 55
        row = {
            "condition": name, "device": "cpu" if name.startswith("cpu_") else "cuda",
            "test_reading_cer_percent": quality["test"]["cer"]["reading"] * 100,
            "practical_reading_cer_percent": quality["eval_texts"]["cer"]["reading"] * 100,
            "t2s_s2m_state_mib": (speed.get("t2s_state_bytes", base_t2s_bytes) + speed["s2m_state_bytes"]) / 2**20,
            "process_peak_rss_mib": memory["process_peak_rss_bytes"] / 2**20,
            "inference_peak_vram_mib": memory.get("long_inference_peak_vram_bytes", 0) / 2**20 or None,
            "load_convert_peak_vram_mib": memory.get("load_convert_peak_vram_bytes", 0) / 2**20 or None,
        }
        for case, values in speed["cases"].items():
            row[f"{case}_median_s"] = values["stages"]["end_to_end_s"]["median_s"]
            row[f"{case}_p90_s"] = values["stages"]["end_to_end_s"]["p90_s"]
            row[f"{case}_rtf"] = values["rtf_median"]
        rows.append(row)
    (OUT / "c_summary.json").write_text(json.dumps({"quality_seed": 0, "speed_repeats": 3,
        "memory_processes_per_condition": 1, "conditions": rows}, indent=2) + "\n")
    with (OUT / "c_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("condition | medium s | test CER% | practical CER% | state MiB | VRAM MiB | RSS MiB")
    for r in rows:
        print(f"{r['condition']} | {r['medium_median_s']:.3f} | {r['test_reading_cer_percent']:.2f} | "
              f"{r['practical_reading_cer_percent']:.2f} | {r['t2s_s2m_state_mib']:.1f} | "
              f"{r['inference_peak_vram_mib']} | {r['process_peak_rss_mib']:.1f}")


if __name__ == "__main__":
    main()
