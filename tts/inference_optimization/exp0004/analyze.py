"""既存の区間計測と演算profileからボトルネックの表を作る。"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    records = {}
    baseline = json.loads(
        (
            ROOT / "results" / "measured" / "bench_cpu_torch_int8.json"
        ).read_text()
    )
    cpu_t2s = baseline["cases"]["medium"]["stages"]["t2s_tokens_s"]["median_s"]
    cpu_s2m = baseline["cases"]["medium"]["stages"]["s2m_s"]["median_s"]
    lines = [
        "### TTS全体のブロック別時間",
        "",
        "中文の区間別計測では、CPUの旧ORT版はT2Sが2.706秒、"
        f"S2Mが3.748秒だった。最速PyTorch版の{cpu_t2s:.3f}秒・"
        f"{cpu_s2m:.3f}秒と比べ、"
        "両方に大きな差があるため、CPUではこの2区間の計算本体を改善対象にした。"
        "一方、GPUでは最速PyTorch版でもORT版でもT2Sが全体の約8割を占めるため、"
        "GPUではT2Sを優先して改善する。区間別の測定値を以下に示す。",
        "",
        "正式計測の中文・10回の中央値。各区間と全体は独立の中央値で、"
        "丸めや試行間変動により和は完全には一致しない。",
        "",
        (
            "| 条件 | 前処理＋長さ予測 秒 | T2S 秒 | S2M 秒 | "
            "ボコーダ 秒 | 全体 秒 |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    cases = [
        "cpu_torch_int8",
        "cpu_legacy",
        "cpu_attn_shared_16_front1",
        "cpu_attn8_nospin_8",
        "cuda_torch",
        "cuda_legacy",
        "cuda_attn_nospin_8_front1",
        "cuda_opt_nospin_8_front1_cg",
        "cuda_opt_nospin_8_front1_cg128",
        "cpu_native8_nospin_8",
        "cpu_native7_nospin_8",
    ]
    for name in cases:
        path = ROOT / "results" / "measured" / f"bench_{name}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        stages = data["cases"]["medium"]["stages"]
        values = {k: v["median_s"] for k, v in stages.items()}
        records[name] = values
        cells = [values["text_preprocess_s"] + values["duration_s"]]
        cells += [
            values[k]
            for k in ["t2s_tokens_s", "s2m_s", "vocoder_s", "end_to_end_s"]
        ]
        lines.append(
            "| " + name + " | " + " | ".join(f"{v:.3f}" for v in cells) + " |"
        )
    lines += [
        "",
        "また、初回のthread探索ではPyTorch側を1 threadにしたため、"
        "ボコーダが約0.055→0.229秒へ逆に遅くなった。"
        "この結果から、ONNX本体と前後処理のthread数を一律に減らす設定は避け、"
        "新しいCPU候補では前後処理を8 threadsへ戻した。",
        "",
        "### CPUの演算別時間と残存ボトルネック",
        "",
        "演算別profileでは、Attention融合後もMatMulがT2Sで約1,920 ms、"
        "S2Mで約2,413 msを占め、node単位ではMLPのLinearが上位に残った。"
        "さらにSliceにも約425/596 msかかっていた。"
        "この結果から、MLPの行列積をINT8化し、RoPE・時刻条件分割の小演算を"
        "専用演算へ置き換える方針を選んだ。以下に演算別の時間と割合を示す。",
        "",
        "以下はORTのnode event累積。各profileのwarmup＋中文1回を含む。"
        "profileありの診断値を、上のprofileなしの全体時間に足し合わせない。"
        "比率はそのgraph内の全演算event時間を分母とする。",
    ]
    for config in [
        "legacy",
        "attn_shared_16_front1",
        "attn8_nospin_8",
        "native8_nospin_8",
        "native7_nospin_8",
    ]:
        path = ROOT / "results" / "screen" / f"cpu_{config}_profile.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        lines += [
            "",
            f"**{config}**",
            "",
            "| 計算graph | 演算 | 累積 ms | graph内比率 % |",
            "|---|---|---:|---:|",
        ]
        for graph, profile in data["profiles"].items():
            if graph == "encoder":
                continue
            ops = profile["operator_ms"]
            total = sum(ops.values())
            selected = list(ops.items())[:6]
            rotary = "RotaryEmbedding:CPUExecutionProvider"
            if rotary in ops and rotary not in dict(selected):
                selected.append((rotary, ops[rotary]))
            for op, duration in selected:
                op = op.replace(":CPUExecutionProvider", "")
                lines.append(
                    f"| {graph} | {op} | {duration:.1f} "
                    f"| {100 * duration / total:.1f} |"
                )
    lines += [
        "",
        "**観測結果と対策**: 旧S2MではWhereが1,732.5 ms、"
        "IsNaNが291.1 msを占め、Attention分解の補助演算が無視できなかった。"
        "融合後もT2S/S2MのMatMulはそれぞれ約1,920/2,413 msで最大。"
        "node単位ではMLPの2つのLinearが上位を占める。"
        "加えてSliceが約425/596 ms残り、RoPEや固定幅分割が"
        "多数の小演算へ分解されている。これを専用演算へまとめる。",
        "",
        "元データ: `results/screen/*_profile.json`。"
        "CPUのthread条件が異なるprofileのため、演算時間の差すべてを"
        "Attention融合だけの効果と解釈しない。GPUのORT node eventは"
        "GPU純kernel時間とは区別し、CPUの表をGPUの内訳として流用しない。",
    ]
    lines += [
        "",
        "### GPUのCUDA eventとCPU起動APIの内訳",
        "",
        "旧ORT版の中文では、CUDA event累積が約219 msだったのに対し、"
        "CPU側のCUDA起動APIには約633 msかかっていた。"
        "この結果から、GPU演算だけを速くしても起動負荷が残ると判断し、"
        "専用演算への置換で小kernelをまとめ、CUDA Graphで起動処理を減らした。"
        "以下に、その判断に用いた測定値と改善後の内訳を示す。",
        "",
        "同じ中文1回をtorch profilerで採取。profileによる遅延増加があるため、"
        "正式速度ではなく負荷の切り分けに使う。CUDA eventはkernelとcopy等の"
        "device eventの累積で、重なりを除いたGPU稼働率ではない。",
        "",
        "| 診断対象 | CUDA event累積 ms | 起動APIのCPU累積 ms "
        "| CUDA event数 |",
        "|---|---:|---:|---:|",
    ]
    kernel_records = []
    for config in ["legacy", "opt_nospin_8_front1_cg"]:
        path = ROOT / "results" / "screen" / f"cuda_{config}_kernels.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())["kernel_profile"]
        kernel_records.append((config, data))
        lines.append(
            f"| {config} | {data['cuda_event_ms']:.1f} "
            f"| {data['cuda_launch_api_ms']:.1f} | {data['event_count']} |"
        )
    for config, data in kernel_records:
        lines += [
            "",
            f"**{config}: CUDA eventの上位内訳**",
            "",
            "| kernel・処理 | 累積 ms |",
            "|---|---:|",
        ]
        groups = {}
        for name, duration in data["top_events_ms"].items():
            lowered = name.lower()
            if "attention" in lowered:
                label = "Attention"
            elif "gemm" in lowered or "cutlass::kernel" in lowered:
                label = "行列積（GEMM）"
            elif "norm" in lowered:
                label = "正規化"
            elif "rotary" in lowered:
                label = "RoPE"
            elif "elementwise" in lowered:
                label = "要素演算"
            else:
                label = "その他の上位event"
            groups[label] = groups.get(label, 0) + duration
        for label, duration in sorted(groups.items(), key=lambda x: -x[1]):
            lines.append(f"| {label} | {duration:.1f} |")
        lines += [
            "",
            "分類は元JSONの時間上位15 kernel名をまとめたもの。"
            "上位外のeventは含まない。",
            "",
        ]
    lines += [
        "**対策の効果**: CUDA event累積の減少より、CPUの起動API負荷の"
        "減少が大きい。従来のT2Sは多数の小kernelの起動が支配的だった。"
        "RoPE・固定分割の直接演算化でevent数を減らし、CUDA Graphで"
        "反復ごとのCPU dispatchを削減した。CUDA Graph自体が演算回数を"
        "削るわけではない。step数・モデル重み・FP32精度を変えていない。",
    ]
    path = ROOT / "README.md"
    text = path.read_text()
    begin, end = "<!-- ANALYSIS_START -->", "<!-- ANALYSIS_END -->"
    text = (
        text[: text.index(begin) + len(begin)]
        + "\n"
        + "\n".join(lines)
        + "\n"
        + text[text.index(end) :]
    )
    path.write_text(text)
    (ROOT / "results" / "block_analysis.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
