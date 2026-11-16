"""T2S内訳から改善対象とcache候補を整理し、READMEへ反映する。"""

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONDITIONS = [
    ("cpu_torch_int8", "CPU・PyTorch INT8"),
    ("cpu_native8_nospin_8", "CPU・ORT INT8（exp4）"),
    ("cuda_torch", "GPU・PyTorch FP32"),
    ("cuda_opt_nospin_8_front1_cg", "GPU・ORT FP32＋CUDA Graph（exp4）"),
]
GROUPS = {
    "Transformer群": ["transformer_blocks"],
    "固定テキストの準備・Embedding": ["hidden_text_setup", "text_lookup"],
    "音声トークンのEmbedding": ["audio_embedding"],
    "音声条件加算・系列結合・padding": [
        "audio_condition_add",
        "sequence_assembly",
    ],
    "Attention用padding maskの構築・転送": ["padding_mask"],
    "時刻条件の構築・転送とMLP": ["time_condition", "time_embedding_nn"],
    "出力のslice・padding・有効位置mask": ["output_assembly"],
    "hiddenの正規化・残り": ["hidden_total", "hidden_norm"],
    "CFGの合成・標準偏差": ["cfg_mix"],
    "語彙への射影": ["vocab_projection"],
    "top-kによる候補絞り込み": ["topk_filter"],
    "トークンsampling": ["sampling"],
    "確信度の計算": ["confidence_softmax", "confidence_score"],
    "再マスクの順位計算（argsort 2回）": ["remask_rank"],
    "停止判定（max→Python整数）": ["stop_check"],
    "トークン・mask更新とschedule": [
        "token_update",
        "mask_update",
        "remask_schedule",
    ],
    "その他のループ処理": ["masked_generate"],
}


CANDIDATES = (
    "### 結論と優先順位\n\n計測なしの中文推論では、T2S"
    "はTTS全体の約50〜52%（CPU）、約70〜81%（"
    "GPU）を占めた。T2S内部の計測では約99%がMask"
    "edT2S.generateであり、生成ループが主要な改"
    "善対象であることを確認できた。ただし、CPUではS2Mな"
    "ども約半分残るため、T2Sだけの改善をTTS全体の改善率"
    "とは扱わない。\n\nCPUではMasked生成内のTran"
    "sformer群が約92〜95%を占めるため、大幅な短縮"
    "にはTransformerの反復計算自体を減らす必要があ"
    "る。一方、GPUのexp4条件ではTransformer"
    "群が約34%、時刻条件の構築・MLPが約10%、系列結合"
    "が約4%、出力整形が約4%となった。この結果から、まず出"
    "力を維持できる固定条件の再利用と生成ループの起動回数削減"
    "を優先し、その後に精度との交換条件が生じる近似cache"
    "を検証する。\n\n**今回完了したのは内訳計測と候補の洗い"
    "出しであり、以下の最適化の実装・高速化効果・CER評価は"
    "未実施である。** GPUのexp4条件では細分化計測の"
    "経過時間が未計測版より32.6%増加しており、表の割合は"
    "候補の選定用である。通常推論で同じ時間が削減できるとは断"
    "定しない。\n\n### 出力を維持する方針の候補\n\n| 優"
    "先度・対象 | 計測・コードから分かったこと | 改善策"
    " | 効果の見込みと検証事項 |\n|---|---|--"
    "-|---|\n| 1・GPU | 各stepのCFG有／"
    "無で同じ時刻条件を2回計算。exp4条件では時刻条件全体"
    "が約42 ms | timestep特徴とtime_ml"
    "p出力をstepごとに1回計算して再利用 | MLPの重"
    "複と毎回の時刻転送を削減。動的INT8は入力batch形"
    "状を維持し、一括計算による量子化尺度の変化を避ける |\n"
    "| 1・GPU | テキスト、系列長、padding m"
    "ask、出力の有効位置は反復中に不変 | CFG分岐ごと"
    "にtext embedding・padding mask"
    "・slice位置・有効位置maskを保持 | Trans"
    "former入力前の繰り返し構築を削減。変化する音声em"
    "beddingやTransformer出力は使い回さない"
    " |\n| 1・GPU | exp4のCUDA Graph"
    "はTransformer群が中心で、その外側に細かい演算"
    "が残る | 固定shapeの入力構築・CFG・sampl"
    "ing・mask更新を融合、またはcapture範囲を拡"
    "大 | 多数のCPU起動を削減する候補。乱数状態、ORT"
    "とのstream同期、停止分岐を含むため、全ループをその"
    "ままcaptureできるとは限らない |\n| 2・GPU"
    " | 次stepのmask数は時刻scheduleと系列"
    "長だけで決まる。毎回maxをPython整数に変換 | "
    "mask数と停止stepを事前計算し、ループ内のCPU/"
    "GPU同期を除去 | 停止判定区間は約7 msだが、先行"
    "GPU処理の待機を含むため全量は消えない。同じFP32丸"
    "めと整数化を再現する |\n| 2・GPU | 再マスク順"
    "位にargsortを2回実行。exp4条件では約10 m"
    "s | 最初のargsortを残し、2回目の逆順列計算を"
    "scatterで置換 | 同点の順序を維持して順位を復元"
    "できる。出力一致と通常推論の時間で判定し、top-kへの"
    "変更とは分ける |\n| 2・CPU/GPU | 各blo"
    "ckの時刻依存AdaLNパラメータもCFG両分岐で同じ "
    "| time_mlpだけでなく各blockの時刻射影も再"
    "利用 | ORTではgraph入出力の変更が必要。gra"
    "ph分割による起動費用が削減量を上回らないか確認する |"
    "\n| 3・CPU/GPU | 長さ予測と生成前に同じte"
    "xt encoderを2回実行 | encoder出力を"
    "長さ予測から生成側へ渡す | 重複encoderはT2S"
    "の約0.2〜0.4%で、これだけで大幅な改善は見込めない"
    " |\n\nこれらは入力・数式を維持する方針だが、浮動小数点"
    "演算順や乱数消費の変更で生成結果が変わる可能性がある。ま"
    "ず同一seedの生成ID一致を検査し、不一致なら原因を確"
    "認したうえでCER評価が必要な条件として扱う。速度は今回"
    "と同じ短・中・長文、CPU/GPUで通常推論を測り、各デ"
    "バイスの最速PyTorch条件およびexp4採用条件に対"
    "するTTS全体の効果を示す。\n\n### Transfor"
    "merの再計算を減らす候補（精度評価が必要）\n\nこの生成"
    "方式は、マスク位置を並列に予測し、確信度に応じてマスクし"
    "直すMaskGCT方式で、マスク型diffusion L"
    "LMに近い。通常の自己回帰KV cacheでは過去位置の"
    "K/Vが変わらないことを利用するが、本実装では双方向At"
    "tentionで周囲の音声トークンが更新され、さらに各s"
    "tepで時刻依存のAdaLNも変わる。したがって、トーク"
    "ンIDが同じ位置でもK/Vが同じとは限らない。**通常の"
    "AR用KV cacheの単純適用はできないが、近似的な再"
    "利用は検証候補になる。**\n\n| 優先度・対象 | 改善"
    "策 | 適用理由と具体的な調査 | 精度・実装上の注意 "
    "|\n|---|---|---|---|\n| 1・CPU中"
    "心 | 更新間隔を設けたK/V・中間特徴cache | "
    "Transformerが約92〜95%を占めるため、反復"
    "間の再計算削減が重要。まず層・位置・生成前半/後半別にK"
    "/Vとhiddenの変化量を調べ、変化の小さい範囲だけ再"
    "利用する | CFG有／無のcacheを分離。毎回更新・"
    "2step間隔・4step間隔などを比較し、速度・CER"
    "・メモリを評価。K/V射影だけの省略ではMLP計算は残る"
    " |\n| 2・CPU/GPU | 確定位置のhidden"
    "を保持し、未確定位置を優先更新 | 確定したトークンも毎"
    "回全層で再計算しているため、その省略余地を調査する | "
    "確定位置も周囲と時刻の影響を受ける。双方向Attenti"
    "onとCFGの全位置の標準偏差にも影響し、単なる同値な計"
    "算省略にはならない |\n| 2・CPU/GPU | 信頼"
    "度に応じたcache更新・早期終了 | 一律の更新間隔よ"
    "り、変化量や確信度に応じて再計算する範囲を選ぶ | 判定"
    "処理自体の費用も含める。step削減はこの生成方式に依存"
    "するため、TTS共通の最適化とは区別する |\n| 3・方"
    "式変更 | block単位の生成とcache | 先行b"
    "lockの再計算を抑える構造を検討する | 現在の全系列"
    "並列生成からscheduleが変わり、学習条件との整合性"
    "確認も必要。最初の実装対象にはしない |\n\n関連する一次"
    "資料として、[MaskGCT](https://arxi"
    "v.org/abs/2409.00750)はマスクによる"
    "音声生成、[dKV-Cache](https://arx"
    "iv.org/abs/2505.15781)はdiffu"
    "sionモデルの特徴変化を踏まえた遅延・条件付きcach"
    "e、[Fast-dLLM](https://arxiv."
    "org/abs/2505.22618)は双方向モデルでの"
    "block単位の近似KV cacheを扱う。後二者は主に"
    "言語モデルの研究であり、その高速化率や品質が本TTSへそ"
    "のまま当てはまるとは仮定しない。\n\n次の実装順は、**固"
    "定条件・時刻条件の再利用 → 同期・順位計算・起動費用の"
    "削減 → 変化量の調査に基づく近似cache**とする。"
    "前二者でGPUの残存費用を減らし、後者でCPUの主要ボト"
    "ルネックであるTransformer反復計算の削減を狙う"
    "。"
)


def value(stats, name, key):
    return stats.get(name, {}).get(key, 0.0)


def main():
    records = []
    for condition, label in CONDITIONS:
        path = ROOT / "results" / "t2s_profile" / f"{condition}_all.json"
        data = json.loads(path.read_text())
        if data.get("measurement_version") != 2:
            raise RuntimeError(f"Profile version mismatch: {condition}")
        records.append((condition, label, data))
    lines = [
        "## ハイブリッドT2Sの反復処理とcache候補",
        "",
        "### 目的と対象",
        "",
        "現在の日本語HuBERT版は、長さ予測（duration_model=nar）と"
        "MaskGCT方式の音声トークン生成を組み合わせたハイブリッド型である。"
        "生成ループに時間がかかるという予想を確かめるため、まずTTS全体に対する"
        "T2Sの割合と、T2S内部の処理時間をCPU/GPUで分解した。"
        "その結果から、再計算を減らせる箇所と近似cacheが必要な箇所を区別する。",
        "",
        "| ファイル | 役割 | 実行・出力 |",
        "|---|---|---|",
        "| [profile_t2s.py](profile_t2s.py) | 現行コードへ計測scopeを挿入し、"
        "生成IDの一致も検査 | `--device cpu/cuda --config ...` → "
        "`results/t2s_profile/*_all.json` |",
        "| [summarize_t2s.py](summarize_t2s.py) | 内訳と改善候補を集約 | "
        "`python summarize_t2s.py` → 本節 |",
        "",
        "### 測定方法",
        "",
        "exp4と同じcheckpoint・短/中/長文・FP32の入出力・生成設定を使い、"
        "各文warmup 1回＋3回を測定した。CPUはPyTorch動的INT8とexp4の"
        "ORT INT8、GPUはPyTorch FP32とexp4のORT FP32＋CUDA Graphを比較した。"
        "各条件は独立プロセスで、他の速度・品質計測と排他実行した。",
        "",
        "計測なしでTTS全体を測った後、T2Sにだけ計測を挿入した。"
        "各試行では直前にT2S単独版を計測なしで実行し、同じseedの計測版と"
        "生成トークンのhashが一致することを検査した。TTS全体とのhash一致も"
        "別に記録した。モデルの数式・生成step数・cacheの有無は変えていない。",
        "",
        "CPUはhost経過時間、GPUは同じCUDA stream上のevent間経過時間を使う。"
        "GPU値はCPUからの起動待ちも含み、純粋なkernel実行時間ではない。"
        "細分化のたびにGPU同期を追加していない。計測scopeの追加費用は"
        "下表の差で確認し、内訳の絶対値を通常推論の速度と混同しない。"
        "CUDA eventは区間計測の前に確保・初期化する。",
        "",
        "### TTS全体に占めるT2Sの割合",
        "",
        "| 条件 | 入力 | TTS全体 秒 | T2S 秒 | T2S割合 % |",
        "|---|---|---:|---:|---:|",
    ]
    for _, label, data in records:
        for case, row in data["cases"].items():
            base = row["baseline_median_s"]
            t2s = base["duration_s"] + base["t2s_tokens_s"]
            total = base["end_to_end_s"]
            lines.append(
                f"| {label} | {case} | {total:.3f} | {t2s:.3f} "
                f"| {100 * t2s / total:.1f} |"
            )
    lines += [
        "",
        "### T2S内部の内訳（中文）",
        "",
        "| 条件 | 長さ予測 ms | 生成前準備 ms | Masked生成 ms "
        "| Masked比率 % | 未計測版との経過時間差 % |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition, label, data in records:
        row = data["cases"]["medium"]
        stats = row["median_profile"]
        key = "cpu_inclusive_ms"
        total = value(stats, "t2s_total", key)
        duration = value(stats, "duration_total", key)
        masked = value(stats, "masked_generate", key)
        setup = value(stats, "hybrid_generate", key) - masked
        base = row["baseline_median_s"]
        original = statistics.median(row["paired_reference_s"])
        overhead = 100 * (total / (1000 * original) - 1)
        lines.append(
            f"| {label} | {duration:.2f} | {setup:.2f} "
            f"| {masked:.2f} | {100 * masked / total:.1f} "
            f"| {overhead:+.1f} |"
        )
    for condition, label, data in records:
        stats = data["cases"]["medium"]["median_profile"]
        axis = "stream" if condition.startswith("cuda") else "cpu"
        total = value(stats, "masked_generate", f"{axis}_inclusive_ms")
        lines += [
            "",
            f"**{label}: Masked生成内の内訳**",
            "",
            "| 処理 | CPU経過 ms | CUDA stream経過 ms | 割合 % |",
            "|---|---:|---:|---:|",
        ]
        for group, names in GROUPS.items():
            cpu = sum(value(stats, n, "cpu_self_ms") for n in names)
            stream = sum(value(stats, n, "stream_self_ms") for n in names)
            selected = stream if axis == "stream" else cpu
            gpu = f"{stream:.2f}" if axis == "stream" else "—"
            lines.append(
                f"| {group} | {cpu:.2f} | {gpu} "
                f"| {100 * selected / total:.1f} |"
            )
        encoder = value(stats, "generate_encoder", f"{axis}_inclusive_ms")
        t2s = value(stats, "t2s_total", f"{axis}_inclusive_ms")
        lines += [
            "",
            f"長さ予測後に再実行したencoderは{encoder:.2f} msで、"
            f"T2Sの{100 * encoder / t2s:.2f}%だった。"
            "この重複だけを除いても、その割合を超える短縮は得られない。",
        ]
    lines += [
        "",
        "親区間と子区間を足すと二重計上になるため、詳細表では子区間を"
        "引いたself時間をまとめた。各値は試行ごとの中央値なので、"
        "丸めと試行差により割合の合計は厳密に100%にならない。"
        "GPUの停止判定のCPU時間には先行GPU処理を待つ時間も含まれるため、"
        "その全時間が判定の除去で消えるとは解釈しない。",
        "",
        "<!-- T2S_CANDIDATES_START -->",
        *CANDIDATES.splitlines(),
        "<!-- T2S_CANDIDATES_END -->",
        "",
        "### 再実行",
        "",
        "```sh",
        "python profile_t2s.py --device cpu --config torch_int8",
        "python profile_t2s.py --device cpu --config native8_nospin_8",
        "python profile_t2s.py --device cuda --config torch",
        "python profile_t2s.py --device cuda --config opt_nospin_8_front1_cg",
        "python summarize_t2s.py",
        "```",
    ]
    path = ROOT / "README.md"
    text = path.read_text()
    heading = "## 量子化実験後: KV cache・再計算の削減"
    new_heading = "## ハイブリッドT2Sの反復処理とcache候補"
    start = text.index(heading if heading in text else new_heading)
    path.write_text(text[:start] + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
