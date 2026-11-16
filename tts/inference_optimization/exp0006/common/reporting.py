"""段階別の実測結果からREADMEの結果節を生成する。"""

import json

from .paths import RESULTS, ROOT, write_json

MARKER = "## 実施結果"


def summarize():
    records = {}
    for device in ("cpu", "cuda"):
        path = RESULTS / "stage_a" / f"{device}.json"
        records[device] = json.loads(path.read_text())
        if not records[device]["complete"]:
            raise ValueError(f"Incomplete stage A: {device}")
    audit = json.loads((RESULTS / "data_audit.json").read_text())
    first = records["cuda"]
    write_json(
        RESULTS / "environment.json",
        {device: row["environment"] for device, row in records.items()},
    )
    lines = [
        MARKER,
        "",
        "### A: 基準・学習基盤",
        "",
        "**段階Aは完了。** CPU/GPUの両方で6層studentの通常学習・蒸留を"
        "各2更新行い、有限loss・有限gradient、生成Transformerの重み更新、"
        "固定部品の不変、保存前後の全重みと生成IDの一致を確認した。"
        "これは学習経路の検証であり、小型モデルの品質評価ではない。",
        "",
        "#### 重みとデータ",
        "",
        "| モデル | パラメータ数 M | FP32 state_dict MiB | 教師比削減 % |",
        "|---|---:|---:|---:|",
    ]
    teacher_params = first["inventory"]["teacher"]["parameters"]
    for name, row in first["inventory"].items():
        lines.append(
            f"| {name} | {row['parameters'] / 1e6:.2f} "
            f"| {row['state_dict_bytes'] / 2**20:.2f} "
            f"| {100 * (1 - row['parameters'] / teacher_params):.1f} |"
        )
    lines += [
        "",
        "codec・S2Mは上表の容量に含めない。6層・4層モデルも"
        "固定したtext encoderと長さ予測器を含むため、総容量は生成層数に"
        "比例しては減らない。checkpointのパス・SHA-256と部品別内訳は"
        "[stage_a/cuda.json](results/stage_a/cuda.json)に記録した。",
        "",
        "既存の離散トークンcache・テキスト・音素ラベルを使用し、"
        "学習準備で元音声を再処理していない。評価文との重複はNFKC正規化と"
        "記号除去後の同文照合で検査し、studentの学習・検証候補から除外した。",
        "",
        "| split | 元の発話数 | 同文のため除外 | 音素整合で除外 "
        "| 使用発話数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in audit["groups"].items():
        lines.append(
            f"| {name} | {row['original']} "
            f"| {len(row['excluded_text_overlap_ids'])} "
            f"| {row['alignment_dropped']} | {row['used']} |"
        )
    lines += [
        "",
        "trainでは検証文との同文も除外した。詳細な発話IDは"
        "[data_audit.json](results/data_audit.json)に保存した。"
        "教師は既存checkpointをそのまま使用しており、今回の除外によって"
        "教師の過去の学習データまで変更したわけではない。したがって、"
        "教師を含む結果を完全な未見文性能と断定しない。",
        "",
        "#### 最速PyTorch条件の再計測",
        "",
        "CPUは動的INT8、GPUはFP32、8 threads、batch 1、warm-up 1回、"
        "各入力10回の中央値。下表は教師の基準で、小型化の効果ではない。",
        "",
        "| Device | 短文 秒 | 中文 秒 | 長文 秒 | テストCER % | 実用文CER % |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for device, row in records.items():
        times = [
            row["benchmark"]["cases"][case]["median_s"]["end_to_end_s"]
            for case in ("short", "medium", "long")
        ]
        quality = row["quality_reference"]
        lines.append(
            f"| {device} | {times[0]:.3f} | {times[1]:.3f} "
            f"| {times[2]:.3f} | {quality['test_cer_pct']:.3f} "
            f"| {quality['practical_cer_pct']:.3f} |"
        )
    lines += [
        "",
        "速度は新規測定。CERは重み・生成条件が同じ既存評価を参照し、"
        "新規測定とは区別する。出典は各JSONのquality_referenceに記録した。"
        "各試行の部品別時間と生成ID hashも保存している。",
        "",
        "| ファイル | 役割 |",
        "|---|---|",
        "| [run.py](run.py) | 共通CLIと実験間の排他実行 |",
        "| [common/audit.py](common/audit.py) | 重み・環境・段階Aの検証 |",
        "| [common/paths.py](common/paths.py) | 保存先・相対パス・checksum |",
        "| [common/data.py](common/data.py) | データ分割・重複除外・cache |",
        "| [common/models.py](common/models.py) | 教師とstudentの構築・保存 |",
        "| [common/reporting.py](common/reporting.py) | 実測表と本節の生成 |",
        "",
        "### B: 層数縮小・蒸留",
        "",
        "6層・4層とも幅512・8 heads。教師の層を均等に選び、"
        "text encoder・長さ予測器・音声条件への射影は固定する。"
        "まず各構造で通常学習と蒸留を各1,000更新比較する。"
        "学習率3e-5、AdamW、weight decay 0.01、最大4,096 tokens、"
        "BF16 autocast、検証250更新ごと。最初の50更新をwarm-upとし、"
        "その後cosineで学習率を下げる。蒸留は温度1、CE＋KL（重み1）。"
        "checkpointはstep 0も含め、共通の検証CEが最小のものを採用する。",
        "",
        "| ファイル | 役割 |",
        "|---|---|",
        "| [common/distillation.py](common/distillation.py) | 同じマスク・"
        "時刻・条件のCEと教師分布のKL |",
        "| [common/training.py](common/training.py) | 共通の更新・検証・"
        "checkpoint選択と再ロード検査 |",
        "| [common/evaluation.py](common/evaluation.py) | CPU/GPU速度・"
        "155文CER・代表MP3 |",
        "",
    ]
    training = sorted((RESULTS / "training").glob("*.json"))
    if training:
        lines += [
            "| 条件 | 更新数 | 選択step | 検証CE | 学習 秒 |",
            "|---|---:|---:|---:|---:|",
        ]
        for path in training:
            row = json.loads(path.read_text())
            lines.append(
                f"| {row['condition']} | {row['steps']} | {row['best_step']} "
                f"| {row['best_validation_ce']:.4f} "
                f"| {row['training_s']:.1f} |"
            )
    else:
        lines.append(
            "本学習の結果は未確定。段階Aの2更新を採用評価には使わない。"
        )
    lines += [
        "",
        "### 再現手順",
        "",
        "リポジトリルートから実行する。",
        "",
        "```sh",
        "python -m tts.inference_optimization.exp0006.run stage-a "
        "--device cuda",
        "python -m tts.inference_optimization.exp0006.run stage-a "
        "--device cpu",
        "python -m tts.inference_optimization.exp0006.run train "
        "--layers 6 --mode plain",
        "python -m tts.inference_optimization.exp0006.run train "
        "--layers 6 --mode kd",
        "python -m tts.inference_optimization.exp0006.run evaluate "
        "--layers 6 --mode kd --device cuda",
        "python -m tts.inference_optimization.exp0006.run summarize",
        "```",
    ]
    path = ROOT / "README.md"
    text = path.read_text().split(MARKER)[0].rstrip()
    path.write_text(text + "\n\n" + "\n".join(lines) + "\n")
