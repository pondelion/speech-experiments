# C完了レポート: 重みのみINT8/INT4、AWQ、GPTQ

本文のWはweight(重み)を表し、W8は重みのINT8量子化、W4は重みのINT4量子化を指す。
対象Linear層の重みの表現であり、モデル全体の演算精度を指すものではない。
今回の活性値はW8条件でFP32、W4条件でBF16。g128/g64/g32は、量子化のscale等を共有する重みのgroup sizeを示す。
RTNはround-to-nearest(最も近い量子化値への丸め)、qparamsはscale・zero pointなどの量子化パラメータを指す。

[実験全体の計画・A/Bの結果へ戻る](README.md)

## 結論: Cの結果から何を採用するか

**GPUでは速度ならFP32、メモリとのバランスならBF16を選ぶ。重みファイルの小ささが必要な場合だけ、単純INT4 g128(Tinygemm)を採用候補にする。**

| 用途 | 選ぶ条件 | 要点 |
|---|---|---|
| GPU低レイテンシ | FP32 | Cの中文で最速。重み量子化は今回すべて遅くなった |
| GPUのバランス重視 | BF16 | FP32に近い速度・テストCERで、peak VRAMを約39%削減 |
| 重みファイル最小化 | 単純INT4 g128(Tinygemm) | 約90.4 MiB。BF16比で約50%減るが、速度低下とテストCER +0.54ptを受け入れる必要がある。calibration不要で構成も比較的単純 |
| 実用文セットの読みCERを最優先 | MSLK RTN INT4→tileを追加評価 | seed 0では5.66%で最小。ただしテストCERは7.16%。一般的な精度最良とは判断しない |

AWQ/GPTQの追加コストに見合う一貫した優位は確認できず、既定には推奨しない。
CPUで速度を重視するなら、Cの重みのみINT8より[Bの両部品動的INT8](README.md)を選ぶ。
元モデルへの数値的な近似を避けるならCPU/GPUともFP32を使う。
読みCERの順位は評価集合で変わり、聴感品質も未評価のため、「精度全般で最良」と断定できる条件はない。
以上はこのモデル・機器・eager実行の範囲での暫定推奨。

## 完了範囲

CPU/GPUの単純W8、GPUのW4 group size 32/64/128、AWQ W4、GPTQ W8/W4と対応する基準を比較した。
下表の12条件で速度・読みCER・重みサイズ・メモリが揃った。CPU INT4など、実行できなかった経路は別途記録した。
これは選定したCの比較行列の完了であり、全backend・全bit幅・全ハイパーパラメータの網羅ではない。

## 結果

| 条件 | 中文秒 | 中文RTF | テストCER | 実用文CER | 重みMiB | 推論peak VRAM / CPU peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| CPU FP32 | 4.855 | 0.898 | 6.90% | 6.40% | 362.0 | 1639.5 MiB |
| CPU W8 | 4.445 | 0.822 | 6.81% | 6.26% | 105.4 | 1638.3 MiB |
| GPU FP32 | 1.411 | 0.261 | 6.46% | 6.73% | 362.0 | 564.9 MiB |
| GPU BF16 | 1.492 | 0.276 | 6.51% | 6.26% | 181.0 | 347.3 MiB |
| GPU W8 | 1.793 | 0.332 | 6.94% | 6.20% | 105.5 | 311.9 MiB |
| GPU GPTQ W8 | 1.828 | 0.338 | 6.70% | 6.46% | 105.5 | 310.7 MiB |
| GPU W4 g128 (Tinygemm) | 1.856 | 0.343 | 7.05% | 6.20% | 90.4 | 291.9 MiB |
| GPU W4 g64 | 2.018 | 0.373 | 7.00% | 6.40% | 95.1 | 296.7 MiB |
| GPU W4 g32 | 1.985 | 0.367 | 7.72% | 6.40% | 104.6 | 306.1 MiB |
| GPU AWQ W4 g128 | 2.141 | 0.396 | 7.33% | 7.20% | 90.6 | 310.2 MiB |
| GPU W4 g128 (MSLK→tile) | 1.881 | 0.348 | 7.16% | 5.66% | 90.4 | 287.0 MiB |
| GPU GPTQ W4 g128→tile | 2.107 | 0.390 | 6.92% | 5.93% | 90.4 | 288.7 MiB |

- 重みサイズはT2S＋S2Mのstate_dict。codec・runtimeを含まない。
- GPU行のメモリは長文推論中のPyTorch peak allocated VRAM、CPU行はプロセス開始以来のpeak RSS。
  単位は同じでも意味が異なるため、CPU/GPUの数値を直接比較しない。変換中のpeakとreserved VRAMも元JSONに保存した。
- 速度は同じ中文、batch 1、8 threads、warm-up 1回＋3回のmedian。短文・長文・p90もCSV/JSONに保存。
- CERはテスト100文・実用文55文、seed 0、T2S batch 16/S2M batch 4。CPU FP32のCERはAの同一設定の結果を参照。
- 全条件の生成Mel・波形は有限値。GPU FP32とINT8、GPU BF16とINT4をそれぞれ対比する。
- 重み量子化で生成長も変わりうる。RTFは実際の生成音声長で計算した。速度の入力集合とCERの集合は異なる。

## 観測したこと

1. 重み圧縮と推論速度は別。GPU eagerではW8/W4とも対応する基準より遅かった。
2. W4 g128はBF16基準から重みサイズを約50%減らすが、推論peak VRAMの削減は約16%。
   未量子化の部品や作業領域があるため、全体のメモリ削減率は重み圧縮率と一致しない。
3. groupを小さくするとscaleの数とサイズが増える。今回の単一seedでは、テストCERが単調に良くならなかった。
4. AWQは今回のcalibration設定では単純W4よりCERが悪化した。
5. GPTQは同じbit幅・量子化器の基準に対してテストCERが改善する一方、実用文CERは悪化した。
   W4はMSLK RTN→tileを対照とする。Tinygemm RTNとはqparamsの求め方も異なる。
6. CPU W8のpeak RSSはFP32とほぼ同じ。保存重みの圧縮をそのままRAM削減とは扱えない。
   CPU速度も、別の10回FP32計測では中文4.474秒だったため、小さな差の高速化を断定しない。

## Calibrationと再現性

- 学習splitを文字数で8区間に分け、seed 0で各区間から1文選んだ。選定IDと元checkpointのSHA256を各条件に記録。
  評価文・評価音声はcalibrationに使用していない。
- 基準モデルから予測トークンとMelを生成し、HuBERT本体や学習データの音声ファイルは使わなかった。
- T2Sは予測トークンへ異なるマスク率を適用した近似状態、S2Mは予測Melとノイズの線形補間を観測した。
  実際の全生成軌跡をキャプチャしたcalibrationではない。
- T2Sの時刻相当は1/0.5/0.04、S2Mは0/0.5/0.96。CFGの条件あり/なしを含む。
- AWQはtorchaoの20候補scale探索を使い、observerの保持activationを各層最大256行に制限した。
  GPTQは観測入力をHessianへ集計し、damping 0.01、block size 256。
  同じ8文でも保持するactivation量・統計が異なるため、厳密に同一サンプル数の探索比較ではない。
- 全層が未量子化の上流出力を観測する方式。blockごとに量子化済み出力を再観測するsequential GPTQは行っていない。
- 変換済みcheckpointはGit管理外のruns/calibration/へ保存。速度とCERは同じcheckpointを使用し、再読込の一致検査を通した。
- calibration時間・除外層・観測行数・変換条件は条件別JSONに保存。calibration時間をwarm推論時間へ含めない。

## 実装上の補正と失敗経路

- AWQのobserverはautocast前の入力を見るため、保存activationを実際のLinearのdtypeへそろえた。
- 変換はinference_modeではなくno_gradで実施し、checkpoint再読込をinference_mode外で行った。
- GPTQのBF16観測ではHessianのmatmulをFP32へ固定し、autocastによる正定値性の破綻を避けた。
  libraryファイルは編集せず、観測中だけのcontext内で補正し、終了時に元へ戻す。
- MSLKの標準INT4推論はRTX 5090で「cutlass cannot initialize」になった。
  signed nibbleとscale/zeroを保持して動作するtinygemm配置へ変換した。再量子化はしていない。
  全対象層でqparams保持とidentity入力によるdense weightの検査を通し、kernel由来のBF16丸め誤差を記録した。
  GPTQ W4だけでなく、対照のMSLK RTN W4にも同じ配置変換を適用した。
- 今回のCPU INT4経路はGPUを要求し動作しなかった。CPU一般でINT4が不可能という結論ではない。

## 環境・成果物

PyTorch 2.14.0+cu130、torchao 0.18.0、RTX 5090、Core Ultra 7 265K。
既存全versionを固定したdry run後に公式CUDA 13.0配布のmslk 1.4.0+cu130のみ追加した。
既存186パッケージのversion変更はなく、追加後も依存チェックは通った。install_audit_mslk.jsonに記録。

- 集計: results/weight_only/c_summary.csv・c_summary.json
- 生の速度・CER・メモリ: 同ディレクトリの条件別JSON
- 試聴: results/samples/ の各条件3文、48kbps MP3とindex.json。A〜C全体で15条件・45本、計1,888,560 bytes(約1.80 MiB)。聴感評価は未実施
- 図は作成しない。中間結果は表に蓄積し、量子化exp全体の最後に散布図へまとめる

## 残る制約と次の段階

C全条件の複数seed・速度10回以上の再測定・聴感確認は未実施。特定条件の小差を一般化せず、既定の推論条件も置き換えない。
calibrationは8文・近似状態に限られ、データ量・実軌跡・sequential GPTQの違いは未探索。
現在のGriffin-Limは暫定処理で、NNボコーダでの音質維持は別途確認する必要がある。

Cの選定比較は完了。次はDの活性値を含む静的PTQ/SmoothQuant系、
Eの敏感な層を高精度に残す比較、FのQAT、Gの配布runtimeへ進む。

参照: [torchao AWQ](https://github.com/pytorch/ao/tree/main/torchao/prototype/awq)、
[torchao GPTQ](https://github.com/pytorch/ao/tree/main/torchao/prototype/gptq)。
実行にはインストール済み0.18.0のAPIを使用した。

