# exp0001: 日本語HuBERT版の推論ベースラインとボトルネック計測

- **カテゴリ**: ベースライン・計測
- **対象**: exp0003の`hubert_ja_k256/tts`
- **変えるもの**: なし。学習・推論時の既定条件をそのまま測る
- **状態**: GPUのFP32/FP16/BF16速度・メモリ・CER評価、およびCPUのFP32速度・メモリ計測まで完了

## 目的

以降の量子化・コンパイル・ONNX/TensorRT・蒸留・ストリーミングを同じ条件で比較できるベースラインを作る。
end-to-endの時間だけでなく、テキスト前処理、長さ予測、T2S、S2M、Griffin-Limに分け、どこが入力長に応じて支配的になるかを調べる。

## 対象モデル

[`../../indextts_bases/exp0003/`](../../indextts_bases/exp0003/) のうち、評価条件ID `hubert_ja_k256/tts` の3 checkpointを使う。

| 部品 | checkpoint | 学習時の規模・設定 |
|---|---|---|
| codec | `../../indextts_bases/exp0003/runs/codec_hubert_ja_k256/codec.pt` | 20.01M parameters、codebook 256 |
| T2S | `../../indextts_bases/exp0003/runs/t2s_hubert_ja_k256/t2s.pt` | 54.15M parameters、ハイブリッド型、長さを一括予測 |
| S2M | `../../indextts_bases/exp0003/runs/s2m_hubert_ja_k256/s2m.pt` | 40.72M parameters、Flow Matching |
| ボコーダ | Griffin-Lim | checkpointなし、64反復 |

この条件の既存評価は、テスト100文CER 6.9%、実用文55文CER 5.9%、正解トークンからS2Mへ通した上限が4.6%。
日本語HuBERT本体と学習用`tokens.pt`は推論に使わない。

### 固定する生成条件

| 区間 | 設定 |
|---|---|
| T2Sの長さ | 非自己回帰のduration predictorで一括予測 |
| T2Sのトークン生成 | 50 step、temperature 0.9、filter threshold 0.98、CFG 2.5、CFG rescale 0.75 |
| S2M | Euler 25 step、CFG 0.7 |
| ボコーダ | Griffin-Lim 64反復、24kHz |
| seed | 0 |
| batch | まず1。後で複数batchを追加する |

## 測定するもの

### 時間

- checkpointの読み込みとモデル構築(cold load)
- テキスト前処理
- T2Sの長さ予測
- T2Sの音声トークン生成
- S2M
- Griffin-Lim
- end-to-end

GPU区間は計測の前後で同期する。warm-up後に複数回実行し、median・p90・最小値・最大値を保存する。

### 速度・資源

- 生成音声長、RTF、x realtime
- peak VRAM、およびプロセス全体のpeak RSS(RAM)。RSSはプロセス開始以降の最大値で、ケース間ではresetしない
- checkpointごとのサイズと合計
- 短文・中文・長文に対する入力長、音声トークン数、Mel frame数

現在の実装は全音声を生成してから返す非ストリーミング構成なので、TTFAはend-to-end完了時間と同じ扱いになる。
真のTTFAはストリーミング対応後に測る。

### 品質

速度測定とは別に、既存のテスト100文・実用文55文でCERを再確認する。FP16/BF16など出力が変わる条件では、同じseedで
中間出力、CER、試聴用音声をFP32基準と比較する。ベースラインの再計測では既存のCERを変えないことを確認する。

## 入力

`inputs.json`に短文・中文・長文を固定した。最初は各1文・batch 1で区間別の傾向を測り、その後に共通評価文とbatch条件を増やす。
入力文を変えた場合は結果を同じ表に混ぜない。

## 実行方法

依存関係を満たすPython環境で、このディレクトリから実行する。測定環境のバージョンは結果JSONを参照。

モデルをロードせず入力とcheckpointの存在だけを確認する場合:

```bash
python benchmark.py --check-only
```

測定前の動作確認として、短文1件・warm-up 1回・計測3回を実行する。

```bash
python benchmark.py --case short --warmup 1 --repeats 3
```

問題がなければ全入力を本計測する。

```bash
python benchmark.py --case all --warmup 3 --repeats 10
```

精度ごとの結果は`results/metrics_{精度}.json`、実行環境は`results/environment_gpu.json`または`environment_cpu.json`へ保存する。既定では既存ファイルを上書きしない。

```bash
python benchmark.py --case all --warmup 3 --repeats 10 --dtype float32 --output results/metrics_fp32.json
python benchmark.py --case all --warmup 3 --repeats 10 --dtype float16 --output results/metrics_fp16.json
python benchmark.py --case all --warmup 3 --repeats 10 --dtype bfloat16 --output results/metrics_bf16.json
python evaluate_precision.py --dtype float32
python evaluate_precision.py --dtype float16
python evaluate_precision.py --dtype bfloat16
python benchmark.py --device cpu --threads 20 --case all --warmup 1 --repeats 3 --output results/metrics_cpu_fp32.json
```

## 実装上の注意

- `codec.pt`には学習時に必要だったencoder/decoder全体が入っているが、TTS推論で使うのは主に`decode_hidden`側。
  推論専用checkpointへの切り出しは、ベースライン測定後の軽量化候補にする
- T2Sは50回、S2Mは25回の反復を含む。反復回数を減らす実験はベースラインに混ぜず、別条件として品質とともに測る
- Griffin-LimはCPU/GPUとdtypeの影響がモデル部品と異なる可能性があるため、必ず独立した区間として扱う
- cold loadとwarm推論を同じ値に混ぜない
- GPUが他の処理に使われている計測結果は正式なベースラインに採用しない

## 結果

2026-11-11にRTX 5090(32GB)、PyTorch 2.14.0+cu130、batch 1で測定した。実行前後ともGPUを使う他プロセスがなく、
warm-up 3回後の10回のmedian。checkpointは3つで合計438.3 MiB、FP32のcold loadは3.82秒だった。

### FP32の区間別ベースライン

| 入力 | 文字数 | 生成音声 | end-to-end | T2S | S2M | Griffin-Lim | RTF | x realtime | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 短文 | 3 | 0.63秒 | 1.59秒 | 1.25秒 | 0.30秒 | 0.04秒 | 2.524 | 0.40x | 475 MiB |
| 中文 | 33 | 5.41秒 | 1.65秒 | 1.35秒 | 0.27秒 | 0.03秒 | 0.305 | 3.28x | 505 MiB |
| 長文 | 104 | 14.67秒 | 1.66秒 | 1.36秒 | 0.26秒 | 0.03秒 | 0.113 | 8.84x | 565 MiB |

- 中文・長文は実時間より速いが、短文は固定の反復処理の比率が大きく、RTF 2.52で実時間に届かない
- T2Sがend-to-endの79〜82%を占め、入力長が増えても約1.3秒かかる。次に優先する対象はT2Sの50-step生成
- S2Mは16〜19%、Griffin-Limは2%前後。現時点のRTX 5090では、ボコーダだけを速くしてもend-to-endへの効果は小さい
- 入力長に対して実行時間がほぼ一定なので、短文の待ち時間短縮にはstep削減、コンパイル、固定overheadの削減が重要

### FP32・FP16・BF16

T2S/S2Mをautocastし、BF16で未対応の複素数演算を含むGriffin-Limは全条件FP32に固定した。

| dtype | 短文 RTF | 中文 RTF | 長文 RTF | 長文 peak VRAM | テスト100文 CER | 実用文55文 CER |
|---|---:|---:|---:|---:|---:|---:|
| **FP32** | **2.524** | **0.305** | **0.113** | 565 MiB | 6.70% | 5.86% |
| FP16 autocast | 2.875 | 0.336 | 0.120 | **523 MiB** | 7.07% | **5.66%** |
| BF16 autocast | 2.624 | 0.328 | 0.121 | **523 MiB** | **6.61%** | **5.66%** |

- この環境ではFP16/BF16は速くならず、FP32が全入力で最速。autocastの変換overheadと、FP32でも高速なGPU kernelの影響が考えられる
- 低精度は長文のpeak VRAMを565 MiBから523 MiBへ約7%削減した。重み自体はFP32のままなので、checkpointサイズとcold loadは変わらない
- CER差は小さく、全条件でMel・波形にNaN/Infはなかった。FP16は同一seedの代表3文でFP32と同じ音声トークン列、
  BF16は中文・長文で一部変わったが、100文のCERは悪化しなかった
- したがって速度を目的にautocastを既定にはしない。メモリ制約がある場合はBF16を候補にし、量子化・重み変換は別実験で評価する

### CPUとGPUの比較

CPUはCore Ultra 7 265K、20 threads、FP32、batch 1、warm-up 1回後の3回のmedian。
GPUは上記RTX 5090の10回のmedian。反復数とseed列が異なるため、厳密な同一条件比較ではなく初期ベースラインとして扱う。

| 入力 | CPU生成時間 | GPU生成時間 | CPU RTF | GPU RTF | CPU peak RSS |
|---|---:|---:|---:|---:|---:|
| 短文 | 2.28秒 | 1.59秒 | 3.623 | 2.524 | 1.53 GiB |
| 中文 | 7.84秒 | 1.65秒 | 1.450 | 0.305 | 1.55 GiB |
| 長文 | 16.81秒 | 1.66秒 | 1.146 | 0.113 | 1.56 GiB |

CPUでも全入力を正常に生成でき、Mel・波形は有限値だった。RAMにも余裕があり、CPUで扱える規模のモデルである。
ただし、この設定では全入力が実時間より遅い。長文ではT2Sが約8.49秒、S2Mが約8.09秒で、両方の最適化が必要になる。
20 threadsが最適かは未検証。今後はthread数、INT8量子化、ONNX Runtime等をCPU/GPUそれぞれで測る。
CPUのCERは今回未計測。CPU/GPUで完全に同じ音声になることは保証しない。

### 次の手(優先順)

1. CPU thread数、実行時overhead、コンパイル、量子化、ONNXなど、方式をまたいで使える最適化を優先する
2. モデル部品の処理時間と、暫定的なGriffin-Limの時間を分けて比較する。学習済みNNボコーダが用意できたら基準を追加する
3. 推論専用checkpointから学習専用部品を除き、配布サイズとcold loadへの効果を測る
4. 反復数・CFGの調整は、その生成方式に限る追加検証とする。一般的なTTSへの結論にはしない

生の計測値は`results/metrics_fp32.json`・`metrics_fp16.json`・`metrics_bf16.json`、CERは
`results/precision_quality.json`、CPU計測は`results/metrics_cpu_fp32.json`、環境は
`results/environment_gpu.json`・`environment_cpu.json`に保存した。

計測上の制約: end-to-endは同期を挟んだ各区間の合計で、無計装の単一wall-clock計測ではない。
cold loadはimport・モデル構築込みで、OSのファイルキャッシュを消していない。CERはseed 0の1回であり、小差の有意性は未検証。
演算別profiler、batch比較、CPU thread数比較は今後の対象とする。
