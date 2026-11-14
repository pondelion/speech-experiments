# inference_optimization: TTS 推論の軽量化・高速化

TTS の品質をできるだけ保ったまま、推論時間・初回音声までの待ち時間・メモリ・モデルサイズを小さくする方法を比較する。
まずは IndexTTS 系の構成を対象にするが、特定のモデルだけに閉じず、T2S・S2M/音響モデル・ボコーダからなる TTS に再利用できる知見としてまとめる。

最適化手法ごとに別の条件で速度を測るのではなく、最初に共通のベンチマークを作り、同じ入力・品質評価・ハードウェアで比較する。
各 `exp` では、速くなったかだけでなく、どの部品が速くなり、品質・移植性・実装の複雑さと何を交換したかを記録する。

最終成果は、ここで得た量子化・軽量化・変換・ストリーミングの知見を統合し、**GitHub Pages 上でモデルをダウンロードして、
サーバーを使わずブラウザ内でリアルタイムに音声を生成・再生できるデモ**にする。

## 目的

- オフライン生成: 1 発話を短時間・少ないメモリで生成する
- 対話用途: リクエストから最初の音声を返すまでの時間を短くする
- サーバー用途: 同時リクエストを効率よく処理する
- ローカル用途: CPU、ブラウザ、エッジデバイスでも実用的に動かす
- 手法ごとの速度・品質・実装コスト・対応環境を、再現可能な形で比較する
- 最終的に、静的ホスティングだけで動くブラウザ TTS として統合・公開する

## 検証の基準モデル

このモデルは最適化手法を比較する最初の題材とする。CPU thread数・コンパイル・量子化・ランタイム変換などの共通軸を優先し、
モデル固有の生成stepやCFG調整は適用方式を限定した補助実験とする。Griffin-LimはNNボコーダが用意できるまでの暫定処理であり、
その性能から学習済みボコーダの性能を推定しない。

最初の基準には、[`../indextts_bases/exp0003/`](../indextts_bases/exp0003/) の `hubert_ja_k256/tts` を使う。
日本語 HuBERT の 9 層目の特徴を256種類に量子化した音声トークンと、ハイブリッド型T2S、S2Mを組み合わせた条件で、
テスト100文の読みCERは6.9%、実用文55文は5.9%。試聴でも候補の中で最も良かった条件を選んだ。

| 部品 | checkpoint | 推論での役割 |
|---|---|---|
| codec | `../indextts_bases/exp0003/runs/codec_hubert_ja_k256/codec.pt` | 音声トークンをS2Mへ渡す特徴に変換する |
| T2S | `../indextts_bases/exp0003/runs/t2s_hubert_ja_k256/t2s.pt` | テキストから長さと音声トークンを生成する |
| S2M | `../indextts_bases/exp0003/runs/s2m_hubert_ja_k256/s2m.pt` | 音声トークンからMelを生成する |
| ボコーダ | checkpointなし(Griffin-Lim 64反復) | Melから波形を生成する。後の実験でNNボコーダも対象にする |

日本語 HuBERT 本体は学習データをトークン化するときだけ使っており、テキストからの推論ではロードしない。学習用の`tokens.pt`も不要。
したがって、推論最適化の対象は上記の自作codec・T2S・S2Mとボコーダだけになる。比較・問題切り分け用には、外部モデルの特徴を
学習にも使わないexp0002のMel codec版(`t2s_hybrid_nar`、CER 11.9%)を残す。

## 最終成果: GitHub Pages で動くリアルタイム TTS

GitHub Pages は静的ファイルの配信だけを担当し、推論は利用者のブラウザ内で行う。主要候補は ONNX モデルを
ONNX Runtime Web の WebGPU Execution Provider で実行する構成とし、WASM/CPU（利用可能なら WebNN）も比較・fallback の候補にする。
必要に応じてネイティブな WebGPU 実装とも比較する。変換済みモデルと tokenizer/G2P を JavaScript から実行し、生成した波形は
全体の完了を待たず、chunk ごとに再生する。

```text
GitHub Pages
  ├── UI・tokenizer/G2P
  ├── 分割・量子化したモデルを取得してブラウザ内に cache
  ├── Web Worker で ONNX Runtime Web(WebGPU/WASM/WebNN)等の推論
  └── 生成済み chunk を AudioWorklet へ渡して逐次再生
```

### 完了条件

- GitHub Pages を開き、文章を入力してブラウザ内だけで音声を生成できる
- 対応ブラウザでは ONNX Runtime Web + WebGPU、非対応環境では可能な範囲で WASM/CPU fallback が動く
- 短文では全音声の生成完了を待たずに再生が始まり、平均生成速度が再生速度を上回る
- UI の main thread を推論で止めず、生成の進捗表示・途中停止・連続生成ができる
- 初回のモデル取得量、初回 TTFA、cache 後の TTFA、RTF、使用メモリを表示または計測できる
- モデルをブラウザ内に cache し、2 回目以降は不要な再 download を避ける
- 長文を処理してもメモリが増え続けず、chunk 境界に目立つノイズや不自然な無音がない
- デモで使うモデル・重み・データ・依存ライブラリのライセンスが公開用途に適合する

「リアルタイム」は、最低条件を `RTF < 1`(1 秒の音声を 1 秒未満で継続して生成)とする。ただし対話時の体感には TTFA も重要なので、
RTF と TTFA を別々に目標・評価する。最初のベースラインを測った後、対象端末別の具体的な目標値を決める。

## 最適化する範囲

TTS 全体の時間だけでなく、次の区間を分けて計測する。

| 区間 | 主な処理 | 主な最適化候補 |
|---|---|---|
| 前処理 | 正規化、G2P、参照音声の特徴抽出 | キャッシュ、並列化、実装の置き換え |
| T2S / テキストモデル | テキストから音声トークン・中間表現を生成 | KV cache、量子化、蒸留、投機的生成、非自己回帰化 |
| S2M / 音響モデル | 音声トークンから Mel などを生成 | step 数削減、蒸留、コンパイル、低精度化 |
| ボコーダ / codec decoder | Mel・トークンから波形を生成 | ONNX/TensorRT、チャンク推論、軽量モデルへの置換 |
| 入出力 | モデル読み込み、転送、音声のエンコード | 重みの共有・遅延読み込み、非同期転送、逐次配信 |

品質を保ったまま実装だけを変える実験と、モデルを学習し直して速度と品質を交換する実験は分ける。

## 共通ベンチマーク

以降の実験ではCPUとGPUの両方を計測する。CPUはthread数とpeak RSS、GPUは機種とpeak VRAMを記録し、
片方で未対応の手法はその理由を残す。CPUでの実時間動作も最適化の目標に含める。

### 入力条件

- 短文・中文・長文を固定し、文字数・音素数・生成音声長を保存する
- 日本語の数字、英字、固有名詞、句読点を含む実用文も使う
- ゼロショット TTS では参照音声も固定し、参照特徴をキャッシュする場合としない場合を分ける
- 単発リクエスト(`batch=1`)と、バッチ・同時リクエストを分けて測る
- 生成結果を比較できるように乱数 seed とデコード設定を固定する

既存の共通評価文は [`../../datasets/eval_texts/practical.tsv`](../../datasets/eval_texts/practical.tsv) を使い、速度測定用の代表文は
最初の実験でその中から長さ別に固定する。

### 測るもの

| 分類 | 指標 | 補足 |
|---|---|---|
| レイテンシ | cold start、warm latency、TTFA(Time to First Audio) | モデル読み込みを含む値と含まない値を分ける |
| 生成速度 | RTF、x realtime、各区間の実行時間 | `RTF = 推論秒 / 生成音声秒`。1 未満なら実時間より速い |
| スループット | requests/s、audio seconds/s | バッチ数・同時数とセットで記録する |
| 資源 | peak CPU RAM、peak VRAM、CPU/GPU 使用率、消費電力 | 消費電力は測定できる環境で追加する |
| 配布 | 重み・ランタイムのサイズ、起動時間 | ブラウザでは初回 download/cache も測る |
| 品質 | CER、UTMOS、話者類似度、F0/音声長、失敗率 | 元モデルとの差を中心に見る。必要に応じて試聴も行う |

平均値だけでなく、warm-up 後に複数回測った median・p90・ばらつきを残す。GPU の非同期実行は計測前後で同期し、
モデル読み込み、前処理、CPU/GPU 転送を計測範囲に含めたかを明記する。

### 結果に必ず残す環境

- CPU、GPU、RAM/VRAM、OS
- Python、PyTorch、CUDA/cuDNN、ONNX Runtime、TensorRT などのバージョン
- dtype、batch size、入力長、生成音声長、デコード設定
- warm-up 回数、計測回数、電源モード
- 使用した commit、モデル、重みの checksum

各実験の `results/metrics.json` は可能な範囲で同じ schema にし、後で結果表とグラフを自動生成できるようにする。

実験では同じPython環境を使い続け、依存パッケージのversionを各実験の`results/environment.json`にも保存する。

## ロードマップ

実験番号は着手時に確定する。まず NVIDIA GPU 上で測定基盤と低リスクな最適化を固め、その後にモデル変更、ストリーミング、
別ランタイム・別デバイスへ広げる。

| exp | カテゴリ | 基準 | 変えるもの | 狙い |
|---|---|---|---|---|
| [exp0001](exp0001/) | ベースライン・計測 | `hubert_ja_k256/tts`の既定実装 | 共通ベンチマーク、区間別 profiler、FP32/FP16/BF16、cold/warm の測定 | 再現可能な基準を作り、支配的な区間を特定する |
| exp0002 | PyTorch・ランタイム最適化 | 0001 | `inference_mode`、不要な転送の除去、KV/参照特徴の cache、`torch.compile`、CUDA Graphs、SDPA/FlashAttention、生成ループの改善 | 学習し直さずに得られる改善と、各変更の寄与を分けて測る |
| exp0003 | 量子化 | 0002 | 重みのみ INT8/INT4、動的/静的量子化、SmoothQuant/AWQ/GPTQ 系、必要なら QAT | T2S・S2M・ボコーダごとに、低下しやすい品質と速度・メモリの境界を調べる |
| exp0004 | ONNX・ランタイム変換 | 0002/0003 | 部品ごとの ONNX export、ONNX Runtime(CPU/CUDA)、graph optimization | export 可能性、演算子の互換性、動的 shape のコストを整理する |
| exp0005 | TensorRT・GPU 最適化 | 0004 | TensorRT(FP16/BF16/INT8)、最適化 profile、固定/動的 shape | NVIDIA GPU での最大性能と、engine 作成・運用上の制約を測る |
| exp0006 | 蒸留・モデル軽量化 | 最も良い基準 | 蒸留、層/幅の縮小、pruning、flow/diffusion step の蒸留、軽量ボコーダ | 学習を伴う圧縮で、モデルサイズ・速度・品質の Pareto front を作る |
| exp0007 | ストリーミング | 最も良い基準 | 文・音素・音声トークン・Mel・波形の chunk 化、先読み幅、状態/cache の引き継ぎ | TTFA を短くし、chunk 境界のノイズや抑揚への影響を測る |
| exp0008 | ブラウザ推論 | 0004/0006/0007 | ONNX Runtime Web(WebGPU/WASM/WebNN)、ネイティブ WebGPU、Web Worker、ブラウザ内キャッシュ | GitHub Pages で動く最小構成を作り、runtime/backend、対応演算、初回 download、端末差、UI を止めない構成を比較する |
| exp0009 | エッジ推論 | 0004/0006/0007 | CPU、Jetson 等の edge、可能なら Core ML/TFLite/ExecuTorch | デバイスごとの変換可能性、実時間動作、消費電力・メモリを比較する |
| exp0010 | サービング最適化 | 有望な構成 | dynamic batching、continuous batching、複数 stream、モデル常駐、queue 制御 | サーバー用途でレイテンシを保ちながら throughput を上げる |
| exp0011(任意) | クラウド推論・デプロイ | 0005/0006/0007/0010 | コンテナ化、managed endpoint/serverless、CPU/GPU instance、autoscaling、streaming API、監視 | 構成ごとの cold start、レイテンシ、throughput、可用性、運用負荷、音声 1 分あたりの費用を比較する |
| exp0012 | ブラウザ統合・公開 | 0006/0007/0008 | 最良の軽量モデル、変換、chunk 推論、Web UI を統合 | GitHub Pages 上で動くリアルタイム TTS を完成させ、端末・ブラウザ別に最終評価する |

exp0004・0005 では全体を一度に変換できない可能性がある。まずボコーダや encoder のような固定 shape に近い部品から変換し、
Python/PyTorch と変換済み engine を組み合わせた構成も正式な比較対象にする。

## 各テーマで比べる条件

### 量子化

- 対象: T2S、S2M、ボコーダを個別・組み合わせで比較する
- 精度: FP32、FP16/BF16、INT8、INT4。活性値まで量子化する場合は別条件にする
- 方法: post-training quantization と quantization-aware training を分ける
- 確認点: 特定層を高精度に残す必要性、GPU/CPU で本当に高速化するか、出力の崩壊・ノイズ・話者類似度低下

重みサイズが小さくなっても、カーネルやハードウェアが対応しなければ速くならないため、圧縮率と実測速度を分けて報告する。

### 蒸留・モデル構造の軽量化

- logits・隠れ状態・音声トークン列・Mel/波形のどこを teacher に合わせるかを比較する
- 自己回帰 T2S は小型 student に加え、投機的生成や非自己回帰 student も候補にする
- flow/diffusion 系は step 数の単純削減を先に測り、その後に consistency/step distillation を試す
- ボコーダは小型化と既存の軽量ボコーダへの置換を比較する
- structured pruning、低 rank 化、層共有は、蒸留なし/ありを分けて効果を測る

### ONNX・TensorRT

- export 前後で同じ中間出力・最終音声になるかを数値で検証する
- dynamic shape、制御フロー、乱数、autoregressive loop、未対応 operator を一覧化する
- 固定長 padding と動的 shape の速度差、engine 数・キャッシュ容量の増加も測る
- 変換時間や保守コストを含め、PyTorch のままの最適化に対する上積みを示す

### ストリーミング

「文章を分割して順に合成するだけ」と、モデル内部の状態を引き継ぐ真のストリーミングを分ける。

- TTFA、chunk ごとの生成時間、再生が推論に追いつくかを測る
- chunk サイズ・lookahead と、品質・抑揚・文脈保持の関係を比較する
- chunk 境界の無音、クリック、音量/F0 の不連続を測り、overlap-add/cross-fade の効果を見る
- 長文でメモリ使用量が増え続けないか、途中キャンセルがすぐ反映されるかを確認する

### ブラウザ・エッジ

- ブラウザの第一候補として、exp0004 で export した ONNX を ONNX Runtime Web で実行する
- ONNX Runtime Web の WebGPU・WASM・利用可能なら WebNN の各 backend で、速度、対応 operator、精度、メモリを比較する
- ONNX Runtime Web とネイティブ WebGPU/その他のブラウザ向け runtime も、同じモデル・入力・品質条件で比較する
- WebGPU 対応/非対応、初回と cache 後、PC/スマートフォンを分ける
- エッジ: GPU/NPU/CPU の delegate 対応、thermal throttling、電池・消費電力を確認する
- モデルだけでなく、G2P・音声 codec・未対応 operator を含む end-to-end の可搬性を評価する
- 重みの分割 download、IndexedDB/cache、Web Worker/AudioWorklet を使った非同期再生も対象にする

GitHub Pages への統合では、ブラウザで動かしやすいモデルを最後に選ぶのではなく、前段の実験から export 可能性・演算子対応・重みサイズを
選定条件に含める。品質が最も高い構成と、ブラウザでリアルタイム動作する構成が異なる場合は、用途別の Pareto front として両方を残す。

### クラウド推論・デプロイ(任意)

最終目標はブラウザ内推論だが、大きなモデル、低性能端末、非対応 operator ではクラウド推論が有利な場合があるため、比較候補として残す。

- Docker/OCI image として再現可能にし、CPU・GPU の instance type ごとに TTFA、RTF、throughput、費用を測る
- 常駐 endpoint、scale-to-zero/serverless、spot/preemptible instance を比較し、cold start と可用性の差を記録する
- model load、重み download、engine build を起動時のどこで行い、何を永続 cache するかを比較する
- autoscaling の指標(queue 長、同時実行数、GPU 使用率)、最小/最大 replica、scale out 時間を評価する
- HTTP の一括応答と WebSocket/streaming response を比較し、ネットワークを含む TTFA と中断処理を測る
- dynamic batching、複数 GPU、TensorRT engine など exp0010 までの最適化がクラウド費用をどれだけ下げるかを見る
- logs、metrics、trace、health check、失敗時の再試行、モデル version/rollback を含む最小限の運用方法を整理する
- browser-only、cloud-only、端末性能に応じて切り替える hybrid の 3 構成を、プライバシー・品質・速度・費用で比較する

## 追加で検証する候補

当初の項目に加え、次も効果が大きい可能性がある。

- **プロファイリングとボトルネック分解**: 最適化の前提。T2S が支配的か、flow step か、ボコーダかを入力長別に判断する
- **生成アルゴリズム**: KV cache、投機的生成、early exit、flow/diffusion の可変 step。モデル圧縮なしでも効く可能性がある
- **cache**: 話者 embedding、参照音声特徴、text encoder 出力、変換済み engine。反復利用時の効果を別に測る
- **長さ別ルーティング**: 短文は低レイテンシ構成、長文は高 throughput 構成など、用途に応じて engine/設定を切り替える
- **サービング最適化**: dynamic/continuous batching、admission control、複数 GPU、音声の逐次送信
- **コスト・電力**: 1 分の音声を作る時間、電力量、クラウド費用。速度だけでは選べない構成の判断材料にする
- **頑健性**: 長文、異常に短い文、記号、連続リクエスト、OOM 後の復旧、決定性を負荷試験する

## 実験の進め方

1. 仮説と、改善したい指標・許容する品質低下を先に書く
2. 比較対象以外のモデル、入力、seed、生成設定を固定する
3. 中間出力の一致を確認してから、速度・メモリを測る
4. 共通評価文で品質を測り、代表音声を `results/samples/` に保存する
5. 結果、失敗した条件、環境、再現手順を各 exp の README に残す
6. 有望な構成は単独の改善だけでなく、他の最適化と組み合わせて再評価する

各 exp は原則として次の形にそろえる。

```text
expXXXX/
├── README.md              目的、条件、手順、結果、考察
├── benchmark.py           共通条件での速度・メモリ測定
├── evaluate.py            品質評価
├── export_or_optimize.py  変換・最適化(必要な場合)
└── results/
    ├── metrics.json       機械可読な結果
    ├── environment.json   ハードウェア・ソフトウェア環境
    └── samples/           比較用の合成音声
```

## 最初に行うこと

最初の `exp0001` では手法を増やさず、対象にする IndexTTS 系モデルと代表的な実行環境を固定する。その上で、

1. end-to-end と各区間の時間を測れる benchmark harness を作る
2. cold start、warm、短/中/長文、単発/バッチを測る
3. FP32・FP16・BF16 を同じ品質評価とともに比較する
4. profiler から、次に着手する対象を決める

という順に進める。以降の優先順位は事前に固定せず、exp0001 で大きかったボトルネックを基準にする。一般には、学習不要で効果を
切り分けやすい exp0002 → 量子化 → 部品単位の ONNX/TensorRT を先に行い、共通ベンチマークが安定してから蒸留・ストリーミングへ進む。

最終成果から逆算すると、ブラウザ向けには `ONNX export 可能性 → ONNX Runtime Web の各 backend での演算子対応 → モデル取得量 → RTF → TTFA → chunk 品質` を
継続的に確認する。exp0008 で初めてブラウザへ移植するのではなく、exp0004 以降の有望なモデルは小さな browser smoke test も通す。

## 結果のまとめ

量子化exp0003のA〜Cは比較済み。[Cの比較表](exp0003/C_REPORT.md)に、重みINT8/INT4・group size・AWQ・GPTQの速度、CER、サイズ、メモリをまとめた。
現時点ではGPU速度重視はFP32、GPUメモリとのバランスはBF16、CPU速度重視は両部品の動的INT8が採用候補。
重みファイル最小化は単純INT4 g128だが、速度低下と読みCERの劣化を伴う。[用途別の結論](exp0003/README.md)を参照。
Dの静的INT8・SmoothQuantも[同READMEの比較表](exp0003/README.md#d-静的ptqsmoothquant)に集約する。
GPUでは重み約106 MiB・peak VRAM約311 MiBまで縮小できたが、今回の実装はeager FP32より遅く、速度重視の推奨は変わらない。
CPU静的INT8は中文で基準の約8.6倍遅いため除外し、CPU SmoothQuantも同じ演算経路では追加測定しない。

Eの混合精度8条件も比較済み。[Eの結果と採用判断](exp0003/README.md#e-高精度に残す部品層の比較)を参照。
GPUはS2MのみINT4で読みとファイル圧縮のバランスを取りやすい一方、CPUは全体の動的INT8を維持する。
入出力側を一律に高精度へ戻す方法は、今回のCPU条件では速度・サイズ・CERのいずれも不利だった。

FのQATも固定200 step・追加学習対照を含むGPU5条件で比較済み。[QATの必要性と結果](exp0003/README.md#f-qatの必要性と固定予算での検証)を参照。
現状の採用候補にはQATは必須でなく、今回の検証でも追加学習後PTQに対する明確な優位を得られなかったため採用しない。

実験開始後、ここに各 exp の代表値と結論を追記する。異なるハードウェアの絶対速度を横並びにはせず、環境ごとに基準との差を示す。

| exp | 手法 | 環境 | TTFA | RTF | peak VRAM/RAM | サイズ | 品質差 | 結論 |
|---|---|---|---:|---:|---:|---:|---:|---|
| [exp0001](exp0001/) | FP32/FP16/BF16ベースライン | RTX 5090 | 非ストリーミングのためend-to-endと同じ | FP32: 短文2.524 / 中文0.305 / 長文0.113 | FP32: 475〜565 MiB | checkpoint合計438.3 MiB | FP32 6.70% / FP16 7.07% / BF16 6.61%(テスト100文CER) | T2Sが約80%。autocastはVRAMを約7%減らすが高速化せず、FP32が最速 |
