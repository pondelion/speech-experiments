# exp0002: 実行環境・ランタイムの最適化

目的は、TTSの方式をまたいで使える最適化の効果と適用条件を整理すること。
最初の題材はexp0001と同じcheckpointだが、特定モデルでの実測結果と一般化できる仮説を区別する。

## 比較項目

| 項目 | 検証すること | 適用範囲・注意 |
|---|---|---|
| CPU thread数 | 1・4・8・20 threadsの速度とRAM | 小さな演算ではthread管理の費用が増える可能性。最適値はCPU・入力・モデル依存 |
| GPU時のCPU thread数 | host側のthread設定による速度差 | GPU処理にもCPU側の実行費用がある |
| コンパイル | eagerとコンパイル後の速度、初回compile時間 | 動的shape、graph break、対応演算を記録する |
| 転送・キャッシュ | CPU/GPU転送、再計算、推論時に不必要な部品 | 入力やモデルの意味を変えずに減らせる処理を対象にする |

Griffin-LimはNNボコーダが未実装のための暫定処理で、専用チューニングは優先しない。
モデル部分(T2S＋長さ予測＋S2M)と波形化の時間を分けて示す。
T2Sのマスク反復とS2MのFlow Matchingは異なる生成方式であり、step数調整を全TTS共通の手法とは扱わない。

## thread数の測定

exp0001のbenchmark.pyを再利用し、短・中・長文、FP32、batch 1、warm-up 1回、計測3回、seed列を固定する。
各条件は独立プロセスで実行する。

```bash
python ../exp0001/benchmark.py --device cpu --threads 4 --case all --warmup 1 --repeats 3 --output results/cpu_threads4/metrics.json
```

今回CPUの4・8 threadsと、GPUの8 threadsを測った。CPUの20 threadsとGPUの既定thread条件はexp0001を参考値にし、
厳密な比較が必要なら同じ反復・seed条件で再計測する。
変更したのはintra-op thread数で、interopは20のまま。1 threadの比較とinteropの調整は未実施。

## 結果

生成時間のmedian:

| 入力 | CPU 4 threads | CPU 8 threads | CPU 20 threads(exp0001) | GPU 8 threads | CPU 8-thread RTF |
|---|---:|---:|---:|---:|---:|
| 短文 | 1.33秒 | 1.10秒 | 2.28秒 | 1.58秒 | 1.754 |
| 中文 | 6.82秒 | 4.89秒 | 7.84秒 | 1.51秒 | 0.904 |
| 長文 | 19.94秒 | 13.02秒 | 16.81秒 | 1.73秒 | 0.888 |

8 threadsは今回のCPU条件では全入力で最速で、20 threadsより約23〜52%短縮した。中文・長文は実時間より速く生成できた。
GPUではexp0001の1.59・1.65・1.66秒に対して1.58・1.51・1.73秒で、一貫した改善は確認できない。
CPUがGPUより短文で速かった点も、この小さな入力・モデル・実行条件での結果であり、一般的な優劣にはしない。

短文・中文では4 threadsが速く、長文では20 threadsが速かった。4-thread長文ではT2Sが7.41秒、S2Mが12.26秒。
20-thread長文の8.49秒・8.09秒に対し、処理段階によって逆の傾向がある。
単純に全core数を指定することも、thread数を減らすことも、一律には最適にならない。
実測はこのCPU・モデル・入力に限られるため、一般的な最適thread数としては扱わない。

thread数によって浮動小数点計算の順序が変わりうるため、有限値検査と代表トークンのhashも比較する。
hash一致は音声品質全体の同等性を保証しない。

今回の全条件でMel・波形は有限値。品質の再評価、他モデルでの再現は未実施。

## S2M演算部のコンパイル

`torch.compile(fullgraph=True, dynamic=False)`をS2Mの`velocity`に適用した。
テンソル演算部だけをコンパイルする方法の最初の検証であり、Flow Matchingのstep数は変更していない。
中文、FP32、8 threads、warm-up 1回、計測3回。比較元は前節のeager条件。

| 環境 | eager S2M | compile S2M | eager全体 | compile全体 | compileを含む初回warm-up |
|---|---:|---:|---:|---:|---:|
| CPU | 2.490秒 | 2.290秒 | 4.888秒 | 4.898秒 | 20.39秒 |
| GPU | 0.253秒 | 0.074秒 | 1.513秒 | 1.429秒 | 12.05秒 |

GPUのS2Mは約3.4倍速くなったが、全体の短縮は約6%。CPUの全体速度には改善が見られなかった。
最適化対象が全体の一部なら、局所の高速化ほどには全体が速くならないことを確認した。
コンパイル費用を回収するには同じプロセスでの反復利用が必要になる。

初回warm-upは推論・初期化も含み、純粋なcompile時間ではない。キャッシュも消していない。
静的shapeのため他の入力長で再compileする可能性があり、短文・長文や別モデルでの効果は未検証。
前節との計測時刻が異なり、3回のみなので小さな全体速度差には再測定が必要。
代表トークンhashは対応するeager条件と一致し、Mel・波形の有限値検査も通ったが、
コンパイルしたS2M出力の数値誤差・CER・聴感の同等性は未確認。採用候補とし、品質維持を確認してから既定化する。

```bash
python ../exp0001/benchmark.py --device cpu --threads 8 --case medium --warmup 1 --repeats 3 --compile-s2m --output results/cpu_compile/metrics.json
python ../exp0001/benchmark.py --device cuda --threads 8 --case medium --warmup 1 --repeats 3 --compile-s2m --output results/gpu_compile/metrics.json
```

### コンパイル前後の数値検証

`check_compile.py`で固定乱数のテンソル(batch 2、508 frames、末尾paddingあり)を比較した。
時刻0・0.5・0.96の単発演算と、同じ初期値からの25回のEuler反復を測定した。
これは合成テンソルによる検査で、実際のTTSやCFG全体の再現ではない。
許容誤差は`atol=1e-4, rtol=1e-4`として実行前に固定した。

| 対象 | CPU 最大絶対誤差 | GPU 最大絶対誤差 | allclose |
|---|---:|---:|---|
| 単発演算(3時刻の最大) | 6.32e-6 | 4.29e-6 | 両方通過 |
| 25回反復 | 1.56e-5 | 2.33e-4 | CPU通過、GPU不通過 |

全出力は有限値。GPU反復の相対L2誤差は5.37e-6と小さいが、一部要素は設定した許容誤差を超えた。
許容値を事後的に緩めず、この条件の検査は失敗として記録する。音声品質の悪化を意味するとは限らないが、
速度のみを根拠に既定化はしない。実際の中間出力・品質評価で許容性を確認する必要がある。

汎用的な知見は、コンパイルの速度測定に加えて単発演算と反復後の誤差を分けて検査すること。
反復型以外のモデルでは、実際の推論経路に対応する後段出力を検査する。
結果は`results/compile_numerics_cpu.json`・`compile_numerics_cuda.json`。

```bash
python check_compile.py --device cpu
python check_compile.py --device cuda
```

次は量子化・ONNXの比較へ進む。コンパイルは既定にせず、eagerを比較基準に保持する。

## ハイブリッドT2Sの反復処理とcache候補

### 目的と対象

現在の日本語HuBERT版は、長さ予測（duration_model=nar）とMaskGCT方式の音声トークン生成を組み合わせたハイブリッド型である。生成ループに時間がかかるという予想を確かめるため、まずTTS全体に対するT2Sの割合と、T2S内部の処理時間をCPU/GPUで分解した。その結果から、再計算を減らせる箇所と近似cacheが必要な箇所を区別する。

| ファイル | 役割 | 実行・出力 |
|---|---|---|
| [profile_t2s.py](profile_t2s.py) | 現行コードへ計測scopeを挿入し、生成IDの一致も検査 | `--device cpu/cuda --config ...` → `results/t2s_profile/*_all.json` |
| [summarize_t2s.py](summarize_t2s.py) | 内訳と改善候補を集約 | `python summarize_t2s.py` → 本節 |

### 測定方法

exp4と同じcheckpoint・短/中/長文・FP32の入出力・生成設定を使い、各文warmup 1回＋3回を測定した。CPUはPyTorch動的INT8とexp4のORT INT8、GPUはPyTorch FP32とexp4のORT FP32＋CUDA Graphを比較した。各条件は独立プロセスで、他の速度・品質計測と排他実行した。

計測なしでTTS全体を測った後、T2Sにだけ計測を挿入した。各試行では直前にT2S単独版を計測なしで実行し、同じseedの計測版と生成トークンのhashが一致することを検査した。TTS全体とのhash一致も別に記録した。モデルの数式・生成step数・cacheの有無は変えていない。

CPUはhost経過時間、GPUは同じCUDA stream上のevent間経過時間を使う。GPU値はCPUからの起動待ちも含み、純粋なkernel実行時間ではない。細分化のたびにGPU同期を追加していない。計測scopeの追加費用は下表の差で確認し、内訳の絶対値を通常推論の速度と混同しない。CUDA eventは区間計測の前に確保・初期化する。

### TTS全体に占めるT2Sの割合

| 条件 | 入力 | TTS全体 秒 | T2S 秒 | T2S割合 % |
|---|---|---:|---:|---:|
| CPU・PyTorch INT8 | short | 0.737 | 0.433 | 58.8 |
| CPU・PyTorch INT8 | medium | 3.248 | 1.617 | 49.8 |
| CPU・PyTorch INT8 | long | 8.343 | 3.172 | 38.0 |
| CPU・ORT INT8（exp4） | short | 0.661 | 0.451 | 68.2 |
| CPU・ORT INT8（exp4） | medium | 2.355 | 1.215 | 51.6 |
| CPU・ORT INT8（exp4） | long | 7.693 | 2.609 | 33.9 |
| GPU・PyTorch FP32 | short | 1.232 | 0.970 | 78.8 |
| GPU・PyTorch FP32 | medium | 1.345 | 1.083 | 80.5 |
| GPU・PyTorch FP32 | long | 1.272 | 1.007 | 79.2 |
| GPU・ORT FP32＋CUDA Graph（exp4） | short | 0.257 | 0.190 | 74.1 |
| GPU・ORT FP32＋CUDA Graph（exp4） | medium | 0.394 | 0.276 | 69.9 |
| GPU・ORT FP32＋CUDA Graph（exp4） | long | 0.568 | 0.334 | 58.9 |

### T2S内部の内訳（中文）

| 条件 | 長さ予測 ms | 生成前準備 ms | Masked生成 ms | Masked比率 % | 未計測版との経過時間差 % |
|---|---:|---:|---:|---:|---:|
| CPU・PyTorch INT8 | 6.30 | 4.43 | 1464.21 | 99.4 | -0.3 |
| CPU・ORT INT8（exp4） | 5.96 | 4.72 | 1247.86 | 99.2 | +2.4 |
| GPU・PyTorch FP32 | 3.53 | 2.78 | 1120.52 | 99.4 | +8.5 |
| GPU・ORT FP32＋CUDA Graph（exp4） | 1.76 | 1.40 | 368.08 | 99.1 | +32.6 |

**CPU・PyTorch INT8: Masked生成内の内訳**

| 処理 | CPU経過 ms | CUDA stream経過 ms | 割合 % |
|---|---:|---:|---:|
| Transformer群 | 1385.22 | — | 94.6 |
| 固定テキストの準備・Embedding | 1.62 | — | 0.1 |
| 音声トークンのEmbedding | 3.51 | — | 0.2 |
| 音声条件加算・系列結合・padding | 5.80 | — | 0.4 |
| Attention用padding maskの構築・転送 | 1.26 | — | 0.1 |
| 時刻条件の構築・転送とMLP | 12.87 | — | 0.9 |
| 出力のslice・padding・有効位置mask | 6.91 | — | 0.5 |
| hiddenの正規化・残り | 5.85 | — | 0.4 |
| CFGの合成・標準偏差 | 9.41 | — | 0.6 |
| 語彙への射影 | 5.67 | — | 0.4 |
| top-kによる候補絞り込み | 7.49 | — | 0.5 |
| トークンsampling | 7.84 | — | 0.5 |
| 確信度の計算 | 3.60 | — | 0.2 |
| 再マスクの順位計算（argsort 2回） | 1.18 | — | 0.1 |
| 停止判定（max→Python整数） | 0.34 | — | 0.0 |
| トークン・mask更新とschedule | 2.34 | — | 0.2 |
| その他のループ処理 | 2.45 | — | 0.2 |

長さ予測後に再実行したencoderは3.62 msで、T2Sの0.25%だった。この重複だけを除いても、その割合を超える短縮は得られない。

**CPU・ORT INT8（exp4）: Masked生成内の内訳**

| 処理 | CPU経過 ms | CUDA stream経過 ms | 割合 % |
|---|---:|---:|---:|
| Transformer群 | 1150.10 | — | 92.2 |
| 固定テキストの準備・Embedding | 1.57 | — | 0.1 |
| 音声トークンのEmbedding | 3.12 | — | 0.2 |
| 音声条件加算・系列結合・padding | 6.00 | — | 0.5 |
| Attention用padding maskの構築・転送 | 1.19 | — | 0.1 |
| 時刻条件の構築・転送とMLP | 19.75 | — | 1.6 |
| 出力のslice・padding・有効位置mask | 9.44 | — | 0.8 |
| hiddenの正規化・残り | 14.66 | — | 1.2 |
| CFGの合成・標準偏差 | 9.85 | — | 0.8 |
| 語彙への射影 | 5.94 | — | 0.5 |
| top-kによる候補絞り込み | 7.50 | — | 0.6 |
| トークンsampling | 7.79 | — | 0.6 |
| 確信度の計算 | 4.25 | — | 0.3 |
| 再マスクの順位計算（argsort 2回） | 1.32 | — | 0.1 |
| 停止判定（max→Python整数） | 0.32 | — | 0.0 |
| トークン・mask更新とschedule | 2.58 | — | 0.2 |
| その他のループ処理 | 2.24 | — | 0.2 |

長さ予測後に再実行したencoderは4.50 msで、T2Sの0.36%だった。この重複だけを除いても、その割合を超える短縮は得られない。

**GPU・PyTorch FP32: Masked生成内の内訳**

| 処理 | CPU経過 ms | CUDA stream経過 ms | 割合 % |
|---|---:|---:|---:|
| Transformer群 | 916.43 | 1020.00 | 81.9 |
| 固定テキストの準備・Embedding | 8.22 | 6.28 | 0.5 |
| 音声トークンのEmbedding | 6.93 | 5.87 | 0.5 |
| 音声条件加算・系列結合・padding | 17.35 | 17.09 | 1.4 |
| Attention用padding maskの構築・転送 | 8.27 | 6.82 | 0.5 |
| 時刻条件の構築・転送とMLP | 33.42 | 35.34 | 2.8 |
| 出力のslice・padding・有効位置mask | 15.77 | 11.92 | 1.0 |
| hiddenの正規化・残り | 12.47 | 25.12 | 2.0 |
| CFGの合成・標準偏差 | 27.79 | 28.46 | 2.3 |
| 語彙への射影 | 3.89 | 3.43 | 0.3 |
| top-kによる候補絞り込み | 5.82 | 5.77 | 0.5 |
| トークンsampling | 9.79 | 10.59 | 0.9 |
| 確信度の計算 | 17.62 | 17.12 | 1.4 |
| 再マスクの順位計算（argsort 2回） | 8.25 | 8.56 | 0.7 |
| 停止判定（max→Python整数） | 6.59 | 6.87 | 0.6 |
| トークン・mask更新とschedule | 11.00 | 9.73 | 0.8 |
| その他のループ処理 | 3.91 | 18.41 | 1.5 |

長さ予測後に再実行したencoderは2.41 msで、T2Sの0.19%だった。この重複だけを除いても、その割合を超える短縮は得られない。

**GPU・ORT FP32＋CUDA Graph（exp4）: Masked生成内の内訳**

| 処理 | CPU経過 ms | CUDA stream経過 ms | 割合 % |
|---|---:|---:|---:|
| Transformer群 | 125.88 | 136.03 | 33.7 |
| 固定テキストの準備・Embedding | 9.50 | 7.41 | 1.8 |
| 音声トークンのEmbedding | 8.26 | 6.87 | 1.7 |
| 音声条件加算・系列結合・padding | 19.96 | 19.51 | 4.8 |
| Attention用padding maskの構築・転送 | 9.61 | 7.86 | 1.9 |
| 時刻条件の構築・転送とMLP | 39.99 | 41.67 | 10.3 |
| 出力のslice・padding・有効位置mask | 16.28 | 15.50 | 3.8 |
| hiddenの正規化・残り | 16.57 | 32.64 | 8.1 |
| CFGの合成・標準偏差 | 34.11 | 33.03 | 8.2 |
| 語彙への射影 | 4.69 | 4.18 | 1.0 |
| top-kによる候補絞り込み | 6.90 | 6.47 | 1.6 |
| トークンsampling | 12.07 | 12.45 | 3.1 |
| 確信度の計算 | 21.36 | 20.60 | 5.1 |
| 再マスクの順位計算（argsort 2回） | 9.40 | 9.75 | 2.4 |
| 停止判定（max→Python整数） | 7.09 | 7.02 | 1.7 |
| トークン・mask更新とschedule | 12.61 | 10.90 | 2.7 |
| その他のループ処理 | 4.66 | 22.66 | 5.6 |

長さ予測後に再実行したencoderは0.87 msで、T2Sの0.21%だった。この重複だけを除いても、その割合を超える短縮は得られない。

親区間と子区間を足すと二重計上になるため、詳細表では子区間を引いたself時間をまとめた。各値は試行ごとの中央値なので、丸めと試行差により割合の合計は厳密に100%にならない。GPUの停止判定のCPU時間には先行GPU処理を待つ時間も含まれるため、その全時間が判定の除去で消えるとは解釈しない。

<!-- T2S_CANDIDATES_START -->
### 結論と優先順位

計測なしの中文推論では、T2SはTTS全体の約50〜52%（CPU）、約70〜81%（GPU）を占めた。T2S内部の計測では約99%がMaskedT2S.generateであり、生成ループが主要な改善対象であることを確認できた。ただし、CPUではS2Mなども約半分残るため、T2Sだけの改善をTTS全体の改善率とは扱わない。

CPUではMasked生成内のTransformer群が約92〜95%を占めるため、大幅な短縮にはTransformerの反復計算自体を減らす必要がある。一方、GPUのexp4条件ではTransformer群が約34%、時刻条件の構築・MLPが約10%、系列結合が約4%、出力整形が約4%となった。この結果から、まず出力を維持できる固定条件の再利用と生成ループの起動回数削減を優先し、その後に精度との交換条件が生じる近似cacheを検証する。

**今回完了したのは内訳計測と候補の洗い出しであり、以下の最適化の実装・高速化効果・CER評価は未実施である。** GPUのexp4条件では細分化計測の経過時間が未計測版より32.6%増加しており、表の割合は候補の選定用である。通常推論で同じ時間が削減できるとは断定しない。

### 出力を維持する方針の候補

| 優先度・対象 | 計測・コードから分かったこと | 改善策 | 効果の見込みと検証事項 |
|---|---|---|---|
| 1・GPU | 各stepのCFG有／無で同じ時刻条件を2回計算。exp4条件では時刻条件全体が約42 ms | timestep特徴とtime_mlp出力をstepごとに1回計算して再利用 | MLPの重複と毎回の時刻転送を削減。動的INT8は入力batch形状を維持し、一括計算による量子化尺度の変化を避ける |
| 1・GPU | テキスト、系列長、padding mask、出力の有効位置は反復中に不変 | CFG分岐ごとにtext embedding・padding mask・slice位置・有効位置maskを保持 | Transformer入力前の繰り返し構築を削減。変化する音声embeddingやTransformer出力は使い回さない |
| 1・GPU | exp4のCUDA GraphはTransformer群が中心で、その外側に細かい演算が残る | 固定shapeの入力構築・CFG・sampling・mask更新を融合、またはcapture範囲を拡大 | 多数のCPU起動を削減する候補。乱数状態、ORTとのstream同期、停止分岐を含むため、全ループをそのままcaptureできるとは限らない |
| 2・GPU | 次stepのmask数は時刻scheduleと系列長だけで決まる。毎回maxをPython整数に変換 | mask数と停止stepを事前計算し、ループ内のCPU/GPU同期を除去 | 停止判定区間は約7 msだが、先行GPU処理の待機を含むため全量は消えない。同じFP32丸めと整数化を再現する |
| 2・GPU | 再マスク順位にargsortを2回実行。exp4条件では約10 ms | 最初のargsortを残し、2回目の逆順列計算をscatterで置換 | 同点の順序を維持して順位を復元できる。出力一致と通常推論の時間で判定し、top-kへの変更とは分ける |
| 2・CPU/GPU | 各blockの時刻依存AdaLNパラメータもCFG両分岐で同じ | time_mlpだけでなく各blockの時刻射影も再利用 | ORTではgraph入出力の変更が必要。graph分割による起動費用が削減量を上回らないか確認する |
| 3・CPU/GPU | 長さ予測と生成前に同じtext encoderを2回実行 | encoder出力を長さ予測から生成側へ渡す | 重複encoderはT2Sの約0.2〜0.4%で、これだけで大幅な改善は見込めない |

これらは入力・数式を維持する方針だが、浮動小数点演算順や乱数消費の変更で生成結果が変わる可能性がある。まず同一seedの生成ID一致を検査し、不一致なら原因を確認したうえでCER評価が必要な条件として扱う。速度は今回と同じ短・中・長文、CPU/GPUで通常推論を測り、各デバイスの最速PyTorch条件およびexp4採用条件に対するTTS全体の効果を示す。

### Transformerの再計算を減らす候補（精度評価が必要）

この生成方式は、マスク位置を並列に予測し、確信度に応じてマスクし直すMaskGCT方式で、マスク型diffusion LLMに近い。通常の自己回帰KV cacheでは過去位置のK/Vが変わらないことを利用するが、本実装では双方向Attentionで周囲の音声トークンが更新され、さらに各stepで時刻依存のAdaLNも変わる。したがって、トークンIDが同じ位置でもK/Vが同じとは限らない。**通常のAR用KV cacheの単純適用はできないが、近似的な再利用は検証候補になる。**

| 優先度・対象 | 改善策 | 適用理由と具体的な調査 | 精度・実装上の注意 |
|---|---|---|---|
| 1・CPU中心 | 更新間隔を設けたK/V・中間特徴cache | Transformerが約92〜95%を占めるため、反復間の再計算削減が重要。まず層・位置・生成前半/後半別にK/Vとhiddenの変化量を調べ、変化の小さい範囲だけ再利用する | CFG有／無のcacheを分離。毎回更新・2step間隔・4step間隔などを比較し、速度・CER・メモリを評価。K/V射影だけの省略ではMLP計算は残る |
| 2・CPU/GPU | 確定位置のhiddenを保持し、未確定位置を優先更新 | 確定したトークンも毎回全層で再計算しているため、その省略余地を調査する | 確定位置も周囲と時刻の影響を受ける。双方向AttentionとCFGの全位置の標準偏差にも影響し、単なる同値な計算省略にはならない |
| 2・CPU/GPU | 信頼度に応じたcache更新・早期終了 | 一律の更新間隔より、変化量や確信度に応じて再計算する範囲を選ぶ | 判定処理自体の費用も含める。step削減はこの生成方式に依存するため、TTS共通の最適化とは区別する |
| 3・方式変更 | block単位の生成とcache | 先行blockの再計算を抑える構造を検討する | 現在の全系列並列生成からscheduleが変わり、学習条件との整合性確認も必要。最初の実装対象にはしない |

関連する一次資料として、[MaskGCT](https://arxiv.org/abs/2409.00750)はマスクによる音声生成、[dKV-Cache](https://arxiv.org/abs/2505.15781)はdiffusionモデルの特徴変化を踏まえた遅延・条件付きcache、[Fast-dLLM](https://arxiv.org/abs/2505.22618)は双方向モデルでのblock単位の近似KV cacheを扱う。後二者は主に言語モデルの研究であり、その高速化率や品質が本TTSへそのまま当てはまるとは仮定しない。

次の実装順は、**固定条件・時刻条件の再利用 → 同期・順位計算・起動費用の削減 → 変化量の調査に基づく近似cache**とする。前二者でGPUの残存費用を減らし、後者でCPUの主要ボトルネックであるTransformer反復計算の削減を狙う。
<!-- T2S_CANDIDATES_END -->

### 再実行

```sh
python profile_t2s.py --device cpu --config torch_int8
python profile_t2s.py --device cpu --config native8_nospin_8
python profile_t2s.py --device cuda --config torch
python profile_t2s.py --device cuda --config opt_nospin_8_front1_cg
python summarize_t2s.py
```
