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

## 量子化実験後: KV cache・再計算の削減

exp0002ではKV cacheは未検証。現行T2Sは自己回帰GPTではなく、双方向attentionでマスクした音声トークンを反復予測する方式。
毎回トークンと時刻条件が変わり、テキスト側の隠れ状態も音声側の影響を受けるため、
自己回帰GPTで過去トークンのK/Vを再利用する方法をそのまま適用しない。
T2SがGPU時間の大部分を占める観測と、KV cacheが有効かどうかは別の問題である。

exp0003終了後、次を分けて検証する。

- 自己回帰TTSの題材で、KV cacheなし/あり、prefill/decode、cacheのメモリ・精度・長文での速度を比較する
- 現行の非自己回帰T2Sで、独立したtext encoderの重複計算、固定embedding、alignment等の再利用を検証する
- 固定部分のみを再利用し、時刻・音声トークン依存の隠れ状態を誤って使い回さないことを数値・CERで確認する

量子化とcacheの単独効果を測った後に組み合わせ、weight量子化とKV cache量子化も区別する。

結果JSONの`experiment`は再利用したharnessの`exp0001`表記を保持する。実験条件は保存ディレクトリと`threads_requested`で識別する。
