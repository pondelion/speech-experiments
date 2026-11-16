# exp0004: ONNX Runtimeの性能改善

## 目的とexp0003との役割分担

exp0003のGは、量子化した計算部分がONNX RuntimeのCPU/CUDAで実行できるかと、精度・サイズ・速度の変化を確認した。
本expはONNX Runtime（以下ORT）のgraphと接続実装を基準に、**遅くなった原因を切り分け、実際のTTS全体を高速化する**。
export成功や設定変更だけを成果にせず、CPU/GPUごとの性能改善とCER維持を検証する。
量子化方式の総当たりは繰り返さず、PyTorch FP32・既存のCPU動的INT8・exp3のORT設定を比較対象にする。

| 項目 | exp0003 G | exp0004 |
|---|---|---|
| 主目的 | 量子化形式と別runtimeの互換性・トレードオフ | 遅延原因の特定と解消、PyTorch版の最速条件に対する性能改善 |
| 共用するもの | T2S encoder/block群・S2M velocityの可変長graph、I/O Binding | 左記を初期基準として再利用 |
| 新しく変えるもの | 量子化・runtime | thread pool、待機時のCPU使用、Attentionの表現と融合、graph最適化、必要な境界処理 |
| 完了判断 | 選定条件の比較 | 有効な変更の切り分けと、短/中/長文の速度・CER・メモリでの確認 |

## 改善策のサマリ

**最速PyTorch比で、GPUはFP32のまま中文を約68%短縮した。CPUの速度優先条件は約27%短縮したが、テストCERが0.346ポイント悪化する。精度とメモリも含めた採否を末尾にまとめる。**

| 改善策としてやったこと（端的に） | その具体的な説明 | 効果（PyTorch最速条件を基準） |
|---|---|---|
| 待機workerのCPU消費を抑える | spinning無効化、pool共有、thread数を切り分ける | CPU消費は減ったが中文6.180秒で、PyTorch動的INT8の3.339秒には届かない |
| 分解Attentionを1演算にする | SDPAのsymbolicを差し替え、MatMul・mask・Softmax群を標準Attentionにする | 初回はCPU中文4.487秒・GPU中文1.181秒。中間条件の参考値として残し、最終的な採否は追加対策を含むC.1の比較で判断する |
| RoPE・固定幅分割を直接演算化する | RoPEのSlice等を専用RotaryEmbeddingへ、時刻条件の6分割を固定幅Splitへ変換する | SliceはT2S/S2Mとも80→0個。GPUの専用演算化のみの探索は0.591秒。CPU FP32は3.944秒で、まだINT8基準には届かない |
| GPUの反復起動をまとめる | 固定I/Oを使うORT CUDA Graph。同期を維持し、CPU fallbackを禁止する | 正式計測0.252/0.363/0.571秒で、PyTorch最速比76/68/53%短縮。CER 6.420/6.396%で悪化なし（今回の評価） |
| 異なる文長でGPU実行を再利用する | 128位置単位でpadし、余白をAttentionから除外。出力を元の長さに戻す。session cacheは各graph最大2個 | 正式計測0.305/0.384/0.605秒で、PyTorch最速比71/66/51%短縮。テストCER -0.130pt、実用文 +0.200pt |
| CPUの量子化条件を合わせる | per-tensor/per-channelと整数範囲を比較し、残存PyTorch部品も最速基準と同じ動的INT8へ合わせる | 正式計測0.675/2.404/7.806秒。中文はPyTorch最速比27%短縮。テストCER +0.346pt、実用文 -0.266pt |
| CPU入力の量子化範囲も合わせる | weightをPyTorchのobserverで量子化し、入力値域0〜127を使うscale・zero point計算をONNX化する | 正式計測0.746/2.803/8.999秒。中文はPyTorch最速比15%短縮。テストCER +1.146pt、実用文 -1.532pt |
| Linearを2D Gemmへ変換する | rank-3のMatMul＋biasをflatten→Gemm→reshapeへ書き換え、CPU融合の効果を検証する | 探索4.04〜4.11秒。改善せず不採用。主要負荷がMatMulであっても、この表現変更だけでは速くならなかった |
| 低精度化だけで速度を上げる | Attention版の定数MatMulをINT8化、またはgraph内部をFP16化する | 先行条件はCERが悪化。品質維持の高速化としては採用しない |

## 基準と評価方法

FP32は32 bit浮動小数点、BF16/FP16は異なる16 bit浮動小数点形式。動的INT8は重みを8 bit整数で保持し、入力のscaleを推論時に求めて整数MatMulを実行する。今回のORT INT8は定数重みMatMulをper-channelで量子化し、Attention本体等は浮動小数点のまま。FP16条件はgraph内部を16 bit化するが、Pythonとの入出力はFP32で共通化する。全演算をINT8/FP16に置換したという意味ではない。

最初の題材はこれまでと同じ学習済みT2S・S2M・codecとし、生成条件・暫定ボコーダを固定する。
固有のstep数やボコーダ変更で結果を作らず、複数sessionの管理や演算の融合など、他のTTSにも再利用できる原因を優先する。
速度はbatch 1で独立に測定し、CPUとGPUを分ける。thread数やdtypeを変えた場合は記録する。品質評価のT2S batch 16・S2M batch 4はCERを比較するための設定で、batch throughputの速度測定ではない。
探索は代表中文・warmup 1回・3回計測のscreeningとし、候補を選んだ後に短/中/長文を反復して正式比較する。
100テスト文＋55実用文のCER、生成の有限値、別プロセスのメモリ、代表3文の48 kbps MP3も確認する。
速度だけの探索結果は品質確認済みと扱わない。元モデルの語彙・checkpointや評価文を学習で変更しない。

## ボトルネックの分析

<!-- ANALYSIS_START -->
### TTS全体のブロック別時間

中文の区間別計測では、CPUの旧ORT版はT2Sが2.706秒、S2Mが3.748秒だった。最速PyTorch版の1.634秒・1.574秒と比べ、両方に大きな差があるため、CPUではこの2区間の計算本体を改善対象にした。一方、GPUでは最速PyTorch版でもORT版でもT2Sが全体の約8割を占めるため、GPUではT2Sを優先して改善する。区間別の測定値を以下に示す。

正式計測の中文・10回の中央値。各区間と全体は独立の中央値で、丸めや試行間変動により和は完全には一致しない。

| 条件 | 前処理＋長さ予測 秒 | T2S 秒 | S2M 秒 | ボコーダ 秒 | 全体 秒 |
|---|---:|---:|---:|---:|---:|
| cpu_torch_int8 | 0.007 | 1.634 | 1.574 | 0.051 | 3.293 |
| cpu_legacy | 0.118 | 2.706 | 3.748 | 0.080 | 6.648 |
| cpu_attn_shared_16_front1 | 0.009 | 1.976 | 2.269 | 0.229 | 4.487 |
| cpu_attn8_nospin_8 | 0.007 | 1.609 | 1.562 | 0.054 | 3.205 |
| cuda_torch | 0.005 | 0.886 | 0.195 | 0.024 | 1.120 |
| cuda_legacy | 0.005 | 0.941 | 0.197 | 0.031 | 1.164 |
| cuda_attn_nospin_8_front1 | 0.003 | 0.938 | 0.198 | 0.035 | 1.181 |
| cuda_opt_nospin_8_front1_cg | 0.002 | 0.253 | 0.076 | 0.030 | 0.363 |
| cuda_opt_nospin_8_front1_cg128 | 0.002 | 0.271 | 0.079 | 0.030 | 0.384 |
| cpu_native8_nospin_8 | 0.007 | 1.252 | 1.105 | 0.047 | 2.404 |
| cpu_native7_nospin_8 | 0.007 | 1.411 | 1.322 | 0.051 | 2.803 |

また、初回のthread探索ではPyTorch側を1 threadにしたため、ボコーダが約0.055→0.229秒へ逆に遅くなった。この結果から、ONNX本体と前後処理のthread数を一律に減らす設定は避け、新しいCPU候補では前後処理を8 threadsへ戻した。

### CPUの演算別時間と残存ボトルネック

演算別profileでは、Attention融合後もMatMulがT2Sで約1,920 ms、S2Mで約2,413 msを占め、node単位ではMLPのLinearが上位に残った。さらにSliceにも約425/596 msかかっていた。この結果から、MLPの行列積をINT8化し、RoPE・時刻条件分割の小演算を専用演算へ置き換える方針を選んだ。以下に演算別の時間と割合を示す。

以下はORTのnode event累積。各profileのwarmup＋中文1回を含む。profileありの診断値を、上のprofileなしの全体時間に足し合わせない。比率はそのgraph内の全演算event時間を分母とする。

**legacy**

| 計算graph | 演算 | 累積 ms | graph内比率 % |
|---|---|---:|---:|
| blocks | MatMul | 5369.0 | 69.2 |
| blocks | Slice | 431.6 | 5.6 |
| blocks | Add | 254.0 | 3.3 |
| blocks | Mul | 252.6 | 3.3 |
| blocks | Where | 241.6 | 3.1 |
| blocks | Transpose | 200.3 | 2.6 |
| velocity | MatMul | 3606.4 | 47.9 |
| velocity | Where | 1732.5 | 23.0 |
| velocity | Slice | 569.9 | 7.6 |
| velocity | IsNaN | 291.1 | 3.9 |
| velocity | Add | 266.1 | 3.5 |
| velocity | Transpose | 230.4 | 3.1 |

**attn_shared_16_front1**

| 計算graph | 演算 | 累積 ms | graph内比率 % |
|---|---|---:|---:|
| blocks | MatMul | 1920.4 | 46.4 |
| blocks | Slice | 425.0 | 10.3 |
| blocks | Attention | 313.2 | 7.6 |
| blocks | Mul | 226.3 | 5.5 |
| blocks | Add | 219.4 | 5.3 |
| blocks | Gemm | 145.4 | 3.5 |
| velocity | MatMul | 2413.2 | 52.9 |
| velocity | Attention | 629.0 | 13.8 |
| velocity | Slice | 595.9 | 13.1 |
| velocity | Transpose | 145.8 | 3.2 |
| velocity | Add | 138.3 | 3.0 |
| velocity | Mul | 133.3 | 2.9 |

**attn8_nospin_8**

| 計算graph | 演算 | 累積 ms | graph内比率 % |
|---|---|---:|---:|
| blocks | DynamicQuantizeMatMul | 1528.7 | 38.5 |
| blocks | Slice | 434.4 | 11.0 |
| blocks | Attention | 405.9 | 10.2 |
| blocks | Mul | 286.3 | 7.2 |
| blocks | Add | 170.5 | 4.3 |
| blocks | Gelu | 141.6 | 3.6 |
| velocity | DynamicQuantizeMatMul | 1540.0 | 36.9 |
| velocity | Attention | 805.7 | 19.3 |
| velocity | Slice | 555.2 | 13.3 |
| velocity | Mul | 267.6 | 6.4 |
| velocity | Transpose | 172.6 | 4.1 |
| velocity | Gelu | 161.8 | 3.9 |

**native8_nospin_8**

| 計算graph | 演算 | 累積 ms | graph内比率 % |
|---|---|---:|---:|
| blocks | DynamicQuantizeMatMul | 1280.2 | 47.9 |
| blocks | Attention | 353.2 | 13.2 |
| blocks | Mul | 141.5 | 5.3 |
| blocks | Gelu | 134.2 | 5.0 |
| blocks | Add | 122.9 | 4.6 |
| blocks | LayerNormalization | 120.4 | 4.5 |
| blocks | RotaryEmbedding | 75.6 | 2.8 |
| velocity | DynamicQuantizeMatMul | 1030.9 | 44.5 |
| velocity | Attention | 587.0 | 25.3 |
| velocity | Transpose | 140.3 | 6.1 |
| velocity | Gelu | 103.2 | 4.5 |
| velocity | Add | 92.7 | 4.0 |
| velocity | Mul | 90.3 | 3.9 |
| velocity | RotaryEmbedding | 43.2 | 1.9 |

**native7_nospin_8**

| 計算graph | 演算 | 累積 ms | graph内比率 % |
|---|---|---:|---:|
| blocks | MatMulIntegerToFloat | 1235.3 | 34.2 |
| blocks | Attention | 381.4 | 10.6 |
| blocks | ReduceMax | 184.4 | 5.1 |
| blocks | Mul | 175.5 | 4.9 |
| blocks | QuantizeLinear | 174.6 | 4.8 |
| blocks | Gelu | 137.3 | 3.8 |
| blocks | RotaryEmbedding | 84.4 | 2.3 |
| velocity | MatMulIntegerToFloat | 1061.6 | 31.0 |
| velocity | Attention | 732.0 | 21.4 |
| velocity | ReduceMax | 328.6 | 9.6 |
| velocity | ReduceMin | 174.5 | 5.1 |
| velocity | Transpose | 171.0 | 5.0 |
| velocity | QuantizeLinear | 135.9 | 4.0 |
| velocity | RotaryEmbedding | 49.2 | 1.4 |

**観測結果と対策**: 旧S2MではWhereが1,732.5 ms、IsNaNが291.1 msを占め、Attention分解の補助演算が無視できなかった。融合後もT2S/S2MのMatMulはそれぞれ約1,920/2,413 msで最大。node単位ではMLPの2つのLinearが上位を占める。加えてSliceが約425/596 ms残り、RoPEや固定幅分割が多数の小演算へ分解されている。これを専用演算へまとめる。

元データ: `results/screen/*_profile.json`。CPUのthread条件が異なるprofileのため、演算時間の差すべてをAttention融合だけの効果と解釈しない。GPUのORT node eventはGPU純kernel時間とは区別し、CPUの表をGPUの内訳として流用しない。

### GPUのCUDA eventとCPU起動APIの内訳

旧ORT版の中文では、CUDA event累積が約219 msだったのに対し、CPU側のCUDA起動APIには約633 msかかっていた。この結果から、GPU演算だけを速くしても起動負荷が残ると判断し、専用演算への置換で小kernelをまとめ、CUDA Graphで起動処理を減らした。以下に、その判断に用いた測定値と改善後の内訳を示す。

同じ中文1回をtorch profilerで採取。profileによる遅延増加があるため、正式速度ではなく負荷の切り分けに使う。CUDA eventはkernelとcopy等のdevice eventの累積で、重なりを除いたGPU稼働率ではない。

| 診断対象 | CUDA event累積 ms | 起動APIのCPU累積 ms | CUDA event数 |
|---|---:|---:|---:|
| legacy | 218.8 | 633.0 | 100249 |
| opt_nospin_8_front1_cg | 173.3 | 77.1 | 49799 |

**legacy: CUDA eventの上位内訳**

| kernel・処理 | 累積 ms |
|---|---:|
| 行列積（GEMM） | 102.7 |
| 要素演算 | 23.9 |
| その他の上位event | 20.2 |
| 正規化 | 6.4 |

分類は元JSONの時間上位15 kernel名をまとめたもの。上位外のeventは含まない。


**opt_nospin_8_front1_cg: CUDA eventの上位内訳**

| kernel・処理 | 累積 ms |
|---|---:|
| 行列積（GEMM） | 93.2 |
| その他の上位event | 18.7 |
| Attention | 12.3 |
| 正規化 | 5.8 |
| RoPE | 3.9 |
| 要素演算 | 2.8 |

分類は元JSONの時間上位15 kernel名をまとめたもの。上位外のeventは含まない。

**対策の効果**: CUDA event累積の減少より、CPUの起動API負荷の減少が大きい。従来のT2Sは多数の小kernelの起動が支配的だった。RoPE・固定分割の直接演算化でevent数を減らし、CUDA Graphで反復ごとのCPU dispatchを削減した。CUDA Graph自体が演算回数を削るわけではない。step数・モデル重み・FP32精度を変えていない。
<!-- ANALYSIS_END -->

## A: CPU session/thread管理の切り分け

exp3では3つのORT sessionにそれぞれ8 threadsを設定し、PyTorchのthread数も8だった。
thread poolは演算を分担するworker群、spinningは待機workerがCPUを使って次の仕事を待つ動作を指す。
呼び出し自体は逐次でも、別sessionやPyTorchの待機threadがCPUを消費して競合する可能性がある。
以下で時間とCPU消費を測り、主因かどうかを切り分けた。

| ファイル | 役割 | 実行・出力 |
|---|---|---|
| [runtime.py](runtime.py) | exp3のgraph・接続を共用し、session設定とPyTorch側のthread数を変更 | 内部モジュール。既存条件の意味を変えない追加引数で共通処理を利用 |
| [screen.py](screen.py) | 区間時間・CPU消費・ORT演算別profile・GPU kernel/起動APIを測定 | `--profile` / `--kernel-profile`。`results/screen/` |
| [analyze.py](analyze.py) | 計測値をブロック別・演算別・GPU起動負荷の表へ集約 | `python analyze.py`。本READMEの分析節と `results/block_analysis.json` |
| [../exp0003/onnx_models.py](../exp0003/onnx_models.py) | 既存graphとI/O Bindingの共通実装 | exp3の既定呼び出しは変更せず、exp4からsession生成方法を指定 |

| 比較する設定 | 切り分けるもの |
|---|---|
| PyTorch FP32 / 動的INT8 | 現行CPU候補の再測定 |
| exp3のORT設定 | 遅延の再現 |
| sessionごとのpool、spinning無効 | 待機threadによる競合 |
| 共有pool、thread数2/4/8 | pool重複と並列度 |
| ORTのpoolとPyTorch側thread数の組合せ | 前後処理のpoolとの競合 |

共有poolのPython APIはthread数の指定で、spinningはORT環境の既定設定に従う。session側へのspinning要求値と区別して記録する。
CPUのprocess CPU時間÷壁時間は旧設定10.66 cores相当からspinning無効4.94へ減ったが、中文遅延は6.480→6.180秒に留まった。待機競合は存在するが、遅さの全ては説明しない。8 threads・spinning無効を共通にしたAttention融合は6.180→4.848秒で、演算表現の変更にも独立した効果があった。これらは探索値で、正式表と混同しない。
探索値は候補の絞り込みに使い、正式な採否は独立した反復計測とCER確認で決める。

<!-- SCREEN_RESULTS_START -->
| 条件 | CPU 中文・秒 | GPU 中文・秒 |
|---|---:|---:|
| torch | 4.574 | 1.394 |
| torch_int8 | 3.662 | — |
| legacy | 6.480 | 1.266 |
| nospin_8 | 6.180 | — |
| shared_8 | 6.014 | — |
| attn_shared_16_front1 | 4.489 | — |
| attn8_nospin_8 | 3.291 | — |
| torch_bf16 | — | 1.434 |
| attn16_nospin_8_front1 | — | 1.017 |

代表中文3回の探索値。実行時刻が異なるため、正式比較では同じ候補を独立に再測定する。
待機threadのspinning無効化だけでは改善が小さく、Attention融合とINT8の併用でCPUの改善候補を得た。GPUはAttention融合＋内部FP16が候補。これは探索時点の候補であり、採否はCの正式結果と末尾の結論による。
<!-- SCREEN_RESULTS_END -->

設定名の `attn` はAttention融合、`attn8`/`attn16` はそのINT8/FP16版。`nospin_N` はsessionごとにN threadsで待機spinning無効、`shared_N` はN threadsの共有pool、`front1` はPyTorch側を1 threadにする条件。`legacy` はexp3の設定を保持する。

## B: ONNX graphと実行経路の改善

前節のprofileでは、CPUの行列積と分解された小演算、GPUのkernel起動処理に時間がかかっていた。この結果を受け、生成step数や学習済み重みを変えず、演算のまとめ方と実行方法を順に改善した。以下に、各対策を選んだ理由、実装内容、効果を示す。

### B.1: 分解されたAttentionを標準演算にまとめる

旧S2Mの演算profileでは、Whereが約1,733 ms、IsNaNが約291 msを占めていた（warmup＋中文1回の累積）。これらはSDPAをMatMul・mask・Softmax等へ分解したgraphに含まれる補助演算である。この費用を減らすため、SDPAを標準ONNX Attentionとしてexportする方法を選んだ。

opset番号を変更するだけではこの置換は行われないため、export時のsymbolicを明示的に差し替えた。標準Attentionを使うためにopset 23へ上げ、ORT CPUのmask形状制約にはquery長を明示して対応した。可変長・padding付きの数値検査と、実際の実行providerの確認を行っている。

この変更だけの初回計測はCPU中文4.487秒、GPU中文1.181秒だった。CPUは既存の動的INT8に届かず、GPUにも小kernelの起動処理が残っていたため、演算の分解と起動負荷をさらに調べた。

| ファイル | 役割 | 実行・出力 |
|---|---|---|
| [export_attention.py](export_attention.py) | SDPAのsymbolicを差し替え、標準Attentionをexport | `python export_attention.py` → `runs/attention_fp32/` |
| [probe_attention.py](probe_attention.py) | 可変長・padding付き誤差と実行providerを検査 | `--device cpu/cuda --config ...` → `results/probes/` |

### B.2: RoPEと固定幅分割の小演算をまとめる

Attention融合後もSliceがT2Sで約425 ms、S2Mで約596 ms残っていた。graphを確認すると、RoPEの回転処理と時刻条件の6分割が多数のSlice・要素演算・shape計算へ分かれていた。そこで、同じ数式を少ない演算で実行するため、RoPEを専用RotaryEmbeddingへ、6分割を固定幅Splitへ置き換えた。

RoPEは隣接する次元を回転するため、`com.microsoft.RotaryEmbedding(interleaved=1)`を使う。sin/cosは入力値によらず位置から決まるのでFP32で事前計算し、繰り返しの再計算を省いた。分割幅も学習済みLinearの出力幅から決まるため、実行時にchunk幅を計算する必要がない。

置換後、T2S block群は2,434→789 nodes、S2M velocityは2,453→808 nodesとなり、各graphのSliceは80→0個になった。Attentionは各8個のままで、この段階の変更はAttentionのopset変更とは別である。専用演算化だけの探索値はGPU中文0.591秒・CPU中文3.944秒だった。cache上限32,768位置を超える入力は明示的に拒否する。

| ファイル | 役割 | 実行・出力 |
|---|---|---|
| [graph_rewrite.py](graph_rewrite.py) | 学習済みblockの数式を保ち、専用RoPE・固定幅Splitを定義 | export時に `install()` を使用 |
| [export_attention.py](export_attention.py) | 上記のblockをONNXへexport | `python export_attention.py --optimized` → `runs/optimized_fp32/` |

### B.3: GPUの反復起動をCUDA Graphにまとめる

GPU profileでは、CUDA event累積が約219 msだったのに対し、CPUのCUDA起動APIに約633 msかかっていた。演算だけを速くしてもこの起動負荷が残るため、反復中のGPU処理をCUDA Graphとしてcaptureし、まとめて再生する方法を選んだ。

CUDA Graphには固定shape・固定アドレスが必要なので、入力shapeをONNXへ設定してshape計算を定数化し、固定I/O bufferを用意した。各回の入力をbufferへcopyし、再生後の出力はcloneして、次回の再生で過去の出力を上書きしないようにしている。CPU fallbackは禁止し、PyTorchとの同期は維持する。以前に非有限値が出た「同期を外す方法」は使わない。

専用RoPEとの組合せでは、CUDA event数が約100,249→49,799、起動APIのCPU時間が約633→77 msへ減った。event数の減少は演算をまとめた効果であり、CUDA Graph自体が計算回数を削るわけではない。最終的なTTS全体の速度とCERはC.1でPyTorch最速条件と比較する。

| ファイル | 役割 | 実行・出力 |
|---|---|---|
| [capture.py](capture.py) | shape特化・固定I/O・CUDA Graph再生と出力の保護 | runtimeから使用。`runs/static/` にshape別graphを保存 |
| [screen.py](screen.py) | CUDA eventとCPU起動APIの内訳を測定 | `--kernel-profile --repeats 1` → `results/screen/*_kernels.json` |

### B.4: 異なる文長でGPU実行を再利用する

完全一致shapeの方式では、文長が変わるたびに新しいsessionの構築が必要になる。この初回費用を繰り返さないよう、文長を128位置単位にそろえ、同じbucketに入る文で実行を再利用する方式も作った。

追加したpaddingはAttentionから除外し、出力は元の長さに戻す。sessionを無制限に保持するとメモリが増えるため、各計算graphにつき最大2個のLRU cacheとした。`quantization.capture` にsession作成回数・cache hit・再生回数を記録する。

高速化率は、対応するshapeのsessionとCUDA Graphを準備した後の定常推論である。未登録bucketでは特化・session構築・captureが必要で、今回の初回warmupは短/中/長文で1.522/0.659/1.333秒だった。新しいbucketへの移行が常に定常速度で完了するとは扱わない。OS cacheを落としたcold起動やブラウザでの起動は未測定である。

| ファイル | 役割 | 実行・出力 |
|---|---|---|
| [capture.py](capture.py) | padding・有効位置のmask・元の長さへの復帰・上限付きcache | `opt_nospin_8_front1_cg128` 条件で使用 |

### B.5: CPUの行列積と量子化条件を合わせる

CPUではMLPの行列積が最大の負荷として残ったため、専用RoPE版へINT8を併用した。ただし先行ORT INT8条件はPyTorchよりCERが悪化したので、速度だけで採用せず、両者の量子化設定の違いを調べた。

PyTorch基準はper-tensor、先行ORT版はper-channelのweight量子化だったため、まず粒度をそろえた。さらに、部分ONNX化で残ったT2Sの時間埋め込み等はFP32のままだったため、最速PyTorch条件と同じ動的INT8へ変更した。整数の飽和対策としてweightの使用範囲を狭めるreduce_rangeも比較したが、この比較だけで飽和が原因だったとは断定しない。

per-tensorへそろえた後もweightを比較すると、T2S block群のscaleは40/40個が異なり、最大差は約0.392%だった。量子化後の整数コードも約8.61%が異なっていた。これは実数へ戻した重みの誤差率ではないが、量子化方式の名称だけでは丸め結果が一致しないことを示す。そのため、weightのscale算出もPyTorchのobserverへ合わせた。監査結果は [quantization_audit.json](results/quantization_audit.json) に保存している。

入力にも違いがあり、今回のPyTorch動的INT8実装は値域0〜127、ONNXのDynamicQuantizeLinearは0〜255を使う。そこで、weightをPyTorchのobserverで量子化し、入力の値域・scale・zero point計算を明示したONNX graphも作った。これはweightを7 bitにする実験ではなく、INT8形式の入力に使う値域を合わせる検証である。単体検査でも完全な数値一致にはならなかったため、155文CERで最終判断する。

| ファイル | 役割 | 実行・出力 |
|---|---|---|
| [graph_rewrite.py](graph_rewrite.py) | per-channel/per-tensorとweightのreduce_rangeを切り分けるgraphを生成 | `python graph_rewrite.py --quantize` → `runs/{opt8,pt8,rr8}_int8/` |
| [native_range.py](native_range.py) | PyTorchのweight量子化と入力値域0〜127をONNXへ適用 | `python native_range.py` → `runs/native7_int8/` |
| [runtime.py](runtime.py) | ONNX外に残った部品の量子化条件を合わせる | `native8_nospin_8` / `native7_nospin_8` |
| [probe_attention.py](probe_attention.py) | FP32参照に加え、PyTorch INT8参照との差を測る | `--reference-int8` → `results/probes/*_reference_int8.json` |

### B.6: 2D Gemm化が効くかを確認する

MLPのMatMulが大きな負荷だったため、rank-3のMatMul＋biasをflatten→2D Gemm→reshapeへ変え、CPU側でbias融合が効くかも調べた。しかし探索では中文4.04〜4.11秒で改善しなかったため、この変更は採用しない。

| ファイル | 役割 | 実行・出力 |
|---|---|---|
| [graph_rewrite.py](graph_rewrite.py) | Linearを2D Gemmへ書き換える | `python graph_rewrite.py` → `runs/gemm_fp32/` と `results/gemm_export.json` |

## C: 採用候補の全体評価

| ファイル | 役割 | 実行・出力 |
|---|---|---|
| [run.py](run.py) | 1候補を短/中/長文・CER・独立メモリで正式評価 | `--device cpu --config ... --phase bench`等。既定10回。`results/measured/` |
| [summarize.py](summarize.py) | 既存CERの出典を保持して比較表・CSV/JSON・試聴ページを更新 | `python summarize.py --require-complete` → `results/summary.{csv,json}`。未完了ならエラー |

集約データ: [CSV](results/summary.csv) / [JSON](results/summary.json)。区間別時間、各反復値、RTF、準備時間・初回warmupは [生の計測結果](results/measured/) に保持する。

速度の比較基準は各device・各文長の**PyTorch版最速条件**。CPUは動的INT8、GPUはFP32で、表とCSVに基準条件と短縮率を明記する。旧ORTの値は低速化の原因を説明する対照であり、目標達成の基準にはしない。

MP3は速度計測のseed 3を保存し、exp3と同じ文章・seedで比較できるようにする。
peak RSSはモデル準備から長文のwarmup・推論までのプロセス最大値。GPUのdevice VRAM増分は初期CUDA contextからの増加分を10 msごとに観測し、allocatorのcacheも含む。GPUメモリはPyTorch管理外のORT割当も含め、exp3 Gと同じdevice使用量増分の定義で比較する。重み・graphサイズはT2S/S2Mの残存PyTorch stateとONNX graph（RoPEの定数cache等も含む）の合計で、codec・暫定ボコーダ・runtime本体を含まない。実行時メモリとは別の指標である。


<!-- FORMAL_RESULTS_START -->
### C.1: PyTorch最速条件との最終比較

| Device | 条件 | 短文 秒 | 中文 秒 | 長文 秒 | テストCER % | 実用文CER % | PyTorch最速比・中文短縮 % |
|---|---|---|---|---|---|---|---|
| cpu | PyTorch 動的INT8 | 0.725 | 3.293 | 9.292 | 6.507 | 6.462 | 0.000 |
| cpu | ORT INT8・RoPE融合＋量子化条件調整 | 0.675 | 2.404 | 7.806 | 6.853 | 6.196 | 27.000 |
| cpu | ORT INT8・入力量子化範囲をPyTorchへ合わせる | 0.746 | 2.803 | 8.999 | 7.652 | 4.930 | 14.861 |
| cuda | PyTorch FP32 | 1.062 | 1.120 | 1.225 | 6.463 | 6.729 | 0.000 |
| cuda | ORT FP32・RoPE融合＋CUDA Graph | 0.252 | 0.363 | 0.571 | 6.420 | 6.396 | 67.605 |
| cuda | ORT FP32・CUDA Graph＋文長bucket | 0.305 | 0.384 | 0.605 | 6.334 | 6.929 | 65.761 |

### C.2: 改善途中の条件（参考）

以下は初回の測定値で、最終比較とは測定時期が異なる。PyTorch最速比の短縮率は最終比較の条件だけに示す。

| Device | 条件 | 短文 秒 | 中文 秒 | 長文 秒 | テストCER % | 実用文CER % |
|---|---|---|---|---|---|---|
| cpu | PyTorch FP32・8 threads | 1.018 | 4.410 | 12.129 | 6.896 | 6.396 |
| cpu | PyTorch FP32・16 threads | 0.984 | 4.261 | 11.141 | 6.874 | 6.396 |
| cpu | ORT FP32・exp3設定 | 1.309 | 6.648 | 25.587 | 6.896 | 6.396 |
| cpu | ORT Attention FP32 | 0.861 | 4.487 | 13.797 | 6.896 | 6.396 |
| cpu | ORT Attention INT8 | 0.831 | 3.205 | 9.410 | 8.020 | 6.529 |
| cuda | PyTorch BF16 | 1.469 | 1.554 | 1.447 | 6.507 | 6.262 |
| cuda | ORT FP32・exp3設定 | 1.067 | 1.164 | 1.369 | 6.485 | 6.729 |
| cuda | ORT Attention FP32 | 1.090 | 1.181 | 1.201 | 6.463 | 6.729 |
| cuda | ORT Attention FP16 | 1.088 | 1.117 | 1.186 | 7.263 | 6.929 |

### C.3: 重みサイズと実行時メモリ

| Device | 条件 | 重み・graph MiB | peak RSS MiB | device VRAM増分 MiB |
|---|---|---|---|---|
| cpu | PyTorch FP32・8 threads | 361.979 | 1639.504 | — |
| cpu | PyTorch FP32・16 threads | 361.979 | 1642.875 | — |
| cpu | PyTorch 動的INT8 | 104.759 | 1739.852 | — |
| cpu | ORT FP32・exp3設定 | 362.655 | 2937.633 | — |
| cpu | ORT Attention FP32 | 362.662 | 2229.156 | — |
| cpu | ORT Attention INT8 | 109.237 | 1948.855 | — |
| cpu | ORT INT8・RoPE融合＋量子化条件調整 | 121.263 | 2122.359 | — |
| cpu | ORT INT8・入力量子化範囲をPyTorchへ合わせる | 121.956 | 2146.562 | — |
| cuda | PyTorch FP32 | 361.983 | 2601.746 | 806.000 |
| cuda | PyTorch BF16 | 181.039 | 2747.902 | 670.000 |
| cuda | ORT FP32・exp3設定 | 362.655 | 3079.359 | 1766.000 |
| cuda | ORT Attention FP32 | 362.662 | 3049.875 | 1318.000 |
| cuda | ORT Attention FP16 | 190.923 | 3329.305 | 994.000 |
| cuda | ORT FP32・RoPE融合＋CUDA Graph | 378.332 | 3110.789 | 1616.000 |
| cuda | ORT FP32・CUDA Graph＋文長bucket | 378.332 | 3143.797 | 1618.000 |
<!-- FORMAL_RESULTS_END -->

既存条件はcheckpoint・生成seed 0・batch条件・実装が同じexp3のCERを参照し、速度とメモリは再測定する。
`results/summary.json` の `quality_source` と `quality_reused` が各行の出典を示す。新しいAttention条件と16 threadsのPyTorch条件は155文を再評価した。
変更のない既存条件のCERをexp4で再測定した値として扱わない。

A/Bで有効な変更を選び、探索での単独変更と正式比較での組合せを区別してCPU/GPUの短・中・長文を測定した。
PyTorchと旧ORT設定に対する区間別/全体の短縮率、初回warmup、CER、メモリをまとめる。初回warmupはモデル準備後の最初の1回であり、OSのファイルcacheを落としたcold起動ではない。準備時間はJSONの `load_prepare_s` に記録する。
同一精度での改善と、低精度化を伴う改善は別条件とする。TensorRTは後続exp0005で扱う。

参照: [ORT thread管理](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)、[ONNX Attention](https://onnx.ai/onnx/operators/onnx__Attention.html)。

## 用途別の結論と適用範囲

<!-- CONCLUSIONS_START -->
最終比較では、GPUのFP32実装がPyTorch最速条件を短文・中文・長文のすべてで上回り、今回の155文CERでも悪化しなかった。一方、CPUの速度優先条件は全体を短縮したが、テスト文と実用文でCERの変化方向が分かれた。このため、GPUは高速化したFP32実装を候補にし、CPUは許容する品質差に応じて選ぶ。

| 優先するもの | 採用候補と結論 |
|---|---|
| GPUの定常速度と、今回のCER維持 | **専用RoPE＋CUDA Graph（FP32・完全一致shape）**。中文1.120→0.363秒、約68%短縮。テスト/実用文CERは6.420/6.396%で、PyTorchの6.463/6.729%に対し悪化なし。未登録shapeには構築費用があるため、準備済みshapeを再利用する用途の候補 |
| GPUで異なる文長をまとめて扱う | **128位置bucket版**。中文0.384秒で約66%短縮。テストCERは0.130ポイント改善したが、実用文は0.200ポイント悪化した。この差を許容でき、同じbucketの再利用が見込める場合の候補 |
| CPUの速度 | **専用RoPE＋per-tensor INT8＋残存部品INT8**。中文3.293→2.404秒、約27%短縮。短文・長文も短縮した。ただしCERはテスト+0.346ポイント・実用文-0.266ポイントで、品質不変の置き換えではない |
| CPUで既存のCERを維持する | **PyTorch動的INT8を残す**。今回のCPU ONNX候補はどちらもテストCERが悪化したため、劣化を避けたい場合に無条件では置き換えない。高速化の結果と品質維持の結果を混同しない |
| 今回の実用55文のCERを重視する | **入力値域0〜127を合わせたCPU版**は実用文CER4.930%、中文2.803秒で約15%短縮。ただしテストCER7.652%はPyTorchより1.146ポイント悪く、短文も僅かに遅い。全般的な品質改善とは扱わず、既定にはしない |
| 実行時メモリ | **PyTorchの方が小さい**。CPUのpeak RSSはPyTorch約1740 MiBに対し、速度優先ORT約2122 MiB。GPUのVRAM増分はPyTorch約806 MiBに対し、完全一致shape版約1616 MiB。速度向上とメモリ増加を交換する |
| 配布ファイルの最小化 | 引き続き[exp3のORT INT4](../exp0003/README.md)（約68.8 MiB）が小さい。今回のGPU FP32 graphは約378.3 MiBで、RoPEの定数cacheも含む。今回の高速化はファイル圧縮の代替にはならない |

GPUでは、分解された小演算をまとめたうえで起動処理を削減したことが、TTS全体の短縮につながった。CPUでもT2S・S2Mの双方が短縮したが、同じINT8・per-tensorという設定名でも丸め結果は一致しなかった。入力範囲とweightのscale計算を近づけた条件も、単体誤差の改善がテストCERの改善に直結しなかったため、速度と生成後CERを組にして判断する必要がある。

評価した15条件の速度は短・中・長文各10回、メモリは独立プロセスで測定した。変更した9条件は各155文でCERを評価し、変更のない6条件は出典を明記してexp3のCERを参照した。最終比較のPyTorch基準とGPU候補は近い時刻に再測定し、以前の値も保存した。代表MP3は45本・合計約1.81 MiB。全条件で生成の有限値を確認した。

これらは固定したcheckpoint・単一seed・同じ評価文での観測であり、別の話者・モデル・hardwareでの改善率や品質の同等性まで保証するものではない。生成step数は変更していない。
<!-- CONCLUSIONS_END -->

今回の実行対象はPython版ONNX Runtime 1.32のCPU/CUDA。標準AttentionとORT専用RotaryEmbeddingを含むgraphのWebGPU/WASM対応は未検証で、今回の高速化をブラウザへそのまま移せるとは扱わない。旧opset 18のgraphも保持し、exp0008で対応演算・必要な表現・実測性能を比較する。

## 再実行

以下はこのディレクトリから、依存ライブラリとexp3のONNX基準graphを用意した環境で実行する。生成物のgraphとWAVは `runs/`、公開対象は計測JSON/CSVと代表MP3とする。速度・メモリはプロセス間lockで排他実行し、CER生成と競合させない。

```sh
python export_attention.py
python export_attention.py --derived-only
python export_attention.py --optimized
python graph_rewrite.py --quantize
python native_range.py
python probe_attention.py --device cpu --config attn8_nospin_8
python probe_attention.py --device cuda --config attn16_nospin_8_front1

for phase in bench memory; do
  for config in torch torch_16 torch_int8 legacy attn_shared_16_front1 attn8_nospin_8 native8_nospin_8 native7_nospin_8; do
    python run.py --device cpu --config "$config" --phase "$phase"
  done
  for config in torch torch_bf16 legacy attn_nospin_8_front1 attn16_nospin_8_front1 opt_nospin_8_front1_cg opt_nospin_8_front1_cg128; do
    python run.py --device cuda --config "$config" --phase "$phase"
  done
done
for config in torch_16 attn_shared_16_front1 attn8_nospin_8 native8_nospin_8 native7_nospin_8; do
  python run.py --device cpu --config "$config" --phase quality
done
for config in attn_nospin_8_front1 attn16_nospin_8_front1 opt_nospin_8_front1_cg opt_nospin_8_front1_cg128; do
  python run.py --device cuda --config "$config" --phase quality
done
python summarize.py --require-complete
```

比較基準は最終候補と同じ実行期間で再測定した。再測定前の速度・メモリJSONは `results/measured/initial_references/` に保持する。

既存の結果ファイルはskipする。条件を変更して再測定する場合は、該当する結果を退避してから実行する。
