# exp0003: 量子化

このREADMEにはA〜Gの計画と各段階の概要を記載する。
Cの12条件を横断する結果表と、calibration・実装・評価上の制約の詳細は、[Cの詳細レポート](C_REPORT.md)を参照。

本文は、結論、計画と進捗、実行方法、A〜Dの段階別結果、今後の検証の順にまとめる。

## 結論: 結局どの条件を選ぶか(A〜D)

現時点の採用候補は、GPUで速度優先ならFP32、VRAMとのバランスならBF16、CPUで速度優先ならT2S＋S2Mの動的INT8。
ファイルをさらに小さくする必要がある場合に、GPUの単純INT4・group size 128(Tinygemm)を選ぶ。

| 優先するもの | 推奨条件 | 判断理由・引き換えになるもの |
|---|---|---|
| GPUでの速度 | GPU FP32 | Cの中文で1.411秒と最速。量子化を足しても速くならなかった |
| GPUの速度・読みCER・VRAMのバランス | GPU BF16 | FP32比でpeak VRAMが565→347 MiB。中文は1.411→1.492秒で、テストCERは6.46→6.51% |
| CPUでの速度 | T2S＋S2Mの動的INT8(B) | 10回計測の中文medianが4.474→3.200秒。2 seed平均の読みCERにも悪化は見られなかった。RAM削減は得られない |
| 配布する重みファイルの小ささ | GPUの単純INT4 g128(Tinygemm) | T2S＋S2Mで約90.4 MiB、BF16の約半分。calibration不要。BF16比で中文は1.492→1.856秒、テストCERは6.51→7.05%となるため、その劣化を許容できる用途向け |
| 元モデルの数値的な変更を抑える | FP32 | 低精度化による近似を加えない。読みCERが全評価集合で最良という意味ではない |

**読みの正確さだけでの総合1位は、まだ決めない。** Cのseed 0ではテスト100文はGPU FP32(6.46%)、
実用文55文はMSLK RTN INT4→tile(5.66%)が最小で、評価集合によって順位が変わった。
実用文に特化する場合は後者が候補だが、テストCER 7.16%への悪化もあるため、対象文章で再評価してから選ぶ。
CERは発音の誤りの指標であり、自然さ・抑揚・ノイズ等の聴感品質は未評価。

AWQ/GPTQを一律の既定には採用しない。今回の少量calibrationでは安定した優位が得られず、追加の変換・実装コストもある。
CPUのRAM削減を目的とする場合も、現状の測定から推奨できる量子化条件はない。
これらは現在のモデル・CPU/GPUでの暫定推奨であり、E〜GやNNボコーダ導入後に更新する。
ファイルサイズはT2S＋S2Mのみで、codec・runtimeを含む配布物全体のサイズではない。

**Dを加えても、速度重視の推奨はGPU FP32・CPU動的INT8のまま。** GPU静的INT8/SmoothQuantは約106 MiBに縮小できるが、
今回のコンパイル実装で中文2.26〜2.52秒と、eager FP32の約1.36秒より遅い。
SmoothQuant α=0.5はテストCER 6.44%・実用文6.40%で、Dの同条件FP32(6.46%・6.73%)に対して読みの悪化は見られなかった。
ただし差は小さく、単一seed・calibration 8文の結果なので、品質改善の確証とはしない。
CPU静的INT8は中文で約8.6倍遅く、採用しない。CPU SmoothQuantは同じ演算経路のため追加測定しない。

## 計画と進捗

A〜Eは学習不要のPTQ(post-training quantization)を扱う。
FのQAT(quantization-aware training)は量子化誤差を模擬して追加学習する方式で、現時点では未実施。

各候補の採否は速度と読みCERを必ず組み合わせて判断する。速度測定だけ済んだ条件は「品質未判定」と表示する。
CERは基準と候補の絶対値、およびポイント差を記録する。部品誤差・有限値検査・MP3はCERの代わりにはしない。
量子化範囲・backend・精度・QAT学習条件を変えた場合は、変更後の条件で速度とCERの両方を再評価する。

| 段階 | 条件 | 目的 |
|---|---|---|
| A | CPU FP32、S2Mのみ動的INT8 | 変換・推論・計測・サンプル出力の動作確認 |
| B | T2Sのみ、S2Mのみ、両方のCPU INT8 | 部品ごとの寄与と劣化箇所を分離 |
| C | CPU/GPU対応backendの重みのみINT8・INT4、AWQ系・GPTQ系 | 単純な重み量子化と、calibrationで誤差を抑える方式を比較 |
| D | 活性値も量子化する静的PTQ、SmoothQuant系 | calibrationデータと外れ値処理の影響を比較 |
| E | 敏感な層をFP32/FP16に残す混合精度(C/Dの各方式と併用) | 小さいサイズと品質維持の両立 |
| F | QATによる追加学習(AWQ/GPTQ/SmoothQuantの学習不要PTQとは別) | PTQでの品質低下を回復できるか |
| G | ONNX等の配布runtimeへ移行 | Python上の結果がブラウザ・エッジでも成立するか |

各段階の具体的なbackendと対応dtypeは、実環境で実行できるかを確認して記録する。
全手法を無条件に総当たりせず、非対応・実利のない条件も理由を残して絞り込む。
GPUで非対応の条件はCPUの数値で代用しない。ボコーダは学習済み版が用意できてから量子化対象へ追加する。

### 実施状況

| 条件 | 速度 | CER | 採用判断 |
|---|---|---|---|
| CPU FP32 / S2M動的INT8 | 代表3文・3回測定済み | 100＋55文のseed 0対比較済み | CER劣化なし。複数seed・聴感は未確認 |
| CPU T2S動的INT8 | 代表3文・3回測定済み | 100＋55文のseed 0対比較済み | 小差あり。複数seed・聴感は未確認 |
| CPU 両方INT8候補 | 代表3文・10回、独立process RAM測定済み | 100＋55文のseed 0・1対比較済み | 今回CER劣化なし。RAMは増加。聴感は未確認 |
| C: 重みINT8/INT4・group size・AWQ・GPTQ | 基準を含む12条件で速度・サイズ・メモリ測定済み | 100＋55文のseed 0評価済み | 選定条件の比較完了。[詳細レポート](C_REPORT.md) |
| D: 静的PTQ・SmoothQuant | GPU4条件・CPU基準の速度・サイズ・メモリ測定済み | 同5条件の100＋55文・seed 0評価済み | 比較完了。CPU静的INT8は速度で除外・CER未評価、CPU SmoothQuantは未実施 |
| E〜G: 混合精度・QAT・配布runtime | 未実施 | 未実施 | 未判定 |

### 比較条件の固定

- FP32 eagerを基準とし、コンパイル等を組み合わせる前に量子化単独の効果を測る
- 生成step・温度・CFG・seed列・入力・batch・CPU thread数をそろえる
- calibrationとQATに評価文を使わず、学習splitから取り出す。選定規則・件数・seedを記録する
- calibrationは短文だけに偏らないよう長さを分け、反復型部品では異なる時刻の活性値も対象にする
- calibration用の重い特徴抽出は必要性を確認してから行い、推論にHuBERTを追加しない
- 量子化範囲、除外層、per-tensor/per-channel、scale、zero point、group size、重み/活性値/cacheのdtypeを明記する
- T2Sでは重みを直接参照する実装があるため、Linearの置換だけで動くか確認する。除外した層は記録する

### 精度と品質

1. 固定入力の部品単独出力: 最大絶対誤差、RMSE、相対L2、NaN/Inf
2. 上流を固定: 同一のテキストID、同一音声トークン、同一ノイズから対象部品を比較する
3. 全体生成: テスト100文と実用文55文の読みCER、失敗数、生成音声長、極端な長短・繰り返しを確認する
4. 良い候補は複数seedで再評価し、小差を1回の偶然で採用しない
5. 同じ文章・seedのMP3を出し、数値に出ないノイズ・発音・抑揚の差を比較できるようにする

Griffin-Limを使う期間はその音質上限を明示し、UTMOS等の改善・劣化だけで量子化の採否を決めない。

### 性能とサイズ

- 区間別時間、全体時間、RTF、median/p90、cold load、変換時間
- peak RSS/VRAM、重みのserializeサイズ、runtime込みの配布サイズを分ける
- CPU thread数とbackend、GPU kernel対応、実際に量子化された層数を記録する
- weightだけ圧縮しても実際のkernelが速くならない場合は、圧縮の成果と速度の成果を分ける
- warm-upと計測回数を条件間で固定する。正式比較は初期3回から増やし、必要なら別processでも再測定する

### QAT

PTQで劣化する部品を優先する。まず元checkpointから小規模な追加学習を試し、学習データ・step・optimizer・LR・seed・時間を記録する。
同じ予算で量子化なしの追加学習も比較し、追加学習そのものの効果とQATの効果を区別する。
fake quantでの検証に加え、実際の量子化kernelへ変換した後の品質・速度も評価する。
学習checkpointはGit管理外に置き、公開するのはレポート・集計JSON・少数のMP3に限定する。

### MP3の容量管理

初期条件は3文のみ。正式候補には既存実験と同じ代表文を追加し、全155文の音声を公開用へコピーしない。
48kbpsなら1分あたり概ね360KBなので、条件数と総秒数を記録して公開容量を管理する。
一覧JSONの条件IDは量子化方式・対象部品・deviceを区別できるものにする。

## 実行方法と出力

```bash
python run.py --device cpu
python run.py --device cpu --quantize
python run.py --device cuda
```

短・中・長文、8 threads、warm-up 1回、3回計測。同一条件の最終run(seed 3)を48kbps MP3として保存する。
音声エンコードは計測区間外。各条件3文のみとし、波形・重みはGitへ追加しない。
サンプル一覧は`results/samples/index.json`で、既存試聴ページと同じ形式。
MP3は生成波形から直接エンコードする。

初期のAではS2M、B以降ではT2S＋S2Mのstate_dictサイズを記録する。TTS全体やruntime込みの配布サイズとは異なる。
3文の有限値検査・試聴サンプルだけでは品質維持を結論づけない。100文CER等の評価は別途実施する。

## A: CPU S2Mの動的INT8

### 速度とサイズ

CPU: Core Ultra 7 265K、8 threads。GPU: RTX 5090。FP32モデルを基準に、CPUではS2MのLinearだけを動的INT8へ変換。
生成条件はexp0001と同一。warm-up 1回・3回計測のmedianで、正式候補の評価前の動作確認とする。

| 入力 | CPU FP32 全体 | CPU INT8 全体 | CPU FP32 S2M | CPU INT8 S2M | GPU FP32 全体 |
|---|---:|---:|---:|---:|---:|
| 短文 | 1.120秒 | 1.012秒 | 0.444秒 | 0.327秒 | 1.487秒 |
| 中文 | 4.855秒 | 4.129秒 | 2.407秒 | 1.706秒 | 1.706秒 |
| 長文 | 13.070秒 | 11.326秒 | 7.481秒 | 5.795秒 | 1.581秒 |

S2M state_dictは162,901,299→46,491,967 bytes(約71.5%減)。CPU全体は約10〜15%短縮した。
これはA段階のCPU用dynamic INT8の結果。GPUの重みINT8はCで別途評価した。
動的INT8はLinearの重みを量子化し、活性値の量子化を実行時に行う。静的calibrationやQATは今回使っていない。
全3条件で生成Mel・波形は有限値。CPUのCER・中間出力の誤差は次節で評価した。聴感は未確認。

MP3は3条件×3文で計9本、合計376,272 bytes(約367 KiB)。同じseed 3の生成を48kbpsで保存した。
`site/manifest.json`に「TTS 推論最適化」を追加し、既存GitHub Pagesの試聴UIで比較できる一覧を登録した。
聴感による品質評価は未実施。

### CERと部品出力の誤差

`evaluate.py`ではCPUのT2Sを1回だけ実行して両条件で同一の音声トークンを使い、S2Mの初期ノイズも固定する。
S2M batch 4、seed 0で100テスト文＋55実用文を比較し、同じGPU ASRで読みCERを計算する。
これはS2M量子化の効果を分離する品質比較で、exp0001のbatch 16のCERと厳密には同条件でない。
速度はASRを含まない別計測。評価WAVはGit管理外の`runs/quality/`、集計は`results/quality.json`へ保存する。

x86 backend、45個のLinear、qint8重み・per-tensor affine量子化。
変換層一覧は`results/quality.json`に保存した。T2S・codec・Griffin-LimはFP32のまま。
両条件で同じ音声トークンとS2Mノイズを使い、CPU S2M batch 4、seed 0で対比較した。

| 条件 | テスト100文 読みCER | FP32との差 | 実用文55文 読みCER | FP32との差 |
|---|---:|---:|---:|---:|
| CPU FP32 | 6.90% | 基準 | 6.40% | 基準 |
| CPU S2M動的INT8 | 6.77% | -0.13pt | 6.40% | 0.00pt |

このseedでは読みCERの劣化はなかった。小差であり、改善の有意性や音質維持を結論づけない。
FP32の数値がexp0001と違うのは、CPU実行・batch・乱数の使われ方が異なるため。量子化の差はこの対比較内で判断する。

合成テンソルでのS2M単発出力誤差は最大絶対値0.252、RMSE 0.0604、相対L2 4.26%。
これは実データ分布を保証しない補助検査。実際の同一トークン・ノイズから生成したlog-Melの相対L2誤差の中央値は、
テスト文0.932%、実用文0.841%だった。全生成のMel・波形は有限値。
Melの数値誤差をそのまま知覚音質の劣化率と解釈しない。

```bash
python evaluate.py
```

## B: 量子化対象部品の比較

同じCPU動的INT8で、T2Sのみ・S2Mのみ・両方を比較する。
T2Sではdevice取得用に`head.weight`を参照するため出力`head`をFP32に残す。Embedding、Conv、LayerNormも対象外。
T2Sの68個のLinear、S2Mの45個のLinearを変換する。除外層と変換名は各結果JSONへ保存する。
ここでの「T2S INT8」は全演算INT8という意味ではない。

| CPU条件 | 短文 全体 | 中文 全体 | 長文 全体 | T2S state | S2M state | テスト100文CER | 実用文55文CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 1.120秒 | 4.855秒 | 13.070秒 | 206.6 MiB | 155.4 MiB | 6.90% | 6.40% |
| S2M INT8 | 1.012秒 | 4.129秒 | 11.326秒 | 206.6 MiB | 44.3 MiB | 6.77% | 6.40% |
| T2S INT8 | 0.931秒 | 4.277秒 | 12.284秒 | 60.4 MiB | 155.4 MiB | 6.79% | 6.20% |
| 両方INT8 | 0.940秒 | 3.747秒 | 10.548秒 | 60.4 MiB | 44.3 MiB | 6.51% | 6.46% |

短文で両方INT8が最速ではない差は小さく、3回計測のばらつきとしても考えられる。両方INT8候補の10回計測結果は次節に示す。
T2S stateは216,661,159→63,355,479 bytes。表のサイズは各部品のstate_dictで、codec・runtimeを含む全体容量ではない。
生成サンプルを各3文追加した。公開用の全評価音声コピーや重みの保存は行わない。

BのCERは量子化T2Sで100＋55文の共通トークンを生成し、それをFP32/INT8のS2Mへ渡す。
Aと同じseed・batch設定で、T2Sの量子化によるトークン・長さの変化を含めて比較する。
BのMel誤差は「量子化T2Sを共有した上でのS2M差」であり、FP32 T2Sからの全体誤差とは異なる。

両方INT8はFP32比で中文約23%、長文約19%短縮。テストCERは-0.39pt、実用文は+0.07ptだった。
単一seedの小差なので精度改善や同等性の統計的な保証はしない。全条件のMel・波形は有限値。
T2SのみINT8は実用文で-0.20ptだが、これも改善と断定しない。
量子化によるサイズ削減は明確だが、この初期比較ではpeak RAMは未測定。次節の独立プロセス測定で確認した。

Bの品質結果は`results/quality_b.json`、速度は`cpu_t2s_int8.json`・`cpu_both_int8.json`。
MP3はGPU FP32を含む5条件×3文で計15本、合計634,608 bytes(約620 KiB)。中間結果は表で蓄積する。

```bash
python run.py --device cpu --target t2s
python run.py --device cpu --target both
python evaluate.py --t2s-int8
```

### 採用候補の追加検証: FP32と両方INT8

速度を各入力10回へ増やし、warm-up 1回、8 threads、同じseed列で独立プロセス測定した。
元の3回計測を上書きせず、結果は`results/validation/`へ保存する。

| 入力 | FP32 median / p90 | 両方INT8 median / p90 | median短縮率 |
|---|---:|---:|---:|
| 短文 | 1.063 / 1.122秒 | 0.784 / 0.797秒 | 約26% |
| 中文 | 4.474 / 4.607秒 | 3.200 / 3.279秒 | 約28% |
| 長文 | 12.241 / 12.496秒 | 9.787 / 10.012秒 | 約20% |

#### RAM

サイズ計測用のserialize bufferを作らない別プロセスで、モデルをロード・必要なら変換後に長文1件を推論した。
RSSはPython・ライブラリ・codec・全モデル・allocator等を含むprocess全体で、純粋な重みサイズではない。

| CPU条件 | load/変換後RSS | 長文推論後RSS | process peak RSS |
|---|---:|---:|---:|
| FP32 | 1.506 GiB | 1.544 GiB | 1.544 GiB |
| 両方INT8 | 1.587 GiB | 1.622 GiB | 1.622 GiB |

この実行方法ではINT8のpeak RSSは約5%増えた。FP32 checkpointからその場で変換する処理や、
allocatorが保持する領域を含むため、重みserializeサイズの減少からprocess RAM削減を結論づけられない。
量子化済みcheckpointの直接ロードやallocator条件を変えた場合は別測定が必要。今回は各条件1プロセスのみ。

#### seed 0・1のCER

| seed | FP32 テストCER | 両方INT8 テストCER | FP32 実用文CER | 両方INT8 実用文CER |
|---|---:|---:|---:|---:|
| 0 | 6.90% | 6.51% | 6.40% | 6.46% |
| 1 | 7.98% | 6.81% | 5.93% | 5.33% |
| 2 seedの平均 | 7.44% | 6.66% | 6.16% | 5.90% |

seed 1でも全生成は有限値で、CER悪化は見られなかった。平均差はテスト-0.78pt、実用文-0.27pt。
seed 0の実用文は微増している。量子化により生成の確率・長さ・トークン列も変わり、
同じseedでも同じ出力にはならないため、2 seedの結果から量子化が精度を改善すると一般化しない。
基準FP32のテストCERにもseed間で約1.08ptの差があり、単一seedの小差を評価する難しさが分かる。

結果は`results/validation/quality_seed1.json`。seed 0はA/Bの`quality.json`・`quality_b.json`を使用した。
公開MP3は15本のまま。試聴ページのCERはseed 0の値を保持し、複数seed平均と混ぜない。

この候補は「CPU速度・配布重みサイズには効果あり、今回のCERは維持、実行時RAM削減は未達」と整理する。
Cでは重みのみINT8/INT4と同じ軸で比較する。

```bash
python run.py --target none --repeats 10 --no-samples --output-dir results/validation
python run.py --target both --repeats 10 --no-samples --output-dir results/validation
python check_memory.py
python check_memory.py --quantize
python evaluate_candidate.py --seed 1
```

## C: 重みのみINT8/INT4と、calibrationを使う方式

最初にtorchaoの単純な重みのみINT8/INT4を測り、次にAWQ系・GPTQ系を比較する。
今回追加したbackendはtorchao 0.18.0。PyTorch 2.14.0+cu130等の既存パッケージは変更していない。

### backend・dtypeの事前確認(C1時点)

| 重み方式 | CPU | GPU | 備考 |
|---|---|---|---|
| Int8WeightOnlyConfig | FP32/BF16の単体Linearで動作 | FP32/BF16の単体Linearで動作 | 本計測ではFP32と比較 |
| Int4WeightOnlyConfig・plain | 依存不足で失敗 | 依存不足で失敗 | C1時点ではmslk不足。C3で追加し、packing変換後の推論を評価 |
| Int4WeightOnlyConfig・tile-packed | 今回の経路では動作せず | BF16で動作 | FP32は未対応。CPUではCUDA/driverを要求する経路に入った |

CPU全体としてINT4非対応という結論ではなく、今回のconfig・経路での結果である。
失敗条件も`results/weight_only/probe_*.json`に保存する。

INT4はgroup size 128、TILE_PACKED_TO_4D、TINYGEMMで、単純なgroup-wise PTQ。
C1ではAWQ/GPTQを使用しない。group境界に合わないS2Mの`in_proj`等を高精度に残す。
T2Sの`head`もBと同様に除外し、量子化・除外層の一覧を結果へ保存する。
INT4の速度・品質は、同じBF16 model/activationの基準と比較してdtype変更の影響を分ける。
Griffin-Limは引き続きFP32で、量子化対象にしない。

計測はeagerのまま。コンパイルとの組合せは別条件とし、量子化単独と混ぜない。
各条件3文のMP3を追加し、同じ100テスト文＋55実用文のCERを評価する。中間結果は表に蓄積する。

参照: [Int8WeightOnlyConfig](https://docs.pytorch.org/ao/stable/api_reference/generated/torchao.quantization.Int8WeightOnlyConfig.html)、
[Int4WeightOnlyConfig](https://docs.pytorch.org/ao/stable/api_reference/generated/torchao.quantization.Int4WeightOnlyConfig.html)。
実際のAPIはインストール済み0.18.0のsignatureと実行結果でも確認した。

### C1: 単純な重み量子化の初期結果

速度は8 threads、batch 1、warm-up 1回・計測3回のmedian。
品質はseed 0、T2S batch 16・S2M batch 4、100テスト文＋55実用文。dtypeをそろえた基準との差を見る。
AWQ/GPTQを使わない重み量子化なので、calibrationデータは不要だった。
量子化対象はINT8でT2S 68層・S2M 45層、INT4ではT2S 67層・S2M 44層。
INT4のS2M `in_proj`とT2S `head`・`duration.out`は高精度のまま。正確な除外一覧は各JSONへ保存した。
重みINT8はper-row。A/Bの動的INT8はper-tensor重みなので、両者の差を活性値量子化の有無だけに帰属させない。

| GPU条件 | 短文 | 中文 | 長文 | T2S＋S2M state | テストCER | 実用文CER |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 1.356秒 | 1.411秒 | 1.484秒 | 362.0 MiB | 6.46% | 6.73% |
| 重みINT8・FP32 activation | 1.730秒 | 1.793秒 | 1.805秒 | 105.5 MiB | 6.94% | 6.20% |
| BF16基準 | 1.527秒 | 1.492秒 | 1.475秒 | 181.0 MiB | 6.51% | 6.26% |
| 重みINT4・BF16 activation | 1.882秒 | 1.856秒 | 1.924秒 | 90.4 MiB | 7.05% | 6.20% |

GPUのINT8・INT4とも今回のeager実行では高速化しなかった。INT4はBF16基準からstateサイズを約50%削減したが、
テストCERが+0.54pt、実用文は-0.07ptだった。INT8のテストCERもFP32比+0.48pt。
単一seedなので有意差は未判定だが、「小さくなれば速くなる」「低bitでも品質は常に維持される」とは扱わない。
この結果は方式共通の結論ではなく、当該モデル・eager backend・GPU・入力での観測である。
Cではeager実行を比較した。静的INT8とコンパイルの組合せはDに示す。

CPUの重みINT8も100＋55文のCER評価を完了した。A/BのFP32と動的INT8を参考として並べる。

| CPU条件 | 短文 | 中文 | 長文 | T2S＋S2M state | テストCER | 実用文CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32(A/Bの3回基準) | 1.120秒 | 4.855秒 | 13.070秒 | 362.0 MiB | 6.90% | 6.40% |
| 動的INT8・両部品(Bの3回) | 0.940秒 | 3.747秒 | 10.548秒 | 104.8 MiB | 6.51% | 6.46% |
| 重みのみINT8・両部品(C1) | 1.077秒 | 4.445秒 | 11.766秒 | 105.4 MiB | 6.81% | 6.26% |

CPU重みINT8はseed 0でFP32比テスト-0.09pt、実用文-0.13pt。全条件で生成Mel・波形は有限値。
CPUの初期速度表では動的INT8の方が速かったが、時刻・計測回数の違いも考慮する。
実際、別の10回検証でFP32の中文medianは4.474秒だったため、C1重みINT8の4.445秒という小差を高速化とは断定しない。
GPU INT8はFP32基準比、INT4はBF16基準比で比較する。CPU/GPU間で同じseedでも出力が完全一致するとは限らない。
C全体のメモリ結果は[詳細レポート](C_REPORT.md)を参照。Cの全条件に対する複数seed評価は未実施で、Bの動的INT8の結果を転用しない。

品質は`results/weight_only/quality_*.json`、速度は同ディレクトリの条件別JSONに保存した。
GPUの基準もC1で再計測した。公開MP3は全9条件×3文で27本、合計1,136,304 bytes(約1.08 MiB)。
公開MP3は各条件3文。中間結果は表のみで、散布図は多数の条件を終えた後の総括で作成する。

再現例:

```bash
python run.py --device cpu --target both --wo-bits 8 --output-dir results/weight_only
python run.py --device cuda --target both --wo-bits 8 --output-dir results/weight_only
python run.py --device cuda --dtype bfloat16 --output-dir results/weight_only
python run.py --device cuda --target both --wo-bits 4 --dtype bfloat16 --output-dir results/weight_only
python evaluate_weight_only.py --device cuda --bits 4 --dtype bfloat16
```

### C2/C3: group size・AWQ・GPTQの比較(完了)

group size 32/64/128、AWQ INT4、GPTQ INT8/INT4と対応する基準を追加した。
選定した12条件で速度・100＋55文CER・重みサイズ・独立プロセスのメモリ計測を完了した。
結果表、calibrationの条件、backendの制約は[C完了レポート](C_REPORT.md)にまとめた。
集計元データは`results/weight_only/c_summary.csv`・`c_summary.json`。
group sizeは量子化対象層をそろえて比較し、層ごとの混合精度はEで扱う。

## D: 静的PTQ・SmoothQuant

DではLinearの重みと活性値をINT8にする静的PTQを対象とする。
平滑化なし、SmoothQuant α=0.5、α=0.75を、CPU/GPUそれぞれのFP32基準と比較する。
重みはper-row、活性値はper-tensor・対称量子化で、calibration後は活性値のscaleを固定する。
T2Sの出力head、出力次元が8の倍数でないLinear、codec、LayerNorm、attentionのsoftmax、暫定ボコーダは量子化対象外。
出力次元の制約はGPU INT8 GEMMに合わせ、CPUにも同じ対象範囲を適用する。

GPUのeager静的INT8は中文で50.035秒、同FP32は1.356秒となり、実用候補から外した。
以降はLinearだけを`torch.compile(fullgraph=True, dynamic=True)`でコンパイルし、FP32にも同じ処理を適用する。
GPUの16行以下の入力は32行へゼロ埋めし、Linear後に余分な行を除く。attentionへは余分な行を渡さない。
初回コンパイルを含むwarmup時間は定常速度と別に記録する。A〜Cのeager結果とは直接同列に比較しない。

CPU静的INT8は短文5.328秒・中文38.554秒で、同条件FP32の1.275秒・4.505秒に対して約4.2倍・8.6倍遅かった。
このtorchao CPU INT8演算経路は速度で不採用とし、長文反復とCER評価を省略した。
同じ演算経路を使うCPU SmoothQuantの2条件も追加測定しない。CPUの品質維持は未判定であり、静的INT8一般や他runtimeへの否定ではない。

calibrationは学習splitのテキスト8文を長さ別に選び、2 passでチャネル別の最大絶対値と平滑化後の範囲を求める。
評価文は使用せず、CPU/GPUの参照入力をそれぞれキャッシュして同じdevice内の3方式で共有する。
T2Sのマスク率とS2Mの時刻を変えた近似状態を観測し、全生成軌跡の収集は行わない。
GPU/CPUの両方で速度・100＋55文CER・独立プロセスのメモリを測り、代表3文を48kbps MP3に保存する。
測定手順は`finish_d.py`、変換処理は`static_models.py`。既存結果は上書きせず、`results/static/`に保存する。
集計は[`d_summary.csv`](results/static/d_summary.csv)・[`d_summary.json`](results/static/d_summary.json)、
表の再生成と試聴ページの条件登録には、集計スクリプト[summarize_d.py](summarize_d.py)を使う。

<!-- D_RESULTS_START -->
| Device | 条件 | 中文 秒 | テストCER % | 実用文CER % | 重み MiB | peak RSS MiB | peak VRAM MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| cuda | FP32 | 2.42 | 6.46 | 6.73 | 361.98 | 2665.66 | 565.33 |
| cuda | 静的INT8 | 2.33 | 6.66 | 5.73 | 105.52 | 2665.59 | 311.22 |
| cuda | SmoothQuant α=0.5 | 2.52 | 6.44 | 6.40 | 105.88 | 2676.59 | 311.55 |
| cuda | SmoothQuant α=0.75 | 2.26 | 6.92 | 5.60 | 105.88 | 2671.49 | 311.55 |
| cpu | FP32 | 4.50 | 6.90 | 6.40 | 361.98 | 1778.07 | — |
| cpu | 静的INT8 | 38.55 | — | — | — | — | — |
| cpu | SmoothQuant α=0.5 | — | — | — | — | — | — |
| cpu | SmoothQuant α=0.75 | — | — | — | — | — | — |
<!-- D_RESULTS_END -->

## 今後の検証

次はEの混合精度量子化で、敏感な層を高精度に残した場合のCER・速度・サイズを比較する。
その後FのQAT、Gの配布runtimeでの検証へ進む。採用候補では複数seedと聴感も確認する。

## 最終総括の方針

このexpで量子化方式・bit幅・対象部品・混合精度・QATなど多数の条件を比較した後、総括として散布図にまとめる。
初期のFP32とS2M動的INT8の2条件は、現段階では表だけで記録する。

各条件で比較可能な速度とCERを蓄積し、最後にCPU/GPU別の速度・精度の関係とPareto frontを可視化する。
軸・速度の評価入力・品質の評価集合をそろえ、方式やbit幅を色・記号で区別する。
小規模な条件追加のたびに散布図を作ること自体を作業目標にはしない。
