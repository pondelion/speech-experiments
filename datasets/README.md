# datasets

音声実験で使うデータセットの置き場。音声データは git の対象外(`.gitignore`)で、この README と、
小さいテキストだけの `eval_texts/` を git で管理する。

| ディレクトリ | データセット | 話者 | 発話数 | 時間 | 形式 | 置き方 |
|---|---|---|---|---|---|---|
| `jsut/jsut_ver1.1/` | JSUT ver1.1 | 女性1人 | 7,696 | 10.27 h | 48kHz / 16bit | 配布データを展開 |
| `jvs/jvs_ver1/` | JVS ver1 | 100人(男49・女51) | 14,997 | 30.25 h | 24kHz / 16bit | 配布 zip を展開 |
| `jvnv/jvnv_v1/` | JVNV v1 | 4人(F1, F2, M1, M2) | 1,615 | 3.94 h | 48kHz / 24bit | 配布 zip を展開 |
| `reazonspeech_tiny/` | ReazonSpeech(tiny) | 多数(テレビ放送) | 5,323 | 8.60 h | 16kHz / 16bit / flac | HuggingFace から取得 |
| `jsut_lab/` | JSUT の音素アライメント(jsut-lab) | 女性1人 | 7,696 | - | テキスト(.lab) | GitHub から取得 |
| `eval_texts/` | TTS の評価用の文章(音声なし) | - | 479 文 | - | テキスト | ITA コーパス + 自作。git で管理する |

発話数・時間・形式は、音声ファイルを全部読んで数えた値(2026-11-06)。

`jsut/` と `reazonspeech_tiny/` は、別の ASR 実験のデータ置き場から実体をここへ移し、元の場所にはこちらへのシンボリックリンクを張っている。

## JSUT

- 出典:https://sites.google.com/site/shinnosuketakamichi/publication/jsut
- 女性1人の読み上げ。サブセットは `basic5000`, `countersuffix26`, `loanword128`, `onomatopee300`, `precedent130`, `repeat500`, `travel1000`, `utparaphrase512`, `voiceactress100`。
- 構成:`jsut_ver1.1/{サブセット}/wav/{ID}.wav` と `jsut_ver1.1/{サブセット}/transcript_utf8.txt`(1行 = `ID:テキスト`)。wav と書き起こしは全部対応している。
- `voiceactress100` の 100 文は、JVS の `parallel100` と同じ文(文面が全行一致)。
- 利用条件:音声は非商用の研究(企業内の研究を含む)と個人利用。商用は要問い合わせ。

## JVS

- 出典:https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus
- 構成:`jvs{001-100}/{セット}/wav24kHz16bit/{ID}.wav` と `{セット}/transcripts_utf8.txt`(1行 = `ID:テキスト`)。セットは次の4つ。

  | セット | 内容 | wav 数 |
  |---|---|---|
  | `parallel100` | 全員共通の 100 文(JSUT の voiceactress100 と同じ文) | 9,997 |
  | `nonpara30` | 話者ごとに違う 30 文 | 3,000 |
  | `whisper10` | ささやき声 10 文 | 1,000 |
  | `falset10` | 裏声 10 文 | 1,000 |

- 付属情報:`gender_f0range.txt`(性別と F0 の範囲)、`duration_info.txt`、`speaker_similarity_{male,female}.csv`、`parallel100`・`nonpara30` の `lab/`(自動で付けた音素の区切り)。
- **注意点(元のデータ側の事情)**:
  - `nonpara30` の書き起こしは話者ごとに 50 行あるが、wav は 30 個。余りの 20 行には wav がない。
  - `parallel100` は jvs030, jvs074, jvs089 の wav が 1 つずつ欠けている(99 個)。
  - `falset10` の jvs001 の 5 個(`BASIC5000_0235` など)と jvs088 の 10 個(`001`〜`010`)は、wav のファイル名が書き起こしの ID と一致しない。
  - 学習用の一覧を作るときは、wav と書き起こしの両方がそろうものだけを使う。
- 利用条件:音声は非商用の研究(企業内の研究を含む)と個人利用。再配布不可(10 ファイル程度の公開は可)。タグは CC BY-SA 4.0。

## JVNV

- 出典:https://sites.google.com/site/shinnosuketakamichi/research-topics/jvnv_corpus 、論文 https://arxiv.org/abs/2310.06072
- 6 感情(`anger`, `disgust`, `fear`, `happy`, `sad`, `surprise`)の演技音声。どの発話にも笑い声・泣き声などの非言語音声が 1 か所以上入る。
- 構成:`{話者}/{感情}/{セッション}/{話者}_{感情}_{セッション}_{番号}.wav`。
  - セッション:`regular`(非言語音声の部分を台本で指定。1,375 発話)と `free`(非言語音声の部分は話者が決める。240 発話)。
  - 書き起こし:`transcription.csv`(1行 = `{感情}_{セッション}_{番号}|{非言語音声の部分}|{全文}`、514 行)。wav と全部対応している。
  - `nv_label/{話者}/`:各発話の非言語音声の区間。
- 利用条件:CC BY-SA 4.0。

## ReazonSpeech(tiny)

- 出典:https://huggingface.co/datasets/reazon-research/reazonspeech (全体は 35,000 時間超。ここにあるのは tiny サブセット)
- テレビ放送から切り出した音声と字幕。雑音や音楽が混じる。話者の情報はない。
- 構成:`000/{ID}.flac` と `tiny.tsv`(1行 = `{パス}\t{テキスト}`)。
  - `tiny.tsv` は 5,324 行で、`000/3b1b48e55b400.flac` が 2 行に重複している。ファイルの欠けはない。
- 利用条件:CDLA-Sharing-1.0。著作権法 30 条の 4 の範囲での利用が条件とされている(HuggingFace のミラーの記載による。公式の全文は未確認)。

## eval_texts(TTS の評価用の文章)

- 音声は無く、文章だけ。合成した音声を ASR で書き起こし、入力した文章との CER で評価するのに使う
- `ita/`: [ITA コーパス](https://github.com/mmorise/ita-corpus)の感情 100 文・朗読 324 文(パブリックドメイン)
- `practical.tsv`: 自作の実用文 55 文(ニュース風・会話・数字・英字・固有名詞・読みが文脈で変わる語・長文・短文・疑問/感嘆・オノマトペ)
- 詳細と注意点(文字入力での `<unk>`、pyopenjtalk の読み間違い)は `eval_texts/README.md`
- 読み込みは `tts/lib_common/eval_sets.py` の `load_eval_texts()`

## jsut_lab(JSUT の音素アライメント)

- 出典: https://github.com/r9y9/jsut-lab(MIT ライセンス)
- JSUT の全 7,696 発話について、音素ごとの開始・終了時刻と、OpenJTalk のフルコンテキストラベルを持つ。
  時刻は音声認識ソフト Julius による強制アライメントで自動的に付けたもの
- 構成: `{サブセット}/lab/{ID}.lab`(JSUT と同じ並び)。1 行 = `開始 終了 フルコンテキストラベル`(時刻は 100 ナノ秒単位)
- 読み込みは `tts/lib_common/data.py` の `load_jsut_alignment()`
