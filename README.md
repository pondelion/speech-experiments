# speech-experiments

日本語の音声合成(TTS)などを、スクラッチ実装で試していく実験のリポジトリ。

| ディレクトリ | 中身 |
|---|---|
| [tts/](tts/) | TTS の実験。ベースにしたモデルごとにまとめる(例: [tts/indextts_bases/](tts/indextts_bases/)) |
| [datasets/](datasets/) | データセットの説明と、全実験で共通の評価用の文章(データ本体は git の対象外) |

## 試聴ページ(GitHub Pages)

リポジトリのトップの `index.html` が試聴ページ。ベースモデル → exp の順に並べ、同じ文章の合成音声を条件どうしで聞き比べられる。

- 中身は `site/`(`app.js`・`style.css`・`manifest.json`)。音声は各 exp の `results/samples/`(48kbps の MP3)と、
  その一覧 `results/samples/index.json`(各 exp の評価スクリプトが作る)をそのまま読む
- **exp を足すとき**: `site/manifest.json` の該当するベースモデルの `exps` に 1 項目足す(試聴用の音声の場所 `samples`、
  条件の表示名・CER・既定で選ぶかを書いた `condition_groups`)。`condition_groups` に書いていない条件も「その他」として表示される
- **ベースモデルを足すとき**: `bases` に 1 項目足す
- 公開: GitHub の Settings → Pages で「Deploy from a branch」・`main` ブランチの `/ (root)` を選ぶ
- 手元で見るとき: リポジトリのトップで `python -m http.server` を動かし、`http://localhost:8000/` を開く
  (ファイルを直接開くと、ブラウザの制限で一覧の JSON を読めない)
