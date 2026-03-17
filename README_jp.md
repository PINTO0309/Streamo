# Streamo
<h1 align="center">Streaming Video Instruction Tuning</h1>
<p align="center"><i>汎用的な対話アシスタントとして機能する、リアルタイムなストリーミング動画 LLM。</i></p>

<p align="center">
         📑 <a href="https://arxiv.org/abs/2512.21334">Paper</a>  &nbsp|&nbsp  🌐 <a href="https://jiaerxia.github.io/Streamo/">Web</a>  &nbsp|&nbsp 🤗 <a href="https://huggingface.co/datasets/maifoundations/Streamo-Instruct-465K">Huggingface</a>
</p>

本リポジトリは、論文「Streaming Video Instruction Tuning」の公式実装です。


# News📰
* **`[2026/2/27]`:**🎉**本論文が CVPR 2026 に採択されました。**
* **`[2026/1/27]`:**🔥**Streamo-Instruct データセットを公開しました。[[HF](https://huggingface.co/datasets/maifoundations/Streamo-Instruct-465K)]**
* **`[2026/1/22]`:**🔥**学習コードを公開しました。**
* **`[2026/1/6]`:**🔥**より多くのデモを含む Web サイトを公開しました。[[Web](https://jiaerxia.github.io/Streamo/)]**
* **`[2025/12/24]`:**🔥**論文を公開しました。[[Arxiv](https://arxiv.org/abs/2512.21334)]**

> **Note:** 制約により、現時点ではモデル重みを一般公開できません。必要があればご連絡ください。


# Demo🎬

<p align="center">
  <a href="https://youtu.be/lGRdBP-SYeo">
    <img src="https://img.youtube.com/vi/lGRdBP-SYeo/maxresdefault.jpg" alt="Demo Video" width="800">
  </a>
</p>



# Training🚀

## Installation

```bash
pip install -r requirements.txt
```

## Data Preparation📊

### Data Pipeline Overview

`train.sh` で使われる学習パイプラインは次の通りです。

`raw annotations + partial local video folders -> preparation script -> stream-format JSON with <stream> -> training-time conversion to <image> + frame cache -> swift sft`

重要なのは、`train.sh` は raw annotation をそのまま学習しないという点です。`train.sh` は `streaming_video` という登録済みデータセットを読み、そのデータセットは前処理時にフレーム単位の `<image>` 入力へ変換されます。ローカルディスクにラベルが参照する動画の一部しか存在しない場合は、準備スクリプトが実際に解決できた動画サンプルだけを残すようにフィルタします。

### How `train.sh` Actually Reads Data

`train.sh` では次を使用しています。

- `--dataset streaming_video`
- `--custom_register_path swift/plugin/loss.py swift/plugin/streaming_dataset.py`
- `--new_special_tokens './special_token_v1.txt'`

現在の `swift/plugin/streaming_dataset.py` における `streaming_video` の登録先は次の通りです。

- データセットファイル: 既定では `./dataset/stream/stream_format.json`
- フォールバック用サンプル: 生成済みデータセットが存在しない場合は `./dataset/example/stream_format.json`
- フレームキャッシュディレクトリ: `./dataset/stream/frames`

`train.sh` は同じ既定値を環境変数としても export しています。

- `STREAMING_DATASET_PATH`
- `STREAMING_FRAME_DIR`
- `STREAMING_DATASET_FPS`

学習時には、登録された streaming-video preprocessor が次を実行します。

1. stream-format JSON を読み込む
2. `messages` 内の `<stream>` トークンを探す
3. 設定された FPS で動画フレームを抽出する
4. `<stream>` を `<image>` に置き換える
5. 抽出済みフレームをディスクに保存または再利用する

`special_token_v1.txt` は、データセット内で使う以下のラベルと必ず一致している必要があります。

- `</Silence>`
- `</Standby>`
- `</Response>`

### Raw Annotation Format

コンバータは単一の JSON オブジェクトまたは JSON 配列のどちらも受け付けますが、実際の学習では通常サンプルの JSON 配列を用意します。1つの raw item が 1つの学習サンプルになります。同じ動画に対して複数の独立した質問がある場合は、同じ `video_path` を使う複数 raw item として表現してください。

raw annotation の例:

```json
{
  "video_name": "video1.mp4",
  "video_path": "/path/to/video.mp4",
  "task_type": "QA",
  "source": "custom",
  "question": [
    {"content": "What happens in the video?", "time": "5"}
  ],
  "response": [
    {"content": "A person walks into the room.", "st_time": 5.0, "end_time": 6.0, "time": ""}
  ]
}
```

`scripts/convert_streaming_video.py` における各フィールドの意味:

| Field | Description |
|-------|-------------|
| `video_path` | 絶対パスでも、`--video-prefix` で解決される相対パスでもよいです。 |
| `question.time` | 質問は `max(0, time - 1)` フレーム目に挿入されます。たとえば `"5"` は `<4s-5s>` の user turn に入ります。 |
| `response.st_time` | イベント区間の開始です。サンプルは `max(0, st_time - 1)` フレーム目から `</Standby>` としてラベル付けされます。 |
| `response.end_time` | イベント区間の終了です。`</Standby>` は `max(0, end_time - 1)` フレーム目まで続き、最終応答はその次のフレームで出力されます。 |
| `response.time` | 即時応答モードです。`st_time` と `end_time` を使わない場合、応答は `max(0, time - 1)` フレーム目で直接出力されます。 |

時間値は整数、浮動小数、`MM:SS`、`HH:MM:SS` として解釈され、選択した `--fps` に応じてフレーム番号へ変換されます。

### Stream-Format Training Data

`train.sh` は次のような stream-format サンプルを想定しています。

```json
{
  "messages": [
    {"role": "system", "content": "System prompt for streaming video assistant"},
    {"role": "user", "content": "Your question\n<0s-1s>\n<stream>"},
    {"role": "assistant", "content": "</Silence>"},
    {"role": "user", "content": "<1s-2s>\n<stream>"},
    {"role": "assistant", "content": "</Standby>"},
    {"role": "user", "content": "<2s-3s>\n<stream>"},
    {"role": "assistant", "content": "</Response> Your answer here"}
  ],
  "videos": ["/path/to/video.mp4"]
}
```

補足:

- `<stream>` は現在フレームのプレースホルダです。
- `<Xs-Ys>` はそのフレームに対応する時間区間です。
- 学習時には `<stream>` は `<image>` に置き換えられます。
- 既定では動画は 1 FPS でサンプリングされるため、各 `<stream>` は 1 枚の抽出フレームに対応します。

raw / converted の例は `dataset/example/` を参照してください。

### Recommended Workflow for `train.sh`

`train.sh` を変更せずに使いたい場合は、準備スクリプトで `./dataset/stream/stream_format.json` を生成する運用にしてください。

#### 1. ラベルとローカル動画フォルダを用意する

既定では以下を前提とします。

- ラベルルート: `/media/lm/NO_NAME/Streamo-Instruct-465K`
- 動画ルート: `/media/lm/NO_NAME`

スクリプトはラベル JSON をすべて走査し、生き残った各行の `video_path` を絶対ローカルパスへ書き換え、ローカルに存在しない動画を参照する行は落とします。

#### 2. ローカルディスク配置から学習データセットを構築する

```bash
python scripts/prepare_streamo_training_data.py \
    --label-root /media/lm/NO_NAME/Streamo-Instruct-465K \
    --media-root /media/lm/NO_NAME \
    --output-raw ./dataset/stream/raw_resolved.json \
    --output-stream ./dataset/stream/stream_format.json \
    --report-json ./dataset/stream/prepare_report.json \
    --fps 1.0 \
    --num-workers 8
```

このステップでは次を行います。

1. ラベルルート配下の JSON をすべて走査する
2. ソースごとのルールでローカル動画パスを解決する
3. 解決不能または動画欠損のサンプルをスキップする
4. 生き残った raw JSON を `./dataset/stream/raw_resolved.json` に書く
5. 生き残った行を stream format に変換する
6. 機械可読なレポートを `./dataset/stream/prepare_report.json` に書く

#### 3. 学習を実行する

```bash
bash train.sh
```

既定のワークフローでは、`swift/plugin/streaming_dataset.py` を手で編集する必要はありません。

### `to-image` Is Not Required for `train.sh`

`scripts/convert_streaming_video.py` には `to-image` モードもありますが、このリポジトリの既定の学習スクリプトでは **不要** です。

`train.sh` は `<stream>` トークンを含む stream-format データを期待します。登録済み preprocessor がそれを `<image>` 形式へ変換し、データセット前処理中にフレームを抽出または再利用します。

新しい準備スクリプトはすでに stream-format JSON を出力します。`to-image` はオフライン確認や独自ワークフロー向けのオプションとして残っています。

### Special Tokens

| Token | Description |
|-------|-------------|
| `</Silence>` | 関連イベントがまだ始まっていない、または現在の入力が質問と無関係であることを示します。 |
| `</Standby>` | イベント進行中、またはまだ回答に十分な情報が揃っていないことを示します。 |
| `</Response>` | イベントが終了した、または回答に十分な情報が揃ったことを示します。 |

### Advanced: Multiple Datasets

基盤の `swift` ローダは `--dataset` に複数データセットを渡せますが、このリポジトリのラッパは既定では `streaming_video` という1つのデータセット名しか登録していません。

このリポジトリでは、複数ソースをまとめて学習したい場合でも、既定では次の方針を推奨します。

- 1つの統合 stream-format JSON を作る
- `train.sh` に `STREAMING_DATASET_PATH` 経由で読ませる
- `train.sh` 自体は変更しない

別々のデータセット ID を持たせたい場合は、独自の登録ファイルを作って複数の streaming dataset を明示的に登録してください。例:

```python
from swift.llm.dataset.dataset import register_streaming_video_dataset

register_streaming_video_dataset(
    dataset_path='./dataset/source_a/stream_format.json',
    dataset_name='streaming_video_a',
    fps=1.0,
    save_frames=True,
    frame_output_dir='./dataset/source_a/frames',
    enable_memory_cache=False,
)

register_streaming_video_dataset(
    dataset_path='./dataset/source_b/stream_format.json',
    dataset_name='streaming_video_b',
    fps=1.0,
    save_frames=True,
    frame_output_dir='./dataset/source_b/frames',
    enable_memory_cache=False,
)
```

その後、標準学習コマンドの dataset 関連部分を次のように差し替えます。

```bash
swift sft \
    --model Qwen2.5-VL-3B-Instruct \
    --dataset streaming_video_a streaming_video_b \
    --custom_register_path swift/plugin/loss.py path/to/custom_streaming_datasets.py
```

混合時の挙動:

- 複数データセットを渡しても `--interleave_prob` を指定しなければ `swift` は単純連結します。
- `--interleave_prob` を指定するとインターリーブになります。
- `--stopping_strategy first_exhausted` はどれか1つのデータセットが尽きた時点で停止します。
- `--stopping_strategy all_exhausted` は全データセットを使い切るまで続けます。

`<stream>` データセットに対して `--custom_dataset_info` を安易に使わないでください。同等の独自前処理を別途入れない限り、この経路は `AutoPreprocessor` を使いますが、本リポジトリではフレーム抽出と `<stream>` から `<image>` への置換を行う streaming-video preprocessor が必要です。

### Local Partial-Video Support

準備スクリプトは、「ラベル木は完全だが、参照先動画は一部しかローカルに保存されていない」という典型的なローカル環境を想定しています。

既定のソース別パス解決ルール:

- `coin` -> `/media/lm/NO_NAME/coin/videos/{basename}`
- `ActivityNet` -> `/media/lm/NO_NAME/activitynet/videos/{basename}`
- `QVHighlight` -> `/media/lm/NO_NAME/QVHighlight/videos/{basename}`
- `queryd` -> `/media/lm/NO_NAME/Queryd/videos/{basename}`
- `didemo` -> `/media/lm/NO_NAME/didemo/videos/{basename}`
- `tacos` -> `/media/lm/NO_NAME/tacos/videos/{basename}` または `.avi`
- `Youcook` / `Youcookv2` -> `/media/lm/NO_NAME/youcook2/videos/{basename}.mp4`
- `how_to_caption` -> `/media/lm/NO_NAME/how_to_caption/how_to_caption/{basename}`
- `how_to_step` -> media root 配下の元のパスをそのまま使用
- `ego_timeqa` -> `/media/lm/NO_NAME/ego4d/videos_3fps_480_noaudio/{uuid_prefix}.mp4`

既定の除外および best-effort 動作:

- `Koala` は、現在見えているローカルツリーから動画ファイルを直接参照できないため、既定では除外されます。
- `LLaVA_Video` はまず厳密なパス解決を試し、失敗した場合は `/media/lm/NO_NAME/LLaVA_Video` 配下で basename フォールバックを行います。
- basename フォールバックで候補が複数見つかった場合、そのサンプルは曖昧としてスキップされます。
- ローカルに一致する動画が存在しない場合も、そのサンプルはスキップされ、レポートに記録されます。

### Validation and Troubleshooting

- 動画ファイルがない場合、準備時または前処理時にサンプルがスキップされます。
- `./dataset/stream/prepare_report.json` には、ソース別・理由別の kept / dropped 件数が入ります。
- よくある除外理由は `missing_file`、`unsupported_source`、`ambiguous_match`、`conversion_failed` です。
- フレーム数は `messages` 内の `<stream>` トークン数と照合されます。
- 差分が許容範囲内であれば、余剰フレームは切り詰め、不足分は最後のフレームを複製します。
- 差分が許容範囲を超える場合、そのサンプルは破棄されます。
- 変換時の FPS と登録済みデータセットの FPS は揃えてください。ずれるとフレーム数不一致や時間ずれの原因になります。
- `save_frames=True` の場合、初回実行時にフレームをディスクへ抽出し、その後はキャッシュを再利用します。
- データセットの内容を差し替えた場合は、キャッシュディレクトリもそのデータセットに対応したものになっていることを確認してください。

## Quick Start▶️

```bash
python scripts/prepare_streamo_training_data.py
bash train.sh
```


# Acknowledgement

本プロジェクトは [ms-swift](https://github.com/modelscope/ms-swift) をベースに構築されています。優れた実装を提供してくれた著者の皆様に感謝します。

# Citation🎓
```
@article{xia2025streaming,
  title={Streaming Video Instruction Tuning},
  author={Xia, Jiaer and Chen, Peixian and Zhang, Mengdan and Sun, Xing and Zhou, Kaiyang},
  journal={arXiv preprint arXiv:2512.21334},
  year={2025}
}
```
