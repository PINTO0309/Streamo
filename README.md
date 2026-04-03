# Streamo
<h1 align="center">Streaming Video Instruction Tuning</h1>
<p align="center"><i>A real-time streaming video LLM that serves as a general-purpose interactive assistant.</i></p>

<p align="center">
         📑 <a href="https://arxiv.org/abs/2512.21334">Paper</a>  &nbsp|&nbsp  🌐 <a href="https://jiaerxia.github.io/Streamo/">Web</a>  &nbsp|&nbsp 🤗 <a href="https://huggingface.co/datasets/maifoundations/Streamo-Instruct-465K">Huggingface</a>
</p>

<p align="center">
  <a href="./README_jp.md">日本語 README</a>
</p>

This is the official implementation of the paper 'Streaming Video Instruction Tuning'.


# News📰
* **`[2026/2/27]`:**🎉**Our paper has been accepted by CVPR 2026!**
* **`[2026/1/27]`:**🔥**We have released the Streamo-Instruct dataset.[[HF](https://huggingface.co/datasets/maifoundations/Streamo-Instruct-465K)].**
* **`[2026/1/22]`:**🔥**We have released our training code.**
* **`[2026/1/6]`:**🔥**We have released our website with more interesting demos [[Web](https://jiaerxia.github.io/Streamo/)].**
* **`[2025/12/24]`:**🔥**We have released our paper [[Arxiv](https://arxiv.org/abs/2512.21334)].**

> **Note:** Due to some restrictions, we are unable to publicly release the model weights at this time. If you have any request, please feel free to contact us.


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
pip install google-cloud-storage
```

## Streaming Inference Demo

Use `examples/infer/streaming_action_caption_demo.py` for round-by-round streaming inference with trained weights. The recommended backend is `pt`; `vllm` is supported as an optional path for the same CLI.

Full checkpoint / merged model example:

```bash
CUDA_VISIBLE_DEVICES=0 \
python examples/infer/streaming_action_caption_demo.py \
--model-path /path/to/full_or_merged_checkpoint \
--video-path demo/cook.mp4 \
--mode caption \
--realtime false \
--save-jsonl output/stream_demo.jsonl
```

Adapter example:

```bash
CUDA_VISIBLE_DEVICES=0 \
python examples/infer/streaming_action_caption_demo.py \
--adapter-path /path/to/adapter_checkpoint \
--video-path demo/cook.mp4 \
--mode qa \
--question "What is being added to the bowl?" \
--realtime false
```

The script prints each round as `<Xs-Ys>` plus the model output, emits a separate event log only for new `</Response>` turns, and can optionally persist one JSONL record per round.

## Data Preparation📊

### Data Pipeline Overview

The training pipeline used by `train.sh` is:

`raw annotations + partial local video folders -> preparation script -> stream-format JSON with <stream> -> training-time conversion to <image> + frame cache -> swift sft`

This is important because `train.sh` does **not** train directly from raw annotations. It trains from a registered dataset named `streaming_video`, and that dataset is converted to frame-level `<image>` inputs during preprocessing. When your local disk only contains a subset of the videos referenced by the labels, the preparation script filters the dataset down to the samples whose video files can actually be resolved.

### How `train.sh` Actually Reads Data

`train.sh` uses:

- `--dataset streaming_video`
- `--custom_register_path swift/plugin/loss.py swift/plugin/streaming_dataset.py`
- `--new_special_tokens './special_token_v1.txt'`

The current registration in `swift/plugin/streaming_dataset.py` points `streaming_video` to:

- dataset file: `./dataset/stream/stream_format.json` by default
- fallback example file: `./dataset/example/stream_format.json` if the generated dataset does not exist
- frame cache directory: `./dataset/stream/frames`

`train.sh` exports the same defaults through environment variables:

- `STREAMING_DATASET_PATH`
- `STREAMING_FRAME_DIR`
- `STREAMING_DATASET_FPS`

During training, the registered streaming-video preprocessor:

1. loads the stream-format JSON,
2. finds `<stream>` tokens in `messages`,
3. extracts video frames at the configured FPS,
4. replaces `<stream>` with `<image>`,
5. stores or reuses cached frames from disk.

`special_token_v1.txt` must stay aligned with the labels used in the dataset:

- `</Silence>`
- `</Standby>`
- `</Response>`

### Raw Annotation Format

The converter accepts either a single JSON object or a JSON array, but for real training you will usually prepare a JSON array of samples. One raw item becomes one training sample. If the same video has multiple independent questions, represent them as multiple raw items that reuse the same `video_path`.

Example raw annotation item:

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

Field semantics implemented by `scripts/convert_streaming_video.py`:

| Field | Description |
|-------|-------------|
| `video_path` | Can be an absolute path, or a relative path resolved with `--video-prefix`. |
| `question.time` | The question is injected at frame `max(0, time - 1)`. For example, `"5"` appears in the user turn for `<4s-5s>`. |
| `response.st_time` | Start of an event span. The sample is labeled `</Standby>` beginning at frame `max(0, st_time - 1)`. |
| `response.end_time` | End of an event span. `</Standby>` continues through frame `max(0, end_time - 1)`, and the final answer is emitted on the next frame. |
| `response.time` | Instant-response mode. If `st_time` and `end_time` are not used, the answer is emitted directly at frame `max(0, time - 1)`. |

Time values are parsed as integers, floats, `MM:SS`, or `HH:MM:SS`, then mapped to frame indices using the selected `--fps`.

### Stream-Format Training Data

`train.sh` expects stream-format samples like this:

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

Notes:

- `<stream>` is a placeholder for the current frame.
- `<Xs-Ys>` is the timestamp interval of that frame.
- At training time, `<stream>` is replaced with `<image>`.
- Videos are sampled at 1 FPS by default, so each `<stream>` corresponds to one extracted frame.

See `dataset/example/` for example raw and converted files.

### Recommended Workflow for `train.sh`

If you want to keep `train.sh` unchanged, use the preparation script and let it generate the dataset at `./dataset/stream/stream_format.json`.

#### 1. Prepare labels and local media folders

The default script assumes:

- label root: `/media/lm/NO_NAME/Streamo-Instruct-465K`
- media root: `/media/lm/NO_NAME`

It scans all label JSON files, rewrites each surviving row to an absolute local `video_path`, and drops rows whose videos are unavailable locally.

#### 2. Build the training dataset from the local disk layout

```bash
source /mnt/data/venv/bin/activate
hf auth login

hf download maifoundations/Streamo-Instruct-465K \
--repo-type dataset \
--local-dir dataset/Streamo-Instruct-465K

python scripts/prepare_streamo_training_data.py \
--label-root /media/lm/NO_NAME/Streamo-Instruct-465K \
--media-root /media/lm/NO_NAME \
--output-raw ./dataset/stream/raw_resolved.json \
--output-stream ./dataset/stream/stream_format.json \
--report-json ./dataset/stream/prepare_report.json \
--fps 1.0 \
--num-workers 8
```

This step:

1. scans every label JSON under the label root,
2. resolves source-specific local video paths,
3. skips unresolved or missing-video samples,
4. writes a merged raw JSON to `./dataset/stream/raw_resolved.json`,
5. converts the surviving rows to stream format,
6. writes a machine-readable report to `./dataset/stream/prepare_report.json`.

If you already have a raw JSON with resolved video paths, you can also convert it directly:

```bash
python scripts/convert_streaming_video.py to-stream \
    --input raw_data.json \
    --output stream_format.json \
    --video-prefix /path/to/videos \
    --fps 1.0
```

`--video-prefix` is the base directory prepended to each `video_path` in the input JSON. Use it when `video_path` is stored as a relative path or filename. For example, if `video_path` is `LLaVA_Video/.../sample.mp4` and `--video-prefix` is `/data`, the script reads `/data/LLaVA_Video/.../sample.mp4`. If `video_path` is already an absolute path, this argument is not needed.

#### 3. Run training

```bash
bash train.sh
```

No manual edits to `swift/plugin/streaming_dataset.py` are required for the default workflow.

### `to-image` Is Not Required for `train.sh`

`scripts/convert_streaming_video.py` also provides a `to-image` mode, but that is **not** required by the default training script in this repo.

`train.sh` expects stream-format data with `<stream>` tokens. The registered preprocessor converts it to `<image>` format and extracts or reuses cached frames during dataset preprocessing.

The new preparation script already writes stream-format JSON. `to-image` remains optional for offline inspection or custom workflows.

### Special Tokens

| Token | Description |
|-------|-------------|
| `</Silence>` | No relevant event has started, or the current input is irrelevant to the question. |
| `</Standby>` | The event is in progress, or the current information is not yet sufficient to answer. |
| `</Response>` | The event has finished, or the available information is now sufficient to answer. |

### Advanced: Multiple Datasets

The underlying `swift` loader supports passing multiple datasets to `--dataset`, but the current repo wrapper registers only one dataset name, `streaming_video`, by default.

For this repo, the recommended default for mixed-source training is still:

- build one combined stream-format JSON,
- let `train.sh` read it through `STREAMING_DATASET_PATH`,
- keep `train.sh` unchanged.

If you need separate dataset identities, create your own registration file and register multiple streaming datasets explicitly. Example:

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

Then replace the dataset-related part of the standard training command with multiple dataset names, for example:

```bash
swift sft \
    --model Qwen2.5-VL-3B-Instruct \
    --dataset streaming_video_a streaming_video_b \
    --custom_register_path swift/plugin/loss.py path/to/custom_streaming_datasets.py
```

Mixing behavior:

- If you pass multiple datasets without `--interleave_prob`, `swift` concatenates them.
- If you set `--interleave_prob`, `swift` interleaves them instead.
- `--stopping_strategy first_exhausted` stops when one dataset runs out.
- `--stopping_strategy all_exhausted` keeps sampling until all datasets are exhausted.

Do **not** rely on `--custom_dataset_info` for `<stream>` datasets unless you also provide equivalent custom preprocessing. That path uses `AutoPreprocessor`, while this repo needs the streaming-video preprocessor that extracts frames and replaces `<stream>` with `<image>`.

### Local Partial-Video Support

The preparation script is designed for the common local setup where the label tree is complete but only part of the referenced videos are stored on disk.

Default source-specific resolution rules include:

- `coin` -> `/media/lm/NO_NAME/coin/videos/{basename}`
- `ActivityNet` -> `/media/lm/NO_NAME/activitynet/videos/{basename}`
- `QVHighlight` -> `/media/lm/NO_NAME/QVHighlight/videos/{basename}`
- `queryd` -> `/media/lm/NO_NAME/Queryd/videos/{basename}`
- `didemo` -> `/media/lm/NO_NAME/didemo/videos/{basename}`
- `tacos` -> `/media/lm/NO_NAME/tacos/videos/{basename}` or `.avi`
- `Youcook` / `Youcookv2` -> `/media/lm/NO_NAME/youcook2/videos/{basename}.mp4`
- `how_to_caption` -> `/media/lm/NO_NAME/how_to_caption/how_to_caption/{basename}`
- `how_to_step` -> original path under the media root
- `ego_timeqa` -> `/media/lm/NO_NAME/ego4d/videos_3fps_480_noaudio/{uuid_prefix}.mp4`

Default exclusions and best-effort behavior:

- `Koala` is excluded by default because the visible local tree does not expose video files directly.
- `LLaVA_Video` uses exact-path resolution first, then a basename fallback under `/media/lm/NO_NAME/LLaVA_Video`.
- If a basename fallback finds multiple candidates, the sample is skipped as ambiguous.
- If no local match exists, the sample is skipped and counted in the report.

### Validation and Troubleshooting

- Missing video files are skipped during preparation or preprocessing.
- `./dataset/stream/prepare_report.json` contains kept and dropped counts by source and reason.
- Common drop reasons are `missing_file`, `unsupported_source`, `ambiguous_match`, and `conversion_failed`.
- Frame count is checked against the number of `<stream>` tokens in `messages`.
- If the difference is within tolerance, preprocessing truncates extra frames or duplicates the last frame.
- If the difference exceeds tolerance, the sample is discarded.
- Keep the conversion FPS and the registered dataset FPS aligned. A mismatch can produce frame-count errors or incorrect timing.
- With `save_frames=True`, the first run extracts frames to disk and later runs reuse the cached frame directory.
- If you replace the dataset with a new version, make sure the cache directory matches that dataset version.

## Detailed Pipeline Flow

### Local Video Training Flow

When `bash train.sh` is executed in the default setup, the training pipeline works as follows:

1. `train.sh` launches `swift sft` with `--dataset streaming_video`.
2. `swift/plugin/streaming_dataset.py` registers the dataset specified by `STREAMING_DATASET_PATH` (default: `./dataset/stream/stream_format.json`).
3. Each sample is loaded from JSON/JSONL and passed to `StreamingVideoPreprocessor`.
4. The preprocessor reads the first path from `videos`, `video`, or `video_path`.
5. For each `<stream>` token in `messages`, the corresponding video is sampled at `STREAMING_DATASET_FPS` fps.
6. `<stream>` is replaced by `<image>`, and the extracted frames are stored in `images`.
7. If `save_frames=True`, extracted frames are cached under `STREAMING_FRAME_DIR` so repeated epochs do not decode the same video again.
8. The final training sample is passed to Swift as a multi-image training example.

In short, the model trains on `messages + images`, while the original `videos` field is only used to lazily produce the frame sequence.

### GCE/GCS Archive Training Flow

This project also supports training from datasets stored in GCS as split archives such as `.tar.gz.00`, `.tar.gz.01`, ...

The expected flow is:

1. Prepare `stream_format.json` as usual. The schema does not change; `videos` / `video_path` should still contain the logical relative path of the video inside the archive.
2. Build a SQLite sidecar index from the GCS archive shards:

```bash
python scripts/build_stream_archive_index.py \
    --gcs-prefix gs://your-bucket/path/to/archives \
    --output ./dataset/stream/archive_index.sqlite
```

3. Enable archive resolution before running training:

```bash
export STREAMING_ENABLE_ARCHIVE_RESOLUTION=1
export STREAMING_ARCHIVE_INDEX_PATH=./dataset/stream/archive_index.sqlite
export STREAMING_ARCHIVE_GCS_PREFIX=gs://your-bucket/path/to/archives
export STREAMING_ARCHIVE_CACHE_DIR=./dataset/stream/archive_cache
export STREAMING_VIDEO_CACHE_DIR=./dataset/stream/video_cache
export STREAMING_DATASET_PATH=./dataset/stream/stream_format.json
export STREAMING_FRAME_DIR=./dataset/stream/frames
```

Legacy `STREAM_*` environment variable names are also accepted for backward compatibility.

4. Run `bash train.sh`.

At training time, each sample is processed in this order:

1. `StreamingVideoPreprocessor` reads the logical path from `videos` / `video_path`.
2. `ArchiveVideoResolver` checks whether that path already exists locally.
3. If not, it looks up the path in `archive_index.sqlite` and finds:
   - which logical archive shard contains the file
   - the member path inside that tar archive
   - the ordered list of split parts such as `.00`, `.01`, ...
4. The required split parts are downloaded from GCS into `STREAMING_ARCHIVE_CACHE_DIR/parts/`.
5. The parts are concatenated into a local `.tar.gz` archive under `STREAMING_ARCHIVE_CACHE_DIR/archives/`.
6. Only the requested video file is extracted from that archive into `STREAMING_VIDEO_CACHE_DIR/`.
7. The extracted local video path is then passed to the normal frame extraction pipeline.
8. Frames are decoded at the configured dataset FPS and cached under `STREAMING_FRAME_DIR/`.
9. `<stream>` tokens are replaced with `<image>`, and Swift receives the resulting `messages + images` sample.

### Cache Behavior

- `STREAMING_ARCHIVE_CACHE_DIR` caches downloaded archive parts and reconstructed `.tar.gz` archives.
- `STREAMING_VIDEO_CACHE_DIR` caches extracted video files from the archives.
- `STREAMING_FRAME_DIR` caches per-frame images used during training.
- On repeated access, the pipeline reuses these caches instead of downloading or decoding again.

This means the first epoch may spend time downloading archives, extracting videos, and decoding frames, while later epochs mostly reuse local cache.

## Quick Start▶️

```bash
python scripts/prepare_streamo_training_data.py
bash train.sh
```


# Acknowledgement

This project is built upon [ms-swift](https://github.com/modelscope/ms-swift). We thank the authors for their excellent work.

# Citation🎓
```
@article{xia2025streaming,
  title={Streaming Video Instruction Tuning},
  author={Xia, Jiaer and Chen, Peixian and Zhang, Mengdan and Sun, Xing and Zhou, Kaiyang},
  journal={arXiv preprint arXiv:2512.21334},
  year={2025}
}
```
