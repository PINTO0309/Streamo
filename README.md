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
```

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
