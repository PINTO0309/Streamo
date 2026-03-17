# Streamo
<h1 align="center">Streaming Video Instruction Tuning</h1>
<p align="center"><i>A real-time streaming video LLM that serves as a general-purpose interactive assistant.</i></p>

<p align="center">
         📑 <a href="https://arxiv.org/abs/2512.21334">Paper</a>  &nbsp|&nbsp  🌐 <a href="https://jiaerxia.github.io/Streamo/">Web</a>  &nbsp|&nbsp 🤗 <a href="https://huggingface.co/datasets/maifoundations/Streamo-Instruct-465K">Huggingface</a>
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

`raw annotations -> stream-format JSON with <stream> -> training-time conversion to <image> + frame cache -> swift sft`

This is important because `train.sh` does **not** train directly from raw annotations. It trains from a registered dataset named `streaming_video`, and that dataset is converted to frame-level `<image>` inputs during preprocessing.

### How `train.sh` Actually Reads Data

`train.sh` uses:

- `--dataset streaming_video`
- `--custom_register_path swift/plugin/loss.py swift/plugin/streaming_dataset.py`
- `--new_special_tokens './special_token_v1.txt'`

The current registration in `swift/plugin/streaming_dataset.py` points `streaming_video` to:

- dataset file: `./dataset/example/stream_format.json`
- frame cache directory: `./dataset/stream/frames`

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

If you want to keep `train.sh` unchanged, use the following workflow.

#### 1. Prepare one or more raw annotation files

Create raw JSON files in the schema above. If you have multiple data sources and want to train on all of them together, the recommended default is to merge them into one combined JSON array before conversion.

#### 2. Merge raw files if needed

Example merge script:

```bash
python - <<'PY'
import json

inputs = ['raw_a.json', 'raw_b.json']
merged = []
for path in inputs:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        merged.append(data)
    else:
        merged.extend(data)

with open('raw_merged.json', 'w', encoding='utf-8') as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)
PY
```

#### 3. Convert raw annotations to stream format

```bash
python scripts/convert_streaming_video.py to-stream \
    --input raw_merged.json \
    --output dataset/stream/stream_format.json \
    --video-prefix /path/to/video/root \
    --fps 1.0 \
    --num-workers 8
```

Use `--video-prefix` when `video_path` in the raw annotations is relative. If `video_path` is already absolute, you can omit this flag.

#### 4. Point the registered dataset to your converted JSON

Edit `swift/plugin/streaming_dataset.py` and update:

- `dataset_path` to your converted stream-format JSON
- `frame_output_dir` to the frame cache directory you want to use
- `fps` if you intentionally trained at a different sampling rate

By default, the repo uses:

```python
register_streaming_video_dataset(
    dataset_path='./dataset/example/stream_format.json',
    dataset_name='streaming_video',
    fps=1.0,
    save_frames=True,
    frame_output_dir='./dataset/stream/frames',
    enable_memory_cache=False,
)
```

#### 5. Run training

```bash
bash train.sh
```

### `to-image` Is Not Required for `train.sh`

`scripts/convert_streaming_video.py` also provides a `to-image` mode, but that is **not** required by the default training script in this repo.

`train.sh` expects stream-format data with `<stream>` tokens. The registered preprocessor converts it to `<image>` format and extracts or reuses cached frames during dataset preprocessing.

### Special Tokens

| Token | Description |
|-------|-------------|
| `</Silence>` | No relevant event has started, or the current input is irrelevant to the question. |
| `</Standby>` | The event is in progress, or the current information is not yet sufficient to answer. |
| `</Response>` | The event has finished, or the available information is now sufficient to answer. |

### Advanced: Multiple Datasets

The underlying `swift` loader supports passing multiple datasets to `--dataset`, but the current repo wrapper registers only one dataset name, `streaming_video`, by default.

For this repo, the recommended default for mixed-source training is still:

- merge all raw or converted samples into one combined stream-format JSON,
- register that combined file as `streaming_video`,
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

### Validation and Troubleshooting

- Missing video files are skipped during conversion or preprocessing.
- Frame count is checked against the number of `<stream>` tokens in `messages`.
- If the difference is within tolerance, preprocessing truncates extra frames or duplicates the last frame.
- If the difference exceeds tolerance, the sample is discarded.
- Keep the conversion FPS and the registered dataset FPS aligned. A mismatch can produce frame-count errors or incorrect timing.
- With `save_frames=True`, the first run extracts frames to disk and later runs reuse the cached frame directory.
- If you replace the dataset with a new version, make sure the cache directory matches that dataset version.

## Quick Start▶️

After updating `swift/plugin/streaming_dataset.py` to point at your converted dataset:

```bash
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
