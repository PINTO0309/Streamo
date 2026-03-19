# Streamo
<h1 align="center">Streaming Video Instruction Tuning</h1>
<p align="center"><i>A real-time streaming video LLM that serves as a general-purpose interactive assistant.</i></p>

<p align="center">
         📑 <a href="https://arxiv.org/abs/2512.21334">Paper</a>  &nbsp&nbsp 🌐 <a href="https://jiaerxia.github.io/Streamo/">Web</a>
</p>

This is the official implementation of the paper 'Streaming Video Instruction Tuning'.


# News📰
* **`[2026/1/22]`:**🔥**We have released our training code.**
* **`[2026/1/6]`:**🔥**We have released our website with more interesting demos [[Web](https://jiaerxia.github.io/Streamo/)].**
* **`[2025/12/24]`:**🔥**We have released our paper [[Arxiv](https://arxiv.org/abs/2512.21334)].**


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

## Data Format📊

### Raw Data Format

The example raw annotation format in `raw_data.json`:

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

| Field | Description |
|-------|-------------|
| `question.time` | The second when the question appears (e.g., "5" means `<4s-5s>`) |
| `response.st_time` | Start time of the event (standby begins) |
| `response.end_time` | End time of the event |
| `response.time` | Response time for instant response |

### Training Data Format (Stream Format)

The training data uses a multi-turn conversation format, where each turn corresponds to one video frame (1fps):

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

### Data Conversion

Use `scripts/convert_streaming_video.py` to convert raw data to training format:

```bash
# Convert raw_data.json to stream format
python scripts/convert_streaming_video.py to-stream \
--input raw_data.json \
--output stream_format.json \
--video-prefix /path/to/videos \
--fps 1.0
```

`--video-prefix` is the base directory prepended to each `video_path` in the input JSON. Use it when `video_path` is stored as a relative path or filename; for example, if `video_path` is `LLaVA_Video/.../sample.mp4` and `--video-prefix` is `/data`, the script will read `/data/LLaVA_Video/.../sample.mp4`. If `video_path` is already an absolute path, this argument is not needed.

See `dataset/example/` for example files.

### Special Tokens

| Token | Description |
|-------|-------------|
| `</Silence>` | No relevant event or current input is irrelevant |
| `</Standby>` | Event is in progress but not yet completed |
| `</Response>` | Event has completed, start outputting the answer |

### Key Points

- `<stream>` is a placeholder for the current frame, replaced with `<image>` during training
- `<Xs-Ys>` indicates the timestamp interval of the current frame
- Videos are sampled at 1fps, each `<stream>` corresponds to one frame

## Detailed Pipeline Flow

### Local Video Training Flow

When `bash train.sh` is executed in the default setup, the training pipeline works as follows:

1. `train.sh` launches `swift sft` with `--dataset streaming_video`.
2. `swift/plugin/streaming_dataset.py` registers the dataset specified by `STREAM_DATASET_PATH` (default: `./dataset/stream/llava.jsonl`).
3. Each sample is loaded from JSON/JSONL and passed to `StreamingVideoPreprocessor`.
4. The preprocessor reads the first path from `videos`, `video`, or `video_path`.
5. For each `<stream>` token in `messages`, the corresponding video is sampled at 1 fps.
6. `<stream>` is replaced by `<image>`, and the extracted frames are stored in `images`.
7. If `save_frames=True`, extracted frames are cached under `STREAM_FRAME_CACHE_DIR` so repeated epochs do not decode the same video again.
8. The final training sample is passed to Swift as a multi-image training example.

In short, the model trains on `messages + images`, while the original `videos` field is only used to lazily produce the frame sequence.

### GCE/GCS Archive Training Flow

This project also supports training from datasets stored in GCS as split archives such as `.tar.gz.00`, `.tar.gz.01`, ...

The expected flow is:

1. Prepare `llava.jsonl` as usual. The schema does not change; `videos` / `video_path` should still contain the logical relative path of the video inside the archive.
2. Build a SQLite sidecar index from the GCS archive shards:

```bash
python scripts/build_stream_archive_index.py \
  --gcs-prefix gs://your-bucket/path/to/archives \
  --output ./dataset/stream/archive_index.sqlite
```

3. Enable archive resolution before running training:

```bash
export STREAM_ENABLE_ARCHIVE_RESOLUTION=1
export STREAM_ARCHIVE_INDEX_PATH=./dataset/stream/archive_index.sqlite
export STREAM_ARCHIVE_GCS_PREFIX=gs://your-bucket/path/to/archives
export STREAM_ARCHIVE_CACHE_DIR=./dataset/stream/archive_cache
export STREAM_VIDEO_CACHE_DIR=./dataset/stream/video_cache
export STREAM_DATASET_PATH=./dataset/stream/llava.jsonl
export STREAM_FRAME_CACHE_DIR=./dataset/stream/frames
```

4. Run `bash train.sh`.

At training time, each sample is processed in this order:

1. `StreamingVideoPreprocessor` reads the logical path from `videos` / `video_path`.
2. `ArchiveVideoResolver` checks whether that path already exists locally.
3. If not, it looks up the path in `archive_index.sqlite` and finds:
   - which logical archive shard contains the file
   - the member path inside that tar archive
   - the ordered list of split parts such as `.00`, `.01`, ...
4. The required split parts are downloaded from GCS into `STREAM_ARCHIVE_CACHE_DIR/parts/`.
5. The parts are concatenated into a local `.tar.gz` archive under `STREAM_ARCHIVE_CACHE_DIR/archives/`.
6. Only the requested video file is extracted from that archive into `STREAM_VIDEO_CACHE_DIR/`.
7. The extracted local video path is then passed to the normal frame extraction pipeline.
8. Frames are decoded at 1 fps and cached under `STREAM_FRAME_CACHE_DIR/`.
9. `<stream>` tokens are replaced with `<image>`, and Swift receives the resulting `messages + images` sample.

### Cache Behavior

- `STREAM_ARCHIVE_CACHE_DIR` caches downloaded archive parts and reconstructed `.tar.gz` archives.
- `STREAM_VIDEO_CACHE_DIR` caches extracted video files from the archives.
- `STREAM_FRAME_CACHE_DIR` caches per-frame images used during training.
- On repeated access, the pipeline reuses these caches instead of downloading or decoding again.

This means the first epoch may spend time downloading archives, extracting videos, and decoding frames, while later epochs mostly reuse local cache.

## Quick Start▶️

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
