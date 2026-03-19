# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Register streaming video dataset with preprocessor.

This plugin registers the streaming video dataset so that:
1. <stream> tokens are replaced with <image> tokens
2. Videos are extracted to frames (with disk caching for efficiency)
3. Data is processed as multi-image input (not video)

Performance optimizations:
- save_frames=True: Frames are cached on disk, avoiding repeated video decoding
- enable_memory_cache=True: LRU cache for in-memory frames in current epoch
- Optimized seek-based frame extraction (only reads target frames, not all frames)
"""

import os

from swift.llm.dataset.dataset import ArchiveVideoResolver, register_streaming_video_dataset


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


dataset_path = os.getenv('STREAM_DATASET_PATH', './dataset/stream/llava.jsonl')
frame_output_dir = os.getenv('STREAM_FRAME_CACHE_DIR', './dataset/stream/frames')

video_resolver = None
if _env_flag('STREAM_ENABLE_ARCHIVE_RESOLUTION', default=False):
    index_path = os.getenv('STREAM_ARCHIVE_INDEX_PATH')
    if index_path:
        video_resolver = ArchiveVideoResolver(
            index_path=index_path,
            archive_cache_dir=os.getenv('STREAM_ARCHIVE_CACHE_DIR', './dataset/stream/archive_cache'),
            video_cache_dir=os.getenv('STREAM_VIDEO_CACHE_DIR', './dataset/stream/video_cache'),
            gcs_prefix=os.getenv('STREAM_ARCHIVE_GCS_PREFIX') or None,
        )
    else:
        print('STREAM_ENABLE_ARCHIVE_RESOLUTION=1 but STREAM_ARCHIVE_INDEX_PATH is not set; '
              'falling back to local video paths.')

# Register the streaming video dataset with optimized settings
# save_frames=True enables disk caching - first run extracts frames, subsequent runs load from disk
register_streaming_video_dataset(
    dataset_path=dataset_path,
    dataset_name='streaming_video',
    fps=1.0,
    max_frames=None,  # No limit
    save_frames=True,  # Enable disk caching for efficiency (recommended for multi-epoch training)
    frame_output_dir=frame_output_dir,  # Frame cache directory
    enable_memory_cache=True,  # Enable in-memory LRU cache
    video_resolver=video_resolver,
)

print(f"Registered streaming_video dataset with optimized frame extraction "
      f"(dataset_path={dataset_path}, frame_output_dir={frame_output_dir}, "
      f"archive_resolution={'enabled' if video_resolver is not None else 'disabled'})")
