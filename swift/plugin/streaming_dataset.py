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

from swift.llm.dataset.dataset import ArchiveVideoResolver, register_streaming_video_dataset
from swift.plugin.streaming_dataset_config import (
    resolve_streaming_archive_cache_dir,
    resolve_streaming_archive_gcs_prefix,
    resolve_streaming_archive_index_path,
    resolve_streaming_dataset_fps,
    resolve_streaming_dataset_path,
    resolve_streaming_enable_archive_resolution,
    resolve_streaming_enable_memory_cache,
    resolve_streaming_frame_dir,
    resolve_streaming_video_cache_dir,
)


DATASET_PATH = resolve_streaming_dataset_path()
FRAME_OUTPUT_DIR = resolve_streaming_frame_dir()
DATASET_FPS = resolve_streaming_dataset_fps()
ENABLE_MEMORY_CACHE = resolve_streaming_enable_memory_cache()

video_resolver = None
if resolve_streaming_enable_archive_resolution():
    index_path = resolve_streaming_archive_index_path()
    if index_path:
        video_resolver = ArchiveVideoResolver(
            index_path=index_path,
            archive_cache_dir=resolve_streaming_archive_cache_dir(),
            video_cache_dir=resolve_streaming_video_cache_dir(),
            gcs_prefix=resolve_streaming_archive_gcs_prefix() or None,
        )
    else:
        print('Archive resolution is enabled but no archive index path is set; '
              'falling back to local video paths.')


# Register the streaming video dataset with optimized settings.
# save_frames=True enables disk caching: first run extracts frames, subsequent runs load from disk.
register_streaming_video_dataset(
    dataset_path=DATASET_PATH,
    dataset_name='streaming_video',
    fps=DATASET_FPS,
    max_frames=None,  # No limit
    save_frames=True,  # Enable disk caching for efficiency (recommended for multi-epoch training)
    frame_output_dir=FRAME_OUTPUT_DIR,  # Frame cache directory
    enable_memory_cache=ENABLE_MEMORY_CACHE,
    video_resolver=video_resolver,
)

print(
    'Registered streaming_video dataset with optimized frame extraction '
    f'(dataset_path={DATASET_PATH}, frame_output_dir={FRAME_OUTPUT_DIR}, fps={DATASET_FPS}, '
    f'memory_cache={"enabled" if ENABLE_MEMORY_CACHE else "disabled"}, '
    f'archive_resolution={"enabled" if video_resolver is not None else "disabled"})')
