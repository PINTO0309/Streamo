import os


DEFAULT_GENERATED_DATASET_PATH = './dataset/stream/stream_format.json'
DEFAULT_EXAMPLE_DATASET_PATH = './dataset/example/stream_format.json'
DEFAULT_FRAME_OUTPUT_DIR = './dataset/stream/frames'
DEFAULT_DATASET_FPS = 1.0
DEFAULT_ARCHIVE_CACHE_DIR = './dataset/stream/archive_cache'
DEFAULT_VIDEO_CACHE_DIR = './dataset/stream/video_cache'
DEFAULT_ENABLE_MEMORY_CACHE = False


def _env_get(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ''


def _env_flag(default: bool, *names: str) -> bool:
    value = _env_get(*names)
    if not value:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


def resolve_streaming_dataset_path() -> str:
    dataset_path = _env_get('STREAMING_DATASET_PATH', 'STREAM_DATASET_PATH')
    if dataset_path:
        return dataset_path
    if os.path.exists(DEFAULT_GENERATED_DATASET_PATH):
        return DEFAULT_GENERATED_DATASET_PATH
    return DEFAULT_EXAMPLE_DATASET_PATH


def resolve_streaming_frame_dir() -> str:
    return _env_get('STREAMING_FRAME_DIR', 'STREAM_FRAME_CACHE_DIR') or DEFAULT_FRAME_OUTPUT_DIR


def resolve_streaming_dataset_fps() -> float:
    value = _env_get('STREAMING_DATASET_FPS', 'STREAM_DATASET_FPS')
    return float(value or str(DEFAULT_DATASET_FPS))


def resolve_streaming_enable_archive_resolution() -> bool:
    return _env_flag(False, 'STREAMING_ENABLE_ARCHIVE_RESOLUTION', 'STREAM_ENABLE_ARCHIVE_RESOLUTION')


def resolve_streaming_archive_index_path() -> str:
    return _env_get('STREAMING_ARCHIVE_INDEX_PATH', 'STREAM_ARCHIVE_INDEX_PATH')


def resolve_streaming_archive_gcs_prefix() -> str:
    return _env_get('STREAMING_ARCHIVE_GCS_PREFIX', 'STREAM_ARCHIVE_GCS_PREFIX')


def resolve_streaming_archive_cache_dir() -> str:
    return _env_get('STREAMING_ARCHIVE_CACHE_DIR', 'STREAM_ARCHIVE_CACHE_DIR') or DEFAULT_ARCHIVE_CACHE_DIR


def resolve_streaming_video_cache_dir() -> str:
    return _env_get('STREAMING_VIDEO_CACHE_DIR', 'STREAM_VIDEO_CACHE_DIR') or DEFAULT_VIDEO_CACHE_DIR


def resolve_streaming_enable_memory_cache() -> bool:
    return _env_flag(DEFAULT_ENABLE_MEMORY_CACHE, 'STREAMING_ENABLE_MEMORY_CACHE', 'STREAM_ENABLE_MEMORY_CACHE')
