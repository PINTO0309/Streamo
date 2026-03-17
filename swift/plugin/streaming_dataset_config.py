import os


DEFAULT_GENERATED_DATASET_PATH = './dataset/stream/stream_format.json'
DEFAULT_EXAMPLE_DATASET_PATH = './dataset/example/stream_format.json'
DEFAULT_FRAME_OUTPUT_DIR = './dataset/stream/frames'
DEFAULT_DATASET_FPS = 1.0


def resolve_streaming_dataset_path() -> str:
    dataset_path = os.environ.get('STREAMING_DATASET_PATH')
    if dataset_path:
        return dataset_path
    if os.path.exists(DEFAULT_GENERATED_DATASET_PATH):
        return DEFAULT_GENERATED_DATASET_PATH
    return DEFAULT_EXAMPLE_DATASET_PATH


def resolve_streaming_frame_dir() -> str:
    return os.environ.get('STREAMING_FRAME_DIR', DEFAULT_FRAME_OUTPUT_DIR)


def resolve_streaming_dataset_fps() -> float:
    return float(os.environ.get('STREAMING_DATASET_FPS', str(DEFAULT_DATASET_FPS)))
