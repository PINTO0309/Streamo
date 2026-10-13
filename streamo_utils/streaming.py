"""The shared interval-end time convention for streaming video.

Each turn covers [i / fps, (i + 1) / fps), clipped to the video duration.
It observes the latest source frame before that interval ends and answers at
the end of the interval. Low-FPS sources repeat the latest available frame.
"""

import math
from typing import Optional, Union


SAMPLING_VERSION = 'interval_end_v1'
STATE_TOKENS = ('</Silence>', '</Response>')
SYSTEM_PROMPT = """You are a helpful assistant specializing in streaming video analysis.
Each input represents a completed time interval, such as <0s-0.25s>,
and contains the latest available frame before the end of that interval.

Use </Silence> when there is nothing new to report, including when an event
is still in progress or the available information is insufficient to answer.
Use </Response> followed by a complete description when an event has concluded
or sufficient information is available to answer. Report each event once.
If several events become reportable in the same interval, describe all of them.
Do not provide partial answers or speculate beyond the available information.
"""


def validate_fps(fps: float) -> float:
    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f'FPS must be finite and positive: {fps}')
    return fps


def parse_seconds(value: Union[str, int, float, None]) -> Optional[float]:
    """Parse seconds uniformly; missing values are distinct from invalid ones."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise ValueError(f'Invalid time: {value!r}')
    try:
        parts = str(value).strip().split(':')
        if len(parts) not in (1, 2, 3):
            raise ValueError()
        values = [float(part) for part in parts]
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError()
        if len(values) > 1 and any(v >= 60 for v in values[1:]):
            raise ValueError()
        seconds = sum(v * 60 ** i for i, v in enumerate(reversed(values)))
        if not math.isfinite(seconds):
            raise ValueError()
        return seconds
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f'Invalid time: {value!r}') from exc


def stable_ceil(value: float) -> int:
    nearest = round(value)
    if abs(value - nearest) <= 1e-9:
        return nearest
    return math.ceil(value)


def interval_index(seconds: float, fps: float, duration: Optional[float] = None) -> int:
    validate_fps(fps)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f'Invalid time: {seconds}')
    if duration is not None and seconds > duration + 1e-9:
        raise ValueError(f'Time {seconds}s exceeds video duration {duration}s')
    return max(0, stable_ceil(seconds * fps) - 1)


def sampling_indices(total_frames: int, source_fps: float, fps: float):
    """Return CFR source indices, including the last partial target interval."""
    validate_fps(source_fps)
    validate_fps(fps)
    if total_frames <= 0:
        raise ValueError(f'Invalid source frame count: {total_frames}')
    duration = total_frames / source_fps
    count = stable_ceil(duration * fps)
    return [
        min(total_frames - 1, max(0, stable_ceil(min((i + 1) / fps, duration) * source_fps) - 1))
        for i in range(count)
    ]


def format_seconds(seconds: float) -> str:
    return f'{seconds:.9f}'.rstrip('0').rstrip('.')


def time_tag(index: int, fps: float, duration: Optional[float] = None) -> str:
    validate_fps(fps)
    end = (index + 1) / fps
    if duration is not None:
        end = min(end, duration)
    return f'<{format_seconds(index / fps)}s-{format_seconds(end)}s>'


def iter_sampled_frames(cap, indices, *, seek: bool = True):
    """Read exact requested indices; fail on decode errors instead of truncating.

Fall back to sequential decoding if the backend cannot seek to an exact index.
Repeated source indices (upsampling) return copies of the last decoded frame.
"""
    import cv2

    previous_index = -1
    previous_frame = None
    next_index = 0
    if not seek:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for index in indices:
        if index == previous_index:
            yield previous_frame.copy()
            continue
        if seek:
            ok = cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            if not ok or abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - index) > 0.5:
                seek = False
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                next_index = 0
        if not seek:
            while next_index < index:
                ok, _ = cap.read()
                if not ok:
                    raise ValueError(f'Cannot decode source frame {next_index}')
                next_index += 1
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f'Cannot decode source frame {index}')
        next_index = index + 1
        previous_index, previous_frame = index, frame
        yield frame
