#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert streaming video data between formats.

Supports two conversion modes:
1. to-stream: Convert format.json to videos+<stream> multi-round format
2. to-image: Convert videos+<stream> format to images+<image> format (with frame extraction)

Usage:
    # Convert from format.json to stream format (multi-round with <stream> tokens)
    python scripts/convert_streaming_video.py to-stream \
        --input format.json \
        --output stream_format.json \
        --fps 1.0

    # Convert from stream format to image format (extract frames, replace <stream> with <image>)
    python scripts/convert_streaming_video.py to-image \
        --input stream_format.json \
        --output converted_output.json \
        --fps 1.0 \
        --frame-dir /path/to/frames

Input format (format.json):
[
    {
        "video_name": "video1",
        "video_path": "/path/to/video.mp4",
        "task_type": "event_detection",
        "source": "custom",
        "question": [
            {"content": "Find all significant events", "time": "0"}
        ],
        "response": [
            {"content": "A man walks in", "st_time": "0", "end_time": "5"},
            {"content": "He picks up a book", "time": "10"}
        ]
    }
]

Stream format output (with <stream> tokens):
{
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "Question<stream>"},
        {"role": "assistant", "content": "</Silence>"},
        {"role": "user", "content": "<stream>"},
        {"role": "assistant", "content": "</Response> ..."},
        ...
    ],
    "videos": ["/path/to/video.mp4"]
}

Image format output (with <image> tokens):
{
    "messages": [...],  // same structure, but <stream> -> <image>
    "images": ["/path/to/frame_0.png", ...]
}
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from streamo_utils.streaming import (
    SAMPLING_VERSION, SYSTEM_PROMPT, interval_index, parse_seconds,
    sampling_indices, time_tag, validate_fps,
)

IMAGE_TOKEN = '<image>'
STREAM_TOKEN = '<stream>'
DEFAULT_SYSTEM_PROMPT = SYSTEM_PROMPT
STATE_SILENCE = "</Silence>"
STATE_RESPONSE = "</Response>"


def get_video_metadata(video_path: str) -> Tuple[int, float]:
    import cv2
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        validate_fps(source_fps)
        if total_frames <= 0:
            raise ValueError(f"No frames in video: {video_path}")
        return total_frames, source_fps
    finally:
        cap.release()


def get_video_frame_count(video_path: str, fps: float = 1.0) -> int:
    total_frames, source_fps = get_video_metadata(video_path)
    return len(sampling_indices(total_frames, source_fps, fps))


def parse_time(time_str: Union[str, int, float, None], fps: float = 1.0) -> Optional[int]:
    """Parse seconds to the zero-based turn ending at or after this time."""
    seconds = parse_seconds(time_str)
    return None if seconds is None else interval_index(seconds, fps)


def build_frame_states(
    num_frames: int,
    questions: List[Dict[str, Any]],
    responses: List[Dict[str, Any]],
    fps: float = 1.0,
    duration: Optional[float] = None,
) -> Tuple[List[str], Dict[int, str]]:
    """Emit each annotation once, at the end of its containing interval."""
    validate_fps(fps)
    duration = num_frames / fps if duration is None else duration
    question_map = {}
    response_map = {}

    def position(seconds):
        index = interval_index(seconds, fps, duration)
        if index >= num_frames:
            raise ValueError(f'Time {seconds}s has no output frame')
        return index

    for q in questions:
        seconds = parse_seconds(q.get('time'))
        if q.get('content'):
            index = position(0 if seconds is None else seconds)
            question_map[index] = '\n'.join(filter(None, [question_map.get(index), q['content']]))

    for resp in responses:
        content = resp.get('content')
        if not isinstance(content, str) or not content.strip():
            raise ValueError('Response content must not be empty')
        start = parse_seconds(resp.get('st_time'))
        end = parse_seconds(resp.get('end_time'))
        direct = parse_seconds(resp.get('time'))
        if start is not None and end is not None:
            if start > end:
                raise ValueError(f'Event starts after its end: {start} > {end}')
            seconds = end
        elif direct is not None:
            seconds = direct
        else:
            raise ValueError('Response requires st_time/end_time or time')
        response_map.setdefault(position(seconds), []).append(content)

    states = [STATE_SILENCE] * num_frames
    for index, contents in response_map.items():
        states[index] = STATE_RESPONSE + ' ' + '\n'.join(contents)
    return states, question_map


def convert_format_to_stream(
    data: Dict[str, Any],
    fps: float = 1.0,
    system_prompt: Optional[str] = None,
    default_question: str = "Find all significant events in the video and describe them.",
    video_prefix: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Convert from format.json to stream format with <stream> tokens.
    
    Input (format.json):
    {
        "video_path": "/path/to/video.mp4",
        "question": [{"content": "...", "time": "0"}],
        "response": [{"content": "...", "st_time": "0", "end_time": "5"}]
    }
    
    Output:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "Question<stream>"},
            {"role": "assistant", "content": "</Silence>"},
            ...
        ],
        "videos": ["/path/to/video.mp4"]
    }
    """
    validate_fps(fps)
    # Get video path
    video_path = data.get('video_path', '')
    if not video_path:
        videos = data.get('videos', [])
        if videos:
            video_path = videos[0] if isinstance(videos, list) else videos
    
    # Apply video prefix if provided
    if video_prefix and video_path:
        video_path = os.path.join(video_prefix, video_path)
    
    if not video_path or not os.path.exists(video_path):
        print(f"Warning: Video not found: {video_path}")
        return None
    
    # Get frame count
    try:
        total_frames, source_fps = get_video_metadata(video_path)
        duration = total_frames / source_fps
        num_frames = len(sampling_indices(total_frames, source_fps, fps))
    except Exception as e:
        print(f"Warning: Cannot read video {video_path}: {e}")
        return None
    
    if num_frames <= 0:
        print(f"Warning: No frames in video: {video_path}")
        return None
    
    # Get questions and responses
    questions = data.get('question', [])
    responses = data.get('response', [])
    
    # Build frame states
    frame_states, question_map = build_frame_states(
        num_frames, questions, responses, fps, duration
    )
    
    # Only supply a default when there is no timed question. Do not move a
    # future question to the beginning of the video.
    if not questions:
        question_map[0] = default_question

    # Build messages
    messages = []
    
    # System message
    messages.append({
        "role": "system",
        "content": system_prompt or DEFAULT_SYSTEM_PROMPT
    })
    
    # Frame-by-frame messages with <stream> token
    for i in range(num_frames):
        # User message with time format: <Xs-Ys>
        interval_tag = time_tag(i, fps, duration)
        if i in question_map:
            user_content = f"{question_map[i]}\n{interval_tag}\n{STREAM_TOKEN}"
        else:
            user_content = f"{interval_tag}\n{STREAM_TOKEN}"
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        # Assistant message
        messages.append({
            "role": "assistant",
            "content": frame_states[i]
        })
    
    result = {
        "messages": messages,
        "videos": [video_path],
        "streaming_metadata": {
            "fps": fps,
            "sampling": SAMPLING_VERSION,
            "state_mode": "silence_response",
            "duration": duration,
            "annotation_count": len(responses),
            "response_frames": sum(state != STATE_SILENCE for state in frame_states),
        },
    }
    
    # Preserve metadata
    for key in ['video_name', 'task_type', 'source']:
        if key in data and data[key]:
            result[key] = data[key]
    
    return result


def convert_stream_to_image(
    data: Dict[str, Any],
    fps: float = 1.0,
    frame_output_dir: Optional[str] = None,
    max_frames: Optional[int] = None,
    max_width: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Convert from stream format to image format (extract frames).
    
    Replaces <stream> with <image> and extracts video frames.
    """
    from swift.llm.dataset.preprocessor import StreamingVideoPreprocessor
    
    preprocessor = StreamingVideoPreprocessor(
        fps=fps,
        max_frames=max_frames,
        frame_output_dir=frame_output_dir,
        use_frame_cache=True,
        save_frames=True,
        max_width=max_width,
    )
    
    return preprocessor.preprocess(data)


def _convert_to_stream_worker(
    args: Tuple[int, Dict[str, Any], float, Optional[str], Optional[str]]
) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Worker function for parallel to-stream conversion."""
    idx, item, fps, system_prompt, video_prefix = args
    result = convert_format_to_stream(
        item,
        fps=fps,
        system_prompt=system_prompt,
        video_prefix=video_prefix,
    )
    return idx, result


def batch_convert_to_stream(
    input_path: str,
    output_path: str,
    fps: float = 1.0,
    system_prompt: Optional[str] = None,
    video_prefix: Optional[str] = None,
    num_workers: int = 8,
) -> List[Dict[str, Any]]:
    """
    Batch convert from format.json to stream format with multi-threading.
    """
    from tqdm import tqdm
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle single object vs list
    is_single = isinstance(data, dict)
    data_list = [data] if is_single else data
    
    # Prepare worker arguments
    worker_args = [
        (idx, item, fps, system_prompt, video_prefix)
        for idx, item in enumerate(data_list)
    ]
    
    # Use thread pool for parallel processing
    results_dict = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_convert_to_stream_worker, args): args[0]
            for args in worker_args
        }
        
        with tqdm(total=len(data_list), desc="Converting to stream format") as pbar:
            for future in as_completed(futures):
                idx, result = future.result()
                if result:
                    results_dict[idx] = result
                pbar.update(1)
    
    # Sort results by original index to maintain order
    results = [results_dict[i] for i in sorted(results_dict.keys())]
    
    # Output
    output_data = results[0] if is_single and len(results) == 1 else results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nConverted {len(results)} samples to stream format")
    print(f"Output: {output_path}")
    
    # Statistics
    if results:
        total_rounds = sum(
            sum(1 for m in r['messages'] if m['role'] == 'user')
            for r in results
        )
        print(f"Total frames/rounds: {total_rounds}")
    
    return results


def _convert_to_image_worker(
    args: Tuple[int, Dict[str, Any], float, Optional[str], Optional[int], Optional[int]]
) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Worker function for parallel to-image conversion."""
    idx, item, fps, frame_output_dir, max_frames, max_width = args
    result = convert_stream_to_image(
        item,
        fps=fps,
        frame_output_dir=frame_output_dir,
        max_frames=max_frames,
        max_width=max_width,
    )
    return idx, result


def batch_convert_to_image(
    input_path: str,
    output_path: str,
    fps: float = 1.0,
    frame_output_dir: Optional[str] = None,
    max_frames: Optional[int] = None,
    num_workers: int = 8,
    max_width: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Batch convert from stream format to image format with multi-threading.
    """
    from tqdm import tqdm
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle single object vs list
    is_single = isinstance(data, dict)
    data_list = [data] if is_single else data
    
    # Prepare worker arguments
    worker_args = [
        (idx, item, fps, frame_output_dir, max_frames, max_width)
        for idx, item in enumerate(data_list)
    ]
    
    # Use thread pool for parallel processing
    results_dict = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_convert_to_image_worker, args): args[0]
            for args in worker_args
        }
        
        with tqdm(total=len(data_list), desc="Converting to image format") as pbar:
            for future in as_completed(futures):
                idx, result = future.result()
                if result:
                    results_dict[idx] = result
                pbar.update(1)
    
    # Sort results by original index to maintain order
    results = [results_dict[i] for i in sorted(results_dict.keys())]
    
    # Output
    output_data = results[0] if is_single and len(results) == 1 else results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nConverted {len(results)} samples to image format")
    print(f"Output: {output_path}")
    
    # Statistics
    if results:
        total_frames = sum(len(r.get('images', [])) for r in results)
        print(f"Total frames extracted: {total_frames}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Convert streaming video data between formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Conversion mode')
    
    # to-stream: Convert format.json to stream format
    stream_parser = subparsers.add_parser(
        'to-stream',
        help='Convert from format.json to videos+<stream> multi-round format'
    )
    stream_parser.add_argument('--input', '-i', required=True, help='Input JSON file (format.json)')
    stream_parser.add_argument('--output', '-o', required=True, help='Output JSON file')
    stream_parser.add_argument('--fps', type=float, default=1.0, help='FPS for frame counting')
    stream_parser.add_argument('--system-prompt', help='Custom system prompt')
    stream_parser.add_argument('--video-prefix', help='Prefix path to prepend to video_path')
    stream_parser.add_argument('--num-workers', type=int, default=8, help='Number of parallel workers')
    
    # to-image: Convert stream format to image format
    image_parser = subparsers.add_parser(
        'to-image',
        help='Convert from videos+<stream> to images+<image> format (extract frames)'
    )
    image_parser.add_argument('--input', '-i', required=True, help='Input JSON file (stream format)')
    image_parser.add_argument('--output', '-o', required=True, help='Output JSON file')
    image_parser.add_argument('--fps', type=float, default=1.0, help='FPS for frame extraction')
    image_parser.add_argument('--frame-dir', help='Directory to save extracted frames')
    image_parser.add_argument('--max-frames', type=int, help='Maximum frames per video')
    image_parser.add_argument('--max-width', type=int, default=640,
                              help='Maximum saved frame width, preserving aspect ratio; 0 disables resizing')
    image_parser.add_argument('--num-workers', type=int, default=8, help='Number of parallel workers')
    
    args = parser.parse_args()
    if args.command == 'to-image' and args.max_width < 0:
        image_parser.error('--max-width must be non-negative (0 disables resizing)')
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'to-stream':
        batch_convert_to_stream(
            input_path=args.input,
            output_path=args.output,
            fps=args.fps,
            system_prompt=args.system_prompt,
            video_prefix=args.video_prefix,
            num_workers=args.num_workers,
        )
    elif args.command == 'to-image':
        batch_convert_to_image(
            input_path=args.input,
            output_path=args.output,
            fps=args.fps,
            frame_output_dir=args.frame_dir,
            max_frames=args.max_frames,
            max_width=args.max_width,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
