from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from swift.llm import InferEngine, InferRequest

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['MIN_PIXELS'] = '3136'
os.environ['MAX_PIXELS'] = '100352'


# SYSTEM = """
# You are a helpful assistant specializing in streaming video analysis.
# You will receive input frame by frame, each labeled with absolute time intervals
# in the exact format <Xs-Ys> (e.g., <0s-1s>). Follow these rules precisely:

# 1. Use </Silence> when:
#     - No relevant event has started, OR
#     - The current input is irrelevant to the given question.

# 2. Use </Standby> when:
#     - An event is in progress but has not yet completed, OR
#     - The current input is relevant but the question cannot yet be answered.

# 3. Use </Response> only when:
#     - An event has fully concluded, OR
#     - The available information is sufficient to fully answer the question.
#     Provide a complete description at this point.

# Do not provide partial answers or speculate beyond the given information.
# Whenever you deliver an answer, begin with </Response>.
# """

SYSTEM = """
You are a helpful assistant specializing in streaming video analysis.
You will receive input frame by frame, each labeled with absolute time intervals
in the exact format <Xs-Ys> (e.g., <0s-1s>). Follow these rules precisely:

1. Use </Silence> when:
    - No relevant event has started, OR
    - The current input is irrelevant to the given question.

2. Use </Standby> when:
    - An event is in progress but has not yet completed, OR
    - The current input is relevant but the question cannot yet be answered.

3. Use </Response> only when:
    - An event has fully concluded, OR
    - The available information is sufficient to fully answer the question.
    Provide a complete description at this point.

Whenever you deliver an answer, begin with </Response>.
"""

STATE_SILENCE = '</Silence>'
STATE_STANDBY = '</Standby>'
STATE_RESPONSE = '</Response>'
RESPONSE_PREFIXES = (STATE_RESPONSE, STATE_STANDBY, STATE_SILENCE)
DEFAULT_SUBTITLE_MAX_LINES = 4
DEFAULT_QUESTION = 'Detect and summarize each event sequence in the video.'
SUBTITLE_FONT_CANDIDATES = (
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc',
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/noto/NotoSerifCJK-Regular.ttc',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
)


class VideoFrameExtractor:
    """Extract frames from video file at specified fps"""

    def __init__(self, video_path: str, target_fps: float = 1.0):
        """
        Args:
            video_path: Path to the video file
            target_fps: Target frame rate, default 1fps (1 frame per second)
        """
        self.video_path = video_path
        self.target_fps = target_fps
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.original_fps if self.original_fps > 0 else 0

        # Calculate frame extraction interval
        self.frame_interval = int(self.original_fps / self.target_fps)
        self.num_extracted_frames = int(self.duration * self.target_fps)

        print(f"Video info: {video_path}")
        print(f"  Original FPS: {self.original_fps:.2f}")
        print(f"  Total frames: {self.total_frames}")
        print(f"  Duration: {self.duration:.2f}s")
        print(f"  Target FPS: {self.target_fps}")
        print(f"  Frames to extract: {self.num_extracted_frames}")

    def get_frame_at_time(self, time_sec: float) -> Image.Image:
        """Get frame at specified time point"""
        frame_idx = int(time_sec * self.original_fps)
        frame_idx = min(frame_idx, self.total_frames - 1)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()

        if not ret:
            raise ValueError(f"Cannot read frame at time {time_sec}s (frame {frame_idx})")

        # BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    def get_frame_at_round(self, round_num: int) -> Image.Image:
        """Get frame at specified round (each round corresponds to 1 second)"""
        time_sec = round_num  # Each round corresponds to 1 second
        return self.get_frame_at_time(time_sec)

    def get_total_rounds(self) -> int:
        """Get total number of rounds (based on video duration and target fps)"""
        return self.num_extracted_frames

    def close(self):
        """Release video resources"""
        if self.cap is not None:
            self.cap.release()

    def __del__(self):
        self.close()


def infer_single(engine: InferEngine, infer_requests: InferRequest) -> str:
    from swift.llm import RequestConfig
    from swift.plugin import InferStats

    request_config = RequestConfig(max_tokens=512, temperature=0.0)
    metric = InferStats()
    resp_list = engine.infer([infer_requests], request_config, metrics=[metric])
    response = resp_list[0].choices[0].message.content
    return response


def get_data_stream_video(
    video_extractor: VideoFrameExtractor,
    round_num: int,
    system: str = None,
    question: str = None,
    question_time: int = 0,
    data: dict = None,
    answer: str = None
) -> dict:
    """
    Build multi-turn conversation data by extracting frames directly from video

    Args:
        video_extractor: Video frame extractor
        round_num: Current round number
        system: System prompt
        question: Question to ask
        question_time: Time point when question appears
        data: Existing conversation data
        answer: Answer from previous round
    """
    # Get frame for current round
    frame = video_extractor.get_frame_at_round(round_num)

    if data is None:
        data = {}
        if round_num != 0:
            raise ValueError("round_num must be 0 when data is None")
        messages = [
            {'role': 'system', 'content': system},
        ]

        if round_num == question_time:
            messages.append({'role': 'user', 'content': f'{question}\n<{round_num}s-{int(round_num)+1}s>\n<image>'})
        else:
            messages.append({'role': 'user', 'content': f"<{round_num}s-{int(round_num)+1}s>\n<image>"})

        data['images'] = [frame]  # Directly use PIL.Image object
        data['messages'] = messages
        return data
    else:
        messages = data['messages']
        messages.append({'role': 'assistant', 'content': answer})
        if round_num == question_time:
            messages.append({'role': 'user', 'content': f'{question}\n<{round_num}s-{int(round_num)+1}s>\n<image>'})
        else:
            messages.append({'role': 'user', 'content': f"<{round_num}s-{int(round_num)+1}s>\n<image>"})

        data['images'].append(frame)
        data['messages'] = messages
        return data


def get_data_stream_video_window(
    video_extractor: VideoFrameExtractor,
    round_num: int,
    system: str = None,
    question: str = None,
    question_time: int = 0,
    data: dict = None,
    answer: str = None,
    max_rounds: int = 120,
    global_question: bool = False
) -> dict:
    """
    Build multi-turn conversation data by extracting frames directly from video, with sliding window

    Args:
        video_extractor: Video frame extractor
        round_num: Current round number
        system: System prompt
        question: Question to ask
        question_time: Time point when question appears
        data: Existing conversation data
        answer: Answer from previous round
        max_rounds: Maximum number of rounds to keep
        global_question: If True, the first user message after truncation always includes the question

    Returns:
        dict: Conversation data containing 'images' and 'messages'
    """
    # Get frame for current round
    frame = video_extractor.get_frame_at_round(round_num)

    def make_user_content(r: int, include_question: bool = False) -> str:
        time_tag = f"<{r}s-{r + 1}s>\n<image>"
        if include_question and question:
            return f"{question}\n{time_tag}"
        return time_tag

    if data is None:
        # Initialize data
        if round_num != 0:
            raise ValueError("round_num must be 0 when data is None")

        data = {}
        messages = []

        if system:
            messages.append({'role': 'system', 'content': system})

        # Add first user message
        include_q = global_question or (round_num == question_time)
        messages.append({
            'role': 'user',
            'content': make_user_content(round_num, include_q)
        })

        data['images'] = [frame]
        data['messages'] = messages
        return data

    else:
        messages = data['messages']

        # Add answer from previous round
        if answer is not None:
            messages.append({'role': 'assistant', 'content': answer})

        # Add current round's user message
        # When adding normally, only include question when question_time matches (non-truncation scenario)
        include_q = (round_num == question_time)
        messages.append({
            'role': 'user',
            'content': make_user_content(round_num, include_q)
        })

        # Add current frame
        data['images'].append(frame)

        # If exceeding max_rounds, perform sliding window truncation
        if len(data['images']) > max_rounds:
            rounds_to_remove = len(data['images']) - max_rounds

            start_round = round_num - max_rounds + 1

            new_messages = messages[:1]
            messages_to_skip = rounds_to_remove * 2  # Each round has user and assistant messages
            new_messages.extend(messages[1 + messages_to_skip:])

            include_q_start = global_question or (question_time == start_round)
            new_messages[1] = {
                'role': 'user',
                'content': make_user_content(start_round, include_q_start)
            }

            new_images = data['images'][rounds_to_remove:]

            data['messages'] = new_messages
            data['images'] = new_images

        return data


def parse_response(response: str) -> Tuple[str, str]:
    for prefix in RESPONSE_PREFIXES:
        if response.startswith(prefix):
            return prefix, response[len(prefix):].lstrip()
    return 'other', response


def str2bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if value in {'0', 'false', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


def build_subtitle_text(response_type: str, response_body: str) -> str:
    if response_type != STATE_RESPONSE:
        return ''
    subtitle_text = response_body.strip()
    return subtitle_text or STATE_RESPONSE


def ensure_parent_dir(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def load_subtitle_font(font_path: Optional[str], font_size: int):
    candidates: List[str] = []
    if font_path:
        candidates.append(font_path)
    candidates.extend(SUBTITLE_FONT_CANDIDATES)
    for candidate in candidates:
        if not candidate or not Path(candidate).exists():
            continue
        try:
            return ImageFont.truetype(candidate, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_width(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    if not text:
        return 0
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def fit_text_with_ellipsis(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> str:
    if text_width(draw, text, font) <= max_width:
        return text
    ellipsis = '...'
    trimmed = text.rstrip()
    while trimmed and text_width(draw, f'{trimmed}{ellipsis}', font) > max_width:
        trimmed = trimmed[:-1].rstrip()
    return f'{trimmed}{ellipsis}' if trimmed else ellipsis


def wrap_subtitle_text(
    *,
    draw: ImageDraw.ImageDraw,
    text: str,
    font,
    max_width: int,
    max_lines: int,
) -> List[str]:
    if not text:
        return []
    lines: List[str] = []
    paragraphs = text.splitlines() or ['']
    for paragraph_idx, paragraph in enumerate(paragraphs):
        current = ''
        for char in paragraph:
            candidate = current + char
            if current and text_width(draw, candidate, font) > max_width:
                lines.append(current.rstrip())
                if len(lines) >= max_lines:
                    lines[-1] = fit_text_with_ellipsis(draw, lines[-1], font, max_width)
                    return lines
                current = char.lstrip() if char.isspace() else char
            else:
                current = candidate
        if current or not paragraph:
            lines.append(current.rstrip())
            if len(lines) >= max_lines:
                has_remaining = paragraph_idx < len(paragraphs) - 1
                if has_remaining:
                    lines[-1] = fit_text_with_ellipsis(draw, lines[-1], font, max_width)
                return lines
    return [line for line in lines if line]


def draw_subtitle_on_frame(
    frame: np.ndarray,
    subtitle_text: str,
    *,
    font_path: Optional[str],
    max_lines: int,
) -> np.ndarray:
    if not subtitle_text:
        return frame

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb).convert('RGBA')
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    width, height = image.size
    horizontal_margin = max(24, width // 18)
    vertical_margin = max(20, height // 20)
    font_size = max(18, height // 20)
    font = load_subtitle_font(font_path, font_size)
    lines = wrap_subtitle_text(
        draw=draw,
        text=subtitle_text,
        font=font,
        max_width=width - (horizontal_margin * 2),
        max_lines=max_lines,
    )
    if not lines:
        return frame

    line_spacing = max(6, font_size // 4)
    line_heights: List[int] = []
    line_widths: List[int] = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])

    padding_x = max(16, width // 40)
    padding_y = max(12, height // 45)
    text_block_height = sum(line_heights) + line_spacing * max(0, len(lines) - 1)
    box_height = text_block_height + padding_y * 2
    box_top = max(0, height - vertical_margin - box_height)
    box_bottom = min(height, height - vertical_margin)
    draw.rounded_rectangle(
        [horizontal_margin, box_top, width - horizontal_margin, box_bottom],
        radius=max(10, min(width, height) // 60),
        fill=(0, 0, 0, 176),
    )

    y = box_top + padding_y
    stroke_width = max(1, font_size // 18)
    for line, line_width, line_height in zip(lines, line_widths, line_heights):
        x = (width - line_width) / 2
        draw.text(
            (x, y),
            line,
            font=font,
            fill=(255, 255, 255, 255),
            stroke_width=stroke_width,
            stroke_fill=(0, 0, 0, 255),
        )
        y += line_height + line_spacing

    composed = Image.alpha_composite(image, overlay).convert('RGB')
    return cv2.cvtColor(np.array(composed), cv2.COLOR_RGB2BGR)


def video_fourcc_for_path(path: str) -> int:
    suffix = Path(path).suffix.lower()
    return cv2.VideoWriter_fourcc(*('mp4v' if suffix == '.mp4' else 'MJPG'))


def subtitle_for_time(*, time_sec: float, fps: float, round_records: Sequence[Dict]) -> str:
    if not round_records:
        return ''
    if time_sec >= round_records[-1]['end_sec']:
        return ''
    round_idx = min(max(int(time_sec * fps), 0), len(round_records) - 1)
    return round_records[round_idx].get('subtitle_text', '')


def render_subtitle_video(
    *,
    video_path: str,
    output_path: str,
    fps: float,
    round_records: Sequence[Dict],
    font_path: Optional[str],
    max_lines: int,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f'Cannot open video for subtitle rendering: {video_path}')

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if original_fps <= 0:
        cap.release()
        raise ValueError(f'Invalid FPS for video: {video_path}')

    writer = cv2.VideoWriter(output_path, video_fourcc_for_path(output_path), original_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise ValueError(f'Cannot open video writer: {output_path}')

    try:
        frame_idx = 0
        while True:
            success, frame = cap.read()
            if not success:
                break
            subtitle_text = subtitle_for_time(
                time_sec=frame_idx / original_fps,
                fps=fps,
                round_records=round_records,
            )
            if subtitle_text:
                frame = draw_subtitle_on_frame(
                    frame,
                    subtitle_text,
                    font_path=font_path,
                    max_lines=max_lines,
                )
            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()


def parse_args():
    parser = argparse.ArgumentParser(description='Streaming inference demo for Streamo checkpoints.')
    parser.add_argument('--backend', choices=['pt', 'vllm'], default='vllm', help='Inference backend.')
    parser.add_argument('--model-path', default='output/v0-20260402-015200/checkpoint-630', help='Full checkpoint / merged model path.')
    parser.add_argument('--video-path', default='./demo/cook.mp4', help='Input video path.')
    parser.add_argument('--fps', type=float, default=1.0, help='Sampling FPS for streaming inference.')
    parser.add_argument('--save-video', default='./output/cook_output.mp4', help='Output video path with rendered subtitles.')
    parser.add_argument('--subtitle-font-path', default=None, help='Optional font path for subtitle rendering.')
    parser.add_argument('--subtitle-max-lines', type=int, default=DEFAULT_SUBTITLE_MAX_LINES, help='Maximum subtitle lines rendered into --save-video.')
    parser.add_argument('--question', default=DEFAULT_QUESTION, help='Question text for the streaming prompt.')
    parser.add_argument('--global-question', type=str2bool, default=True, help='Re-inject the question into the first visible user turn after truncation.')
    parser.add_argument('--system-prompt', default=SYSTEM, help='System prompt text.')
    parser.add_argument('--question-time', type=int, default=0, help='Round index at which the question becomes active.')
    parser.add_argument('--window-size', type=int, default=32, help='Maximum rounds kept in the sliding window.')
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error('--fps must be > 0.')
    if args.subtitle_max_lines <= 0:
        parser.error('--subtitle-max-lines must be > 0.')
    if args.window_size <= 0:
        parser.error('--window-size must be > 0.')
    if args.question_time < 0:
        parser.error('--question-time must be >= 0.')
    return args


def main():
    from swift.llm import InferRequest, PtEngine
    import json

    args = parse_args()

    infer_backend = args.backend
    model = args.model_path
    video_path = args.video_path
    target_fps = args.fps
    save_video_path = args.save_video
    subtitle_font_path = args.subtitle_font_path
    subtitle_max_lines = args.subtitle_max_lines
    question = args.question
    global_question = args.global_question
    system = args.system_prompt
    question_time = args.question_time
    # vLLM hits the 32k context limit quickly with per-round image inputs.
    # Keep the sliding window conservative so truncation happens before add_request fails.
    max_rounds = args.window_size

    print(f'backend: {infer_backend}')
    print(f'model: {model}')
    print(f'video_path: {video_path}')
    print(f'save_video: {save_video_path}')
    print(f'question: {question}')
    print(f'window_size: {max_rounds}')

    if infer_backend == 'pt':
        engine = PtEngine(model, max_batch_size=64)
    elif infer_backend == 'vllm':
        from swift.llm import VllmEngine
        engine = VllmEngine(
            model,
            max_model_len=32768,
            limit_mm_per_prompt={'image': max(max_rounds, 16)},
            tensor_parallel_size=1,
            enable_prefix_caching=True)

    video_extractor = VideoFrameExtractor(video_path, target_fps=target_fps)

    # Get total number of rounds
    round_num = video_extractor.get_total_rounds()
    print(f"Total rounds: {round_num}")
    print(f"Save subtitle video: {save_video_path or 'disabled'}")

    output = {}
    round_records: List[Dict] = []
    data = None

    for i in range(round_num):
        if i == 0:
            data = get_data_stream_video_window(
                video_extractor=video_extractor,
                round_num=i,
                system=system,
                question=question,
                question_time=question_time,
                max_rounds=max_rounds,
                global_question=global_question
            )
        else:
            data = get_data_stream_video_window(
                video_extractor=video_extractor,
                round_num=i,
                system=system,
                question=question,
                question_time=question_time,
                data=data,
                answer=answer,
                max_rounds=max_rounds,
                global_question=global_question
            )

        if i < question_time:
            answer = STATE_SILENCE
        else:
            infer_request = InferRequest(**data)
            answer = infer_single(engine, infer_request)

        output[f"Round {i}"] = answer
        response_type, response_body = parse_response(answer)
        round_records.append({
            'round': i,
            'start_sec': i,
            'end_sec': i + 1,
            'response': answer,
            'response_type': response_type,
            'response_body': response_body,
            'subtitle_text': build_subtitle_text(response_type, response_body),
        })
        print("=====Round", i, "=====")
        print(f"Answer: {answer}")

    # Close video extractor
    video_extractor.close()

    with open('./test_sample_video.jsonl', 'a') as f:
        result = {
            "video_path": video_path,
            "target_fps": target_fps,
            "output": output
        }
        f.write(json.dumps(result) + '\n')

    video_output_path = ensure_parent_dir(save_video_path)
    if video_output_path is not None:
        if subtitle_max_lines <= 0:
            raise ValueError('subtitle_max_lines must be > 0')
        print(f"Rendering subtitle video to {video_output_path}")
        render_subtitle_video(
            video_path=video_path,
            output_path=str(video_output_path),
            fps=target_fps,
            round_records=round_records,
            font_path=subtitle_font_path,
            max_lines=subtitle_max_lines,
        )
        print(f"Saved subtitle video: {video_output_path}")


if __name__ == '__main__':
    main()
