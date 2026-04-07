import argparse
import json
import os
import time
from pathlib import Path
from typing import IO, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


os.environ.setdefault('MIN_PIXELS', '3136')
os.environ.setdefault('MAX_PIXELS', '100352')


SYSTEM_PROMPT = """
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

Do not provide partial answers or speculate beyond the given information.
Whenever you deliver an answer, begin with </Response>.
""".strip()

DEFAULT_CAPTION_QUESTION = 'Detect and summarize each event sequence in the video.'
STATE_SILENCE = '</Silence>'
STATE_STANDBY = '</Standby>'
STATE_RESPONSE = '</Response>'
RESPONSE_PREFIXES = (STATE_RESPONSE, STATE_STANDBY, STATE_SILENCE)
DEFAULT_SUBTITLE_MAX_LINES = 4
SUBTITLE_FONT_CANDIDATES = (
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc',
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/noto/NotoSerifCJK-Regular.ttc',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
)


def str2bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if value in {'0', 'false', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


def format_seconds(seconds: float) -> str:
    if abs(seconds - round(seconds)) < 1e-9:
        return str(int(round(seconds)))
    return f'{seconds:.3f}'.rstrip('0').rstrip('.')


def parse_response(response: str) -> Tuple[str, str]:
    for prefix in RESPONSE_PREFIXES:
        if response.startswith(prefix):
            return prefix, response[len(prefix):].lstrip()
    return 'other', response


class VideoFrameExtractor:

    def __init__(self, video_path: str, target_fps: float = 1.0):
        self.video_path = video_path
        self.target_fps = target_fps
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f'Cannot open video: {video_path}')

        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.original_fps <= 0:
            raise ValueError(f'Invalid FPS for video: {video_path}')
        self.duration = self.total_frames / self.original_fps
        self.num_extracted_frames = max(1, int(self.duration * self.target_fps))

    def get_frame_at_time(self, time_sec: float) -> Image.Image:
        frame_idx = min(int(time_sec * self.original_fps), self.total_frames - 1)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = self.cap.read()
        if not success:
            raise ValueError(f'Cannot read frame at time {time_sec}s (frame {frame_idx})')
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    def get_frame_at_round(self, round_num: int) -> Image.Image:
        time_sec = round_num / self.target_fps
        return self.get_frame_at_time(time_sec)

    def get_total_rounds(self) -> int:
        return self.num_extracted_frames

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __del__(self):
        self.close()


def make_user_content(round_num: int, fps: float, question: str, include_question: bool) -> str:
    start_sec = round_num / fps
    end_sec = (round_num + 1) / fps
    time_tag = f'<{format_seconds(start_sec)}s-{format_seconds(end_sec)}s>\n<image>'
    if include_question:
        return f'{question}\n{time_tag}'
    return time_tag


def build_stream_window(
    *,
    video_extractor: VideoFrameExtractor,
    round_num: int,
    fps: float,
    system: str,
    question: str,
    question_time: int,
    data: Optional[Dict] = None,
    answer: Optional[str] = None,
    max_rounds: int = 120,
    global_question: bool = True,
) -> Dict:
    frame = video_extractor.get_frame_at_round(round_num)

    if data is None:
        if round_num != 0:
            raise ValueError('round_num must be 0 when data is None')
        messages = [{'role': 'system', 'content': system}]
        include_question = global_question or (round_num == question_time)
        messages.append({
            'role': 'user',
            'content': make_user_content(round_num, fps, question, include_question),
        })
        return {'images': [frame], 'messages': messages}

    messages = data['messages']
    if answer is not None:
        messages.append({'role': 'assistant', 'content': answer})
    include_question = round_num == question_time
    messages.append({
        'role': 'user',
        'content': make_user_content(round_num, fps, question, include_question),
    })
    data['images'].append(frame)

    if len(data['images']) > max_rounds:
        rounds_to_remove = len(data['images']) - max_rounds
        start_round = round_num - max_rounds + 1

        new_messages = messages[:1]
        messages_to_skip = rounds_to_remove * 2
        new_messages.extend(messages[1 + messages_to_skip:])
        include_question_start = global_question or (question_time == start_round)
        new_messages[1] = {
            'role': 'user',
            'content': make_user_content(start_round, fps, question, include_question_start),
        }

        data['messages'] = new_messages
        data['images'] = data['images'][rounds_to_remove:]

    return data


def load_engine(
    *,
    model_path: Optional[str],
    adapter_path: Optional[str],
    backend: str,
    window_size: int,
):
    from swift.llm import BaseArguments, PtEngine

    resolved_model_path = model_path
    adapters = None
    if adapter_path:
        if not resolved_model_path:
            args = BaseArguments.from_pretrained(adapter_path)
            resolved_model_path = args.model
            if not resolved_model_path:
                raise ValueError(f'Failed to resolve base model from adapter args.json: {adapter_path}')
        adapters = [adapter_path]

    if not resolved_model_path:
        raise ValueError('Either --model-path or --adapter-path must be provided.')

    if backend == 'pt':
        return PtEngine(resolved_model_path, adapters=adapters, max_batch_size=1), resolved_model_path

    if backend == 'vllm':
        from swift.llm import VllmEngine
        return VllmEngine(
            resolved_model_path,
            adapters=adapters,
            max_model_len=32768,
            limit_mm_per_prompt={'image': max(window_size, 16)},
            tensor_parallel_size=1,
            enable_prefix_caching=True,
        ), resolved_model_path

    raise ValueError(f'Unsupported backend: {backend}')


def run_single_infer(engine, infer_request):
    from swift.llm import RequestConfig

    request_config = RequestConfig(max_tokens=512, temperature=0.0)
    resp_list = engine.infer([infer_request], request_config)
    return resp_list[0].choices[0].message.content


def ensure_parent_dir(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def write_jsonl_record(handle: IO[str], record: Dict) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    handle.flush()


def build_round_record(
    *,
    round_num: int,
    start_sec: float,
    end_sec: float,
    response: str,
    response_type: str,
    response_body: str,
    is_new_event: bool,
) -> Dict:
    return {
        'round': round_num,
        'start_sec': start_sec,
        'end_sec': end_sec,
        'response': response,
        'response_type': response_type,
        'response_body': response_body,
        'is_new_event': is_new_event,
        'subtitle_text': build_subtitle_text(response_type, response_body),
    }


def build_subtitle_text(response_type: str, response_body: str) -> str:
    if response_type != STATE_RESPONSE:
        return ''
    subtitle_text = response_body.strip()
    return subtitle_text or STATE_RESPONSE


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


def subtitle_for_time(
    *,
    time_sec: float,
    fps: float,
    round_records: Sequence[Dict],
    extend_last_subtitle: bool,
) -> str:
    if not round_records:
        return ''
    if time_sec >= round_records[-1]['end_sec'] and not extend_last_subtitle:
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
    extend_last_subtitle: bool,
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
                extend_last_subtitle=extend_last_subtitle,
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
    parser = argparse.ArgumentParser(description='Streaming action_caption demo for Streamo checkpoints.')
    parser.add_argument('--model-path', default=None, help='Full checkpoint / merged model path.')
    parser.add_argument('--adapter-path', default=None, help='Optional LoRA adapter checkpoint path.')
    parser.add_argument('--backend', choices=['pt', 'vllm'], default='pt', help='Inference backend.')
    parser.add_argument('--video-path', required=True, help='Input video path.')
    parser.add_argument('--mode', choices=['caption', 'qa'], default='caption', help='Prompting mode.')
    parser.add_argument('--question', default=None, help='Question text. Required for --mode qa.')
    parser.add_argument('--fps', type=float, default=1.0, help='Sampling FPS for streaming inference.')
    parser.add_argument('--window-size', type=int, default=120, help='Maximum rounds kept in the sliding window.')
    parser.add_argument('--global-question', type=str2bool, default=True,
                        help='Re-inject the question into the first visible user turn after truncation.')
    parser.add_argument('--realtime', type=str2bool, default=True,
                        help='Sleep between rounds to simulate streaming playback.')
    parser.add_argument('--sleep-sec', type=float, default=1.0, help='Sleep duration used when --realtime true.')
    parser.add_argument('--save-jsonl', default=None, help='Optional JSONL output path for round-level records.')
    parser.add_argument('--save-video', default=None, help='Optional output video path with rendered subtitles.')
    parser.add_argument('--subtitle-font-path', default=None, help='Optional font path for subtitle rendering.')
    parser.add_argument('--subtitle-max-lines', type=int, default=DEFAULT_SUBTITLE_MAX_LINES,
                        help='Maximum subtitle lines rendered into --save-video.')
    parser.add_argument('--stop-on-response', type=str2bool, default=False,
                        help='Stop after the first </Response>.')
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error('--fps must be > 0.')
    if args.window_size <= 0:
        parser.error('--window-size must be > 0.')
    if args.subtitle_max_lines <= 0:
        parser.error('--subtitle-max-lines must be > 0.')
    if args.mode == 'qa' and not args.question:
        parser.error('--question is required when --mode qa.')
    if not args.model_path and not args.adapter_path:
        parser.error('Either --model-path or --adapter-path must be provided.')
    return args


def main():
    args = parse_args()
    question = args.question or DEFAULT_CAPTION_QUESTION
    jsonl_path = ensure_parent_dir(args.save_jsonl)
    video_output_path = ensure_parent_dir(args.save_video)
    jsonl_handle: Optional[IO[str]] = None
    video_extractor = None
    round_records: List[Dict] = []

    try:
        engine, resolved_model_path = load_engine(
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            backend=args.backend,
            window_size=args.window_size,
        )
        video_extractor = VideoFrameExtractor(args.video_path, target_fps=args.fps)

        total_rounds = video_extractor.get_total_rounds()
        if jsonl_path is not None:
            jsonl_handle = jsonl_path.open('w', encoding='utf-8')

        print(f'backend: {args.backend}')
        print(f'model: {resolved_model_path}')
        if args.adapter_path:
            print(f'adapter: {args.adapter_path}')
        print(f'video: {args.video_path}')
        print(f'mode: {args.mode}')
        print(f'question: {question}')
        print(f'fps: {args.fps}')
        print(f'window_size: {args.window_size}')
        print(f'total_rounds: {total_rounds}')
        if jsonl_path is not None:
            print(f'save_jsonl: {jsonl_path}')
        if video_output_path is not None:
            print(f'save_video: {video_output_path}')

        from swift.llm import InferRequest

        data = None
        previous_answer = None
        for round_num in range(total_rounds):
            data = build_stream_window(
                video_extractor=video_extractor,
                round_num=round_num,
                fps=args.fps,
                system=SYSTEM_PROMPT,
                question=question,
                question_time=0,
                data=data,
                answer=previous_answer,
                max_rounds=args.window_size,
                global_question=args.global_question,
            )
            response = run_single_infer(engine, InferRequest(**data))
            response_type, response_body = parse_response(response)
            start_sec = round_num / args.fps
            end_sec = (round_num + 1) / args.fps
            is_new_event = response_type == STATE_RESPONSE and previous_answer != response

            print(
                f'[{round_num:04d}] '
                f'<{format_seconds(start_sec)}s-{format_seconds(end_sec)}s> '
                f'{response}'
            )
            if is_new_event:
                print(f'[event] {response}')

            round_record = build_round_record(
                round_num=round_num,
                start_sec=start_sec,
                end_sec=end_sec,
                response=response,
                response_type=response_type,
                response_body=response_body,
                is_new_event=is_new_event,
            )
            round_records.append(round_record)

            if jsonl_handle is not None:
                write_jsonl_record(jsonl_handle, round_record)

            previous_answer = response
            if args.stop_on_response and response_type == STATE_RESPONSE:
                print('stop_on_response=true: stopping after first </Response>.')
                break
            if args.realtime and round_num < total_rounds - 1:
                time.sleep(args.sleep_sec)

        if video_output_path is not None:
            print(f'rendering subtitle video -> {video_output_path}')
            render_subtitle_video(
                video_path=args.video_path,
                output_path=str(video_output_path),
                fps=args.fps,
                round_records=round_records,
                font_path=args.subtitle_font_path,
                max_lines=args.subtitle_max_lines,
                extend_last_subtitle=not args.stop_on_response,
            )
    finally:
        if jsonl_handle is not None:
            jsonl_handle.close()
        if video_extractor is not None:
            video_extractor.close()


if __name__ == '__main__':
    main()
