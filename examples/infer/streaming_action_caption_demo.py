import argparse
import json
import os
import time
from pathlib import Path
from typing import IO, Dict, Optional, Tuple

import cv2
from PIL import Image


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
    parser.add_argument('--stop-on-response', type=str2bool, default=False,
                        help='Stop after the first </Response>.')
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error('--fps must be > 0.')
    if args.window_size <= 0:
        parser.error('--window-size must be > 0.')
    if args.mode == 'qa' and not args.question:
        parser.error('--question is required when --mode qa.')
    if not args.model_path and not args.adapter_path:
        parser.error('Either --model-path or --adapter-path must be provided.')
    return args


def main():
    args = parse_args()
    question = args.question or DEFAULT_CAPTION_QUESTION
    jsonl_path = ensure_parent_dir(args.save_jsonl)
    jsonl_handle: Optional[IO[str]] = None
    video_extractor = None

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

            if jsonl_handle is not None:
                write_jsonl_record(jsonl_handle, {
                    'round': round_num,
                    'start_sec': start_sec,
                    'end_sec': end_sec,
                    'response': response,
                    'response_type': response_type,
                    'response_body': response_body,
                    'is_new_event': is_new_event,
                })

            previous_answer = response
            if args.stop_on_response and response_type == STATE_RESPONSE:
                print('stop_on_response=true: stopping after first </Response>.')
                break
            if args.realtime and round_num < total_rounds - 1:
                time.sleep(args.sleep_sec)
    finally:
        if jsonl_handle is not None:
            jsonl_handle.close()
        if video_extractor is not None:
            video_extractor.close()


if __name__ == '__main__':
    main()
