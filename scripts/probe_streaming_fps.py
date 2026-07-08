#!/usr/bin/env python3
"""Probe the largest streaming dataset FPS that fits a target max_length."""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from convert_streaming_video import convert_format_to_stream  # noqa: E402


def parse_fps_candidates(value: str) -> List[float]:
    values = []
    for item in re.split(r'[\s,]+', value.strip()):
        if item:
            values.append(float(item))
    if not values:
        raise argparse.ArgumentTypeError('At least one FPS value is required.')
    return values


def fps_label(fps: float) -> str:
    text = f'{fps:g}'.replace('.', 'p')
    return f'fps{text}'


def load_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    if not isinstance(data, list):
        raise TypeError(f'Expected JSON object or list in {path}, got {type(data).__name__}')
    return data


def count_stream_tokens(row: Dict[str, Any]) -> int:
    count = 0
    for message in row.get('messages', []):
        content = message.get('content', '')
        if isinstance(content, str):
            count += content.count('<stream>')
    return count


def count_text_chars(row: Dict[str, Any]) -> int:
    total = 0
    for message in row.get('messages', []):
        content = message.get('content', '')
        if isinstance(content, str):
            total += len(content)
    return total


def build_stream_rows(raw_rows: Sequence[Dict[str, Any]], fps: float) -> Tuple[List[Tuple[int, Dict[str, Any]]], List[int]]:
    converted: List[Tuple[int, Dict[str, Any]]] = []
    failed: List[int] = []
    for index, row in enumerate(raw_rows):
        stream_row = convert_format_to_stream(row, fps=fps)
        if stream_row is None:
            failed.append(index)
        else:
            converted.append((index, stream_row))
    return converted, failed


def select_probe_rows(
    converted: Sequence[Tuple[int, Dict[str, Any]]],
    *,
    top_k: int,
    indices: Optional[Sequence[int]] = None,
) -> List[Tuple[int, Dict[str, Any]]]:
    if indices:
        index_set = set(indices)
        return [(index, row) for index, row in converted if index in index_set]
    ranked = sorted(
        converted,
        key=lambda item: (count_stream_tokens(item[1]), count_text_chars(item[1])),
        reverse=True,
    )
    return ranked if top_k <= 0 else ranked[:top_k]


def init_template(args: argparse.Namespace):
    from swift.llm import TrainArguments

    train_args = TrainArguments(
        model=args.model,
        torch_dtype=args.torch_dtype,
        attn_impl=args.attn_impl,
        max_length=args.measurement_max_length,
        new_special_tokens=args.new_special_tokens,
        train_type='full',
        dataset=['streaming_video'],
        output_dir=args.output_dir,
    )
    model, processor = train_args.get_model_processor(load_model=False, download_model=False)
    template = train_args.get_template(processor)
    template.set_mode('train')
    return template


def measure_lengths(
    template,
    rows: Sequence[Tuple[int, Dict[str, Any]]],
    *,
    fps: float,
    frame_cache_root: Path,
) -> List[Dict[str, Any]]:
    from swift.llm.dataset.preprocessor.streaming_video import StreamingVideoPreprocessor

    cache_dir = frame_cache_root / fps_label(fps)
    preprocessor = StreamingVideoPreprocessor(fps=fps, frame_output_dir=str(cache_dir), save_frames=True)
    measurements = []
    for index, stream_row in rows:
        processed = preprocessor.preprocess(stream_row)
        if processed is None:
            measurements.append({
                'index': index,
                'ok': False,
                'error': 'preprocess_failed',
                'stream_tokens': count_stream_tokens(stream_row),
                'text_chars': count_text_chars(stream_row),
            })
            continue
        try:
            encoded = template.encode(processed, return_length=True)
            measurements.append({
                'index': index,
                'ok': True,
                'length': int(encoded['length']),
                'stream_tokens': count_stream_tokens(stream_row),
                'images': len(processed.get('images', [])),
                'text_chars': count_text_chars(stream_row),
            })
        except Exception as exc:
            measurements.append({
                'index': index,
                'ok': False,
                'error': f'{exc.__class__.__name__}: {exc}',
                'stream_tokens': count_stream_tokens(stream_row),
                'images': len(processed.get('images', [])),
                'text_chars': count_text_chars(stream_row),
            })
    return measurements


def summarize_candidate(
    *,
    fps: float,
    converted: Sequence[Tuple[int, Dict[str, Any]]],
    failed_indices: Sequence[int],
    probe_rows: Sequence[Tuple[int, Dict[str, Any]]],
    measurements: Optional[Sequence[Dict[str, Any]]],
    target_max_length: int,
) -> Dict[str, Any]:
    stream_counts = [count_stream_tokens(row) for _, row in converted]
    text_chars = [count_text_chars(row) for _, row in converted]
    summary: Dict[str, Any] = {
        'fps': fps,
        'rows': len(converted),
        'conversion_failed': len(failed_indices),
        'stream_tokens': {
            'min': min(stream_counts) if stream_counts else None,
            'max': max(stream_counts) if stream_counts else None,
            'avg': sum(stream_counts) / len(stream_counts) if stream_counts else None,
        },
        'text_chars': {
            'max': max(text_chars) if text_chars else None,
        },
        'probe_indices': [index for index, _ in probe_rows],
    }
    if measurements is not None:
        ok_measurements = [m for m in measurements if m.get('ok')]
        failed_measurements = [m for m in measurements if not m.get('ok')]
        max_length = max((m['length'] for m in ok_measurements), default=None)
        summary.update({
            'measured': measurements,
            'measured_max_length': max_length,
            'measured_over_limit': any(m.get('length', 0) > target_max_length for m in ok_measurements),
            'measurement_failed': len(failed_measurements),
            'fits_target_on_measured_rows': (
                bool(ok_measurements)
                and not failed_measurements
                and all(m['length'] <= target_max_length for m in ok_measurements)
            ),
        })
    return summary


def write_stream_format(path: Path, converted: Iterable[Tuple[int, Dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [row for _, row in converted]
    with path.open('w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write('\n')


def parse_indices(values: Optional[Sequence[str]]) -> Optional[List[int]]:
    if not values:
        return None
    indices = []
    for value in values:
        for item in re.split(r'[\s,]+', value.strip()):
            if item:
                indices.append(int(item))
    return indices


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=Path('dataset/stream/raw_resolved.json'))
    parser.add_argument('--fps-candidates', type=parse_fps_candidates, default=parse_fps_candidates('1,2,3,4,6,8,12'))
    parser.add_argument('--target-max-length', type=int, default=32768)
    parser.add_argument('--measurement-max-length', type=int, default=1_000_000)
    parser.add_argument('--top-k', type=int, default=16, help='Measure top-k longest rows by stream token count. 0 means all rows.')
    parser.add_argument('--indices', nargs='*', help='Specific raw row indices to measure, overriding --top-k.')
    parser.add_argument('--no-exact', action='store_true', help='Only compute stream-token counts; skip template.encode.')
    parser.add_argument('--frame-cache-root', type=Path, default=Path('dataset/stream/fps_probe_frames'))
    parser.add_argument('--output-json', type=Path, default=Path('dataset/stream/fps_probe_report.json'))
    parser.add_argument('--write-best-stream-format', type=Path)
    parser.add_argument('--write-candidates-dir', type=Path)
    parser.add_argument('--model', default='Qwen/Qwen3-VL-2B-Instruct')
    parser.add_argument('--torch-dtype', default='bfloat16')
    parser.add_argument('--attn-impl', default=None)
    parser.add_argument('--new-special-tokens', default='./special_token_v1.txt')
    parser.add_argument('--output-dir', default='output/fps_probe')
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    raw_rows = load_rows(args.raw)
    indices = parse_indices(args.indices)
    template = None if args.no_exact else init_template(args)
    summaries = []
    converted_by_fps: Dict[float, List[Tuple[int, Dict[str, Any]]]] = {}

    for fps in args.fps_candidates:
        converted, failed_indices = build_stream_rows(raw_rows, fps)
        converted_by_fps[fps] = converted
        probe_rows = select_probe_rows(converted, top_k=args.top_k, indices=indices)
        measurements = None
        if template is not None:
            measurements = measure_lengths(
                template,
                probe_rows,
                fps=fps,
                frame_cache_root=args.frame_cache_root,
            )
        summary = summarize_candidate(
            fps=fps,
            converted=converted,
            failed_indices=failed_indices,
            probe_rows=probe_rows,
            measurements=measurements,
            target_max_length=args.target_max_length,
        )
        summaries.append(summary)
        max_streams = summary['stream_tokens']['max']
        measured = summary.get('measured_max_length')
        status = 'fit' if summary.get('fits_target_on_measured_rows') else 'over/unknown'
        print(f"fps={fps:g} rows={len(converted)} max_streams={max_streams} measured_max={measured} {status}")

        if args.write_candidates_dir:
            write_stream_format(args.write_candidates_dir / f'stream_format_{fps_label(fps)}.json', converted)

    exact_summaries = [s for s in summaries if 'fits_target_on_measured_rows' in s]
    passing = [s for s in exact_summaries if s['fits_target_on_measured_rows']]
    best = max(passing, key=lambda item: item['fps']) if passing else None
    report = {
        'raw': str(args.raw),
        'target_max_length': args.target_max_length,
        'top_k': args.top_k,
        'indices': indices,
        'best_measured_fps': best['fps'] if best else None,
        'candidates': summaries,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'wrote report: {args.output_json}')

    if best and args.write_best_stream_format:
        write_stream_format(args.write_best_stream_format, converted_by_fps[best['fps']])
        print(f"wrote best stream-format: {args.write_best_stream_format} (fps={best['fps']:g})")


if __name__ == '__main__':
    main()
