#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepare a training-ready stream-format dataset from local Streamo-Instruct labels
and partially downloaded video folders.

Workflow:
1. Scan label JSON files under the label root.
2. Resolve each sample's local video path from the media root.
3. Drop unresolved or missing-video samples with a JSON report.
4. Merge the remaining samples into one raw JSON file.
5. Convert the merged raw JSON to stream format.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from convert_streaming_video import convert_format_to_stream  # noqa: E402


VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
KNOWN_SOURCES = {
    'ActivityNet',
    'LLaVA_Video',
    'QVHighlight',
    'Youcook',
    'Youcookv2',
    'coin',
    'didemo',
    'ego_timeqa',
    'how_to_caption',
    'how_to_step',
    'queryd',
    'tacos',
}


def str2bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if value in {'0', 'false', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


def normalize_filter(values: Optional[Sequence[str]]) -> Optional[set]:
    if not values:
        return None
    return {value.strip().lower() for value in values if value and value.strip()}


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def iter_label_files(label_root: Path) -> List[Path]:
    return sorted(path for path in label_root.rglob('*.json') if path.is_file())


def iter_index_roots(media_root: Path) -> List[Path]:
    candidates = [
        media_root / 'coin' / 'videos',
        media_root / 'activitynet' / 'videos',
        media_root / 'QVHighlight' / 'videos',
        media_root / 'Queryd' / 'videos',
        media_root / 'didemo' / 'videos',
        media_root / 'tacos' / 'videos',
        media_root / 'youcook2' / 'videos',
        media_root / 'ego4d' / 'videos_3fps_480_noaudio',
        media_root / 'how_to_caption' / 'how_to_caption',
        media_root / 'how_to_step',
        media_root / 'LLaVA_Video',
    ]
    unique_paths = []
    seen = set()
    for path in candidates:
        if path.exists() and path not in seen:
            unique_paths.append(path)
            seen.add(path)
    return unique_paths


def build_basename_index(media_root: Path) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = defaultdict(list)
    seen_paths = set()
    for root in iter_index_roots(media_root):
        for path in root.rglob('*'):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            resolved = str(path.resolve())
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            index[path.name].append(resolved)
    return dict(index)


def _fallback_matches(
    basename: str,
    basename_index: Optional[Dict[str, List[str]]],
    *,
    required_prefix: Optional[Path] = None,
) -> List[str]:
    if not basename_index or not basename:
        return []
    matches = basename_index.get(basename, [])
    if required_prefix is None:
        return matches
    prefix = str(required_prefix.resolve())
    return [match for match in matches if match.startswith(prefix)]


def _source_candidates(source: str, video_path: str, media_root: Path) -> Tuple[List[Path], Optional[str], Optional[Path]]:
    basename = Path(video_path).name
    stem = Path(video_path).stem if Path(video_path).suffix else Path(video_path).name
    llava_root = media_root / 'LLaVA_Video'

    if source == 'coin':
        return [media_root / 'coin' / 'videos' / basename], None, None
    if source == 'ActivityNet':
        return [media_root / 'activitynet' / 'videos' / basename], None, None
    if source == 'QVHighlight':
        return [media_root / 'QVHighlight' / 'videos' / basename], None, None
    if source == 'queryd':
        return [media_root / 'Queryd' / 'videos' / basename], None, None
    if source == 'didemo':
        return [media_root / 'didemo' / 'videos' / basename], None, None
    if source == 'tacos':
        return [
            media_root / 'tacos' / 'videos' / basename,
            media_root / 'tacos' / 'videos' / f'{stem}.avi',
        ], None, None
    if source in {'Youcook', 'Youcookv2'}:
        filename = basename if basename.endswith('.mp4') else f'{basename}.mp4'
        return [media_root / 'youcook2' / 'videos' / filename], None, None
    if source == 'how_to_caption':
        return [media_root / 'how_to_caption' / 'how_to_caption' / basename], None, None
    if source == 'how_to_step':
        return [media_root / video_path], None, None
    if source == 'ego_timeqa':
        uuid_prefix = basename.split('_', 1)[0]
        return [media_root / 'ego4d' / 'videos_3fps_480_noaudio' / f'{uuid_prefix}.mp4'], None, None
    if source == 'LLaVA_Video':
        return [media_root / video_path], None, llava_root
    if source == 'Koala':
        return [], 'unsupported_source', None
    if source not in KNOWN_SOURCES:
        return [], 'unsupported_source', None
    return [media_root / video_path], None, None


def resolve_local_video_path(
    row: Dict[str, Any],
    media_root: Path,
    basename_index: Optional[Dict[str, List[str]]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    source = str(row.get('source') or '')
    video_path = str(row.get('video_path') or '')
    if not video_path:
        return None, 'missing_video_path'

    candidates, early_reason, fallback_root = _source_candidates(source, video_path, media_root)
    if early_reason is not None:
        return None, early_reason

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve()), None

    basename = Path(video_path).name
    matches = _fallback_matches(basename, basename_index, required_prefix=fallback_root)
    if len(matches) == 1:
        return matches[0], None
    if len(matches) > 1:
        return None, 'ambiguous_match'
    return None, 'missing_file'


def _convert_to_stream_worker(args: Tuple[int, Dict[str, Any], float]) -> Tuple[int, Optional[Dict[str, Any]]]:
    idx, row, fps = args
    return idx, convert_format_to_stream(row, fps=fps)


def convert_rows_to_stream(
    rows: Sequence[Dict[str, Any]],
    output_path: Path,
    *,
    fps: float,
    num_workers: int,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    if not rows:
        ensure_parent_dir(output_path)
        output_path.write_text('[]\n', encoding='utf-8')
        return [], []

    results_by_index: Dict[int, Dict[str, Any]] = {}
    failed_indices: List[int] = []
    worker_args = [(idx, row, fps) for idx, row in enumerate(rows)]

    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
        futures = {executor.submit(_convert_to_stream_worker, args): args[0] for args in worker_args}
        for future in as_completed(futures):
            idx, result = future.result()
            if result is None:
                failed_indices.append(idx)
            else:
                results_by_index[idx] = result

    converted_rows = [results_by_index[idx] for idx in sorted(results_by_index)]
    ensure_parent_dir(output_path)
    output_path.write_text(json.dumps(converted_rows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return converted_rows, sorted(failed_indices)


def _empty_source_stats() -> Dict[str, Any]:
    return {
        'seen': 0,
        'filtered_out': 0,
        'resolved_raw': 0,
        'stream_kept': 0,
        'dropped': 0,
        'drop_reasons': {},
    }


def _incr(mapping: Dict[str, int], key: str, delta: int = 1) -> None:
    mapping[key] = mapping.get(key, 0) + delta


def prepare_streamo_training_data(args: argparse.Namespace) -> Dict[str, Any]:
    label_root = Path(args.label_root).expanduser().resolve()
    media_root = Path(args.media_root).expanduser().resolve()
    output_raw = Path(args.output_raw).expanduser()
    output_stream = Path(args.output_stream).expanduser()
    report_json = Path(args.report_json).expanduser()

    include_tasks = normalize_filter(args.include_tasks)
    include_sources = normalize_filter(args.include_sources)
    exclude_sources = normalize_filter(args.exclude_sources)

    label_files = iter_label_files(label_root)
    basename_index = build_basename_index(media_root)

    source_stats: Dict[str, Dict[str, Any]] = defaultdict(_empty_source_stats)
    drop_reasons: Dict[str, int] = {}
    resolved_rows: List[Dict[str, Any]] = []
    resolved_sources: List[str] = []

    total_rows = 0
    filtered_rows = 0

    for label_file in label_files:
        data = json.loads(label_file.read_text(encoding='utf-8'))
        rows = [data] if isinstance(data, dict) else data
        for row in rows:
            total_rows += 1
            source = str(row.get('source') or 'unknown')
            task_type = str(row.get('task_type') or '')
            source_stat = source_stats[source]
            source_stat['seen'] += 1

            source_key = source.lower()
            task_key = task_type.lower()
            if include_sources is not None and source_key not in include_sources:
                source_stat['filtered_out'] += 1
                filtered_rows += 1
                continue
            if exclude_sources is not None and source_key in exclude_sources:
                source_stat['filtered_out'] += 1
                filtered_rows += 1
                continue
            if include_tasks is not None and task_key not in include_tasks:
                source_stat['filtered_out'] += 1
                filtered_rows += 1
                continue

            resolved_path, reason = resolve_local_video_path(row, media_root, basename_index)
            if resolved_path is None:
                source_stat['dropped'] += 1
                _incr(source_stat['drop_reasons'], reason)
                _incr(drop_reasons, reason)
                continue

            row_copy = dict(row)
            row_copy['video_path'] = resolved_path
            resolved_rows.append(row_copy)
            resolved_sources.append(source)
            source_stat['resolved_raw'] += 1

    ensure_parent_dir(output_raw)
    output_raw.write_text(json.dumps(resolved_rows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    converted_rows, failed_indices = convert_rows_to_stream(
        resolved_rows,
        output_stream,
        fps=args.fps,
        num_workers=args.num_workers,
    )

    for idx in failed_indices:
        source = resolved_sources[idx]
        source_stat = source_stats[source]
        source_stat['dropped'] += 1
        _incr(source_stat['drop_reasons'], 'conversion_failed')
        _incr(drop_reasons, 'conversion_failed')

    failed_set = set(failed_indices)
    for idx, source in enumerate(resolved_sources):
        if idx not in failed_set:
            source_stats[source]['stream_kept'] += 1

    report = {
        'label_root': str(label_root),
        'media_root': str(media_root),
        'output_raw': str(output_raw),
        'output_stream': str(output_stream),
        'report_json': str(report_json),
        'fps': args.fps,
        'num_workers': args.num_workers,
        'label_files': len(label_files),
        'total_rows': total_rows,
        'filtered_rows': filtered_rows,
        'resolved_raw_rows': len(resolved_rows),
        'stream_rows': len(converted_rows),
        'dropped_rows': total_rows - filtered_rows - len(converted_rows),
        'drop_reasons': dict(sorted(drop_reasons.items())),
        'sources': {
            source: {
                **stats,
                'drop_reasons': dict(sorted(stats['drop_reasons'].items())),
            } for source, stats in sorted(source_stats.items())
        },
    }

    ensure_parent_dir(report_json)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    if args.fail_on_empty and report['stream_rows'] == 0:
        raise RuntimeError(f'No training samples survived. See report: {report_json}')
    if args.min_samples and report['stream_rows'] < args.min_samples:
        raise RuntimeError(
            f"Only {report['stream_rows']} training samples survived, below --min-samples={args.min_samples}. "
            f'See report: {report_json}')

    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare local Streamo training data with partial video coverage.')
    parser.add_argument('--label-root', default='/media/lm/NO_NAME/Streamo-Instruct-465K')
    parser.add_argument('--media-root', default='/media/lm/NO_NAME')
    parser.add_argument('--output-raw', default='./dataset/stream/raw_resolved.json')
    parser.add_argument('--output-stream', default='./dataset/stream/stream_format.json')
    parser.add_argument('--report-json', default='./dataset/stream/prepare_report.json')
    parser.add_argument('--fps', type=float, default=1.0)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--include-tasks', nargs='*')
    parser.add_argument('--include-sources', nargs='*')
    parser.add_argument('--exclude-sources', nargs='*')
    parser.add_argument('--min-samples', type=int, default=0)
    parser.add_argument('--fail-on-empty', type=str2bool, default=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    report = prepare_streamo_training_data(args)
    print(json.dumps({
        'stream_rows': report['stream_rows'],
        'resolved_raw_rows': report['resolved_raw_rows'],
        'dropped_rows': report['dropped_rows'],
        'report_json': report['report_json'],
        'output_stream': report['output_stream'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
