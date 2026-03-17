import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / 'scripts' / 'prepare_streamo_training_data.py'
PLUGIN_CONFIG_PATH = REPO_ROOT / 'swift' / 'plugin' / 'streaming_dataset_config.py'


def load_prepare_module():
    spec = importlib.util.spec_from_file_location('prepare_streamo_training_data', SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'')
    return path


def create_video(path: Path, *, fps: float = 1.0, frames: int = 4) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*('mp4v' if path.suffix.lower() == '.mp4' else 'MJPG'))
    writer = cv2.VideoWriter(str(path), fourcc, fps, (16, 16))
    assert writer.isOpened(), f'Failed to create video: {path}'
    for idx in range(frames):
        frame = np.full((16, 16, 3), idx * 30, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    assert path.exists()
    return path


def load_plugin_config_module():
    spec = importlib.util.spec_from_file_location('streaming_dataset_config', PLUGIN_CONFIG_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestPrepareStreamoTrainingData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.module = load_prepare_module()

    def test_resolve_video_path_supported_sources(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            media_root = Path(tmp_dir) / 'media'
            cases = [
                ({'source': 'coin', 'video_path': 'coin/example.mp4'}, media_root / 'coin' / 'videos' / 'example.mp4'),
                ({'source': 'ActivityNet', 'video_path': 'ActivityNet/video/clip.mp4'},
                 media_root / 'activitynet' / 'videos' / 'clip.mp4'),
                ({'source': 'QVHighlight', 'video_path': 'QVHighlight/videos/qv.mp4'},
                 media_root / 'QVHighlight' / 'videos' / 'qv.mp4'),
                ({'source': 'queryd', 'video_path': 'queryd/query.mp4'},
                 media_root / 'Queryd' / 'videos' / 'query.mp4'),
                ({'source': 'didemo', 'video_path': 'didemo/demo.mp4'},
                 media_root / 'didemo' / 'videos' / 'demo.mp4'),
                ({'source': 'tacos', 'video_path': 'tacos/taco.mp4'},
                 media_root / 'tacos' / 'videos' / 'taco.avi'),
                ({'source': 'Youcookv2', 'video_path': 'YouCookv2/raw_videos/training/101/ycook'},
                 media_root / 'youcook2' / 'videos' / 'ycook.mp4'),
                ({'source': 'how_to_caption', 'video_path': 'how_to_caption/caption.mp4'},
                 media_root / 'how_to_caption' / 'how_to_caption' / 'caption.mp4'),
                ({'source': 'how_to_step', 'video_path': 'how_to_step/step.mp4'},
                 media_root / 'how_to_step' / 'step.mp4'),
                ({'source': 'ego_timeqa', 'video_path': 'ego_timeqa/uuid123_18_168.mp4'},
                 media_root / 'ego4d' / 'videos_3fps_480_noaudio' / 'uuid123.mp4'),
            ]

            for _, expected in cases:
                touch(expected)

            basename_index = self.module.build_basename_index(media_root)
            for row, expected in cases:
                with self.subTest(source=row['source']):
                    resolved, reason = self.module.resolve_local_video_path(row, media_root, basename_index)
                    self.assertIsNone(reason)
                    self.assertEqual(resolved, str(expected.resolve()))

    def test_llava_basename_fallback_success(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            media_root = Path(tmp_dir) / 'media'
            actual = touch(media_root / 'LLaVA_Video' / 'academic_source' / 'clip.mp4')
            basename_index = self.module.build_basename_index(media_root)

            resolved, reason = self.module.resolve_local_video_path(
                {'source': 'LLaVA_Video', 'video_path': 'LLaVA_Video/missing/tree/clip.mp4'},
                media_root,
                basename_index,
            )

            self.assertIsNone(reason)
            self.assertEqual(resolved, str(actual.resolve()))

    def test_llava_basename_fallback_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            media_root = Path(tmp_dir) / 'media'
            touch(media_root / 'LLaVA_Video' / 'academic_source' / 'clip.mp4')
            touch(media_root / 'LLaVA_Video' / 'youtube_source' / 'clip.mp4')
            basename_index = self.module.build_basename_index(media_root)

            resolved, reason = self.module.resolve_local_video_path(
                {'source': 'LLaVA_Video', 'video_path': 'LLaVA_Video/missing/tree/clip.mp4'},
                media_root,
                basename_index,
            )

            self.assertIsNone(resolved)
            self.assertEqual(reason, 'ambiguous_match')

    def test_koala_is_unsupported(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            media_root = Path(tmp_dir) / 'media'
            basename_index = self.module.build_basename_index(media_root)

            resolved, reason = self.module.resolve_local_video_path(
                {'source': 'Koala', 'video_path': 'Koala/clip.mp4'},
                media_root,
                basename_index,
            )

            self.assertIsNone(resolved)
            self.assertEqual(reason, 'unsupported_source')

    def test_prepare_streamo_training_data_integration(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            label_root = tmp_path / 'labels'
            media_root = tmp_path / 'media'
            output_raw = tmp_path / 'dataset' / 'stream' / 'raw_resolved.json'
            output_stream = tmp_path / 'dataset' / 'stream' / 'stream_format.json'
            report_json = tmp_path / 'dataset' / 'stream' / 'prepare_report.json'

            create_video(media_root / 'coin' / 'videos' / 'coin_clip.mp4')
            create_video(media_root / 'tacos' / 'videos' / 'taco_clip.avi')

            rows = [
                {
                    'video_name': 'coin_clip.mp4',
                    'video_path': 'coin/coin_clip.mp4',
                    'task_type': 'QA',
                    'source': 'coin',
                    'question': [{'content': 'What happens?', 'time': '0'}],
                    'response': [{'content': 'A coin task.', 'st_time': 0, 'end_time': 1, 'time': ''}],
                },
                {
                    'video_name': 'taco_clip.mp4',
                    'video_path': 'tacos/taco_clip.mp4',
                    'task_type': 'Event_Grounding',
                    'source': 'tacos',
                    'question': [{'content': 'Tell me when it ends.', 'time': '0'}],
                    'response': [{'content': 'A tacos task.', 'st_time': 0, 'end_time': 1, 'time': ''}],
                },
                {
                    'video_name': 'missing.mp4',
                    'video_path': 'queryd/missing.mp4',
                    'task_type': 'QA',
                    'source': 'queryd',
                    'question': [{'content': 'Missing?', 'time': '0'}],
                    'response': [{'content': 'Missing.', 'st_time': 0, 'end_time': 1, 'time': ''}],
                },
                {
                    'video_name': 'koala.mp4',
                    'video_path': 'Koala/koala.mp4',
                    'task_type': 'QA',
                    'source': 'Koala',
                    'question': [{'content': 'Unsupported?', 'time': '0'}],
                    'response': [{'content': 'Unsupported.', 'st_time': 0, 'end_time': 1, 'time': ''}],
                },
            ]
            label_file = label_root / 'qa' / 'labels.json'
            label_file.parent.mkdir(parents=True, exist_ok=True)
            label_file.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

            args = self.module.parse_args([
                '--label-root', str(label_root),
                '--media-root', str(media_root),
                '--output-raw', str(output_raw),
                '--output-stream', str(output_stream),
                '--report-json', str(report_json),
                '--num-workers', '2',
            ])
            report = self.module.prepare_streamo_training_data(args)

            self.assertTrue(output_raw.exists())
            self.assertTrue(output_stream.exists())
            self.assertTrue(report_json.exists())

            raw_rows = json.loads(output_raw.read_text(encoding='utf-8'))
            stream_rows = json.loads(output_stream.read_text(encoding='utf-8'))
            report_from_disk = json.loads(report_json.read_text(encoding='utf-8'))

            self.assertEqual(len(raw_rows), 2)
            self.assertEqual(len(stream_rows), 2)
            self.assertEqual(report['stream_rows'], 2)
            self.assertEqual(report_from_disk['drop_reasons']['missing_file'], 1)
            self.assertEqual(report_from_disk['drop_reasons']['unsupported_source'], 1)

    def test_streaming_dataset_plugin_uses_env_vars(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            old_cwd = os.getcwd()
            old_env = os.environ.copy()
            try:
                os.chdir(tmp_path)
                os.environ['STREAMING_DATASET_PATH'] = str(tmp_path / 'custom.json')
                os.environ['STREAMING_FRAME_DIR'] = str(tmp_path / 'frames')
                os.environ['STREAMING_DATASET_FPS'] = '2.5'
                module = load_plugin_config_module()
                self.assertEqual(module.resolve_streaming_dataset_path(), str(tmp_path / 'custom.json'))
                self.assertEqual(module.resolve_streaming_frame_dir(), str(tmp_path / 'frames'))
                self.assertEqual(module.resolve_streaming_dataset_fps(), 2.5)
            finally:
                os.chdir(old_cwd)
                os.environ.clear()
                os.environ.update(old_env)

    def test_streaming_dataset_plugin_prefers_generated_dataset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            touch(tmp_path / 'dataset' / 'stream' / 'stream_format.json')
            old_cwd = os.getcwd()
            old_env = os.environ.copy()
            try:
                os.chdir(tmp_path)
                os.environ.pop('STREAMING_DATASET_PATH', None)
                os.environ.pop('STREAMING_FRAME_DIR', None)
                os.environ.pop('STREAMING_DATASET_FPS', None)
                module = load_plugin_config_module()
                self.assertEqual(module.resolve_streaming_dataset_path(), './dataset/stream/stream_format.json')
            finally:
                os.chdir(old_cwd)
                os.environ.clear()
                os.environ.update(old_env)

    def test_streaming_dataset_plugin_falls_back_to_example(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            touch(tmp_path / 'dataset' / 'example' / 'stream_format.json')
            old_cwd = os.getcwd()
            old_env = os.environ.copy()
            try:
                os.chdir(tmp_path)
                os.environ.pop('STREAMING_DATASET_PATH', None)
                os.environ.pop('STREAMING_FRAME_DIR', None)
                os.environ.pop('STREAMING_DATASET_FPS', None)
                module = load_plugin_config_module()
                self.assertEqual(module.resolve_streaming_dataset_path(), './dataset/example/stream_format.json')
            finally:
                os.chdir(old_cwd)
                os.environ.clear()
                os.environ.update(old_env)


if __name__ == '__main__':
    unittest.main()
