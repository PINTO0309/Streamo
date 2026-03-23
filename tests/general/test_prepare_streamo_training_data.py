import importlib.util
import json
import os
import sqlite3
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


def _create_archive_index(db_path, entries):
    """Create a minimal SQLite archive index for testing.

    ``entries`` is a list of (logical_path, archive_id, member_path) tuples.
    """
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript("""
            CREATE TABLE archives (
                archive_id TEXT PRIMARY KEY,
                gcs_prefix TEXT NOT NULL,
                archive_stem TEXT NOT NULL,
                parts_json TEXT NOT NULL
            );
            CREATE TABLE files (
                logical_path TEXT PRIMARY KEY,
                archive_id TEXT NOT NULL,
                member_path TEXT NOT NULL,
                FOREIGN KEY (archive_id) REFERENCES archives(archive_id)
            );
            CREATE INDEX idx_files_archive_id ON files(archive_id);
        """)
        archive_ids = {e[1] for e in entries}
        for aid in archive_ids:
            conn.execute(
                'INSERT INTO archives (archive_id, gcs_prefix, archive_stem, parts_json) VALUES (?, ?, ?, ?)',
                (aid, 'gs://test-bucket/datasets', aid, '[]'))
        conn.executemany(
            'INSERT INTO files (logical_path, archive_id, member_path) VALUES (?, ?, ?)',
            entries)
        conn.commit()


class TestArchiveVideoResolution(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.module = load_prepare_module()

    def test_resolve_archive_video_path_coin(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / 'index.sqlite'
            _create_archive_index(db_path, [
                ('videos/example.mp4', 'coin/coin/videos.tar.gz', 'videos/example.mp4'),
            ])
            lookup = self.module.build_archive_lookup(db_path)
            resolved, reason = self.module.resolve_archive_video_path(
                {'source': 'coin', 'video_path': 'coin/example.mp4'}, lookup)
            self.assertIsNone(reason)
            self.assertEqual(resolved, 'videos/example.mp4')

    def test_resolve_archive_video_path_activitynet(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / 'index.sqlite'
            _create_archive_index(db_path, [
                ('videos/v_clip.mp4', 'activitynet/videos.tar.gz', 'videos/v_clip.mp4'),
            ])
            lookup = self.module.build_archive_lookup(db_path)
            resolved, reason = self.module.resolve_archive_video_path(
                {'source': 'ActivityNet', 'video_path': 'ActivityNet/v_clip.mp4'}, lookup)
            self.assertIsNone(reason)
            self.assertEqual(resolved, 'videos/v_clip.mp4')

    def test_resolve_archive_video_path_ego_timeqa(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / 'index.sqlite'
            _create_archive_index(db_path, [
                ('uuid123.mp4', 'ego4d/v2/videos_3fps_480_noaudio.tar.gz', 'uuid123.mp4'),
            ])
            lookup = self.module.build_archive_lookup(db_path)
            resolved, reason = self.module.resolve_archive_video_path(
                {'source': 'ego_timeqa', 'video_path': 'ego_timeqa/uuid123_18_168.mp4'}, lookup)
            self.assertIsNone(reason)
            self.assertEqual(resolved, 'uuid123.mp4')

    def test_resolve_archive_video_path_youcook_adds_mp4(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / 'index.sqlite'
            _create_archive_index(db_path, [
                ('videos/ycook.mp4', 'youcook2/videos.tar.gz', 'videos/ycook.mp4'),
            ])
            lookup = self.module.build_archive_lookup(db_path)
            resolved, reason = self.module.resolve_archive_video_path(
                {'source': 'Youcookv2', 'video_path': 'YouCookv2/raw_videos/training/101/ycook'},
                lookup)
            self.assertIsNone(reason)
            self.assertEqual(resolved, 'videos/ycook.mp4')

    def test_resolve_archive_video_path_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / 'index.sqlite'
            _create_archive_index(db_path, [])
            lookup = self.module.build_archive_lookup(db_path)
            resolved, reason = self.module.resolve_archive_video_path(
                {'source': 'coin', 'video_path': 'coin/missing.mp4'}, lookup)
            self.assertIsNone(resolved)
            self.assertEqual(reason, 'missing_file')

    def test_resolve_archive_video_path_koala_unsupported(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / 'index.sqlite'
            _create_archive_index(db_path, [])
            lookup = self.module.build_archive_lookup(db_path)
            resolved, reason = self.module.resolve_archive_video_path(
                {'source': 'Koala', 'video_path': 'Koala/clip.mp4'}, lookup)
            self.assertIsNone(resolved)
            self.assertEqual(reason, 'unsupported_source')

    def test_resolve_archive_video_path_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / 'index.sqlite'
            _create_archive_index(db_path, [
                ('dir_a/clip.mp4', 'coin/coin/videos.tar.gz', 'dir_a/clip.mp4'),
                ('dir_b/clip.mp4', 'coin/coin/videos.tar.gz', 'dir_b/clip.mp4'),
            ])
            lookup = self.module.build_archive_lookup(db_path)
            resolved, reason = self.module.resolve_archive_video_path(
                {'source': 'coin', 'video_path': 'coin/clip.mp4'}, lookup)
            self.assertIsNone(resolved)
            self.assertEqual(reason, 'ambiguous_match')

    def test_prepare_with_archive_index_integration(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            label_root = tmp_path / 'labels'
            media_root = tmp_path / 'media'
            output_raw = tmp_path / 'dataset' / 'stream' / 'raw_resolved.json'
            output_stream = tmp_path / 'dataset' / 'stream' / 'stream_format.json'
            report_json = tmp_path / 'dataset' / 'stream' / 'prepare_report.json'
            db_path = tmp_path / 'archive_index.sqlite'

            _create_archive_index(db_path, [
                ('videos/coin_clip.mp4', 'coin/coin/videos.tar.gz', 'videos/coin_clip.mp4'),
                ('videos/qv_clip.mp4', 'qvhighlights/videos.tar.gz', 'videos/qv_clip.mp4'),
            ])

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
                    'video_name': 'qv_clip.mp4',
                    'video_path': 'QVHighlight/qv_clip.mp4',
                    'task_type': 'QA',
                    'source': 'QVHighlight',
                    'question': [{'content': 'What?', 'time': '0'}],
                    'response': [{'content': 'QV task.', 'st_time': 0, 'end_time': 1, 'time': ''}],
                },
                {
                    'video_name': 'missing.mp4',
                    'video_path': 'queryd/missing.mp4',
                    'task_type': 'QA',
                    'source': 'queryd',
                    'question': [{'content': 'Missing?', 'time': '0'}],
                    'response': [{'content': 'Missing.', 'st_time': 0, 'end_time': 1, 'time': ''}],
                },
            ]
            label_file = label_root / 'qa' / 'labels.json'
            label_file.parent.mkdir(parents=True, exist_ok=True)
            label_file.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

            args = self.module.parse_args([
                '--label-root', str(label_root),
                '--media-root', str(media_root),
                '--archive-index', str(db_path),
                '--output-raw', str(output_raw),
                '--output-stream', str(output_stream),
                '--report-json', str(report_json),
                '--num-workers', '2',
                '--fail-on-empty', 'false',
            ])
            report = self.module.prepare_streamo_training_data(args)

            self.assertTrue(output_raw.exists())
            self.assertTrue(report_json.exists())

            raw_rows = json.loads(output_raw.read_text(encoding='utf-8'))
            self.assertEqual(len(raw_rows), 2)
            self.assertEqual(raw_rows[0]['video_path'], 'videos/coin_clip.mp4')
            self.assertEqual(raw_rows[1]['video_path'], 'videos/qv_clip.mp4')
            self.assertEqual(report['resolved_raw_rows'], 2)
            self.assertEqual(report['drop_reasons'].get('missing_file', 0), 1)
            self.assertIsNotNone(report.get('archive_index'))


if __name__ == '__main__':
    unittest.main()
