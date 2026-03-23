import importlib.util
import io
import logging
import os
import sqlite3
import sys
import tarfile
import tempfile
import threading
import types
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

if importlib.util.find_spec('modelscope') is None:
    modelscope_module = types.ModuleType('modelscope')
    modelscope_hub_module = types.ModuleType('modelscope.hub')
    modelscope_hub_api_module = types.ModuleType('modelscope.hub.api')
    modelscope_hub_utils_module = types.ModuleType('modelscope.hub.utils')
    modelscope_hub_utils_utils_module = types.ModuleType('modelscope.hub.utils.utils')
    modelscope_utils_module = types.ModuleType('modelscope.utils')
    modelscope_config_ds_module = types.ModuleType('modelscope.utils.config_ds')
    modelscope_logger_module = types.ModuleType('modelscope.utils.logger')

    def _get_test_logger(*args, **kwargs):
        del args, kwargs
        logger = logging.getLogger('streaming-video-archive-test')
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler())
        return logger

    class _ModelScopeConfig:

        @staticmethod
        def get_cookies():
            return {}

    modelscope_hub_api_module.ModelScopeConfig = _ModelScopeConfig
    modelscope_hub_utils_utils_module.get_cache_dir = lambda *args, **kwargs: tempfile.gettempdir()
    modelscope_logger_module.get_logger = _get_test_logger
    modelscope_config_ds_module.MS_CACHE_HOME = tempfile.gettempdir()
    modelscope_hub_utils_module.utils = modelscope_hub_utils_utils_module
    modelscope_utils_module.logger = modelscope_logger_module
    modelscope_utils_module.config_ds = modelscope_config_ds_module
    modelscope_hub_module.api = modelscope_hub_api_module
    modelscope_hub_module.utils = modelscope_hub_utils_module
    modelscope_module.hub = modelscope_hub_module
    modelscope_module.utils = modelscope_utils_module

    sys.modules['modelscope'] = modelscope_module
    sys.modules['modelscope.hub'] = modelscope_hub_module
    sys.modules['modelscope.hub.api'] = modelscope_hub_api_module
    sys.modules['modelscope.hub.utils'] = modelscope_hub_utils_module
    sys.modules['modelscope.hub.utils.utils'] = modelscope_hub_utils_utils_module
    sys.modules['modelscope.utils'] = modelscope_utils_module
    sys.modules['modelscope.utils.config_ds'] = modelscope_config_ds_module
    sys.modules['modelscope.utils.logger'] = modelscope_logger_module

if importlib.util.find_spec('json_repair') is None:
    json_repair_module = types.ModuleType('json_repair')

    def _repair_json(value, *args, **kwargs):
        del args, kwargs
        return value

    json_repair_module.repair_json = _repair_json
    sys.modules['json_repair'] = json_repair_module

from swift.llm.dataset.preprocessor import ArchiveVideoResolver, StreamingVideoPreprocessor


def _load_build_index_module():
    script_path = Path(__file__).resolve().parents[2] / 'scripts' / 'build_stream_archive_index.py'
    spec = importlib.util.spec_from_file_location('build_stream_archive_index', script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BUILD_INDEX_MODULE = _load_build_index_module()


class FakeBlob:

    def __init__(self, client, name: str):
        self._client = client
        self.name = name
        self.size = len(client.objects.get(name, b''))

    def reload(self):
        self.size = len(self._client.objects.get(self.name, b''))

    def open(self, mode: str = 'rb'):
        assert mode == 'rb'
        with self._client._lock:
            self._client.download_counts[self.name] += 1
            data = self._client.objects[self.name]
        return io.BytesIO(data)

    def download_to_filename(self, filename: str):
        with self._client._lock:
            self._client.download_counts[self.name] += 1
            data = self._client.objects[self.name]
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            f.write(data)


class FakeBucket:

    def __init__(self, client):
        self._client = client

    def blob(self, name: str):
        if name not in self._client.objects:
            raise FileNotFoundError(name)
        return FakeBlob(self._client, name)


class FakeListedBlob:

    def __init__(self, name: str):
        self.name = name


class FakeStorageClient:

    def __init__(self, objects):
        self.objects = objects
        self.download_counts = Counter()
        self._lock = threading.Lock()

    def list_blobs(self, bucket_name: str, prefix: str = ''):
        del bucket_name
        return [
            FakeListedBlob(name)
            for name in sorted(self.objects)
            if name.startswith(prefix)
        ]

    def bucket(self, bucket_name: str):
        del bucket_name
        return FakeBucket(self)


def _create_test_video(video_path: str, frame_count: int = 2, fps: float = 1.0):
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*'MJPG'),
        fps,
        (16, 16),
    )
    if not writer.isOpened():
        raise RuntimeError(f'Failed to create test video at {video_path}')
    for idx in range(frame_count):
        frame = np.full((16, 16, 3), (idx + 1) * 40, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _create_split_archive_objects(root_dir: str, gcs_prefix: str):
    source_video_path = os.path.join(root_dir, 'source', 'videos', 'sample.avi')
    _create_test_video(source_video_path)

    archive_dir = os.path.join(root_dir, 'archives')
    os.makedirs(archive_dir, exist_ok=True)
    archive_path = os.path.join(archive_dir, 'shard-000.tar.gz')
    with tarfile.open(archive_path, 'w:gz') as tar:
        tar.add(source_video_path, arcname='videos/sample.avi')

    with open(archive_path, 'rb') as f:
        archive_bytes = f.read()

    midpoint = max(1, len(archive_bytes) // 2)
    bucket, prefix = BUILD_INDEX_MODULE._parse_gs_uri(gcs_prefix)
    del bucket
    return {
        f'{prefix}/shard-000.tar.gz.00': archive_bytes[:midpoint],
        f'{prefix}/shard-000.tar.gz.01': archive_bytes[midpoint:],
    }


class TestStreamingVideoArchive(unittest.TestCase):

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir = self._tmp_dir.name
        self.gcs_prefix = 'gs://test-bucket/archive-root'
        self.objects = _create_split_archive_objects(self.tmp_dir, self.gcs_prefix)
        self.client = FakeStorageClient(self.objects)
        self.index_path = os.path.join(self.tmp_dir, 'stream_index.sqlite')
        BUILD_INDEX_MODULE.build_archive_index(
            gcs_prefix=self.gcs_prefix,
            output_path=self.index_path,
            scratch_dir=os.path.join(self.tmp_dir, 'scratch'),
            storage_client=self.client,
        )
        self.client.download_counts.clear()

    def tearDown(self):
        self._tmp_dir.cleanup()

    def test_build_archive_index_records_member_paths(self):
        with sqlite3.connect(self.index_path) as conn:
            archive_row = conn.execute(
                'SELECT archive_stem, parts_json FROM archives WHERE archive_id = ?',
                ('shard-000.tar.gz', )).fetchone()
            file_row = conn.execute(
                'SELECT archive_id, member_path FROM files WHERE logical_path = ?',
                ('videos/sample.avi', )).fetchone()

        self.assertIsNotNone(archive_row)
        self.assertEqual(archive_row[0], 'shard-000.tar.gz')
        self.assertEqual(file_row[0], 'shard-000.tar.gz')
        self.assertEqual(file_row[1], 'videos/sample.avi')

    def test_archive_resolver_downloads_once_and_reuses_cache(self):
        resolver = ArchiveVideoResolver(
            index_path=self.index_path,
            archive_cache_dir=os.path.join(self.tmp_dir, 'archive_cache'),
            video_cache_dir=os.path.join(self.tmp_dir, 'video_cache'),
            storage_client=self.client,
        )

        resolved_path = resolver.resolve('videos/sample.avi')
        self.assertTrue(os.path.exists(resolved_path))
        self.assertEqual(sum(self.client.download_counts.values()), 2)

        resolved_path_second = resolver.resolve('videos/sample.avi')
        self.assertEqual(resolved_path, resolved_path_second)
        self.assertEqual(sum(self.client.download_counts.values()), 2)

    def test_archive_resolver_is_thread_safe(self):
        resolver = ArchiveVideoResolver(
            index_path=self.index_path,
            archive_cache_dir=os.path.join(self.tmp_dir, 'archive_cache_threads'),
            video_cache_dir=os.path.join(self.tmp_dir, 'video_cache_threads'),
            storage_client=self.client,
        )

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda _: resolver.resolve('videos/sample.avi'), range(4)))

        self.assertEqual(1, len(set(results)))
        self.assertEqual(sum(self.client.download_counts.values()), 2)

    def test_preprocessor_resolves_archive_video_before_frame_extraction(self):
        resolver = ArchiveVideoResolver(
            index_path=self.index_path,
            archive_cache_dir=os.path.join(self.tmp_dir, 'archive_cache_pre'),
            video_cache_dir=os.path.join(self.tmp_dir, 'video_cache_pre'),
            storage_client=self.client,
        )
        preprocessor = StreamingVideoPreprocessor(
            fps=1.0,
            frame_output_dir=os.path.join(self.tmp_dir, 'frames'),
            save_frames=True,
            video_resolver=resolver,
        )

        row = {
            'messages': [
                {'role': 'user', 'content': '<0s-1s>\n<stream>'},
                {'role': 'assistant', 'content': '</Silence>'},
                {'role': 'user', 'content': '<1s-2s>\n<stream>'},
                {'role': 'assistant', 'content': '</Response> done'},
            ],
            'videos': ['videos/sample.avi'],
        }
        result = preprocessor.preprocess(row)

        self.assertIsNotNone(result)
        self.assertEqual(2, len(result['images']))
        self.assertTrue(all('<image>' in message['content'] for message in result['messages'] if message['role'] == 'user'))

        result_second = preprocessor.preprocess(row)
        self.assertIsNotNone(result_second)
        self.assertEqual(result['images'], result_second['images'])
        self.assertEqual(sum(self.client.download_counts.values()), 2)

    def test_preprocessor_returns_none_when_index_entry_is_missing(self):
        resolver = ArchiveVideoResolver(
            index_path=self.index_path,
            archive_cache_dir=os.path.join(self.tmp_dir, 'archive_cache_missing'),
            video_cache_dir=os.path.join(self.tmp_dir, 'video_cache_missing'),
            storage_client=self.client,
        )
        preprocessor = StreamingVideoPreprocessor(video_resolver=resolver)
        row = {
            'messages': [
                {'role': 'user', 'content': '<0s-1s>\n<stream>'},
                {'role': 'assistant', 'content': '</Silence>'},
            ],
            'videos': ['videos/missing.avi'],
        }

        self.assertIsNone(preprocessor.preprocess(row))


if __name__ == '__main__':
    unittest.main()
