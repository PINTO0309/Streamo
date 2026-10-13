import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))

import convert_streaming_video as converter
import probe_streaming_fps as probe
from swift.llm.dataset.preprocessor.streaming_video import (
    StreamingVideoMessagesPreprocessor, StreamingVideoPreprocessor, VideoFrameExtractor,
    clear_frame_cache, convert_streaming_video_data,
)
from swift.plugin.streaming_dataset_config import resolve_streaming_frame_max_width


def make_video(path, size):
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 8, size)
    assert writer.isOpened()
    try:
        for i in range(8):
            frame = np.full((height, width, 3), (i * 20, 80, 150), np.uint8)
            frame[:, width // 2:] = (200, 100, i * 20)
            writer.write(frame)
    finally:
        writer.release()
    return str(path)


@pytest.fixture
def video_and_row(tmp_path):
    path = make_video(tmp_path / 'wide.avi', (1280, 720))
    row = converter.convert_format_to_stream({
        'video_path': path,
        'response': [{'content': 'done', 'time': 1}],
    }, fps=4)
    return path, row


@pytest.mark.parametrize('source_size,expected_size', [
    ((1280, 720), (640, 360)),
    ((720, 1280), (640, 1138)),
    ((1108, 830), (640, 479)),
    ((640, 480), (640, 480)),
    ((320, 240), (320, 240)),
])
def test_resize_preserves_aspect_ratio_and_frames_in_all_modes(tmp_path, source_size, expected_size):
    path = make_video(tmp_path / 'frames.avi', source_size)
    extractor = VideoFrameExtractor(fps=4, max_width=640, enable_memory_cache=False)
    memory = extractor.extract(path)
    cap = cv2.VideoCapture(path)
    try:
        sequential = extractor._extract_to_memory_sequential(cap)
    finally:
        cap.release()
    disk = VideoFrameExtractor(fps=4, max_width=640, save_frames=True, output_dir=str(tmp_path / 'cache'))
    paths = disk.extract(path)
    assert len(memory) == len(sequential) == len(paths) == extractor.get_frame_count(path) == 4
    for in_memory, from_sequential, saved_path in zip(memory, sequential, paths):
        with Image.open(saved_path) as saved:
            assert in_memory.size == from_sequential.size == saved.size == expected_size
            np.testing.assert_array_equal(np.asarray(in_memory), np.asarray(saved))
        np.testing.assert_array_equal(np.asarray(in_memory), np.asarray(from_sequential))
    # The sampled source frames must stay at indices 1, 3, 5, 7, even when resized.
    cap = cv2.VideoCapture(path)
    try:
        decoded = [cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB) for _ in range(8)]
    finally:
        cap.release()
    for actual, index in zip(memory, (1, 3, 5, 7)):
        np.testing.assert_array_equal(np.asarray(actual)[0, 0], decoded[index][0, 0])
        if source_size[0] <= 640:
            np.testing.assert_array_equal(np.asarray(actual), decoded[index])


@pytest.mark.parametrize('save_frames', [False, True])
def test_width_settings_isolate_caches_and_reuse_matching_entries(tmp_path, video_and_row, monkeypatch, save_frames):
    path, _ = video_and_row
    clear_frame_cache()
    try:
        frames_by_width = {}
        extractors = {}
        for width, expected in [(None, (1280, 720)), (640, (640, 360)), (320, (320, 180)), (0, (1280, 720))]:
            extractor = VideoFrameExtractor(
                fps=4, max_width=width, save_frames=save_frames, output_dir=str(tmp_path / 'cache'))
            frames = extractor.extract(path)
            for frame in frames:
                if save_frames:
                    with Image.open(frame) as saved:
                        assert saved.size == expected
                else:
                    assert frame.size == expected
            frames_by_width[width] = frames
            extractors[width] = extractor
        if save_frames:
            assert len({frames_by_width[w][0] for w in (None, 640, 320)}) == 3
            assert frames_by_width[0] == frames_by_width[None]

        def cannot_decode(*args, **kwargs):
            raise AssertionError('Matching cached frames must not require decoding')

        monkeypatch.setattr(cv2, 'VideoCapture', cannot_decode)
        for width, extractor in extractors.items():
            assert extractor.extract(path) == frames_by_width[width]
    finally:
        clear_frame_cache()


@pytest.mark.parametrize('width', [-1, 640.5, '640', True])
def test_invalid_width_fails_before_extraction(width):
    with pytest.raises(ValueError, match='max_width'):
        VideoFrameExtractor(max_width=width)


@pytest.mark.parametrize('value,expected', [(None, 640), ('480', 480), ('0', 0)])
def test_training_and_probe_width_configuration(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv('STREAMING_FRAME_MAX_WIDTH', raising=False)
    else:
        monkeypatch.setenv('STREAMING_FRAME_MAX_WIDTH', value)
    assert resolve_streaming_frame_max_width() == expected
    assert probe.parse_args([]).max_width == expected
    assert probe.parse_args(['--max-width', '320']).max_width == 320


@pytest.mark.parametrize('value', ['-1', '640.5', 'bad'])
def test_invalid_width_configuration(monkeypatch, value):
    monkeypatch.setenv('STREAMING_FRAME_MAX_WIDTH', value)
    with pytest.raises(ValueError, match='STREAMING_FRAME_MAX_WIDTH'):
        resolve_streaming_frame_max_width()
    with pytest.raises(SystemExit):
        probe.parse_args([])


@pytest.mark.parametrize('width_args,expected', [([], (640, 360)), (['--max-width', '0'], (1280, 720))])
def test_to_image_cli_writes_resized_pngs_and_serializable_json(tmp_path, video_and_row, monkeypatch, width_args, expected):
    _, row = video_and_row
    input_path, output_path = tmp_path / 'stream.json', tmp_path / 'images.json'
    input_path.write_text(json.dumps([row]), encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', [
        'convert_streaming_video.py', 'to-image', '--input', str(input_path), '--output', str(output_path),
        '--frame-dir', str(tmp_path / 'cache'), '--fps', '4', '--num-workers', '2', *width_args,
    ])
    converter.main()
    result = json.loads(output_path.read_text(encoding='utf-8'))[0]
    expected_messages = copy.deepcopy(row['messages'])
    for message in expected_messages:
        message['content'] = message['content'].replace('<stream>', '<image>')
    assert result['messages'] == expected_messages
    assert len(result['images']) == 4
    for path in result['images']:
        with Image.open(path) as saved:
            assert saved.size == expected


def test_preprocessor_wrappers_and_registration_apply_width(tmp_path, video_and_row, monkeypatch):
    import swift.llm.dataset.dataset.streaming_video_dataset as registration

    _, row = video_and_row
    registered = []
    monkeypatch.setattr(registration, 'register_dataset', registered.append)
    meta = registration.register_streaming_video_dataset(
        dataset_path=str(tmp_path / 'stream.json'), fps=4, max_width=640,
        save_frames=True, frame_output_dir=str(tmp_path / 'cache'))
    assert registered == [meta]
    processors = [
        meta.preprocess_func,
        StreamingVideoPreprocessor(fps=4, max_width=640, enable_memory_cache=False),
        StreamingVideoMessagesPreprocessor(fps=4, max_width=640, enable_memory_cache=False),
    ]
    for processor in processors:
        result = processor.preprocess(copy.deepcopy(row))
        assert len(result['images']) == 4
        for frame in result['images']:
            if isinstance(frame, str):
                with Image.open(frame) as saved:
                    assert saved.size == (640, 360)
            else:
                assert frame.size == (640, 360)


def test_direct_conversion_and_probe_use_resized_frames(tmp_path, video_and_row):
    _, row = video_and_row
    input_path, output_path = tmp_path / 'stream.json', tmp_path / 'images.json'
    input_path.write_text(json.dumps(row), encoding='utf-8')
    convert_streaming_video_data(
        str(input_path), str(output_path), fps=4, output_dir=str(tmp_path / 'cache'), max_width=640)
    result = json.loads(output_path.read_text(encoding='utf-8'))
    for path in result['images']:
        with Image.open(path) as saved:
            assert saved.size == (640, 360)

    class InspectingTemplate:
        def encode(self, processed, return_length):
            assert return_length
            for path in processed['images']:
                with Image.open(path) as saved:
                    assert saved.size == (640, 360)
            return {'length': 123}

    measurements = probe.measure_lengths(
        InspectingTemplate(), [(0, row)], fps=4, frame_cache_root=tmp_path / 'probe', max_width=640)
    assert measurements[0]['ok']
    assert measurements[0]['images'] == 4
    assert measurements[0]['length'] == 123
