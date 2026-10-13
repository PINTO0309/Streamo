import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from streamo_utils.streaming import (
    SAMPLING_VERSION, SYSTEM_PROMPT, interval_index, iter_sampled_frames,
    parse_seconds, sampling_indices, time_tag,
)


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from convert_streaming_video import build_frame_states, convert_format_to_stream
from prepare_streamo_training_data import convert_rows_to_stream


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_video(path, fps, frames):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), fps, (32, 32))
    assert writer.isOpened()
    for i in range(frames):
        writer.write(np.full((32, 32, 3), (i * 7) % 240, np.uint8))
    writer.release()
    return str(path)


@pytest.mark.parametrize('value', [1, 1.0, '1', '1.0', '00:01', '00:00:01.0'])
def test_time_types_are_seconds(value):
    assert parse_seconds(value) == 1.0
    assert interval_index(parse_seconds(value), 4) == 3


@pytest.mark.parametrize('value', [-1, float('nan'), 'inf', 'bad', True, '00:60'])
def test_invalid_time_is_not_silently_ignored(value):
    with pytest.raises(ValueError):
        parse_seconds(value)


def test_missing_times_and_fractional_boundaries():
    assert parse_seconds('') is None
    assert parse_seconds(None) is None
    assert interval_index(0, 4) == 0
    assert interval_index(3.75, 4) == 14
    assert interval_index(3.76, 4) == 15
    assert interval_index(15, 4, 15) == 59
    assert interval_index(0.1 + 0.2, 10) == 2
    with pytest.raises(ValueError):
        interval_index(15.01, 4, 15)


def test_events_overlap_and_merge_without_losing_any_response():
    responses = [
        {'content': 'first', 'st_time': 3.25, 'end_time': 3.75},
        {'content': 'overlap', 'st_time': 3.5, 'end_time': 4},
        {'content': 'same turn', 'time': 3.75},
        {'content': 'terminal', 'st_time': 14.5, 'end_time': 15},
    ]
    states, questions = build_frame_states(60, [{'content': 'Q', 'time': '0'}], responses, 4)
    assert questions == {0: 'Q'}
    assert states[14] == '</Response> first\nsame turn'
    assert states[15] == '</Response> overlap'
    assert states[59] == '</Response> terminal'
    assert states.count('</Silence>') == 57
    assert '</Standby>' not in '\n'.join(states)


@pytest.mark.parametrize('response', [
    {'content': 'bad', 'st_time': 4, 'end_time': 3},
    {'content': 'bad', 'st_time': 14, 'end_time': 15.01},
    {'content': 'bad'},
    {'content': '', 'time': 1},
])
def test_invalid_annotations_fail(response):
    with pytest.raises(ValueError):
        build_frame_states(60, [], [response], 4, 15)


def test_sampling_schedule_handles_nondivisible_fps_and_partial_intervals():
    assert sampling_indices(8, 8, 4) == [1, 3, 5, 7]
    assert sampling_indices(30, 30, 4) == [7, 14, 22, 29]
    assert sampling_indices(2, 2, 4) == [0, 0, 1, 1]
    assert sampling_indices(9, 8, 4) == [1, 3, 5, 7, 8]
    assert time_tag(4, 4, 9 / 8) == '<1s-1.125s>'
    for fps in (0, -1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            sampling_indices(8, 8, fps)


def test_fifteen_second_conversion_preserves_endpoint(tmp_path):
    path = make_video(tmp_path / 'clip.avi', 8, 120)
    row = convert_format_to_stream({
        'video_path': path,
        'question': [{'content': 'Q', 'time': 0}],
        'response': [
            {'content': 'event', 'st_time': 3.25, 'end_time': 3.75},
            {'content': 'last', 'st_time': 14.5, 'end_time': 15},
        ],
    }, fps=4)
    messages = row['messages']
    assert len(messages) == 121
    assert messages[1]['content'] == 'Q\n<0s-0.25s>\n<stream>'
    assert messages[30]['content'] == '</Response> event'
    assert messages[-2]['content'] == '<14.75s-15s>\n<stream>'
    assert messages[-1]['content'] == '</Response> last'
    assert row['streaming_metadata']['annotation_count'] == 2
    assert '</Standby>' not in json.dumps(row)


def test_future_question_is_not_injected_at_start(tmp_path):
    path = make_video(tmp_path / 'question.avi', 8, 8)
    row = convert_format_to_stream({
        'video_path': path, 'question': [{'content': 'later', 'time': 0.75}], 'response': [],
    }, fps=4)
    assert row['messages'][1]['content'] == '<0s-0.25s>\n<stream>'
    assert row['messages'][5]['content'].startswith('later\n<0.5s-0.75s>')
    _, questions = build_frame_states(4, [{'content': 'untimed'}], [], 4)
    assert questions == {0: 'untimed'}


def test_one_fps_uses_same_interval_end_convention():
    states, _ = build_frame_states(4, [], [{'content': 'end', 'st_time': 3, 'end_time': 4}], 1)
    assert states == ['</Silence>'] * 3 + ['</Response> end']
    assert time_tag(3, 1) == '<3s-4s>'


def test_report_counts_merged_and_rejected_annotations(tmp_path):
    path = make_video(tmp_path / 'report.avi', 8, 8)
    output = tmp_path / 'output.json'
    output.write_text('[]\n')
    output.chmod(0o640)
    diagnostics = {}
    kept, failed = convert_rows_to_stream([
        {'video_path': path, 'response': [{'content': 'a', 'time': 1}, {'content': 'b', 'time': 1}]},
        {'video_path': path, 'response': [{'content': 'bad', 'time': 1.25}]},
    ], output, fps=4, num_workers=2, diagnostics=diagnostics)
    assert kept == 1 and failed == [1]
    assert output.stat().st_mode & 0o777 == 0o640
    assert diagnostics['input_responses'] == 3
    assert diagnostics['retained_responses'] == 2
    assert diagnostics['dropped_responses'] == 1
    assert diagnostics['merged_responses'] == 1
    assert diagnostics['response_frames'] == 1
    assert diagnostics['silence_frames'] == 3
    assert 'exceeds video duration' in diagnostics['conversion_errors'][0]['error']


@pytest.mark.parametrize('source_fps,frames,expected_indices', [
    (8, 8, [1, 3, 5, 7]), (30, 30, [7, 14, 22, 29]),
    (2, 2, [0, 0, 1, 1]), (8, 9, [1, 3, 5, 7, 8]),
])
def test_training_and_inference_extract_identical_interval_end_frames(
        tmp_path, source_fps, frames, expected_indices):
    from swift.llm.dataset.preprocessor.streaming_video import StreamingVideoPreprocessor, VideoFrameExtractor

    path = make_video(tmp_path / 'frames.avi', source_fps, frames)
    cap = cv2.VideoCapture(path)
    decoded = [cap.read()[1] for _ in range(frames)]
    cap.release()
    expected = [cv2.cvtColor(decoded[i], cv2.COLOR_BGR2RGB) for i in expected_indices]
    extractor = VideoFrameExtractor(fps=4, enable_memory_cache=False)
    assert SAMPLING_VERSION in extractor._get_cache_key(path)
    assert extractor.get_frame_count(path) == len(expected)
    memory = extractor.extract(path)
    cap = cv2.VideoCapture(path)
    sequential = extractor._extract_to_memory_sequential(cap)
    cap.release()
    disk = VideoFrameExtractor(fps=4, save_frames=True, output_dir=str(tmp_path / 'cache'))
    paths = disk.extract(path)
    assert disk.extract(path) == paths
    for i, frame in enumerate(expected):
        np.testing.assert_array_equal(np.asarray(memory[i]), frame)
        np.testing.assert_array_equal(np.asarray(sequential[i]), frame)
        np.testing.assert_array_equal(cv2.cvtColor(cv2.imread(paths[i]), cv2.COLOR_BGR2RGB), frame)
    for module_path in ('inference.py', 'examples/infer/streaming_action_caption_demo.py'):
        module = load_module('interval_inference', module_path)
        infer = module.VideoFrameExtractor(path, target_fps=4)
        try:
            assert infer.get_total_rounds() == len(expected)
            for i, frame in enumerate(expected):
                np.testing.assert_array_equal(np.asarray(infer.get_frame_at_round(i)), frame)
        finally:
            infer.close()
    row = convert_format_to_stream({'video_path': path, 'response': []}, fps=4)
    processed = StreamingVideoPreprocessor(fps=4, frame_tolerance=0).preprocess(row)
    assert len(processed['images']) == len(expected)
    assert StreamingVideoPreprocessor(fps=1).preprocess(row) is None
    # New datasets never silently duplicate or trim a frame to fix a mismatch.
    row['messages'] = row['messages'][:-2]
    assert StreamingVideoPreprocessor(fps=4, frame_tolerance=1).preprocess(row) is None


def test_sequential_fallback_and_decode_failure():
    class Capture:
        def __init__(self):
            self.index = 0

        def set(self, prop, index):
            self.index = 0
            return index == 0

        def get(self, prop):
            return self.index

        def read(self):
            if self.index >= 8:
                return False, None
            frame = np.array([self.index])
            self.index += 1
            return True, frame

    assert [int(f[0]) for f in iter_sampled_frames(Capture(), [1, 3, 3, 7])] == [1, 3, 3, 7]
    with pytest.raises(ValueError, match='Cannot decode'):
        list(iter_sampled_frames(Capture(), [8]))


def test_inference_tags_and_prompts_match_training(tmp_path):
    path = make_video(tmp_path / 'prompt.avi', 8, 8)
    module = load_module('interval_prompt_inference', 'inference.py')
    extractor = module.VideoFrameExtractor(path, target_fps=4)
    try:
        for builder in (module.get_data_stream_video, module.get_data_stream_video_window):
            row = builder(extractor, 0, system=SYSTEM_PROMPT, question='Q')
            row = builder(extractor, 1, question='Q', data=row, answer='</Silence>')
            assert row['messages'][-1]['content'] == '<0.25s-0.5s>\n<image>'
        assert module.SYSTEM == SYSTEM_PROMPT
        assert '</Standby>' not in SYSTEM_PROMPT
    finally:
        extractor.close()
    demo = load_module('interval_prompt_demo', 'examples/infer/streaming_action_caption_demo.py')
    assert demo.SYSTEM_PROMPT == SYSTEM_PROMPT
    extractor = demo.VideoFrameExtractor(path, target_fps=4)
    try:
        row = None
        for i in range(3):
            row = demo.build_stream_window(
                video_extractor=extractor, round_num=i, fps=4, system=SYSTEM_PROMPT,
                question='Q', question_time=0, data=row,
                answer=None if i == 0 else '</Silence>', max_rounds=2)
        assert len(row['images']) == 2
        assert row['messages'][1]['content'] == 'Q\n<0.25s-0.5s>\n<image>'
        assert row['messages'][-1]['content'] == '<0.5s-0.75s>\n<image>'
    finally:
        extractor.close()


def test_loss_uses_two_state_tokens_with_finite_gradients():
    import torch
    from types import SimpleNamespace

    loss_module = load_module('interval_loss', 'swift/plugin/loss.py')
    requested = []

    class Tokenizer:
        def encode(self, token, add_special_tokens):
            requested.append(token)
            return [{'</Silence>': 1, '</Response>': 2}[token]]

    logits = torch.zeros((1, 5, 6), requires_grad=True)
    labels = torch.tensor([[-100, 1, 1, 2, 4]])
    loss = loss_module.stream_focal_loss(
        SimpleNamespace(logits=logits), labels,
        trainer=SimpleNamespace(processing_class=Tokenizer()))
    assert requested == ['</Silence>', '</Response>']
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
