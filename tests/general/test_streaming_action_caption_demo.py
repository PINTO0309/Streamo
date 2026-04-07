import importlib.util
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / 'examples' / 'infer' / 'streaming_action_caption_demo.py'


def load_demo_module():
    spec = importlib.util.spec_from_file_location('streaming_action_caption_demo', SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def create_constant_video(path: Path, *, fps: float, frames: int, value: int = 96) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (160, 90))
    assert writer.isOpened(), f'Failed to create video: {path}'
    frame = np.full((90, 160, 3), value, dtype=np.uint8)
    for _ in range(frames):
        writer.write(frame)
    writer.release()
    assert path.exists()
    return path


def read_frame(video_path: Path, index: int):
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f'Failed to open video: {video_path}'
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    success, frame = cap.read()
    cap.release()
    assert success, f'Failed to read frame {index} from {video_path}'
    return frame


class TestStreamingActionCaptionDemo(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.module = load_demo_module()

    def test_build_subtitle_text_only_uses_response_body(self):
        self.assertEqual(self.module.build_subtitle_text(self.module.STATE_RESPONSE, 'A caption.'), 'A caption.')
        self.assertEqual(self.module.build_subtitle_text(self.module.STATE_STANDBY, 'Pending.'), '')
        self.assertEqual(self.module.build_subtitle_text(self.module.STATE_SILENCE, ''), '')

    def test_render_subtitle_video_burns_caption_into_output(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_video = create_constant_video(tmp_path / 'input.mp4', fps=4.0, frames=8)
            output_video = tmp_path / 'output.mp4'
            round_records = [
                {'end_sec': 0.5, 'subtitle_text': ''},
                {'end_sec': 1.0, 'subtitle_text': 'Test subtitle'},
                {'end_sec': 1.5, 'subtitle_text': 'Test subtitle'},
                {'end_sec': 2.0, 'subtitle_text': ''},
            ]

            self.module.render_subtitle_video(
                video_path=str(input_video),
                output_path=str(output_video),
                fps=2.0,
                round_records=round_records,
                font_path=None,
                max_lines=2,
                extend_last_subtitle=False,
            )

            self.assertTrue(output_video.exists())

            frame_without_subtitle = read_frame(output_video, 0)
            frame_with_subtitle = read_frame(output_video, 3)
            pixel_delta = np.abs(frame_with_subtitle.astype(np.int16) - frame_without_subtitle.astype(np.int16)).sum()
            self.assertGreater(pixel_delta, 0)

