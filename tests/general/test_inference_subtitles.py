import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / 'inference.py'


def load_inference_module():
    spec = importlib.util.spec_from_file_location('streamo_inference', SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestInferenceSubtitles(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.module = load_inference_module()

    def test_silence_body_becomes_subtitle_but_tag_only_does_not(self):
        self.assertEqual(
            self.module.build_subtitle_text(self.module.STATE_SILENCE, 'Customer is waiting.'),
            'Customer is waiting.',
        )
        self.assertEqual(self.module.build_subtitle_text(self.module.STATE_SILENCE, ''), '')
        self.assertEqual(self.module.build_subtitle_text(self.module.STATE_STANDBY, 'Pending.'), '')

    def test_caption_is_rendered_for_eight_seconds(self):
        records = [
            {
                'start_sec': 0.0,
                'end_sec': 1.0,
                'subtitle_text': 'Customer is waiting.',
                'subtitle_end_sec': 8.0,
            },
            *[
                {
                    'start_sec': float(second),
                    'end_sec': float(second + 1),
                    'subtitle_text': '',
                    'subtitle_end_sec': float(second),
                }
                for second in range(1, 9)
            ],
        ]

        self.assertEqual(
            self.module.subtitle_for_time(time_sec=7.999, fps=1.0, round_records=records),
            'Customer is waiting.',
        )
        self.assertEqual(self.module.subtitle_for_time(time_sec=8.0, fps=1.0, round_records=records), '')

    def test_non_empty_subtitle_duration_defaults_to_eight_seconds(self):
        self.assertEqual(
            self.module.subtitle_duration_sec(
                self.module.STATE_RESPONSE,
                'An agent responds.',
                fps=1.0,
            ),
            8.0,
        )
        self.assertEqual(
            self.module.subtitle_duration_sec(
                self.module.STATE_SILENCE,
                'Customer is waiting.',
                fps=1.0,
            ),
            8.0,
        )

    def test_newer_caption_replaces_active_silence_caption(self):
        records = [
            {
                'start_sec': 0.0,
                'end_sec': 1.0,
                'subtitle_text': 'Customer is waiting.',
                'subtitle_end_sec': 4.0,
            },
            {'start_sec': 1.0, 'end_sec': 2.0, 'subtitle_text': '', 'subtitle_end_sec': 1.0},
            {
                'start_sec': 2.0,
                'end_sec': 3.0,
                'subtitle_text': 'An agent responds.',
                'subtitle_end_sec': 3.0,
            },
            {'start_sec': 3.0, 'end_sec': 4.0, 'subtitle_text': '', 'subtitle_end_sec': 3.0},
        ]

        self.assertEqual(
            self.module.subtitle_for_time(time_sec=2.5, fps=1.0, round_records=records),
            'An agent responds.',
        )
        self.assertEqual(self.module.subtitle_for_time(time_sec=3.5, fps=1.0, round_records=records), '')


if __name__ == '__main__':
    unittest.main()
