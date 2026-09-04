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

    def test_no_target_user_response_is_excluded_from_subtitles(self):
        self.assertEqual(
            self.module.build_subtitle_text(
                self.module.STATE_RESPONSE,
                '  no TARGET user is visible.',
            ),
            '',
        )
        self.assertEqual(
            self.module.build_subtitle_text(
                self.module.STATE_RESPONSE,
                'The result mentions No target user later.',
            ),
            'The result mentions No target user later.',
        )

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

    def test_response_marker_times_only_returns_responses(self):
        records = [
            {'start_sec': 1.0, 'response_type': self.module.STATE_SILENCE},
            {
                'start_sec': 2.0,
                'response_type': self.module.STATE_RESPONSE,
                'response_body': 'A target user is waiting.',
            },
            {
                'start_sec': 3.0,
                'response_type': self.module.STATE_RESPONSE,
                'response_body': 'No target user is visible.',
            },
            {
                'start_sec': 4.0,
                'response_type': self.module.STATE_RESPONSE,
                'response': '</Response> No target user appears in this interval.',
            },
            {'start_sec': 4.5, 'response_type': self.module.STATE_RESPONSE},
        ]

        self.assertEqual(self.module.response_marker_times(records), [2.0, 4.5])

    def test_no_target_user_response_stops_an_older_caption(self):
        records = [
            {
                'start_sec': 0.0,
                'end_sec': 1.0,
                'response_type': self.module.STATE_RESPONSE,
                'response_body': 'A customer is waiting.',
                'subtitle_text': 'A customer is waiting.',
                'subtitle_end_sec': 8.0,
            },
            {
                'start_sec': 1.0,
                'end_sec': 2.0,
                'response_type': self.module.STATE_RESPONSE,
                'response_body': 'No target user is visible.',
                'subtitle_text': '',
                'subtitle_end_sec': 1.0,
            },
        ]

        self.assertEqual(
            self.module.subtitle_for_time(time_sec=1.5, fps=1.0, round_records=records),
            '',
        )

    def test_adjacent_standby_rounds_are_combined_into_ranges(self):
        records = [
            {'start_sec': 0.0, 'end_sec': 1.0, 'response_type': self.module.STATE_SILENCE},
            {'start_sec': 1.0, 'end_sec': 2.0, 'response_type': self.module.STATE_STANDBY},
            {'start_sec': 2.0, 'end_sec': 3.0, 'response_type': self.module.STATE_STANDBY},
            {'start_sec': 3.0, 'end_sec': 4.0, 'response_type': self.module.STATE_RESPONSE},
            {'start_sec': 4.0, 'end_sec': 5.0, 'response_type': self.module.STATE_STANDBY},
        ]

        self.assertEqual(self.module.standby_time_ranges(records), [(1.0, 3.0), (4.0, 5.0)])

    def test_timeline_shows_future_response_markers_from_first_frame(self):
        frame = self.module.np.full((180, 320, 3), 255, dtype=self.module.np.uint8)

        rendered = self.module.draw_timeline_on_frame(
            frame,
            time_sec=0.0,
            duration_sec=10.0,
            response_times=[8.0],
        )

        red_response_pixels = (
            (rendered[:, :, 2] > 220)
            & (rendered[:, :, 1] < 130)
            & (rendered[:, :, 0] < 80)
        )
        self.assertTrue(red_response_pixels.any())

    def test_timeline_colors_standby_ranges_blue(self):
        frame = self.module.np.full((180, 320, 3), 255, dtype=self.module.np.uint8)

        rendered = self.module.draw_timeline_on_frame(
            frame,
            time_sec=0.0,
            duration_sec=10.0,
            response_times=[],
            standby_ranges=[(4.0, 6.0)],
        )

        blue_standby_pixels = (
            (rendered[:, :, 0] > 180)
            & (rendered[:, :, 1] > 110)
            & (rendered[:, :, 1] < 180)
            & (rendered[:, :, 2] < 120)
        )
        self.assertTrue(blue_standby_pixels.any())


if __name__ == '__main__':
    unittest.main()
