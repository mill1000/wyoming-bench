"""Tests for TTS RTF and cross-server normalization (wyoming_bench.tts.reporting)."""

from __future__ import annotations

import unittest

from wyoming_bench.tts.reporting import Measurement, normalized_stats, summarize_mode


def _meas(server: str, total_s: float, audio_duration_s: float, text: str = "hello world") -> Measurement:
    m = Measurement(server=server, mode="non_streaming", text=text)
    m.ok = True
    m.t_sent = 0.0
    m.t_first = total_s / 2
    m.t_end = total_s
    m.audio_duration_s = audio_duration_s
    return m


class TestTrueRtf(unittest.TestCase):
    def test_rtf_is_total_over_audio(self):
        self.assertAlmostEqual(_meas("s", 2.0, 10.0).rtf, 0.2)

    def test_rtf_none_when_not_ok(self):
        m = _meas("s", 1.0, 1.0)
        m.ok = False
        self.assertIsNone(m.rtf)

    def test_rtf_none_when_no_audio(self):
        self.assertIsNone(_meas("s", 1.0, 0.0).rtf)


class TestNormalizedStats(unittest.TestCase):
    def _rows(self):
        a = [_meas("a", 2.0, 19.09)]
        b = [_meas("b", 5.0, 9.27)]
        return [
            ("a", summarize_mode(a, "non_streaming")),
            ("b", summarize_mode(b, "non_streaming")),
        ]

    def test_rtf_norm_ratio_equals_wall_time_ratio(self):
        norm = normalized_stats(self._rows())
        self.assertAlmostEqual(norm["b"]["rtf_norm"] / norm["a"]["rtf_norm"], 5.0 / 2.0)

    def test_rtf_norm_uses_mean_audio(self):
        norm = normalized_stats(self._rows())
        mean_audio = (19.09 + 9.27) / 2
        self.assertAlmostEqual(norm["a"]["rtf_norm"], 2.0 / mean_audio)
        self.assertAlmostEqual(norm["b"]["rtf_norm"], 5.0 / mean_audio)

    def test_audio_dur_norm_scales_to_mean(self):
        # Each server's audio duration is scaled to the cross-server mean, so
        # the average server is 1.0 and the slow/fast speakers deviate.
        norm = normalized_stats(self._rows())
        mean_audio = (19.09 + 9.27) / 2
        self.assertAlmostEqual(norm["a"]["audio_dur_norm"], 19.09 / mean_audio)
        self.assertAlmostEqual(norm["b"]["audio_dur_norm"], 9.27 / mean_audio)
        # The mean of the two (equal weight) is exactly 1.0.
        self.assertAlmostEqual((norm["a"]["audio_dur_norm"] + norm["b"]["audio_dur_norm"]) / 2, 1.0)

    def test_true_rtf_still_biased(self):
        # The per-server RTF divides by each server's own audio duration, so a
        # server that speaks slowly (more audio) looks faster than it is
        # relative to a fast-speaking server. Normalization fixes this.
        a, b = _meas("a", 2.0, 19.09), _meas("b", 5.0, 9.27)
        self.assertNotAlmostEqual(b.rtf / a.rtf, 5.0 / 2.0)

    def test_empty(self):
        self.assertEqual(normalized_stats([]), {})


if __name__ == "__main__":
    unittest.main()
