"""Tests for the shared formatting helpers in wyoming_bench.reporting."""

from __future__ import annotations

import unittest

from wyoming_bench.reporting import SERVER_COL_WIDTH, truncate_label


class TestTruncateLabel(unittest.TestCase):
    def test_short_label_unchanged(self):
        self.assertEqual(truncate_label("10.0.0.1:10700"), "10.0.0.1:10700")

    def test_label_at_width_unchanged(self):
        label = "x" * SERVER_COL_WIDTH
        self.assertEqual(truncate_label(label), label)

    def test_long_label_truncated_with_ellipsis(self):
        label = "10.100.1.20:10300 (faster-whisper-large-v3-turbo)"
        out = truncate_label(label)
        self.assertEqual(len(out), SERVER_COL_WIDTH)
        self.assertTrue(out.endswith("..."))

    def test_custom_width(self):
        self.assertEqual(truncate_label("abcdef", 5), "ab...")


if __name__ == "__main__":
    unittest.main()
