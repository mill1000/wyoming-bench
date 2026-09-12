"""Tests for the shared formatting helpers in wyoming_bench.reporting."""

from __future__ import annotations

import unittest

from wyoming_bench.reporting import (
    SERVER_COL_WIDTH,
    _fmt_cell,
    stat_table,
    stats,
    truncate_label,
)


class TestStatTable(unittest.TestCase):
    def test_labels_are_column_headers(self):
        # distinct min/mean/median/p95/max so each value is found exactly once
        s = stats([1.0, 2.0, 3.0, 4.0, 9.0])
        header, row = stat_table([("m", s, _fmt_cell)])
        for label, value in (
            ("min", "1.00"),
            ("mean", "3.80"),
            ("median", "3.00"),
            ("p95", "8.00"),
            ("max", "9.00"),
        ):
            # stat names appear only in the header, right-aligned above their values
            self.assertIn(label, header)
            self.assertNotIn(label, row)
            self.assertEqual(
                header.index(label) + len(label),
                row.index(value) + len(value),
                f"column for {label!r} misaligned: {header!r} / {row!r}",
            )

    def test_multiple_rows_share_one_header(self):
        s = stats([1.0, 2.0, 3.0, 4.0, 9.0])
        lines = stat_table([("a", s, _fmt_cell), ("b", s, _fmt_cell)])
        self.assertEqual(len(lines), 3)
        self.assertEqual(len(lines[0]), len(lines[1]))

    def test_empty_stats_render_n_a(self):
        _, row = stat_table([("m", stats([]), _fmt_cell)])
        self.assertEqual(row.count("n/a"), 5)


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
