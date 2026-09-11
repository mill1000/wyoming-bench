"""Tests for wyoming_tts_bench.tts.texts (default texts + sentence splitting)."""

from __future__ import annotations

import unittest

from wyoming_bench.tts.texts import DEFAULT_TEXTS, split_sentences


class TestSplitSentences(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(split_sentences(""), [])

    def test_whitespace_only(self):
        self.assertEqual(split_sentences("   \n\t  "), [])

    def test_single_with_terminator(self):
        self.assertEqual(split_sentences("Hello there."), ["Hello there."])

    def test_single_no_terminator(self):
        self.assertEqual(split_sentences("No terminator here"), ["No terminator here"])

    def test_multiple(self):
        self.assertEqual(split_sentences("One. Two! Three? Four"), ["One. Two!", "Three?", "Four"])

    def test_strips_outer_whitespace(self):
        self.assertEqual(split_sentences("  A.   B  "), ["A.   B"])

    def test_unspaced_ellipsis_not_a_boundary(self):
        self.assertEqual(len(split_sentences("Wait... what?!")), 1)

    def test_abbreviation_held(self):
        self.assertEqual(
            split_sentences("Mr. Smith went to Washington. He arrived early."),
            ["Mr. Smith went to Washington.", "He arrived early."],
        )

    def test_acronym_held(self):
        self.assertEqual(
            split_sentences("The U.S. government said hi. It was fast."),
            ["The U.S. government said hi.", "It was fast."],
        )

    def test_short_fragments_merged(self):
        # sentence-stream errs toward whole sentences: runs of one-word
        # fragments are merged rather than over-split.
        self.assertEqual(split_sentences("One. Two. Three!"), ["One. Two. Three!"])
        self.assertEqual(
            split_sentences("One two. Three four! Five six?"),
            ["One two.", "Three four!", "Five six?"],
        )


class TestDefaultTexts(unittest.TestCase):
    def test_has_four_entries(self):
        self.assertEqual(len(DEFAULT_TEXTS), 4)

    def test_sentence_counts(self):
        for i, expected in enumerate((1, 2, 3, 4)):
            with self.subTest(i=i):
                self.assertEqual(len(split_sentences(DEFAULT_TEXTS[i])), expected)


if __name__ == "__main__":
    unittest.main()
