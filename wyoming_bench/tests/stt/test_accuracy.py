"""Tests for wyoming_tts_bench.stt.accuracy (word/char error metrics)."""

from __future__ import annotations

import unittest

from wyoming_bench.stt.accuracy import (
    char_accuracy,
    normalize_text,
    tokenize_words,
    word_accuracy,
)


class TestNormalize(unittest.TestCase):
    def test_lowercase_and_punctuation(self):
        self.assertEqual(normalize_text("Hello, World!"), "hello world")

    def test_collapse_whitespace(self):
        self.assertEqual(normalize_text("a\t\t b\n c"), "a b c")

    def test_strip(self):
        self.assertEqual(normalize_text("  hi  "), "hi")

    def test_emoji_dropped(self):
        self.assertEqual(normalize_text("hi :) there"), "hi there")

    def test_empty(self):
        self.assertEqual(normalize_text(""), "")
        self.assertEqual(normalize_text("   "), "")


class TestTokenize(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(tokenize_words("Hello,  world"), ["hello", "world"])

    def test_empty(self):
        self.assertEqual(tokenize_words(""), [])


class TestWordAccuracy(unittest.TestCase):
    def test_exact(self):
        r = word_accuracy("hello world", "hello world")
        self.assertTrue(r["exact"])
        self.assertEqual(r["errors"], 0)
        self.assertEqual(r["n_ref"], 2)
        self.assertEqual(r["wer"], 0.0)

    def test_normalization_ignores_punct_case(self):
        r = word_accuracy("Hello,  World!", "hello world")
        self.assertTrue(r["exact"])
        self.assertEqual(r["wer"], 0.0)

    def test_substitution(self):
        r = word_accuracy("the cat ran", "the dog ran")
        self.assertEqual((r["sub"], r["ins"], r["del"]), (1, 0, 0))
        self.assertEqual(r["n_ref"], 3)
        self.assertEqual(r["errors"], 1)
        self.assertAlmostEqual(r["wer"], 1 / 3)
        self.assertFalse(r["exact"])

    def test_insertion(self):
        r = word_accuracy("the cat sat", "cat sat")
        self.assertEqual((r["sub"], r["ins"], r["del"]), (0, 1, 0))
        self.assertEqual(r["n_ref"], 2)
        self.assertAlmostEqual(r["wer"], 0.5)

    def test_deletion(self):
        r = word_accuracy("the sat", "the cat sat")
        self.assertEqual((r["sub"], r["ins"], r["del"]), (0, 0, 1))
        self.assertEqual(r["n_ref"], 3)
        self.assertAlmostEqual(r["wer"], 1 / 3)

    def test_single_substitution(self):
        r = word_accuracy("a b c d", "a x c d")
        self.assertEqual(r["sub"], 1)
        self.assertEqual(r["errors"], 1)
        self.assertAlmostEqual(r["wer"], 0.25)

    def test_empty_reference(self):
        r = word_accuracy("hello world", "")
        self.assertIsNone(r["wer"])
        self.assertEqual(r["n_ref"], 0)
        self.assertEqual(r["errors"], 2)
        self.assertFalse(r["exact"])

    def test_both_empty(self):
        r = word_accuracy("", "")
        self.assertIsNone(r["wer"])
        self.assertEqual(r["errors"], 0)
        self.assertTrue(r["exact"])

    def test_whole_mismatch(self):
        r = word_accuracy("foo bar", "baz qux")
        self.assertEqual(r["errors"], 2)
        self.assertEqual(r["sub"], 2)
        self.assertAlmostEqual(r["wer"], 1.0)


class TestCharAccuracy(unittest.TestCase):
    def test_exact(self):
        r = char_accuracy("hello", "hello")
        self.assertEqual(r["errors"], 0)
        self.assertEqual(r["n_ref"], 5)
        self.assertEqual(r["cer"], 0.0)

    def test_normalization(self):
        r = char_accuracy("Hello,  World!", "hello world")
        self.assertEqual(r["errors"], 0)
        self.assertEqual(r["n_ref"], 11)
        self.assertEqual(r["cer"], 0.0)

    def test_deletion(self):
        r = char_accuracy("helo", "hello")
        self.assertEqual(r["errors"], 1)
        self.assertEqual(r["n_ref"], 5)
        self.assertAlmostEqual(r["cer"], 0.2)

    def test_empty_reference(self):
        r = char_accuracy("abc", "")
        self.assertIsNone(r["cer"])
        self.assertEqual(r["n_ref"], 0)
        self.assertEqual(r["errors"], 3)


if __name__ == "__main__":
    unittest.main()
