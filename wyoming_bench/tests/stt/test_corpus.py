"""Tests for wyoming_tts_bench.stt.corpus (WAV + transcript corpus loading)."""

from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

from wyoming_bench.stt.corpus import load_corpus, load_recordings


def _write_wav(path: Path, rate: int = 16000, width: int = 2, channels: int = 1, frames: int = 1600) -> bytes:
    """Write a tiny PCM WAV and return the exact raw PCM bytes written."""
    pcm = bytes((i % 256) for i in range(frames * width * channels))
    with wave.open(str(path), "wb") as wf:
        wf.setframerate(rate)
        wf.setsampwidth(width)
        wf.setnchannels(channels)
        wf.writeframes(pcm)
    return pcm


class TestLoadCorpus(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_loads_matched_pair(self):
        pcm = _write_wav(self.dir / "data_01.wav")
        (self.dir / "data_01.txt").write_text("  hello world\n", encoding="utf-8")

        samples = load_corpus(self.dir)
        self.assertEqual(len(samples), 1)
        s = samples[0]
        self.assertEqual(s.sample_id, "data_01")
        self.assertEqual(s.reference, "hello world")  # transcript is stripped
        self.assertEqual((s.rate, s.width, s.channels), (16000, 2, 1))
        self.assertEqual(s.pcm, pcm)
        self.assertAlmostEqual(s.duration_s, len(pcm) / (16000 * 2 * 1))

    def test_skips_unmatched_files(self):
        _write_wav(self.dir / "only_wav.wav")  # no transcript
        (self.dir / "only_txt.txt").write_text("orphan", encoding="utf-8")  # no wav
        pcm = _write_wav(self.dir / "good.wav")
        (self.dir / "good.txt").write_text("good one", encoding="utf-8")

        samples = load_corpus(self.dir)
        self.assertEqual([s.sample_id for s in samples], ["good"])
        self.assertEqual(samples[0].pcm, pcm)

    def test_sorted_by_id(self):
        for name in ("b_2.wav", "a_1.wav"):
            _write_wav(self.dir / name)
        (self.dir / "a_1.txt").write_text("a", encoding="utf-8")
        (self.dir / "b_2.txt").write_text("b", encoding="utf-8")

        samples = load_corpus(self.dir)
        self.assertEqual([s.sample_id for s in samples], ["a_1", "b_2"])

    def test_missing_directory(self):
        with self.assertRaises(FileNotFoundError):
            load_corpus(self.dir / "nope")

    def test_no_pairs(self):
        (self.dir / "orphan.txt").write_text("x", encoding="utf-8")
        with self.assertRaises(FileNotFoundError):
            load_corpus(self.dir)


class TestLoadRecordings(unittest.TestCase):
    """seed's input: every .wav is loaded, paired or not."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_loads_wavs_without_transcripts(self):
        pcm = _write_wav(self.dir / "rec_01.wav")

        samples = load_recordings(self.dir)
        self.assertEqual(len(samples), 1)
        s = samples[0]
        self.assertEqual(s.sample_id, "rec_01")
        self.assertEqual(s.reference, "")  # no transcript yet; seed will write one
        self.assertEqual(s.pcm, pcm)
        self.assertEqual((s.rate, s.width, s.channels), (16000, 2, 1))

    def test_loads_wavs_that_already_have_transcripts(self):
        # Already-seeded recordings must still be listed (seed then skips
        # them via the existing .txt, or re-transcribes with --overwrite).
        _write_wav(self.dir / "rec_01.wav")
        (self.dir / "rec_01.txt").write_text("hello", encoding="utf-8")

        samples = load_recordings(self.dir)
        self.assertEqual([s.sample_id for s in samples], ["rec_01"])

    def test_skips_unreadable_wav(self):
        (self.dir / "bad.wav").write_bytes(b"not a wav")
        pcm = _write_wav(self.dir / "good.wav")

        samples = load_recordings(self.dir)
        self.assertEqual([s.sample_id for s in samples], ["good"])
        self.assertEqual(samples[0].pcm, pcm)

    def test_sorted_by_id(self):
        for name in ("b_2.wav", "a_1.wav"):
            _write_wav(self.dir / name)

        samples = load_recordings(self.dir)
        self.assertEqual([s.sample_id for s in samples], ["a_1", "b_2"])

    def test_missing_directory(self):
        with self.assertRaises(FileNotFoundError):
            load_recordings(self.dir / "nope")

    def test_no_wavs(self):
        (self.dir / "orphan.txt").write_text("x", encoding="utf-8")
        with self.assertRaises(FileNotFoundError):
            load_recordings(self.dir)


if __name__ == "__main__":
    unittest.main()
