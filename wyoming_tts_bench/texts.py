"""Default benchmark texts and a small sentence splitter for streaming chunks."""

from __future__ import annotations

import re

# One sample per sentence count (1..4). Used as the default corpus.
DEFAULT_TEXTS: list[str] = [
    "The weather today is sunny and warm with a light breeze.",
    "The quick brown fox jumps over the lazy dog. This sentence tests two-sentence synthesis.",
    "Text to speech is a fascinating field. Modern neural models produce remarkably natural audio. We benchmark them to compare performance.",
    "Welcome to the Wyoming protocol benchmark. We measure latency, streaming behavior, and throughput. Each sample is between one and four sentences long. Results help us choose the best engine for real-time use.",
]

# Sentence = run of non-terminators followed by one or more terminators (and a
# space or end), plus a trailing run with no terminator. Imperfect about
# abbreviations (``e.g.``), which is acceptable for a benchmark.
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]+(?:\s+|$)|[^.!?]+$")


def split_sentences(text: str) -> list[str]:
    """Split *text* into one or more sentences.

    Returns an empty list for empty/whitespace-only input.
    """
    text = text.strip()
    if not text:
        return []
    parts = [part.strip() for part in _SENTENCE_RE.findall(text)]
    return [part for part in parts if part]
