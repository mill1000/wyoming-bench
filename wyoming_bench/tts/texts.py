"""Default benchmark texts; sentence splitting via ``sentence-stream``.

Streaming synthesis sends text sentence-by-sentence as ``synthesize-chunk``
events, so the split quality directly shapes what the server sees mid-stream.
``sentence-stream`` (Home Assistant, Apache-2.0) is a heuristic splitter that
holds abbreviations/initials (``Mr.``, ``U.S.``) together, handles CJK and
other scripts, and merges very short fragments rather than over-splitting.
"""

from __future__ import annotations

from sentence_stream import stream_to_sentences

# One sample per sentence count (1..4). Used as the default corpus.
DEFAULT_TEXTS: list[str] = [
    "The weather today is sunny and warm with a light breeze.",
    "The quick brown fox jumps over the lazy dog. This sentence tests two-sentence synthesis.",
    "Text to speech is a fascinating field. Modern neural models produce remarkably natural audio. We benchmark them to compare performance.",
    "Welcome to the Wyoming protocol benchmark. We measure latency, streaming behavior, and throughput. Each sample is between one and four sentences long. Results help us choose the best engine for real-time use.",
]


def split_sentences(text: str) -> list[str]:
    """Split *text* into one or more sentences (sentence-stream heuristics).

    Returns an empty list for empty/whitespace-only input.
    """
    text = text.strip()
    if not text:
        return []
    return [part.strip() for part in stream_to_sentences(text) if part.strip()]
