"""Load an STT benchmark corpus: WAV recordings paired with transcript text.

A corpus directory contains ``<stem>.wav`` recordings and matching
``<stem>.txt`` transcripts (e.g. ``data_01.wav`` + ``data_01.txt``). Files
without a matching counterpart are skipped with a warning.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AudioInput:
    """A single benchmark sample: a PCM recording plus its reference transcript."""

    sample_id: str
    reference: str
    rate: int
    width: int
    channels: int
    pcm: bytes
    audio_path: Path

    @property
    def duration_s(self) -> float:
        denom = self.rate * self.width * self.channels
        if denom <= 0:
            return 0.0
        return len(self.pcm) / denom


def _read_wav(path: Path) -> tuple[int, int, int, bytes]:
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        width = wf.getsampwidth()
        channels = wf.getnchannels()
        pcm = wf.readframes(wf.getnframes())
    return rate, width, channels, pcm


def _list_wavs(directory: Path) -> list[Path]:
    """All regular ``.wav`` files directly under *directory*, sorted by name."""
    return sorted(
        (p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".wav"),
        key=lambda p: p.name.lower(),
    )


def load_corpus(directory: Path | str) -> list[AudioInput]:
    """Load all ``.wav``/``.txt`` pairs under *directory*, sorted by sample id.

    Raises ``FileNotFoundError`` if the directory is missing or contains no
    valid pairs.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {directory}")

    wavs = _list_wavs(directory)
    samples: list[AudioInput] = []
    for wav_path in wavs:
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.is_file():
            print(f"  [corpus] no transcript for {wav_path.name}; skipping", flush=True)
            continue
        reference = txt_path.read_text(encoding="utf-8").strip()
        try:
            rate, width, channels, pcm = _read_wav(wav_path)
        except (wave.Error, EOFError) as e:
            print(f"  [corpus] cannot read {wav_path.name}: {e}; skipping", flush=True)
            continue
        samples.append(
            AudioInput(
                sample_id=wav_path.stem,
                reference=reference,
                rate=rate,
                width=width,
                channels=channels,
                pcm=pcm,
                audio_path=wav_path,
            )
        )

    # Warn about transcripts that have no recording.
    wav_stem_names = {w.stem for w in wavs}
    for txt_path in sorted(
        (p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".txt"),
        key=lambda p: p.name.lower(),
    ):
        if txt_path.stem not in wav_stem_names:
            print(f"  [corpus] no recording for {txt_path.name}; skipping", flush=True)

    samples.sort(key=lambda s: s.sample_id)
    if not samples:
        raise FileNotFoundError(f"no .wav/.txt pairs found in {directory}")
    return samples


def load_recordings(directory: Path | str) -> list[AudioInput]:
    """Load all ``.wav`` recordings under *directory*, paired or not.

    Unlike :func:`load_corpus`, recordings **without** a ``.txt`` transcript
    are included (with an empty reference): this is the input for the
    ``seed`` command, which writes one transcript per recording. Files that
    cannot be read as WAV are skipped with a warning.

    Raises ``FileNotFoundError`` if the directory is missing or contains no
    readable recordings.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {directory}")

    samples: list[AudioInput] = []
    for wav_path in _list_wavs(directory):
        try:
            rate, width, channels, pcm = _read_wav(wav_path)
        except (wave.Error, EOFError) as e:
            print(f"  [corpus] cannot read {wav_path.name}: {e}; skipping", flush=True)
            continue
        samples.append(
            AudioInput(
                sample_id=wav_path.stem,
                reference="",
                rate=rate,
                width=width,
                channels=channels,
                pcm=pcm,
                audio_path=wav_path,
            )
        )

    if not samples:
        raise FileNotFoundError(f"no .wav recordings found in {directory}")
    return samples
