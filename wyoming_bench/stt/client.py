"""Wyoming STT (ASR) client wrapper that performs a single timed measurement.

Uses the official async ``wyoming`` library (``AsyncTcpClient``). Each
measurement opens a fresh TCP connection so results are isolated per round.

Wyoming ASR event flows (see the Wyoming protocol spec):

- non-streaming:
    -> transcribe, audio-start, audio-chunk*, audio-stop
    <- transcript
- streaming:
    -> transcribe, audio-start, audio-chunk*   (concurrent with reading)
    <- transcript-start, transcript-chunk*
    -> audio-stop
    <- transcript (final, sent for backwards compatibility), transcript-stop

The final ``transcript`` event carries the authoritative full text, so both
modes score accuracy against it. In streaming mode TTFT is measured to the
first ``transcript-chunk``.
"""

from __future__ import annotations

import asyncio
import time

from wyoming.asr import (
    Transcribe,
    Transcript,
    TranscriptChunk,
    TranscriptStart,
    TranscriptStop,
)
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.error import Error

from ..const import MODE_NON_STREAMING, MODE_STREAMING
from .accuracy import char_accuracy, word_accuracy
from .corpus import AudioInput
from .reporting import SttMeasurement


class TranscriptionError(Exception):
    """Raised when the server does not return a complete transcript response."""


def _fill_accuracy(m: SttMeasurement) -> None:
    """Compute word/char error metrics for a successful measurement."""
    wa = word_accuracy(m.hypothesis, m.reference)
    ca = char_accuracy(m.hypothesis, m.reference)
    m.sub, m.ins, m.del_ = wa["sub"], wa["ins"], wa["del"]
    m.n_ref_words = wa["n_ref"]
    m.wer = wa["wer"]
    m.exact = wa["exact"]
    m.char_errors = ca["errors"]
    m.n_ref_chars = ca["n_ref"]
    m.cer = ca["cer"]


async def measure_stt_once(
    host: str,
    port: int,
    mode: str,
    audio: AudioInput,
    transcribe: Transcribe,
    chunk_samples: int,
    connect_timeout: float,
    read_timeout: float,
    chunk_delay: float = 0.0,
    trailing_silence: float = 0.0,
) -> SttMeasurement:
    """Run one transcription against *host*:*port* and return a SttMeasurement.

    Failures are captured on the returned measurement (``ok=False`` +
    ``error``) rather than raised, so a single bad run cannot abort a batch.
    A failed TCP connect additionally sets ``connection_failed``, so callers
    can stop re-attempting a down server instead of paying the timeout again.
    """
    m = SttMeasurement(
        server=f"{host}:{port}",
        mode=mode,
        sample_id=audio.sample_id,
        reference=audio.reference,
        audio_duration_s=audio.duration_s,
    )
    client = AsyncTcpClient(host, port, connect_timeout=connect_timeout, read_timeout=read_timeout)
    try:
        try:
            await client.connect()
        except asyncio.TimeoutError:
            m.error = f"unavailable: connection timed out (>{connect_timeout:g}s)"
            m.connection_failed = True
            return m
        except OSError as e:
            m.error = f"unavailable: {type(e).__name__}: {e}"
            m.connection_failed = True
            return m
        if mode == MODE_STREAMING:
            await _run_streaming(client, m, audio, transcribe, chunk_samples, chunk_delay, trailing_silence)
        else:
            await _run_non_streaming(client, m, audio, transcribe, chunk_samples, trailing_silence)
        if m.t_end is not None:
            m.ok = True
            _fill_accuracy(m)
    except asyncio.TimeoutError:
        m.error = f"timed out (>{read_timeout:.0f}s per event)"
    except TranscriptionError as e:
        m.error = str(e)
    except Exception as e:  # noqa: BLE001 - report any failure as a failed measurement
        m.error = f"{type(e).__name__}: {e}"
    finally:
        try:
            await client.disconnect()
        except Exception:  # noqa: BLE001
            pass
    return m


async def transcribe_audio(
    host: str,
    port: int,
    audio: AudioInput,
    transcribe: Transcribe,
    chunk_samples: int,
    connect_timeout: float,
    read_timeout: float,
) -> str:
    """Transcribe *audio* once and return the final transcript text.

    Non-streaming path only (seeding does not need partial results). Opens a
    fresh connection. Raises ``TranscriptionError`` (or ``TimeoutError``) on
    failure rather than returning a partial result.
    """
    client = AsyncTcpClient(host, port, connect_timeout=connect_timeout, read_timeout=read_timeout)
    try:
        await client.connect()
        m = SttMeasurement(
            server=f"{host}:{port}",
            mode=MODE_NON_STREAMING,
            sample_id=audio.sample_id,
            reference=audio.reference,
            audio_duration_s=audio.duration_s,
        )
        await _run_non_streaming(client, m, audio, transcribe, chunk_samples)
        if not m.hypothesis:
            raise TranscriptionError("empty transcript")
        return m.hypothesis
    finally:
        try:
            await client.disconnect()
        except Exception:  # noqa: BLE001
            pass


async def _send_audio(
    client: AsyncTcpClient,
    audio: AudioInput,
    chunk_samples: int,
    chunk_delay: float = 0.0,
    trailing_silence: float = 0.0,
) -> None:
    """Send the audio as ``audio-start`` / ``audio-chunk``* / ``audio-stop``.

    Chunks are sized to *chunk_samples* (a whole number of samples), so every
    block is valid PCM. *chunk_delay* paces writes (used to simulate real-time
    streaming); ``0`` sends as fast as the socket allows.

    *trailing_silence* (seconds) appends that much zero PCM to the end of the
    recording **on the wire only**, so streaming (online) ASR models see enough
    trailing context to finalize their last words. It does not change
    ``audio.duration_s``, which is what RTF is computed from.
    """
    await client.write_event(
        AudioStart(rate=audio.rate, width=audio.width, channels=audio.channels, timestamp=0).event()
    )
    chunk_bytes = max(1, chunk_samples * audio.width * audio.channels)
    pcm = audio.pcm
    if trailing_silence > 0:
        sil_samples = round(trailing_silence * audio.rate)
        pcm = pcm + b"\x00" * (sil_samples * audio.width * audio.channels)
    offset = 0
    n = len(pcm)
    while offset < n:
        block = pcm[offset : offset + chunk_bytes]
        await client.write_event(
            AudioChunk(
                rate=audio.rate,
                width=audio.width,
                channels=audio.channels,
                audio=block,
                timestamp=None,
            ).event()
        )
        offset += chunk_bytes
        if chunk_delay > 0:
            await asyncio.sleep(chunk_delay)
    await client.write_event(AudioStop().event())


async def _read_transcript(
    client: AsyncTcpClient,
    m: SttMeasurement,
    terminal_cls: type,
) -> None:
    """Read events until the terminal event, capturing the transcript.

    *terminal_cls* is the event that ends the measurement:

    - non-streaming: ``Transcript`` (a single final result);
    - streaming: ``TranscriptStop`` (the final ``transcript`` arrives first and
      is captured on the way through).

    ``read_event()`` yields base ``Event`` objects, so we dispatch on
    ``event.type`` and rebuild typed events via ``*.from_event()``. ``t_first``
    is set when the first ``transcript-chunk`` arrives; ``t_end`` when the
    terminal event is read. A server ``error`` event aborts the run. If the
    connection closes early, a streaming run that already produced partial
    results is taken as finished; anything else is an error.
    """
    chunk_text = ""
    while True:
        event = await client.read_event()
        if event is None:
            if m.t_end is None:
                if terminal_cls is TranscriptStop and m.t_first is not None:
                    m.t_end = time.perf_counter()
                else:
                    raise TranscriptionError(f"connection closed before {terminal_cls.__name__.lower()}")
            break
        if Error.is_type(event.type):
            err = Error.from_event(event)
            code = f" ({err.code})" if err.code else ""
            raise TranscriptionError(f"server error{code}: {err.text}")
        if TranscriptStart.is_type(event.type):
            # Streaming start signal; no timing action.
            pass
        elif TranscriptChunk.is_type(event.type):
            chunk = TranscriptChunk.from_event(event)
            chunk_text += chunk.text
            if m.t_first is None:
                m.t_first = time.perf_counter()
            m.n_transcript_chunks += 1
        elif Transcript.is_type(event.type):
            transcript = Transcript.from_event(event)
            m.hypothesis = transcript.text
            if terminal_cls is Transcript:
                m.t_end = time.perf_counter()
                break
        elif TranscriptStop.is_type(event.type):
            if terminal_cls is TranscriptStop:
                m.t_end = time.perf_counter()
                break
    if not m.hypothesis and chunk_text:
        # No final transcript event; fall back to the streamed chunks.
        m.hypothesis = chunk_text


async def _run_non_streaming(
    client: AsyncTcpClient,
    m: SttMeasurement,
    audio: AudioInput,
    transcribe: Transcribe,
    chunk_samples: int,
    trailing_silence: float = 0.0,
) -> None:
    m.t_sent = time.perf_counter()
    await client.write_event(transcribe.event())
    await _send_audio(client, audio, chunk_samples, trailing_silence=trailing_silence)
    await _read_transcript(client, m, Transcript)


async def _run_streaming(
    client: AsyncTcpClient,
    m: SttMeasurement,
    audio: AudioInput,
    transcribe: Transcribe,
    chunk_samples: int,
    chunk_delay: float,
    trailing_silence: float = 0.0,
) -> None:
    m.t_sent = time.perf_counter()
    # Read concurrently so TTFT is measured from the moment the server starts
    # producing transcript chunks, even while we are still writing audio.
    reader = asyncio.create_task(_read_transcript(client, m, TranscriptStop))
    try:
        await client.write_event(transcribe.event())
        await _send_audio(client, audio, chunk_samples, chunk_delay, trailing_silence)
    except BaseException:
        reader.cancel()
        raise
    await reader
