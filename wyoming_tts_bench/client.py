"""Wyoming TTS client wrapper that performs a single timed measurement.

Uses the official async ``wyoming`` library (``AsyncTcpClient``). Each
measurement opens a fresh TCP connection so results are isolated per round.
"""

from __future__ import annotations

import asyncio
import time

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.error import Error
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
    SynthesizeTextFormat,
    SynthesizeVoice,
)

from .reporting import Measurement
from .texts import split_sentences

MODE_NON_STREAMING = "non_streaming"
MODE_STREAMING = "streaming"

_TEXT_FORMATS: dict[str, SynthesizeTextFormat] = {
    "text": SynthesizeTextFormat.TEXT,
    "ssml": SynthesizeTextFormat.SSML,
}


class SynthesisError(Exception):
    """Raised when the server does not return a complete audio response."""


def _to_format(text_format: str | None) -> SynthesizeTextFormat | None:
    if text_format is None:
        return None
    try:
        return _TEXT_FORMATS[text_format.lower()]
    except KeyError:
        raise SynthesisError(f"unknown text_format {text_format!r}") from None


async def measure_once(
    host: str,
    port: int,
    mode: str,
    text: str,
    voice: SynthesizeVoice | None,
    text_format: str | None,
    connect_timeout: float,
    read_timeout: float,
    chunk_delay: float = 0.0,
) -> Measurement:
    """Run one synthesis against *host*:*port* and return a Measurement.

    Failures are captured on the returned Measurement (``ok=False`` +
    ``error``) rather than raised, so a single bad run cannot abort a batch.
    """
    m = Measurement(server=f"{host}:{port}", mode=mode, text=text)
    m.sentence_count = len(split_sentences(text))
    try:
        text_fmt = _to_format(text_format)
    except SynthesisError as e:
        m.error = str(e)
        return m

    client = AsyncTcpClient(
        host, port, connect_timeout=connect_timeout, read_timeout=read_timeout
    )
    try:
        await client.connect()
        if mode == MODE_STREAMING:
            await _run_streaming(client, m, text, voice, text_fmt, chunk_delay)
        else:
            await _run_non_streaming(client, m, text, voice, text_fmt)
        if m.t_first is not None and m.t_end is not None:
            m.ok = True
    except asyncio.TimeoutError:
        m.error = f"timed out (>{read_timeout:.0f}s per event)"
    except SynthesisError as e:
        m.error = str(e)
    except Exception as e:  # noqa: BLE001 - report any failure as a failed measurement
        m.error = f"{type(e).__name__}: {e}"
    finally:
        try:
            await client.disconnect()
        except Exception:  # noqa: BLE001
            pass
    return m


async def _read_audio(
    client: AsyncTcpClient,
    m: Measurement,
    stop_cls: type[AudioStop | SynthesizeStopped],
) -> None:
    """Read events until the terminal event, accumulating audio metrics.

    *stop_cls* marks the event that ends the measurement:

    - non-streaming: ``AudioStop`` (a single audio cycle);
    - streaming: ``SynthesizeStopped`` (server -> client). Streaming servers
      (e.g. wyoming-piper) emit one ``audio-start``/``audio-chunk``*/
      ``audio-stop`` cycle per synthesized sentence, so intermediate
      ``audio-stop`` events are read through and the audio is accumulated
      across all cycles.

    ``read_event()`` yields base ``Event`` objects (not typed subclasses), so
    we dispatch on ``event.type`` and rebuild the typed event via the
    ``*.from_event()`` helpers. ``t_first`` is set when the first
    ``audio-chunk`` arrives and ``t_end`` when the terminal event is read.
    Server ``error`` events abort the run with the server's message. If the
    connection closes before the terminal event, a run that already received
    at least one complete audio cycle is still taken as finished (at the
    last ``audio-stop``); anything else is an error.
    """
    t_last_audio_stop: float | None = None
    while True:
        event = await client.read_event()
        if event is None:
            if m.t_end is None and t_last_audio_stop is not None and m.n_chunks > 0:
                m.t_end = t_last_audio_stop
            elif m.t_end is None:
                raise SynthesisError(
                    f"connection closed before {stop_cls.__name__.lower()}"
                )
            break
        if Error.is_type(event.type):
            err = Error.from_event(event)
            code = f" ({err.code})" if err.code else ""
            raise SynthesisError(f"server error{code}: {err.text}")
        if AudioStart.is_type(event.type):
            start = AudioStart.from_event(event)
            m.sample_rate = start.rate
            m.sample_width = start.width
            m.channels = start.channels
        elif AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            if m.t_first is None:
                m.t_first = time.perf_counter()
            m.audio_bytes += len(chunk.audio)
            m.audio_duration_s += chunk.seconds
            m.n_chunks += 1
        elif AudioStop.is_type(event.type):
            m.n_audio_cycles += 1
            t_last_audio_stop = time.perf_counter()
            if stop_cls is AudioStop:
                m.t_end = time.perf_counter()
                break
        elif SynthesizeStopped.is_type(event.type):
            m.t_end = time.perf_counter()
            break
    if m.n_chunks == 0:
        raise SynthesisError("no audio chunks received")


async def _run_non_streaming(
    client: AsyncTcpClient,
    m: Measurement,
    text: str,
    voice: SynthesizeVoice | None,
    text_fmt: SynthesizeTextFormat | None,
) -> None:
    m.t_sent = time.perf_counter()
    await client.write_event(Synthesize(text, voice, text_fmt).event())
    await _read_audio(client, m, AudioStop)


async def _run_streaming(
    client: AsyncTcpClient,
    m: Measurement,
    text: str,
    voice: SynthesizeVoice | None,
    text_fmt: SynthesizeTextFormat | None,
    chunk_delay: float,
) -> None:
    m.t_sent = time.perf_counter()
    # Read concurrently so TTFT is measured from the moment the server starts
    # producing audio, even while we are still writing chunks. The stream
    # ends when the server sends ``synthesize-stopped`` (after the final
    # per-sentence audio cycle); we never send it ourselves.
    reader = asyncio.create_task(_read_audio(client, m, SynthesizeStopped))
    try:
        await client.write_event(SynthesizeStart(voice, text_fmt).event())
        # Send each chunk with a trailing space: the server's
        # SentenceBoundaryDetector needs the "period + whitespace" cue to
        # yield a sentence mid-stream. Stripped chunks concatenate with no
        # space ("...dog.This sentence..."), so it never sees a boundary and
        # buffers everything until synthesize-stop.
        for sentence in split_sentences(text):
            await client.write_event(SynthesizeChunk(sentence + " ").event())
            if chunk_delay > 0:
                await asyncio.sleep(chunk_delay)
        # No full-text ``synthesize`` fallback: this is a pure streaming flow.
        # A server that does not implement synthesize-start/-chunk/-stop will
        # never see a synthesize event, emit no audio, and time out -> fail.
        await client.write_event(SynthesizeStop().event())
    except BaseException:
        reader.cancel()
        raise
    await reader
