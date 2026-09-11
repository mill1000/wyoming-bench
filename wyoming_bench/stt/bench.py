"""Async orchestration of the STT benchmark across samples, modes, and rounds."""

from __future__ import annotations

import asyncio

from wyoming.asr import (
    Transcribe,
    Transcript,
    TranscriptChunk,
    TranscriptStart,
    TranscriptStop,
)
from wyoming.client import AsyncTcpClient
from wyoming.error import Error
from wyoming.info import Info

from ..const import MODE_STREAMING
from ..info import advertised_streaming
from .client import _send_audio, measure_stt_once
from .corpus import AudioInput
from .reporting import SttMeasurement

# Per-event timeout (s) for the one-off streaming capability probe. Kept short
# so a non-streaming server is detected quickly instead of timing out once per
# round for the full --timeout.
STREAM_PROBE_TIMEOUT = 8.0


async def _probe_stt_streaming(
    host: str,
    port: int,
    audio: AudioInput,
    transcribe: Transcribe,
    chunk_samples: int,
    connect_timeout: float,
    read_timeout: float,
    probe_timeout: float = STREAM_PROBE_TIMEOUT,
) -> bool:
    """Return True if *host:port* streams transcript chunks.

    Sends a full ``transcribe`` + audio request and reads events until it sees
    the first transcript-related event. A streaming server answers with
    ``transcript-start``/``transcript-chunk`` (-> True); a non-streaming server
    answers with a single ``transcript`` (-> False). Uses a short per-event
    timeout so the probe is fast either way.
    """
    client = AsyncTcpClient(
        host, port, connect_timeout=connect_timeout, read_timeout=min(read_timeout, probe_timeout)
    )
    try:
        await client.connect()
        await client.write_event(transcribe.event())
        await _send_audio(client, audio, chunk_samples)
        while True:
            event = await client.read_event()
            if event is None:
                return False
            if Error.is_type(event.type):
                return False
            if (
                TranscriptStart.is_type(event.type)
                or TranscriptChunk.is_type(event.type)
                or TranscriptStop.is_type(event.type)
            ):
                return True
            if Transcript.is_type(event.type):
                return False
            # Ignore unrelated events; keep reading for a transcript event.
    except asyncio.TimeoutError:
        return False
    except Exception:  # noqa: BLE001 - any probe failure means "no streaming"
        return False
    finally:
        try:
            await client.disconnect()
        except Exception:  # noqa: BLE001
            pass


async def bench_stt_server(
    host: str,
    port: int,
    samples: list[AudioInput],
    modes: list[str],
    transcribe: Transcribe,
    rounds: int,
    warmup: int,
    verbose: bool,
    chunk_samples: int,
    chunk_delay: float,
    connect_timeout: float,
    read_timeout: float,
    probe_timeout: float = STREAM_PROBE_TIMEOUT,
    info: Info | None = None,
    trailing_silence: float = 0.0,
) -> list[SttMeasurement]:
    """Benchmark one STT server.

    Runs warmup (untimed) then ``rounds`` x ``samples`` per mode, sequentially.
    Streaming mode is skipped immediately when the server's advertised info
    says it is unsupported; otherwise it is probed first (on the shortest
    sample) and skipped if the server does not stream transcript chunks.
    """
    measurements: list[SttMeasurement] = []
    probe_sample = min(samples, key=lambda s: s.duration_s) if samples else None
    for mode in modes:
        if mode == MODE_STREAMING:
            if advertised_streaming(info, "asr") is False:
                print(
                    f"  [streaming] NOT SUPPORTED by {host}:{port} "
                    f"(not advertised by server); skipping streaming mode",
                    flush=True,
                )
                continue
            supported = probe_sample is not None and await _probe_stt_streaming(
                host,
                port,
                probe_sample,
                transcribe,
                chunk_samples,
                connect_timeout,
                read_timeout,
                probe_timeout,
            )
            if not supported:
                print(
                    f"  [streaming] NOT SUPPORTED by {host}:{port} "
                    f"(no transcript-chunk in response); skipping streaming mode",
                    flush=True,
                )
                continue
        for _ in range(warmup):
            w = await measure_stt_once(
                host,
                port,
                mode,
                samples[0],
                transcribe,
                chunk_samples,
                connect_timeout,
                read_timeout,
                chunk_delay,
                trailing_silence,
            )
            if verbose:
                status = "ok" if w.ok else f"FAIL ({w.error})"
                print(f"[warmup:{mode}] {w.sample_id}: {status}", flush=True)
        for round_no in range(rounds):
            for sample in samples:
                m = await measure_stt_once(
                    host,
                    port,
                    mode,
                    sample,
                    transcribe,
                    chunk_samples,
                    connect_timeout,
                    read_timeout,
                    chunk_delay,
                    trailing_silence,
                )
                measurements.append(m)
                if verbose:
                    _print_verbose_stt(mode, round_no, m)
    return measurements


def _print_verbose_stt(mode: str, round_no: int, m: SttMeasurement) -> None:
    if m.ok:
        total = m.total_s or 0.0
        ttft = m.ttft_s
        ttft_str = f"{ttft * 1000:.0f}ms" if ttft is not None else "n/a"
        wer_str = f"{m.wer * 100:.1f}%" if m.wer is not None else "n/a"
        print(
            f"[{mode}] r{round_no} {m.sample_id} "
            f"total={total * 1000:.0f}ms ttft={ttft_str} WER={wer_str} "
            f"audio={m.audio_duration_s:.2f}s",
            flush=True,
        )
    else:
        print(f"[{mode}] r{round_no} {m.sample_id} FAIL: {m.error}", flush=True)
