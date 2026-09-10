"""Async orchestration of the benchmark across texts, modes, and rounds."""

from __future__ import annotations

import secrets

from wyoming.tts import SynthesizeVoice

from .client import MODE_STREAMING, measure_once
from .reporting import Measurement
from .texts import split_sentences

# Per-event timeout (s) for the one-off streaming capability probe. Kept short
# so a non-streaming server is detected quickly instead of timing out once per
# round for the full --timeout.
STREAM_PROBE_TIMEOUT = 8.0


def _maybe_unique(text: str, unique: bool) -> str:
    """Return *text* as-is, or with a random nonce appended.

    Appending a fresh nonce per measurement defeats server-side synthesis
    caching so every run reflects a real, cold synthesis.
    """
    if not unique:
        return text
    return f"{text} {secrets.token_hex(4)}"


async def _probe_streaming(
    host: str,
    port: int,
    texts: list[str],
    voice: SynthesizeVoice | None,
    text_format: str | None,
    unique: bool,
    connect_timeout: float,
    read_timeout: float,
    chunk_delay: float,
    probe_timeout: float = STREAM_PROBE_TIMEOUT,
) -> bool:
    """Return True if *host:port* responds to a streaming synthesize.

    Runs a single streaming measurement with a short per-event timeout
    (*probe_timeout*). A server that ignores synthesize-start/-chunk/-stop
    never emits audio, so the probe fails fast and we can bail on streaming
    mode for that server instead of paying the full read timeout once per
    round.
    """
    w = await measure_once(
        host, port, MODE_STREAMING, _maybe_unique(texts[0], unique), voice, text_format,
        connect_timeout, min(read_timeout, probe_timeout), chunk_delay,
    )
    return w.ok


async def bench_server(
    host: str,
    port: int,
    texts: list[str],
    modes: list[str],
    voice: SynthesizeVoice | None,
    text_format: str | None,
    rounds: int,
    warmup: int,
    verbose: bool,
    chunk_delay: float,
    connect_timeout: float,
    read_timeout: float,
    unique: bool = False,
    probe_timeout: float = STREAM_PROBE_TIMEOUT,
) -> list[Measurement]:
    """Benchmark one server.

    Runs warmup (untimed) then ``rounds`` x ``texts`` per mode, sequentially.
    When *unique* is set, each measurement gets a nonce-suffixed copy of its
    text so repeated identical inputs do not hit a server-side synthesis cache.
    """
    measurements: list[Measurement] = []
    for mode in modes:
        if mode == MODE_STREAMING and not await _probe_streaming(
            host, port, texts, voice, text_format, unique,
            connect_timeout, read_timeout, chunk_delay, probe_timeout,
        ):
            print(
                f"  [streaming] NOT SUPPORTED by {host}:{port} "
                f"(no audio in response to synthesize-*); skipping streaming mode",
                flush=True,
            )
            continue
        for _ in range(warmup):
            w = await measure_once(
                host, port, mode, _maybe_unique(texts[0], unique), voice, text_format,
                connect_timeout, read_timeout, chunk_delay,
            )
            if verbose:
                status = "ok" if w.ok else f"FAIL ({w.error})"
                print(f"[warmup:{mode}] {status}", flush=True)
        for round_no in range(rounds):
            for text in texts:
                m = await measure_once(
                    host, port, mode, _maybe_unique(text, unique), voice, text_format,
                    connect_timeout, read_timeout, chunk_delay,
                )
                # Report the sample's real sentence count, not the nonce-suffixed
                # copy: --unique appends a hex nonce that split_sentences would
                # otherwise count as an extra sentence.
                m.sentence_count = len(split_sentences(text))
                measurements.append(m)
                if verbose:
                    _print_verbose(mode, round_no, m)
    return measurements


def _print_verbose(mode: str, round_no: int, m: Measurement) -> None:
    if m.ok:
        ttft = m.ttft_s or 0.0
        total = m.total_s or 0.0
        print(
            f"[{mode}] r{round_no} sent={m.sentence_count} "
            f"ttft={ttft * 1000:.0f}ms total={total * 1000:.0f}ms "
            f"audio={m.audio_duration_s:.2f}s bytes={m.audio_bytes} "
            f"cycles={m.n_audio_cycles}",
            flush=True,
        )
    else:
        print(f"[{mode}] r{round_no} FAIL: {m.error}", flush=True)
