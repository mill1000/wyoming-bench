"""Command-line interface for the Wyoming TTS and STT benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import wave
from pathlib import Path

from wyoming.asr import Transcribe
from wyoming.info import Info
from wyoming.tts import SynthesizeVoice

from . import __version__
from .const import MODE_NON_STREAMING, MODE_STREAMING
from .info import (
    advertised_streaming,
    available_services,
    describe_services,
    fetch_info,
    server_label,
)
from .stt.bench import STREAM_PROBE_TIMEOUT as STT_PROBE_TIMEOUT
from .stt.bench import bench_stt_server
from .stt.client import transcribe_audio
from .stt.corpus import AudioInput, load_corpus, load_recordings
from .stt.reporting import SttMeasurement, print_stt_server_report, print_stt_summary
from .tts.bench import STREAM_PROBE_TIMEOUT as TTS_PROBE_TIMEOUT
from .tts.bench import bench_server
from .tts.client import synthesize_audio
from .tts.reporting import Measurement, print_server_report, print_summary
from .tts.texts import DEFAULT_TEXTS

DEFAULT_PORT = 10700

_MODE_BY_FLAG = {"non_streaming": MODE_NON_STREAMING, "streaming": MODE_STREAMING}


def parse_server(spec: str) -> tuple[str, int]:
    """Parse ``HOST`` or ``HOST:PORT`` into a ``(host, port)`` tuple."""
    if ":" in spec:
        host, _, port_s = spec.rpartition(":")
    else:
        host, port_s = spec, str(DEFAULT_PORT)
    if not host:
        raise argparse.ArgumentTypeError(f"missing host in {spec!r}")
    try:
        port = int(port_s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid port in {spec!r}") from None
    if not 0 < port <= 65535:
        raise argparse.ArgumentTypeError(f"port out of range in {spec!r}")
    return host, port


# --- shared argument groups -------------------------------------------------


def _add_servers(p: argparse.ArgumentParser, help_text: str) -> None:
    p.add_argument(
        "servers",
        nargs="+",
        type=parse_server,
        metavar="SERVER",
        help=help_text,
    )


def _add_timeout(p: argparse.ArgumentParser, default: float = 60.0, help: str | None = None) -> None:
    """--timeout, defined once and shared by every subcommand."""
    p.add_argument(
        "--timeout",
        type=float,
        default=default,
        metavar="SEC",
        help=help
        or f"Connect and per-event read timeout in seconds; the describe preflight "
        f"and streaming-capability probes are additionally capped at 8s (default {default:g}).",
    )


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--rounds", type=int, default=3, help="Timed measurement rounds per sample per mode (default 3)."
    )
    p.add_argument("--warmup", type=int, default=1, help="Untimed warmup runs per mode (default 1).")
    _add_timeout(p)
    p.add_argument("-v", "--verbose", action="store_true", help="Print one line per measurement.")


# --- TTS --------------------------------------------------------------------


def _add_tts_args(p: argparse.ArgumentParser) -> None:
    _add_servers(p, "Server(s) to benchmark, as HOST or HOST:PORT (default port: 10700).")
    p.add_argument(
        "--mode",
        choices=["non_streaming", "streaming", "both"],
        default="both",
        help="Synthesis path(s) to exercise (default: both).",
    )
    p.add_argument(
        "--texts", nargs="+", default=None, metavar="TEXT", help="Text samples to synthesize (one or more)."
    )
    p.add_argument(
        "--texts-file",
        type=str,
        default=None,
        metavar="FILE",
        help="File with one text sample per line (blank lines ignored).",
    )
    p.add_argument(
        "--texts-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory of .txt files, each used as one text sample (sorted by file name).",
    )
    p.add_argument(
        "--config",
        type=str,
        default="{}",
        metavar="JSON",
        help='Voice/format JSON, e.g. \'{"voice":"name","text_format":"text"}\'.',
    )
    p.add_argument(
        "--chunk-delay",
        type=float,
        default=0.0,
        metavar="SEC",
        help="Seconds between synthesize-chunk writes in streaming mode (default 0).",
    )
    p.add_argument(
        "--unique",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append a random nonce to each text to defeat server-side synthesis caching (cold-cache timings). "
        "On by default; use --no-unique to disable.",
    )
    _add_run_args(p)


def _load_texts_dir(directory: Path | str) -> list[str]:
    """Each ``.txt`` file in *directory* becomes one text sample (content stripped)."""
    directory = Path(directory)
    if not directory.is_dir():
        raise SystemExit(f"directory not found: {directory}")
    samples: list[str] = []
    for path in sorted(directory.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_file() or path.suffix.lower() != ".txt":
            continue
        sample = path.read_text(encoding="utf-8").strip()
        if sample:
            samples.append(sample)
    if not samples:
        raise SystemExit(f"no .txt text files found in {directory}")
    return samples


def load_texts(args: argparse.Namespace) -> list[str]:
    texts: list[str] = []
    if args.texts_dir:
        texts.extend(_load_texts_dir(args.texts_dir))
    if args.texts_file:
        with open(args.texts_file, "r", encoding="utf-8") as f:
            texts.extend(line.strip() for line in f if line.strip())
    if args.texts:
        texts.extend(args.texts)
    if texts:
        return texts
    return list(DEFAULT_TEXTS)


def parse_tts_config(raw: str) -> tuple[SynthesizeVoice | None, str | None]:
    cfg = json.loads(raw)
    if not isinstance(cfg, dict):
        raise SystemExit("--config must be a JSON object")
    voice: SynthesizeVoice | None = None
    if "voice" in cfg:
        v = cfg["voice"]
        if isinstance(v, str):
            voice = SynthesizeVoice(name=v)
        elif isinstance(v, dict):
            voice = SynthesizeVoice(
                name=v.get("name"),
                language=v.get("language"),
                speaker=v.get("speaker"),
            )
        else:
            raise SystemExit("--config voice must be a string or object")
    text_format = cfg.get("text_format")
    if text_format is not None and not isinstance(text_format, str):
        raise SystemExit("--config text_format must be a string")
    return voice, text_format


# --- STT --------------------------------------------------------------------


def _add_stt_args(p: argparse.ArgumentParser) -> None:
    _add_servers(p, "Server(s) to benchmark, as HOST or HOST:PORT (default port: 10700).")
    p.add_argument(
        "--corpus",
        type=str,
        required=True,
        metavar="DIR",
        help="Directory of .wav recordings paired with .txt transcripts " "(e.g. data_01.wav + data_01.txt).",
    )
    p.add_argument(
        "--mode",
        choices=["non_streaming", "streaming", "both"],
        default="both",
        help="Transcription path(s) to exercise (default: both).",
    )
    p.add_argument(
        "--config",
        type=str,
        default="{}",
        metavar="JSON",
        help='Transcribe JSON, e.g. \'{"name":"model","language":"en"}\'.',
    )
    p.add_argument(
        "--chunk-samples",
        type=int,
        default=1024,
        metavar="N",
        help="Samples per audio-chunk when sending audio (default 1024).",
    )
    p.add_argument(
        "--chunk-delay",
        type=float,
        default=0.0,
        metavar="SEC",
        help="Seconds between audio-chunk writes in streaming mode (default 0).",
    )
    p.add_argument(
        "--trailing-silence",
        type=float,
        default=0.5,
        metavar="SEC",
        help="Seconds of silence appended to each recording before transcription, "
        "so streaming (online) ASR models can finalize their last words. RTF is "
        "still computed on the original audio length. Use 0 to send recordings "
        "exactly as stored (default 0.5).",
    )
    _add_run_args(p)


def parse_stt_config(raw: str) -> Transcribe:
    cfg = json.loads(raw)
    if not isinstance(cfg, dict):
        raise SystemExit("--config must be a JSON object")
    name = cfg.get("name")
    language = cfg.get("language")
    context = cfg.get("context")
    vad_sensitivity = cfg.get("vad_sensitivity")
    transcript_names = cfg.get("transcript_names")
    transcript_terms = cfg.get("transcript_terms")

    for key, value in (
        ("name", name),
        ("language", language),
        ("vad_sensitivity", vad_sensitivity),
    ):
        if value is not None and not isinstance(value, str):
            raise SystemExit(f"--config {key} must be a string")
    for key, value in (("transcript_names", transcript_names), ("transcript_terms", transcript_terms)):
        if value is not None and not (isinstance(value, list) and all(isinstance(x, str) for x in value)):
            raise SystemExit(f"--config {key} must be a list of strings")

    return Transcribe(
        name=name,
        language=language,
        context=context,
        vad_sensitivity=vad_sensitivity,
        transcript_names=transcript_names,
        transcript_terms=transcript_terms,
    )


# --- gen-corpus -------------------------------------------------------------


def _add_gen_args(p: argparse.ArgumentParser) -> None:
    _add_servers(p, "TTS server(s) to synthesize with, as HOST or HOST:PORT (default port: 10700).")
    p.add_argument(
        "--out",
        type=str,
        required=True,
        metavar="DIR",
        help="Output directory for the generated corpus (created if missing).",
    )
    p.add_argument(
        "--mode",
        choices=["non_streaming", "streaming"],
        default="non_streaming",
        help="TTS synthesis path to use (default: non_streaming).",
    )
    p.add_argument(
        "--texts", nargs="+", default=None, metavar="TEXT", help="Text samples to synthesize (one or more)."
    )
    p.add_argument(
        "--texts-file",
        type=str,
        default=None,
        metavar="FILE",
        help="File with one text sample per line (blank lines ignored).",
    )
    p.add_argument(
        "--texts-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory of .txt files, each used as one text sample (sorted by file name).",
    )
    p.add_argument(
        "--config",
        type=str,
        default="{}",
        metavar="JSON",
        help="Voice/format JSON (same as the tts subcommand).",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default="sample",
        metavar="PFX",
        help="File name prefix (default: sample -> sample_01.wav).",
    )
    p.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files in the output directory."
    )
    _add_timeout(p)


# --- seed -------------------------------------------------------------------


def _add_seed_args(p: argparse.ArgumentParser) -> None:
    _add_servers(p, "STT server(s) to transcribe with, as HOST or HOST:PORT (default port: 10700).")
    p.add_argument(
        "--corpus",
        type=str,
        required=True,
        metavar="DIR",
        help="Directory of .wav recordings to transcribe. Each transcript is "
        "written next to its recording as <stem>.txt for manual review.",
    )
    p.add_argument(
        "--config",
        type=str,
        default="{}",
        metavar="JSON",
        help='Transcribe JSON, e.g. \'{"name":"model","language":"en"}\' (same as stt).',
    )
    p.add_argument(
        "--chunk-samples",
        type=int,
        default=1024,
        metavar="N",
        help="Samples per audio-chunk when sending audio (default 1024).",
    )
    p.add_argument(
        "--trailing-silence",
        type=float,
        default=0.5,
        metavar="SEC",
        help="Seconds of silence appended to each recording before transcription, "
        "so streaming (online) ASR models can finalize their last words "
        "(default 0.5; use 0 to send recordings exactly as stored).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .txt transcripts (they are skipped by default).",
    )
    _add_timeout(p)


# --- shared parsing helpers -------------------------------------------------


def _collect_servers(args: argparse.Namespace) -> list[tuple[str, int]]:
    """De-duplicate the positional servers while preserving order."""
    seen: set[tuple[str, int]] = set()
    unique: list[tuple[str, int]] = []
    for s in args.servers:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def _resolve_probe_timeout(args: argparse.Namespace, default: float) -> float:
    """Probes respect --timeout, but never exceed the 8s probe budget."""
    return min(args.timeout, default)


def _exc_detail(e: Exception) -> str:
    """Render *e* as ``TypeName: message``, dropping an empty message."""
    return f"{type(e).__name__}: {e}".rstrip(": ")


async def _preflight_check(host: str, port: int, service: str, timeout: float) -> tuple[bool, Info | None]:
    """Return ``(proceed, info)`` for *host:port* advertising *service*.

    *service* is ``"tts"`` or ``"asr"``. Queries the server once via
    ``describe``/``info``. If the connection itself fails (unreachable,
    refused, connect timeout) the server is skipped with a notice: it is
    down, and re-attempting every round would only burn the full timeout
    per measurement. If the server connects but cannot be queried (read
    timeout, no describe support), the check passes, *info* is ``None``,
    and the real benchmark is attempted, where the usual timeouts apply.
    If the server answers but does not list the requested service, a notice
    is printed and ``(False, info)`` is returned so the caller skips that
    server. When the service is advertised (or unknown), ``(True, info)``
    is returned.
    """
    try:
        info = await fetch_info(host, port, timeout)
    except (OSError, asyncio.TimeoutError) as e:
        print(
            f"  [preflight] {host}:{port} unavailable ({_exc_detail(e)}); skipping server",
            flush=True,
        )
        return False, None
    if info is None:
        return True, None
    services = available_services(info)
    if service in services:
        return True, info
    print(
        f"  [preflight] {host}:{port} advertises no {service.upper()} service "
        f"(available: {', '.join(services) or 'none'}); skipping",
        flush=True,
    )
    return False, info


# --- TTS runner -------------------------------------------------------------


async def _async_main_tts(
    args: argparse.Namespace,
    servers: list[tuple[str, int]],
    texts: list[str],
    modes: list[str],
    voice: SynthesizeVoice | None,
    text_format: str | None,
    probe_timeout: float,
) -> int:
    print(f"wyoming-bench {__version__} (tts)", flush=True)
    print(f"servers: {[f'{h}:{p}' for h, p in servers]}", flush=True)
    print(f"modes:   {modes}", flush=True)
    print(f"rounds:  {args.rounds}  warmup: {args.warmup}", flush=True)
    print(f"probe:   {probe_timeout:g}s  unique: {args.unique}", flush=True)
    print(f"texts:   {len(texts)} sample(s)", flush=True)
    if voice is not None:
        print(f"voice:   {voice.name}", flush=True)
    if text_format is not None:
        print(f"format:  {text_format}", flush=True)

    by_server: dict[str, list[Measurement]] = {}
    labels: list[str] = []
    for host, port in servers:
        print(flush=True)
        proceed, info = await _preflight_check(host, port, "tts", probe_timeout)
        label = server_label(host, port, info, "tts")
        print(f"=== {label} ===", flush=True)
        labels.append(label)
        if not proceed:
            continue
        ms = await bench_server(
            host,
            port,
            texts,
            modes,
            voice,
            text_format,
            args.rounds,
            args.warmup,
            args.verbose,
            args.chunk_delay,
            args.timeout,
            args.timeout,
            args.unique,
            probe_timeout,
            info,
        )
        by_server[label] = ms
        print_server_report(label, ms)

    print_summary(labels, by_server)

    if not by_server:
        return 1
    all_ms = [m for ms in by_server.values() for m in ms]
    if all(not m.ok for m in all_ms):
        return 1
    return 0


def _main_tts(args: argparse.Namespace) -> int:
    servers = _collect_servers(args)
    if args.rounds < 1:
        raise SystemExit("--rounds must be >= 1")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be > 0")
    probe_timeout = _resolve_probe_timeout(args, TTS_PROBE_TIMEOUT)

    try:
        voice, text_format = parse_tts_config(args.config)
    except json.JSONDecodeError as e:
        raise SystemExit(f"invalid --config JSON: {e}")

    texts = load_texts(args)
    if not texts:
        raise SystemExit("no texts to benchmark")

    if args.mode == "both":
        modes = [MODE_NON_STREAMING, MODE_STREAMING]
    else:
        modes = [_MODE_BY_FLAG[args.mode]]

    return asyncio.run(_async_main_tts(args, servers, texts, modes, voice, text_format, probe_timeout))


# --- STT runner -------------------------------------------------------------


async def _async_main_stt(
    args: argparse.Namespace,
    servers: list[tuple[str, int]],
    samples,
    modes: list[str],
    transcribe: Transcribe,
    probe_timeout: float,
) -> int:
    print(f"wyoming-bench {__version__} (stt)", flush=True)
    print(f"servers: {[f'{h}:{p}' for h, p in servers]}", flush=True)
    print(f"modes:   {modes}", flush=True)
    print(f"rounds:  {args.rounds}  warmup: {args.warmup}", flush=True)
    print(
        f"probe:   {probe_timeout:g}s  chunk_samples: {args.chunk_samples}  "
        f"chunk_delay: {args.chunk_delay:g}s  trailing_silence: {args.trailing_silence:g}s",
        flush=True,
    )
    total_audio = sum(s.duration_s for s in samples)
    print(f"corpus:  {len(samples)} sample(s), {total_audio:.1f}s audio", flush=True)
    if transcribe.name is not None:
        print(f"model:   {transcribe.name}", flush=True)
    if transcribe.language is not None:
        print(f"language: {transcribe.language}", flush=True)

    by_server: dict[str, list[SttMeasurement]] = {}
    labels: list[str] = []
    for host, port in servers:
        print(flush=True)
        proceed, info = await _preflight_check(host, port, "asr", probe_timeout)
        label = server_label(host, port, info, "asr")
        print(f"=== {label} ===", flush=True)
        labels.append(label)
        if not proceed:
            continue
        ms = await bench_stt_server(
            host,
            port,
            samples,
            modes,
            transcribe,
            args.rounds,
            args.warmup,
            args.verbose,
            args.chunk_samples,
            args.chunk_delay,
            args.timeout,
            args.timeout,
            probe_timeout,
            info,
            args.trailing_silence,
        )
        by_server[label] = ms
        print_stt_server_report(label, ms)

    print_stt_summary(labels, by_server)

    if not by_server:
        return 1
    all_ms = [m for ms in by_server.values() for m in ms]
    if all(not m.ok for m in all_ms):
        return 1
    return 0


def _main_stt(args: argparse.Namespace) -> int:
    servers = _collect_servers(args)
    if args.rounds < 1:
        raise SystemExit("--rounds must be >= 1")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be > 0")
    if args.chunk_samples < 1:
        raise SystemExit("--chunk-samples must be >= 1")
    if args.chunk_delay < 0:
        raise SystemExit("--chunk-delay must be >= 0")
    probe_timeout = _resolve_probe_timeout(args, STT_PROBE_TIMEOUT)

    try:
        transcribe = parse_stt_config(args.config)
    except json.JSONDecodeError as e:
        raise SystemExit(f"invalid --config JSON: {e}")

    try:
        samples = load_corpus(Path(args.corpus))
    except (FileNotFoundError, OSError) as e:
        raise SystemExit(str(e))

    if args.mode == "both":
        modes = [MODE_NON_STREAMING, MODE_STREAMING]
    else:
        modes = [_MODE_BY_FLAG[args.mode]]

    return asyncio.run(_async_main_stt(args, servers, samples, modes, transcribe, probe_timeout))


# --- gen-corpus runner ------------------------------------------------------


async def _async_gen_corpus(
    args: argparse.Namespace,
    servers: list[tuple[str, int]],
    texts: list[str],
    mode: str,
    voice: SynthesizeVoice | None,
    text_format: str | None,
    probe_timeout: float,
) -> int:
    host, port = servers[0]
    print(f"wyoming-bench {__version__} (generate-corpus)", flush=True)
    print(f"tts server: {host}:{port}", flush=True)
    print(f"out:      {args.out}   mode: {mode}   prefix: {args.prefix}", flush=True)
    print(f"texts:    {len(texts)} sample(s)", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    proceed, info = await _preflight_check(host, port, "tts", probe_timeout)
    if not proceed:
        return 1
    if mode == MODE_STREAMING and advertised_streaming(info, "tts") is False:
        print(
            "  [preflight] server does not advertise synthesize streaming; " "use --mode non_streaming",
            flush=True,
        )
        return 1
    n_ok = 0
    for i, text in enumerate(texts, start=1):
        stem = f"{args.prefix}_{i:02d}"
        wav_path = out / f"{stem}.wav"
        txt_path = out / f"{stem}.txt"
        if wav_path.exists() and not args.overwrite:
            print(f"  [skip] {stem}: {wav_path.name} exists (use --overwrite)", flush=True)
            continue
        try:
            pcm, rate, width, channels = await synthesize_audio(
                host,
                port,
                mode,
                text,
                voice,
                text_format,
                args.timeout,
                args.timeout,
            )
        except asyncio.TimeoutError:
            print(f"  [fail] {stem}: timed out (>{args.timeout:.0f}s per event)", flush=True)
            continue
        except Exception as e:  # noqa: BLE001
            print(f"  [fail] {stem}: {type(e).__name__}: {e}", flush=True)
            continue
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(width)
            wf.setframerate(rate)
            wf.writeframes(pcm)
        txt_path.write_text(text + "\n", encoding="utf-8")
        n_ok += 1
        dur = len(pcm) / (rate * width * channels) if rate * width * channels else 0.0
        print(f"  [ok]   {stem}: {dur:.2f}s @ {rate}Hz/{width * 8}bit/{channels}ch", flush=True)
    if n_ok:
        print(f"\nWrote {n_ok} sample(s) to {out}", flush=True)
        print("Now benchmark it, e.g.:", flush=True)
        print(f"  wyoming-bench stt STT_HOST[:PORT] --corpus {out}", flush=True)
    return 0 if n_ok else 1


def _main_gen_corpus(args: argparse.Namespace) -> int:
    servers = _collect_servers(args)
    if len(servers) > 1:
        print("note: generate-corpus uses the first listed server only", file=sys.stderr)
    if args.timeout <= 0:
        raise SystemExit("--timeout must be > 0")
    probe_timeout = _resolve_probe_timeout(args, TTS_PROBE_TIMEOUT)
    try:
        voice, text_format = parse_tts_config(args.config)
    except json.JSONDecodeError as e:
        raise SystemExit(f"invalid --config JSON: {e}")
    texts = load_texts(args)
    if not texts:
        raise SystemExit("no texts to synthesize")
    mode = _MODE_BY_FLAG[args.mode]
    return asyncio.run(_async_gen_corpus(args, servers, texts, mode, voice, text_format, probe_timeout))


# --- seed runner ------------------------------------------------------------


async def _async_seed_corpus(
    args: argparse.Namespace,
    servers: list[tuple[str, int]],
    samples: list[AudioInput],
    transcribe: Transcribe,
    probe_timeout: float,
) -> int:
    host, port = servers[0]
    print(f"wyoming-bench {__version__} (seed-corpus)", flush=True)
    print(f"stt server: {host}:{port}", flush=True)
    print(f"corpus:   {args.corpus}", flush=True)
    total_audio = sum(s.duration_s for s in samples)
    print(f"recordings: {len(samples)} sample(s), {total_audio:.1f}s audio", flush=True)
    if transcribe.name is not None:
        print(f"model:   {transcribe.name}", flush=True)
    if transcribe.language is not None:
        print(f"language: {transcribe.language}", flush=True)

    proceed, _ = await _preflight_check(host, port, "asr", probe_timeout)
    if not proceed:
        return 1

    n_ok = n_skip = n_fail = 0
    for sample in samples:
        txt_path = sample.audio_path.with_suffix(".txt")
        if txt_path.exists() and not args.overwrite:
            print(f"  [skip] {sample.sample_id}: {txt_path.name} exists (use --overwrite)", flush=True)
            n_skip += 1
            continue
        try:
            text = await transcribe_audio(
                host,
                port,
                sample,
                transcribe,
                args.chunk_samples,
                args.timeout,
                args.timeout,
            )
        except asyncio.TimeoutError:
            print(f"  [fail] {sample.sample_id}: timed out (>{args.timeout:.0f}s per event)", flush=True)
            n_fail += 1
            continue
        except Exception as e:  # noqa: BLE001
            print(f"  [fail] {sample.sample_id}: {type(e).__name__}: {e}", flush=True)
            n_fail += 1
            continue
        txt_path.write_text(text.strip() + "\n", encoding="utf-8")
        n_ok += 1
        n_words = len(text.split())
        print(f"  [ok]   {sample.sample_id}: {sample.duration_s:.2f}s -> {n_words} word(s)", flush=True)

    print(flush=True)
    print(f"Wrote {n_ok} transcript(s) to {args.corpus} ({n_skip} skipped, {n_fail} failed)", flush=True)
    if n_ok or n_skip:
        print("Review the .txt files, then benchmark, e.g.:", flush=True)
        print(f"  wyoming-bench stt STT_HOST[:PORT] --corpus {args.corpus}", flush=True)
    return 0 if n_ok else 1


def _main_seed_corpus(args: argparse.Namespace) -> int:
    servers = _collect_servers(args)
    if len(servers) > 1:
        print("note: seed-corpus uses the first listed server only", file=sys.stderr)
    if args.timeout <= 0:
        raise SystemExit("--timeout must be > 0")
    if args.chunk_samples < 1:
        raise SystemExit("--chunk-samples must be >= 1")
    probe_timeout = _resolve_probe_timeout(args, STT_PROBE_TIMEOUT)
    try:
        transcribe = parse_stt_config(args.config)
    except json.JSONDecodeError as e:
        raise SystemExit(f"invalid --config JSON: {e}")
    try:
        samples = load_recordings(Path(args.corpus))
    except (FileNotFoundError, OSError) as e:
        raise SystemExit(str(e))
    return asyncio.run(_async_seed_corpus(args, servers, samples, transcribe, probe_timeout))


# --- info -------------------------------------------------------------------


def _add_info_args(p: argparse.ArgumentParser) -> None:
    _add_servers(p, "Server(s) to query, as HOST or HOST:PORT (default port: 10700).")
    _add_timeout(p, default=5.0, help="Timeout in seconds for the describe exchange (default 5).")


async def _async_main_info(servers: list[tuple[str, int]], timeout: float) -> int:
    print(f"wyoming-bench {__version__} (info)", flush=True)
    failed = False
    for host, port in servers:
        print(flush=True)
        print(f"=== {host}:{port} ===", flush=True)
        try:
            info = await fetch_info(host, port, timeout)
        except (OSError, asyncio.TimeoutError) as e:
            print(f"  (unavailable: {_exc_detail(e)})", flush=True)
            failed = True
            continue
        if info is None:
            print("  (no info response: read timeout, no describe support, or closed connection)", flush=True)
            failed = True
            continue
        for line in describe_services(info):
            print(line, flush=True)
    return 1 if failed else 0


def _main_info(args: argparse.Namespace) -> int:
    if args.timeout <= 0:
        raise SystemExit("--timeout must be > 0")
    return asyncio.run(_async_main_info(_collect_servers(args), args.timeout))


# --- parser & entry point ---------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wyoming-bench",
        description="Benchmark Wyoming-protocol TTS and STT servers " "(non-streaming and streaming).",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="task", required=True, metavar="{tts,stt,generate-corpus,seed-corpus,info}")

    tts_p = sub.add_parser(
        "tts",
        help="benchmark text-to-speech servers",
        description="Benchmark Wyoming TTS servers (non-streaming and streaming).",
    )
    _add_tts_args(tts_p)

    stt_p = sub.add_parser(
        "stt",
        help="benchmark speech-to-text (ASR) servers",
        description="Benchmark Wyoming STT servers (non-streaming and streaming) "
        "for speed and word-level accuracy.",
    )
    _add_stt_args(stt_p)

    gen_p = sub.add_parser(
        "generate-corpus",
        help="generate an STT test corpus from a TTS server",
        description="Synthesize text with a Wyoming TTS server and save paired "
        ".wav/.txt files to use with the stt subcommand.",
    )
    _add_gen_args(gen_p)

    seed_p = sub.add_parser(
        "seed-corpus",
        help="seed an STT corpus by transcribing existing recordings",
        description="Transcribe every .wav in a directory with a Wyoming STT server and write a "
        ".txt next to each recording, for manual review before benchmarking with stt.",
    )
    _add_seed_args(seed_p)

    info_p = sub.add_parser(
        "info",
        help="show the services a server advertises (describe/info)",
        description="Connect to a Wyoming server and print the services it advertises "
        "(ASR/STT and TTS programs, models/voices, streaming support).",
    )
    _add_info_args(info_p)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "tts": _main_tts,
        "stt": _main_stt,
        "generate-corpus": _main_gen_corpus,
        "seed-corpus": _main_seed_corpus,
        "info": _main_info,
    }
    try:
        return handlers[args.task](args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
