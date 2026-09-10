"""Command-line interface for the Wyoming TTS benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from wyoming.tts import SynthesizeVoice

from . import __version__
from .bench import STREAM_PROBE_TIMEOUT, bench_server
from .client import MODE_NON_STREAMING, MODE_STREAMING
from .reporting import Measurement, print_comparison, print_server_report
from .texts import DEFAULT_TEXTS

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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wyoming-tts-bench",
        description="Benchmark Wyoming-protocol TTS servers (non-streaming and streaming).",
    )
    p.add_argument(
        "-s", "--server", action="append", type=parse_server, default=[],
        metavar="HOST[:PORT]",
        help="TTS server to benchmark (repeatable). Default port: 10700.",
    )
    p.add_argument(
        "--servers", type=str, default=None, metavar="HOST[:PORT],[HOST[:PORT]...]",
        help="Comma-separated server list (alternative to repeated -s).",
    )
    p.add_argument("--rounds", type=int, default=3,
                   help="Timed measurement rounds per text per mode (default 3).")
    p.add_argument("--warmup", type=int, default=1,
                   help="Untimed warmup runs per mode (default 1).")
    p.add_argument("--mode", choices=["non_streaming", "streaming", "both"], default="both",
                   help="Synthesis path(s) to exercise (default: both).")
    p.add_argument("--texts", nargs="+", default=None, metavar="TEXT",
                   help="Text samples to synthesize (one or more).")
    p.add_argument("--texts-file", type=str, default=None, metavar="FILE",
                   help="File with one text sample per line (blank lines ignored).")
    p.add_argument("--config", type=str, default="{}", metavar="JSON",
                   help='Voice/format JSON, e.g. \'{"voice":"name","text_format":"text"}\'.')
    p.add_argument("--chunk-delay", type=float, default=0.0, metavar="SEC",
                   help="Seconds between synthesize-chunk writes in streaming mode (default 0).")
    p.add_argument("--unique", action="store_true",
                   help="Append a random nonce to each text to defeat server-side synthesis caching (cold-cache timings).")
    p.add_argument("--probe-timeout", type=float, default=STREAM_PROBE_TIMEOUT, metavar="SEC",
                   help="Per-event timeout for the one-off streaming capability probe. A "
                        "non-streaming server is detected in one probe instead of timing out "
                        f"per round (default {STREAM_PROBE_TIMEOUT:g}).")
    p.add_argument("--timeout", type=float, default=60.0, metavar="SEC",
                   help="Connect and per-event read timeout in seconds (default 60).")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Print one line per measurement.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def load_texts(args: argparse.Namespace) -> list[str]:
    texts: list[str] = []
    if args.texts_file:
        with open(args.texts_file, "r", encoding="utf-8") as f:
            texts.extend(line.strip() for line in f if line.strip())
    if args.texts:
        texts.extend(args.texts)
    if texts:
        return texts
    return list(DEFAULT_TEXTS)


def parse_config(raw: str) -> tuple[SynthesizeVoice | None, str | None]:
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


async def _async_main(
    args: argparse.Namespace,
    servers: list[tuple[str, int]],
    texts: list[str],
    modes: list[str],
    voice: SynthesizeVoice | None,
    text_format: str | None,
) -> int:
    print(f"wyoming-tts-bench {__version__}", flush=True)
    print(f"servers: {[f'{h}:{p}' for h, p in servers]}", flush=True)
    print(f"modes:   {modes}", flush=True)
    print(f"rounds:  {args.rounds}  warmup: {args.warmup}", flush=True)
    print(f"probe:   {args.probe_timeout:g}s  unique: {args.unique}", flush=True)
    print(f"texts:   {len(texts)} sample(s)", flush=True)
    if voice is not None:
        print(f"voice:   {voice.name}", flush=True)
    if text_format is not None:
        print(f"format:  {text_format}", flush=True)

    by_server: dict[str, list[Measurement]] = {}
    for host, port in servers:
        print(flush=True)
        print(f"=== {host}:{port} ===", flush=True)
        ms = await bench_server(
            host, port, texts, modes, voice, text_format,
            args.rounds, args.warmup, args.verbose, args.chunk_delay,
            args.timeout, args.timeout, args.unique, args.probe_timeout,
        )
        by_server[f"{host}:{port}"] = ms
        print_server_report(f"{host}:{port}", ms)

    if len(servers) > 1:
        print_comparison([f"{h}:{p}" for h, p in servers], by_server)

    all_ms = [m for ms in by_server.values() for m in ms]
    if all_ms and all(not m.ok for m in all_ms):
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    servers: list[tuple[str, int]] = list(args.server)
    if args.servers:
        for part in args.servers.split(","):
            part = part.strip()
            if part:
                servers.append(parse_server(part))
    # De-duplicate while preserving order.
    seen: set[tuple[str, int]] = set()
    unique: list[tuple[str, int]] = []
    for s in servers:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    servers = unique
    if not servers:
        parser.error("at least one server is required (-s or --servers)")

    if args.rounds < 1:
        parser.error("--rounds must be >= 1")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.timeout <= 0:
        parser.error("--timeout must be > 0")
    if args.probe_timeout <= 0:
        parser.error("--probe-timeout must be > 0")

    try:
        voice, text_format = parse_config(args.config)
    except json.JSONDecodeError as e:
        parser.error(f"invalid --config JSON: {e}")

    texts = load_texts(args)
    if not texts:
        parser.error("no texts to benchmark")

    if args.mode == "both":
        modes = [MODE_NON_STREAMING, MODE_STREAMING]
    else:
        modes = [_MODE_BY_FLAG[args.mode]]

    try:
        return asyncio.run(_async_main(args, servers, texts, modes, voice, text_format))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
