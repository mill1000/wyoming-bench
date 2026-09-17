"""Query the services a wyoming server advertises (``describe`` -> ``info``).

Wyoming servers answer a ``describe`` event with an ``info`` event listing
their ASR, TTS, and other programs. That lets us (a) report what a
server offers (the ``info`` subcommand) and (b) fail fast when a benchmark is
pointed at a server that does not implement the requested service, instead of
burning a per-event timeout on every round.
"""

from __future__ import annotations

import asyncio

from wyoming.client import AsyncTcpClient
from wyoming.error import Error
from wyoming.info import AsrProgram, Describe, Info, TtsProgram

# Service groups other than asr/tts, as (Info field, label) pairs.
_OTHER_SERVICES = (
    ("handle", "handle"),
    ("intent", "intent"),
    ("wake", "wake"),
    ("mic", "mic"),
    ("snd", "snd"),
)


async def fetch_info(host: str, port: int, timeout: float) -> Info | None:
    """Return the services advertised by *host:port*, or ``None`` if unknown.

    Sends one ``describe`` event and reads until the ``info`` response
    arrives. Connection-level failures (refused, unreachable, connect
    timeout) are raised as ``OSError``/``TimeoutError``: the server is
    down, so callers should skip it rather than re-attempt every
    measurement. ``None`` means "reachable but cannot be determined": the
    read timed out, the server sent an error event, or the connection
    closed without an info response (e.g. a server that predates describe
    support). Callers should still attempt the real benchmark in that case,
    where the usual connect/read timeouts apply.
    """
    client = AsyncTcpClient(host, port, connect_timeout=timeout, read_timeout=timeout)
    # Let connect() failures propagate: a failed connect means the server is
    # unavailable, which callers handle differently from "unknown".
    await client.connect()
    try:
        await client.write_event(Describe().event())
        while True:
            event = await client.read_event()
            if event is None:
                return None
            if Error.is_type(event.type):
                return None
            if Info.is_type(event.type):
                return Info.from_event(event)
            # Ignore unrelated events; keep reading for info.
    except Exception:  # noqa: BLE001 - reachable, but no usable info
        return None
    finally:
        try:
            await client.disconnect()
        except Exception:  # noqa: BLE001
            pass


def available_services(info: Info) -> list[str]:
    """Names of the service groups *info* advertises (e.g. ``["asr", "tts"]``)."""
    names = [name for name in ("asr", "tts") if getattr(info, name)]
    names.extend(label for field, label in _OTHER_SERVICES if getattr(info, field))
    return names


def _find_program(programs: list, name: str | None):
    """The advertised program named *name*, or the first one when *name* is None.

    Name matching is case-insensitive.
    """
    if name is None:
        return programs[0] if programs else None
    for p in programs:
        if p.name.lower() == name.lower():
            return p
    return None


def advertised_streaming(info: Info | None, service: str, program: str | None = None) -> bool | None:
    """Whether the server advertises streaming support for *service*.

    *service* is ``"tts"`` or ``"asr"``. When *program* is given, the flag is
    read off the advertised program with that name (``None`` when the name is
    not advertised); without it the protocol uses the first program of each
    type, so the flag is read off the first advertised program. Returns
    ``None`` when unknown (no info, or no program for that service), in which
    case callers should fall back to a behavioral probe.
    """
    if info is None:
        return None
    if service == "tts":
        selected = _find_program(info.tts, program)
        if selected is None:
            return None
        return selected.supports_synthesize_streaming
    selected = _find_program(info.asr, program)
    if selected is None:
        return None
    return selected.supports_transcript_streaming


def program_name(info: Info | None, service: str) -> str | None:
    """Name of the first advertised program of *service* (``"tts"`` or ``"asr"``).

    Returns ``None`` when there is no info or no program for that service.
    """
    if info is None:
        return None
    programs = info.tts if service == "tts" else info.asr
    return programs[0].name if programs else None


def server_label(host: str, port: int, info: Info | None, service: str, program: str | None = None) -> str:
    """Human-readable label for a benchmarked server.

    ``host:port``, with the program name in parentheses: *program* when given,
    else the first advertised program of *service* when the info request
    provided one (e.g. ``10.0.0.1:10700 (piper)``).
    """
    name = program if program is not None else program_name(info, service)
    if name:
        return f"{host}:{port} ({name})"
    return f"{host}:{port}"


def describe_services(info: Info) -> list[str]:
    """Render the services *info* advertises as human-readable lines."""
    lines: list[str] = []
    lines.extend(_asr_lines(info.asr) if info.asr else ["  asr  (none)"])
    lines.extend(_tts_lines(info.tts) if info.tts else ["  tts  (none)"])
    other = [label for field, label in _OTHER_SERVICES if getattr(info, field)]
    if other:
        lines.append("  other  " + ", ".join(other))
    return lines


def _asr_lines(programs: list[AsrProgram]) -> list[str]:
    out: list[str] = []
    for p in programs:
        flag = "transcript streaming" if p.supports_transcript_streaming else "no streaming"
        out.append(f"  asr  {p.name} ({flag})")
        if p.models:
            for m in p.models:
                langs = ", ".join(m.languages) or "n/a"
                out.append(f"       model {m.name} [{langs}]")
        else:
            out.append("       (no models)")
    return out


def _tts_lines(programs: list[TtsProgram]) -> list[str]:
    out: list[str] = []
    for p in programs:
        flag = "synthesize streaming" if p.supports_synthesize_streaming else "no streaming"
        out.append(f"  tts  {p.name} ({flag})")
        if p.voices:
            for v in p.voices:
                langs = ", ".join(v.languages) or "n/a"
                out.append(f"       voice {v.name} [{langs}]")
        else:
            out.append("       (no voices)")
    return out
