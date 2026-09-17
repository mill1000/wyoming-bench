"""TTS measurement container and plain-text reporting (no CSV export)."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from ..reporting import (
    SERVER_COL_WIDTH,
    _fmt_cell,
    _fmt_ms_cell,
    stat_table,
    stats,
    truncate_label,
)


@dataclass
class Measurement:
    """A single timed synthesis run against one server in one mode."""

    server: str
    mode: str
    text: str
    ok: bool = False
    error: str = ""
    connection_failed: bool = False  # TCP connect failed: server down/unreachable
    t_sent: float = 0.0
    t_first: float | None = None
    t_end: float | None = None
    audio_bytes: int = 0
    audio_duration_s: float = 0.0
    sample_rate: int = 0
    sample_width: int = 0
    channels: int = 0
    n_chunks: int = 0
    n_audio_cycles: int = 0
    sentence_count: int = 0

    @property
    def ttft_s(self) -> float | None:
        if not self.ok or self.t_first is None:
            return None
        return max(0.0, self.t_first - self.t_sent)

    @property
    def total_s(self) -> float | None:
        if not self.ok or self.t_end is None:
            return None
        return max(0.0, self.t_end - self.t_sent)

    @property
    def rtf(self) -> float | None:
        total = self.total_s
        if total is None or self.audio_duration_s <= 0:
            return None
        return total / self.audio_duration_s


def summarize_mode(measurements: list[Measurement], mode: str) -> dict:
    """Aggregate the successful measurements for *mode* into stat dicts."""
    in_mode = [m for m in measurements if m.mode == mode]
    ok = [m for m in in_mode if m.ok]
    ttft = [m.ttft_s for m in ok if m.ttft_s is not None]
    total = [m.total_s for m in ok if m.total_s is not None]
    rtf = [m.rtf for m in ok if m.rtf is not None]
    audio_duration = [m.audio_duration_s for m in ok]
    audio_bytes = [m.audio_bytes for m in ok]
    return {
        "mode": mode,
        "ok": len(ok),
        "fail": len(in_mode) - len(ok),
        "ttft": stats(ttft),
        "total": stats(total),
        "rtf": stats(rtf),
        "audio_duration": stats(audio_duration),
        "audio_bytes": stats(audio_bytes),
    }


def normalized_stats(rows: list[tuple[str, dict]]) -> dict[str, dict]:
    """Cross-server normalized values for fair comparison, keyed by server.

    *rows* is a list of ``(server, summary)`` pairs as produced by
    :func:`summarize_mode`. The shared denominator is the mean audio duration
    across all servers (equal weight per server). For each server:

    - ``rtf_norm``: the server's mean total time divided by that shared mean
      audio duration. Because the denominator is shared, the ratio between two
      servers reduces to the ratio of their wall times, so servers that speak
      the same text at different rates are compared fairly.
    - ``audio_dur_norm``: the server's mean audio duration divided by that
      shared mean audio duration (1.0 for the average server).

    The "actual" per-server values (mean total time, audio duration, and the
    real-time RTF) remain available in the summary itself.
    """
    audio_means = [
        s["audio_duration"]["mean"]
        for _, s in rows
        if s["ok"] > 0 and s["audio_duration"]["mean"] is not None
    ]
    if not audio_means:
        return {}
    mean_audio = statistics.fmean(audio_means)
    out: dict[str, dict] = {}
    for server, s in rows:
        if s["ok"] == 0:
            continue
        total = s["total"]["mean"]
        audio = s["audio_duration"]["mean"]
        out[server] = {
            "rtf_norm": total / mean_audio if total is not None else None,
            "audio_dur_norm": audio / mean_audio if audio is not None else None,
        }
    return out


# --- plain-text rendering ---------------------------------------------------


def print_server_report(server: str, measurements: list[Measurement]) -> None:
    """Print a per-mode summary block for a single server."""
    print()
    print(f"Server: {server}")
    print("-" * 78)
    if not measurements:
        print("  (no measurements)")
        return
    modes: list[str] = []
    for m in measurements:
        if m.mode not in modes:
            modes.append(m.mode)
    for mode in modes:
        s = summarize_mode(measurements, mode)
        if s["ok"] == 0 and s["fail"] == 0:
            continue
        print(f"[{mode}] ok={s['ok']} fail={s['fail']}")
        for line in stat_table(
            [
                ("TTFT (ms)", s["ttft"], _fmt_ms_cell),
                ("Total (ms)", s["total"], _fmt_ms_cell),
                ("RTF (x)", s["rtf"], _fmt_cell),
            ]
        ):
            print(line)
        d = s["audio_duration"]
        b = s["audio_bytes"]
        mean_dur = d["mean"] if d["mean"] is not None else 0.0
        mean_bytes = b["mean"] if b["mean"] is not None else 0.0
        print(f"  {'Audio':<12} mean={mean_dur:.2f}s ({mean_dur * 1000:.0f} ms), {mean_bytes:.0f} bytes")


def print_summary(servers: list[str], by_server: dict[str, list[Measurement]]) -> None:
    """Print a summary table, one per mode."""
    modes: list[str] = []
    for ms in by_server.values():
        for m in ms:
            if m.mode not in modes:
                modes.append(m.mode)
    if not modes:
        return
    print()
    print("=" * 78)
    print("Summary (lower is better)")
    print("=" * 78)
    for mode in modes:
        rows = [(server, summarize_mode(by_server.get(server, []), mode)) for server in servers]
        rows.sort(
            key=lambda item: (
                item[1]["ttft"]["mean"] is None,
                item[1]["ttft"]["mean"] if item[1]["ttft"]["mean"] is not None else 0.0,
            )
        )
        norm = normalized_stats(rows)
        print()
        print(f"[{mode}]")
        print(
            f"  {'server':<{SERVER_COL_WIDTH}}{'ok':>4}{'TTFT_mean':>12}{'TTFT_p95':>12}"
            f"{'Total_mean':>12}{'RTF_norm':>10}{'RTF_adj':>10}{'RTF_act':>10}{'audio_dur':>12}"
        )
        for server, s in rows:
            n = norm.get(server, {})
            print(
                f"  {truncate_label(server):<{SERVER_COL_WIDTH}}{s['ok']:>4}"
                f"{_fmt_ms_cell(s['ttft']['mean']):>12}"
                f"{_fmt_ms_cell(s['ttft']['p95']):>12}"
                f"{_fmt_ms_cell(s['total']['mean']):>12}"
                f"{_fmt_cell(n.get('rtf_norm')):>10}"
                f"{_fmt_cell(n.get('audio_dur_norm')):>10}"
                f"{_fmt_cell(s['rtf']['mean']):>10}"
                f"{_fmt_cell(s['audio_duration']['mean']):>11}s"
            )
