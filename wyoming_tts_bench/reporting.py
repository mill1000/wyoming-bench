"""Measurement container and plain-text reporting (no CSV export)."""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class Measurement:
    """A single timed synthesis run against one server in one mode."""

    server: str
    mode: str
    text: str
    ok: bool = False
    error: str = ""
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


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolation percentile. *pct* is in [0, 100]."""
    if not sorted_values:
        raise ValueError("no values")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def stats(values: Sequence[float]) -> dict:
    """Compute min/mean/median/p95/max over *values*. Empty -> all None."""
    if not values:
        return {"n": 0, "min": None, "mean": None, "median": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {
        "n": len(values),
        "min": ordered[0],
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": _percentile(ordered, 95),
        "max": ordered[-1],
    }


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


# --- plain-text rendering ---------------------------------------------------


def _fmt_cell(value: float | None, width: int = 10, decimals: int = 2) -> str:
    if value is None:
        return f"{'n/a':>{width}}"
    return f"{value:>{width}.{decimals}f}"


def _fmt_ms_cell(value: float | None, width: int = 10) -> str:
    if value is None:
        return f"{'n/a':>{width}}"
    return f"{value * 1000:>{width}.1f}"


def _stat_line(label: str, s: dict, cell=_fmt_cell) -> str:
    return (
        f"  {label:<12}"
        f"min{cell(s['min']):>12} mean{cell(s['mean']):>12}"
        f"median{cell(s['median']):>12} p95{cell(s['p95']):>12} max{cell(s['max']):>12}"
    )


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
        print(_stat_line("TTFT (ms)", s["ttft"], _fmt_ms_cell))
        print(_stat_line("Total (ms)", s["total"], _fmt_ms_cell))
        print(_stat_line("RTF (x)", s["rtf"]))
        d = s["audio_duration"]
        b = s["audio_bytes"]
        mean_dur = d["mean"] if d["mean"] is not None else 0.0
        mean_bytes = b["mean"] if b["mean"] is not None else 0.0
        print(f"  {'Audio':<12} mean={mean_dur:.2f}s ({mean_dur * 1000:.0f} ms), {mean_bytes:.0f} bytes")


def print_comparison(servers: list[str], by_server: dict[str, list[Measurement]]) -> None:
    """Print a cross-server comparison table, one per mode."""
    modes: list[str] = []
    for ms in by_server.values():
        for m in ms:
            if m.mode not in modes:
                modes.append(m.mode)
    if not modes:
        return
    print()
    print("=" * 78)
    print("Comparison (lower is better)")
    print("=" * 78)
    for mode in modes:
        rows = [(server, summarize_mode(by_server.get(server, []), mode)) for server in servers]
        rows.sort(key=lambda item: (
            item[1]["ttft"]["mean"] is None,
            item[1]["ttft"]["mean"] if item[1]["ttft"]["mean"] is not None else 0.0,
        ))
        print()
        print(f"[{mode}]")
        print(
            f"  {'server':<24}{'ok':>4}{'TTFT_mean':>12}{'TTFT_p95':>12}"
            f"{'Total_mean':>12}{'RTF_mean':>10}{'audio_dur':>12}"
        )
        for server, s in rows:
            print(
                f"  {server:<24}{s['ok']:>4}"
                f"{_fmt_ms_cell(s['ttft']['mean']):>12}"
                f"{_fmt_ms_cell(s['ttft']['p95']):>12}"
                f"{_fmt_ms_cell(s['total']['mean']):>12}"
                f"{_fmt_cell(s['rtf']['mean']):>10}"
                f"{_fmt_cell(s['audio_duration']['mean']):>11}s"
            )
