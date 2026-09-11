"""Shared statistics and formatting helpers for TTS and STT reporting.

Domain-specific measurement containers and report renderers live in
``tts.reporting`` and ``stt.reporting``; both build on these helpers.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence


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


def _fmt_cell(value: float | None, width: int = 10, decimals: int = 2) -> str:
    if value is None:
        return f"{'n/a':>{width}}"
    return f"{value:>{width}.{decimals}f}"


def _fmt_ms_cell(value: float | None, width: int = 10) -> str:
    if value is None:
        return f"{'n/a':>{width}}"
    return f"{value * 1000:>{width}.1f}"


# Fixed width of the server column in summary tables. Labels are truncated
# (with an ellipsis) to this width so the columns stay aligned.
SERVER_COL_WIDTH = 34


def truncate_label(label: str, width: int = SERVER_COL_WIDTH) -> str:
    """Truncate *label* to *width* characters, ending in "..." when cut."""
    if len(label) <= width:
        return label
    return label[: width - 3] + "..."


def _stat_line(label: str, s: dict, cell=_fmt_cell) -> str:
    return (
        f"  {label:<12}"
        f"min{cell(s['min']):>12} mean{cell(s['mean']):>12}"
        f"median{cell(s['median']):>12} p95{cell(s['p95']):>12} max{cell(s['max']):>12}"
    )
