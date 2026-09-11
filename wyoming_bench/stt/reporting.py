"""STT measurement container and plain-text reporting (no CSV export)."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from ..reporting import (
    SERVER_COL_WIDTH,
    _fmt_cell,
    _fmt_ms_cell,
    _stat_line,
    stats,
    truncate_label,
)


@dataclass
class SttMeasurement:
    """A single timed transcription run against one server in one mode."""

    server: str
    mode: str
    sample_id: str
    ok: bool = False
    error: str = ""
    t_sent: float = 0.0
    t_first: float | None = None  # time to first transcript-chunk (streaming)
    t_end: float | None = None  # time to final result
    audio_duration_s: float = 0.0
    n_transcript_chunks: int = 0
    hypothesis: str = ""
    reference: str = ""
    # Accuracy components (populated from the accuracy module when ok).
    sub: int = 0
    ins: int = 0
    del_: int = 0
    n_ref_words: int = 0
    char_errors: int = 0
    n_ref_chars: int = 0
    wer: float | None = None
    cer: float | None = None
    exact: bool = False

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


def summarize_stt_mode(measurements: list[SttMeasurement], mode: str) -> dict:
    """Aggregate the successful STT measurements for *mode*.

    Speed metrics use the same min/mean/median/p95/max shape as TTS. Accuracy
    uses the pooled (macro) error rates: total edits divided by total reference
    units across all samples, which is more stable than averaging per-sample
    rates when sample lengths vary.
    """
    in_mode = [m for m in measurements if m.mode == mode]
    ok = [m for m in in_mode if m.ok]
    total = [m.total_s for m in ok if m.total_s is not None]
    ttft = [m.ttft_s for m in ok if m.ttft_s is not None]
    rtf = [m.rtf for m in ok if m.rtf is not None]
    audio_duration = [m.audio_duration_s for m in ok]

    word_errors = sum(m.sub + m.ins + m.del_ for m in ok)
    ref_words = sum(m.n_ref_words for m in ok)
    char_errors = sum(m.char_errors for m in ok)
    ref_chars = sum(m.n_ref_chars for m in ok)
    wers = [m.wer for m in ok if m.wer is not None]
    exact = sum(1 for m in ok if m.exact)

    return {
        "mode": mode,
        "ok": len(ok),
        "fail": len(in_mode) - len(ok),
        "total": stats(total),
        "ttft": stats(ttft),
        "rtf": stats(rtf),
        "audio_duration": stats(audio_duration),
        "wer": (word_errors / ref_words) if ref_words else None,
        "cer": (char_errors / ref_chars) if ref_chars else None,
        "mean_wer": statistics.fmean(wers) if wers else None,
        "sar": (exact / len(ok)) if ok else None,
        "sub": sum(m.sub for m in ok),
        "ins": sum(m.ins for m in ok),
        "del": sum(m.del_ for m in ok),
        "ref_words": ref_words,
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def print_stt_server_report(server: str, measurements: list[SttMeasurement]) -> None:
    """Print a per-mode summary block for one STT server."""
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
        s = summarize_stt_mode(measurements, mode)
        if s["ok"] == 0 and s["fail"] == 0:
            continue
        print(f"[{mode}] ok={s['ok']} fail={s['fail']}")
        print(_stat_line("Total (ms)", s["total"], _fmt_ms_cell))
        if s["ttft"]["n"] > 0:
            print(_stat_line("TTFT (ms)", s["ttft"], _fmt_ms_cell))
        print(_stat_line("RTF (x)", s["rtf"]))
        mean_dur = s["audio_duration"]["mean"]
        mean_dur = mean_dur if mean_dur is not None else 0.0
        print(f"  {'Audio':<12} mean={mean_dur:.2f}s input")
        print(f"  {'Accuracy':<12} WER={_pct(s['wer'])}  " f"CER={_pct(s['cer'])}  SAR={_pct(s['sar'])}")
        print(f"  {'':<12} edits S={s['sub']} I={s['ins']} D={s['del']}" f"  (ref words={s['ref_words']})")


def print_stt_summary(servers: list[str], by_server: dict[str, list[SttMeasurement]]) -> None:
    """Print an STT summary table, one per mode.

    Rows are sorted by pooled WER (accuracy) first, then mean total time.
    """
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

    def _sort_key(item: tuple[str, dict]):
        s = item[1]
        return (
            s["wer"] is None,
            s["wer"] if s["wer"] is not None else 0.0,
            s["total"]["mean"] is None,
            s["total"]["mean"] if s["total"]["mean"] is not None else 0.0,
        )

    for mode in modes:
        rows = [(server, summarize_stt_mode(by_server.get(server, []), mode)) for server in servers]
        rows.sort(key=_sort_key)
        print()
        print(f"[{mode}]")
        print(
            f"  {'server':<{SERVER_COL_WIDTH}}{'ok':>4}{'WER':>9}{'CER':>9}{'SAR':>8}"
            f"{'Total_mean':>12}{'TTFT_p95':>11}{'RTF_mean':>10}"
        )
        for server, s in rows:
            print(
                f"  {truncate_label(server):<{SERVER_COL_WIDTH}}{s['ok']:>4}"
                f"{_pct(s['wer']):>9}"
                f"{_pct(s['cer']):>9}"
                f"{_pct(s['sar']):>8}"
                f"{_fmt_ms_cell(s['total']['mean']):>12}"
                f"{_fmt_ms_cell(s['ttft']['p95']):>11}"
                f"{_fmt_cell(s['rtf']['mean']):>10}"
            )
