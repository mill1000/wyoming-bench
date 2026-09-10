# wyoming-tts-bench

Benchmark text-to-speech servers that speak the [Wyoming protocol](https://github.com/OHF-Voice/wyoming).

The tool drives each server through the **non-streaming** (`synthesize`) and
**streaming** (`synthesize-start` / `synthesize-chunk` / `synthesize-stop`)
synthesis paths and reports latency and throughput:

- **TTFT** — time to first audio byte (from when the request is sent).
- **Total** — total wall time until the final `audio-stop` event.
- **RTF** — total time divided by the duration of the returned audio
  (lower is better; `< 1` means faster than real time).
- **Audio** — returned audio duration and byte count.

Every measurement uses a fresh TCP connection, so results are isolated per
round. Statistics (min / mean / median / p95 / max) are aggregated across
rounds.

## Install

```sh
pip install .
```

or run in place without installing:

```sh
python3 -m wyoming_tts_bench --help
```

## Usage

```sh
# One server, both modes, default texts (one per 1..4 sentences)
wyoming-tts-bench -s 10.100.1.20:10210

# Multiple servers, streaming only, 5 rounds
wyoming-tts-bench -s a:10700 -s b:10700 --mode streaming --rounds 5

# Comma-separated servers, specific voice from a JSON config
wyoming-tts-bench --servers a:10700,b:10700 \
    --config '{"voice": {"name": "en_US-lessac-medium"}, "text_format": "text"}'

# Verbose per-measurement output, extra chunk delay for streaming
wyoming-tts-bench -s a:10700 --mode streaming --chunk-delay 0.2 -v
```

## Options

| Flag | Description | Default |
| --- | --- | --- |
| `-s, --server HOST[:PORT]` | Server to benchmark (repeatable). | — |
| `--servers A,B` | Comma-separated server list (alt to `-s`). | — |
| `--rounds N` | Timed measurement rounds per text per mode. | `3` |
| `--warmup N` | Untimed warmup runs per mode. | `1` |
| `--mode {non_streaming,streaming,both}` | Synthesis path(s) to exercise. | `both` |
| `--texts TEXT [TEXT ...]` | Text samples to synthesize. | built-in 1–4 sentence set |
| `--texts-file FILE` | One text sample per line. | — |
| `--config JSON` | `{"voice": "name"}` or `{"voice": {"name","language","speaker"}, "text_format": "text"}`. | `{}` |
| `--chunk-delay SEC` | Seconds to wait between `synthesize-chunk` writes (streaming). | `0` |
| `--timeout SEC` | Connect and per-event read timeout. | `60` |
| `-v, --verbose` | Print one line per measurement. | off |

A bare `HOST` uses port `10700` (the Wyoming default).

## Exit codes

`0` — at least one measurement succeeded.
`1` — every measurement on every server failed.
`130` — interrupted.
