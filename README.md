# wyoming-bench

Benchmark servers that speak the [Wyoming protocol](https://github.com/OHF-Voice/wyoming):

- **TTS** — text-to-speech servers, via the **non-streaming** (`synthesize`) and
  **streaming** (`synthesize-start` / `synthesize-chunk` / `synthesize-stop`)
  synthesis paths. Reports latency and throughput.
- **STT** — speech-to-text (ASR) servers, via the **non-streaming** and
  **streaming** (`transcript-start` / `transcript-chunk` / `transcript-stop`)
  transcription paths. Reports speed **and** word-level accuracy.
- **generate-corpus** — synthesize a test corpus from a TTS server to feed the
  STT benchmark (a realistic TTS→STT round-trip).
- **seed-corpus** — transcribe existing recordings (no transcripts) with an STT
  server, writing a `.txt` next to each `.wav` for manual review.
- **info** — query a server's advertised services (`describe`/`info`), so you
  can see whether it speaks TTS, STT, or both, and whether it supports
  streaming.

Before benchmarking, sends a `describe` to
confirm the server advertises the service being benchmarked. A server that
answers but lacks it (e.g. an STT server for a TTS benchmark) is skipped with
a notice instead of timing out on every round. Servers that do not answer
describe at all are benchmarked as before. The same describe response also
reports whether the server advertises streaming support: when it does not,
streaming mode is skipped immediately with a notice (no behavioral probe, no
per-round timeout). A server that advertises streaming — or does not report it
at all — is probed once to confirm before the streaming rounds run.

Every measurement uses a fresh TCP connection, so results are isolated per
round. Statistics (min / mean / median / p95 / max) are aggregated across
rounds.

## Vibe Warning
This project was **vibe coded** with Qwen 3.8 27B and the Zed Agent on local hardware.


## Install

```sh
pipx install wyoming-bench
```

## Usage

```sh
# TTS: one server, both modes, default texts (one per 1..4 sentences)
wyoming-bench tts 10.100.1.20:10210

# TTS: multiple servers, streaming only, 5 rounds
wyoming-bench tts a:10700 b:10700 --mode streaming --rounds 5

# TTS: specific voice from a JSON config
wyoming-bench tts a:10700 \
    --config '{"voice": {"name": "en_US-lessac-medium"}, "text_format": "text"}'

# TTS: verbose per-measurement output, extra chunk delay for streaming
wyoming-bench tts a:10700 --mode streaming --chunk-delay 0.2 -v

# STT: one server, both modes, over a corpus of paired recordings
wyoming-bench stt 10.100.1.20:10700 --corpus ./samples

# STT: multiple servers, accuracy + speed comparison, 3 rounds
wyoming-bench stt a:10700 b:10700 --corpus ./samples --rounds 3

# STT: streaming only, pace audio-chunk writes at 0.1s (simulated real-time)
wyoming-bench stt a:10700 --mode streaming --corpus ./samples --chunk-delay 0.1 -v

# Generate a corpus from a TTS server, then benchmark an STT server on it
wyoming-bench generate-corpus tts:10210 --out ./tts-samples \
    --texts "The quick brown fox jumps over the lazy dog."
wyoming-bench stt asr:10700 --corpus ./tts-samples

# Seed a corpus: transcribe existing recordings (no transcripts) with an STT server
wyoming-bench seed-corpus asr:10700 --corpus ./recordings
# ... review/correct the .txt files, then benchmark on the seeded corpus
wyoming-bench stt asr:10700 --corpus ./recordings

# Inspect what a server offers before benchmarking it
wyoming-bench info 10.100.1.20:10700
```

### TTS metrics

| Metric | Meaning |
| --- | --- |
| **TTFT** | Time to first audio byte (from when the request is sent). |
| **Total** | Total wall time until the final `audio-stop` event. |
| **RTF** | Total time divided by the duration of the returned audio (lower is better; `< 1` means faster than real time). |
| **Audio** | Returned audio duration and byte count. |

In streaming mode the text is split into sentences (via the
[sentence-stream](https://github.com/OHF-Voice/sentence-stream) heuristics,
which hold abbreviations such as `Mr.` and `U.S.` together) and sent one
sentence per `synthesize-chunk` event, so servers that buffer until a
certain boundary are exercised the way an LLM-driven pipeline would drive
them. `--chunk-delay` paces those writes.

### STT metrics

Speed:

| Metric | Meaning |
| --- | --- |
| **Total** | Wall time until the final transcript result. |
| **TTFT** | Time to the first `transcript-chunk` (streaming mode only). |
| **RTF** | Total time divided by the duration of the input audio (lower is better; `< 1` means faster than real time). |

Accuracy (word-level by default, character-level as a bonus):

| Metric | Meaning |
| --- | --- |
| **WER** | Pooled word error rate: `(substitutions + insertions + deletions)` over total reference words across all samples. |
| **CER** | Pooled character error rate (over total reference characters). |
| **SAR** | Sentence accuracy: the fraction of samples whose normalized transcript exactly matches the reference. |
| **S / I / D** | Total substitution / insertion / deletion word counts. |

Reference and hypothesis text are normalized before comparison (lowercased,
punctuation stripped, whitespace collapsed).

By default the bench appends `--trailing-silence` (0.5s) of silence to the end
of each recording **before** sending it. Streaming (online) ASR models commit
words incrementally and need a little trailing context to finalize their last
words; without it they tend to drop the final one or two, which inflates WER.
RTF is still computed on the original recording length, so the padding does not
speed up the timing. Set `--trailing-silence 0` to send recordings exactly as
stored.

## Corpus format (STT)

`--corpus DIR` points at a directory of `.wav` recordings paired with `.txt`
transcripts that share the same stem:

```
samples/
  data_01.wav   +   data_01.txt
  data_02.wav   +   data_02.txt
  ...
```

- `data_01.txt` holds the reference transcript for `data_01.wav`.
- Files without a matching counterpart are skipped with a warning.
- The WAV is sent to the server as-is (its sample rate / width / channels are
  read from the header). Record the corpus in the format the ASR server expects
  — typically **16 kHz, 16-bit, mono**.

## Generating a test corpus from a TTS server

Prefer not to record audio? `generate-corpus` synthesizes text with a TTS server and
saves each sample as a `.wav` plus the source text as its `.txt` reference. A
TTS→STT round-trip then yields a realistic word-error-rate measurement:

```sh
wyoming-bench generate-corpus tts:10210 --out ./tts-samples \
    --config '{"voice": "en_US-lessac-medium"}'
wyoming-bench stt asr:10700 --corpus ./tts-samples
```

The WAV is saved in whatever format the TTS server emits — make sure that format
is acceptable to the STT server (most resample internally, but check yours).

## Seeding a corpus from existing recordings

Already have recordings but no transcripts? `seed-corpus` transcribes every `.wav` in a
directory with a Wyoming STT server (non-streaming) and writes the result next
to each recording (`<stem>.txt`):

```sh
wyoming-bench seed-corpus asr:10700 --corpus ./recordings
```

- Recordings that already have a `.txt` are **skipped** by default, so reviewed
  transcripts are never clobbered; pass `--overwrite` to re-transcribe them.
- The transcripts are written as the server returned them (trimmed, one per
  line) and are not verified against anything — reviewing and correcting the
  `.txt` files is a manual step before benchmarking:
  `wyoming-bench stt asr:10700 --corpus ./recordings`.

## Options

All subcommands take the server(s) as positional arguments (`SERVER` is `HOST`
or `HOST:PORT`, repeatable). `tts` and `stt` share these flags (in addition
to their own):

| Flag | Description | Default |
| --- | --- | --- |
| `--rounds N` | Timed measurement rounds per sample per mode. | `3` |
| `--warmup N` | Untimed warmup runs per mode. | `1` |
| `--mode {non_streaming,streaming,both}` | Path(s) to exercise. | `both` |
| `-v, --verbose` | Print one line per measurement. | off |

Every subcommand also takes `--timeout SEC` — the connect and per-event read
timeout (for `info`, the whole describe exchange). It defaults to `60` (`5`
for `info`); the one-off describe/streaming probes are additionally capped at
`8s`, so a server that never answers is skipped quickly.

TTS-specific flags:

| Flag | Description | Default |
| --- | --- | --- |
| `--texts TEXT [TEXT ...]` | Text samples to synthesize. | built-in 1–4 sentence set |
| `--texts-file FILE` | One text sample per line. | — |
| `--texts-dir DIR` | Directory of .txt files, one text sample per file. | — |
| `--config JSON` | `{"voice": "name"}` or `{"voice": {"name","language","speaker"}, "text_format": "text"}`. | `{}` |
| `--chunk-delay SEC` | Seconds to wait between `synthesize-chunk` writes (streaming). | `0` |
| `--unique` / `--no-unique` | Append a random nonce to each text to defeat server-side synthesis caching (cold-cache timings). | on |

STT-specific flags:

| Flag | Description | Default |
| --- | --- | --- |
| `--corpus DIR` | Directory of paired `.wav`/`.txt` samples (required). | — |
| `--config JSON` | Transcribe JSON, e.g. `{"name": "model", "language": "en"}` (fields: `name`, `language`, `context`, `vad_sensitivity`, `transcript_names`, `transcript_terms`). | `{}` |
| `--chunk-samples N` | Samples per `audio-chunk` when sending audio. | `1024` |
| `--chunk-delay SEC` | Seconds to wait between `audio-chunk` writes (streaming). | `0` |
| `--trailing-silence SEC` | Seconds of silence appended to each recording before transcription (helps streaming/online models finalize their last words; RTF still uses the original length). | `0.5` |

generate-corpus-specific flags:

| Flag | Description | Default |
| --- | --- | --- |
| `--out DIR` | Output directory for the corpus (created if missing, required). | — |
| `--mode {non_streaming,streaming}` | TTS synthesis path to use. | `non_streaming` |
| `--texts TEXT [TEXT ...]` | Text to synthesize. | built-in 1–4 sentence set |
| `--texts-file FILE` | One text sample per line. | — |
| `--texts-dir DIR` | Directory of .txt files, one text sample per file. | — |
| `--config JSON` | Voice/format JSON (same as `tts`). | `{}` |
| `--prefix PFX` | File name prefix (`sample_01.wav`, …). | `sample` |
| `--overwrite` | Overwrite existing files. | off |

seed-corpus-specific flags:

| Flag | Description | Default |
| --- | --- | --- |
| `--corpus DIR` | Directory of `.wav` recordings to transcribe; each transcript is written next to its recording (required). | — |
| `--config JSON` | Transcribe JSON (same as `stt`). | `{}` |
| `--chunk-samples N` | Samples per `audio-chunk` when sending audio. | `1024` |
| `--trailing-silence SEC` | Seconds of silence appended to each recording before transcription (helps streaming/online models finalize their last words). | `0.5` |
| `--overwrite` | Overwrite existing `.txt` transcripts (skipped by default). | off |

`info` takes no flags beyond the shared ones (its `--timeout` default is `5`).

A bare `HOST` uses port `10700` (the Wyoming default).

## Exit codes

`0` — at least one measurement succeeded.
`1` — every measurement on every server failed (or a server was skipped / no
info response).
`130` — interrupted.
