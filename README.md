# wyoming-bench

Benchmark TTS & STT servers that speak the [Wyoming protocol](https://github.com/OHF-Voice/wyoming).

## Quick Start

Install via pipx

```sh
pipx install wyoming-bench
```

Query a server to see what it supports

```sh
wyoming-bench info <server>:10200
```

Benchmark a TTS server like [wyoming-piper](https://github.com/OHF-Voice/wyoming-piper).

```sh
wyoming-bench tts <tts-server>:10200
```

Benchmark an STT server like [wyoming-faster-whisper](https://github.com/OHF-Voice/wyoming-faster-whisper).

```sh
# Generate synthetic corpus
wyoming-bench generate-corpus <tts-server>:10200 --out ./corpus
# Test with synthetic corpus
wyoming-bench stt <stt-server>:10300 --corpus ./corpus
```

Multiple servers can be tested in one command.

```sh
wyoming-bench tts <tts-server-a>:10200 <tts-server-b>:10200
```

## Vibe Warning

This project was **vibe coded** with Qwen 3.8 27B and the Zed Agent on local hardware.

## Details

### TTS Benchmarks

Benchmark text-to-speech servers in both non-streaming and streaming mode.
In streaming mode the text is split with [sentence-stream](https://github.com/OHF-Voice/sentence-stream) to mimic an LLM generating a response.

By default, a built-in set of texts is used. Custom texts can be provided with `--texts` or `--texts-dir`.

Metrics, per mode, aggregated across rounds as min / mean / median / p95 / max:

| Metric | Meaning |
| --- | --- |
| TTFT | Time to first audio byte (from when the request is sent). |
| Total | Wall time until all audio has been received. |
| RTF | Total time divided by the duration of the returned audio. `< 1` means faster than real time (lower is better). The summary also reports `RTF_norm`, which divides each server's total time by the mean audio duration across servers, so servers speaking the same text at different rates are compared fairly. |
| Audio | Returned audio duration and byte count. |

### STT Benchmarks

Benchmark speech-to-text servers in both non-streaming and streaming mode, reporting speed and accuracy metrics.

A benchmark requires a corpus. See [corpus format](#stt-corpus-format) for more information.

A synthetic corpus can be [generated using a TTS server](#generating-a-synthetic-corpus-from-a-tts-server) or existing recordings can be [transcribed to seed a corpus](#seeding-a-corpus-from-existing-recordings) with this tool.

The benchmark transcribes every recording in the corpus and compares the result against the transcripts to determine accuracy metrics.
Texts are normalized before comparison (lowercased, punctuation stripped, whitespace collapsed).

By default 0.5 seconds of silence is appended to the end of each recording before sending it. Use `--trailing-silence 0` to send recordings exactly as stored.

Speed, per mode, aggregated across rounds as min / mean / median / p95 / max:

| Metric | Meaning |
| --- | --- |
| Total | Wall time until the final transcript. |
| TTFT | Time to the first partial result (streaming mode only). |
| RTF | Total time divided by the duration of the input audio. `< 1` means faster than real time (lower is better). |

Accuracy, per mode, across all samples (word-, character-, and sentence-level):

| Metric | Meaning |
| --- | --- |
| WER | Word error rate: (substitutions + insertions + deletions) over total reference words. |
| CER | Character error rate over total reference characters. |
| SAR | Sentence accuracy: the fraction of samples whose normalized transcript exactly matches the reference. |
| S / I / D | Total substitution / insertion / deletion word counts. |

### Program Selection

A single Wyoming server can support multiple programs (e.g. multiple TTS or STT backends). Use `wyoming-bench info` to list the supported programs.

The `--programs` argument specifies which programs to test in a benchmark. If unspecified, the each server's default program is used.
- `--programs all` benchmarks every program the server supports.
- `--programs NAME[,NAME…]` benchmarks the named programs on each server.
  - If a server doesn't support the named program, a warning is emitted and the server is skipped.
  - Add `any` to the program list to allow fallback to the server's default if the named program is not supported.

## Examples

```sh
# TTS: one server, both modes, default texts
wyoming-bench tts 127.0.0.1:10200

# TTS: multiple servers, streaming only, 5 rounds
wyoming-bench tts a:10200 b:10200 --mode streaming --rounds 5

# TTS: specific voice from a JSON config
wyoming-bench tts a:10200 --config '{"voice": {"name": "en_US-lessac-medium"}, "text_format": "text"}'

# STT: one server, both modes
wyoming-bench stt 127.0.0.1:10300 --corpus ./samples

# STT: multiple servers, 3 rounds
wyoming-bench stt a:10300 b:10300 --corpus ./samples --rounds 3

# STT: streaming only, pace audio-chunk writes at 0.1s
wyoming-bench stt a:10300 --mode streaming --corpus ./samples --chunk-delay 0.1

# Inspect what a server offers before benchmarking it
wyoming-bench info 127.0.0.1:10200

# TTS: a server hosting multiple TTS programs — pick some, or benchmark them all
wyoming-bench tts a:10200 --programs piper,kokoro
wyoming-bench tts a:10200 --programs all

# TTS: servers hosting different programs — each benchmarks the ones it advertises
wyoming-bench tts a:10200 b:10201 --programs piper,kokoro

# TTS: piper where available, each server's default program otherwise
wyoming-bench tts a:10200 b:10201 --programs piper,any

# STT: a server hosting multiple STT (ASR) programs — benchmark them all
wyoming-bench stt a:10300 --corpus ./samples --programs all
```

## STT Corpus Format

A corpus is a set of test samples, each sample an audio recording and a transcript of what it says.

A corpus should be a directory of `.wav` recordings, with a corresponding `.txt` file:

```
samples/
  data_01.wav
  data_01.txt
  data_02.wav
  data_02.txt
  ...
```

- Files without a matching counterpart are skipped with a warning.
- The audio file is sent to the server as-is, so recordings should be in a format your STT server supports.

### Generating a synthetic corpus from a TTS server

A synthetic corpus can be generated by synthesizing text with a TTS server. By default, a built-in set of texts is used. Custom texts can be provided with `--texts` or `--texts-dir`.

```sh
wyoming-bench generate-corpus tts:10200 --out ./tts-samples --config '{"voice": "en_US-lessac-medium"}'
```

### Seeding a corpus from existing recordings

Seed a corpus by transcribing existing recordings with an STT server.

```sh
wyoming-bench seed-corpus stt:10300 --corpus ./recordings
```

- Transcriptions should be checked for accuracy before utilizing in a benchmark.
- Recordings that already have a transcript (`.txt` file) are skipped by default, so reviewed transcripts aren't overwritten.
