"""Tests for wyoming_bench.stt.bench_stt_server streaming-skip via advertised info.

A dead port (127.0.0.1:1) is used throughout: it proves the "not advertised"
path makes no connection at all, and it makes the behavioral probe fail fast
when the flag says streaming is supported (or unknown).
"""

from __future__ import annotations

import contextlib
import io
import unittest
from unittest.mock import patch

from wyoming.asr import Transcribe

from wyoming_bench.const import MODE_NON_STREAMING, MODE_STREAMING
from wyoming_bench.stt.bench import bench_stt_server
from wyoming_bench.stt.reporting import SttMeasurement
from wyoming_bench.tests.stt.test_client import _make_audio
from wyoming_bench.tests.test_info import make_info, make_info_two_asr


async def _run(info):
    """Run one streaming bench against a dead port; return ``(ms, output)``."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ms = await bench_stt_server(
            "127.0.0.1",
            1,
            [_make_audio()],
            [MODE_STREAMING],
            Transcribe(),
            3,
            0,
            False,
            1024,
            0.0,
            1.0,
            1.0,
            1.0,
            info,
        )
    return ms, out.getvalue()


class TestBenchSttServerStreamingSkip(unittest.IsolatedAsyncioTestCase):
    async def test_skips_when_not_advertised(self):
        # make_info() advertises asr streaming; clear it so streaming is skipped
        # immediately without any connection attempt.
        info = make_info()
        info.asr[0].supports_transcript_streaming = False
        ms, out = await _run(info)
        self.assertEqual(ms, [])
        self.assertIn("not advertised by server", out)

    async def test_probes_when_advertised(self):
        # When the server advertises asr streaming, the behavioral probe still
        # runs; against a dead port it fails fast with no transcript-chunk.
        ms, out = await _run(make_info())
        self.assertEqual(ms, [])
        self.assertIn("no transcript-chunk in response", out)

    async def test_probes_when_info_unknown(self):
        # info=None (server unreachable / no describe support) must NOT skip;
        # the probe runs and fails fast against the dead port.
        ms, out = await _run(None)
        self.assertEqual(ms, [])
        self.assertIn("no transcript-chunk in response", out)


class TestBenchSttServerProgram(unittest.IsolatedAsyncioTestCase):
    async def _fake_measure(self, calls: list):
        async def fake_measure_stt_once(*args, **kwargs):
            calls.append(kwargs.get("program", args[10] if len(args) > 10 else None))
            audio = args[3]
            return SttMeasurement(server="127.0.0.1:1", mode=args[2], sample_id=audio.sample_id)

        return fake_measure_stt_once

    async def test_program_forwarded_to_measure_once(self):
        calls: list[str | None] = []

        with patch("wyoming_bench.stt.bench.measure_stt_once", side_effect=await self._fake_measure(calls)):
            with contextlib.redirect_stdout(io.StringIO()):
                await bench_stt_server(
                    "127.0.0.1",
                    1,
                    [_make_audio()],
                    [MODE_NON_STREAMING],
                    Transcribe(),
                    1,
                    0,
                    False,
                    1024,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    None,
                    0.0,
                    program="asr-prog",
                )
        self.assertTrue(calls)
        self.assertTrue(all(p == "asr-prog" for p in calls))

    async def test_streaming_skip_uses_selected_program_flag(self):
        # asr-prog (the first program) advertises no streaming, but whisper
        # does: the skip decision must follow the selected program, not the
        # first advertised one.
        info = make_info_two_asr()
        calls: list[str | None] = []

        kwargs = dict(
            host="127.0.0.1",
            port=1,
            samples=[_make_audio()],
            modes=[MODE_STREAMING],
            transcribe=Transcribe(),
            rounds=1,
            warmup=0,
            verbose=False,
            chunk_samples=1024,
            chunk_delay=0.0,
            connect_timeout=1.0,
            read_timeout=1.0,
            probe_timeout=1.0,
            info=info,
        )
        with patch("wyoming_bench.stt.bench.measure_stt_once", side_effect=await self._fake_measure(calls)):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                ms = await bench_stt_server(**kwargs, program="whisper")
        # whisper advertises streaming, so the probe runs (and fails against
        # the dead port: no transcript-chunk) instead of skipping as
        # "not advertised".
        self.assertEqual(ms, [])
        self.assertNotIn("not advertised by server", out.getvalue())
        self.assertIn("no transcript-chunk in response", out.getvalue())

        calls.clear()
        with patch("wyoming_bench.stt.bench.measure_stt_once", side_effect=await self._fake_measure(calls)):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                ms = await bench_stt_server(**kwargs, program="asr-prog")
        # asr-prog advertises no streaming: skipped immediately, no probe.
        self.assertEqual(ms, [])
        self.assertEqual(calls, [])
        self.assertIn("not advertised by server", out.getvalue())

    async def test_probe_receives_selected_program(self):
        info = make_info_two_asr()
        probe_calls: list[str | None] = []

        async def fake_probe(*args, **kwargs):
            probe_calls.append(kwargs.get("program", args[-1] if args else None))
            return False

        with patch("wyoming_bench.stt.bench._probe_stt_streaming", side_effect=fake_probe):
            with contextlib.redirect_stdout(io.StringIO()):
                ms = await bench_stt_server(
                    "127.0.0.1",
                    1,
                    [_make_audio()],
                    [MODE_STREAMING],
                    Transcribe(),
                    1,
                    0,
                    False,
                    1024,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    info,
                    0.0,
                    "whisper",
                )
        self.assertEqual(ms, [])
        self.assertEqual(probe_calls, ["whisper"])


class TestBenchSttServerUnreachable(unittest.IsolatedAsyncioTestCase):
    async def test_skips_remaining_runs_after_connection_failure(self):
        calls: list[str] = []

        async def fake_measure_stt_once(*args):
            mode, audio = args[2], args[3]
            calls.append(f"{mode}:{audio.sample_id}")
            m = SttMeasurement(server="127.0.0.1:1", mode=mode, sample_id=audio.sample_id)
            m.error = "unavailable: ConnectionRefusedError: [Errno 111] Connection refused"
            m.connection_failed = True
            return m

        out = io.StringIO()
        with patch("wyoming_bench.stt.bench.measure_stt_once", side_effect=fake_measure_stt_once):
            with contextlib.redirect_stdout(out):
                ms = await bench_stt_server(
                    "127.0.0.1",
                    1,
                    [_make_audio()],
                    [MODE_NON_STREAMING, MODE_STREAMING],
                    Transcribe(),
                    3,
                    1,
                    False,
                    1024,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    None,
                )
        # The first warmup already proved the server is down: the streaming
        # probe and all rounds must not be attempted.
        self.assertEqual(calls, ["non_streaming:sample"])
        self.assertEqual(ms, [])
        self.assertIn("unreachable", out.getvalue())


if __name__ == "__main__":
    unittest.main()
