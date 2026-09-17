"""Tests for wyoming_bench.tts.bench_server streaming-skip via advertised info.

A dead port (127.0.0.1:1) is used throughout: it proves the "not advertised"
path makes no connection at all, and it makes the behavioral probe fail fast
when the flag says streaming is supported (or unknown).
"""

from __future__ import annotations

import contextlib
import io
import unittest
from unittest.mock import patch

from wyoming_bench.const import MODE_NON_STREAMING, MODE_STREAMING
from wyoming_bench.tests.test_info import make_info, make_info_two_tts
from wyoming_bench.tts.bench import bench_server
from wyoming_bench.tts.reporting import Measurement


async def _run(info):
    """Run one streaming bench against a dead port; return ``(ms, output)``."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ms = await bench_server(
            "127.0.0.1",
            1,
            ["hello"],
            [MODE_STREAMING],
            None,
            None,
            3,
            0,
            False,
            0.0,
            1.0,
            1.0,
            False,
            1.0,
            info,
        )
    return ms, out.getvalue()


class TestBenchServerStreamingSkip(unittest.IsolatedAsyncioTestCase):
    async def test_skips_when_not_advertised(self):
        # make_info() defaults its tts program to supports_synthesize_streaming=False,
        # so streaming must be skipped immediately without any connection attempt.
        ms, out = await _run(make_info())
        self.assertEqual(ms, [])
        self.assertIn("not advertised by server", out)

    async def test_probes_when_advertised(self):
        # When the server advertises streaming, the behavioral probe still runs;
        # against a dead port it fails fast with no audio.
        info = make_info()
        info.tts[0].supports_synthesize_streaming = True
        ms, out = await _run(info)
        self.assertEqual(ms, [])
        self.assertIn("no audio in response", out)

    async def test_probes_when_info_unknown(self):
        # info=None (server unreachable / no describe support) must NOT skip;
        # the probe runs and fails fast against the dead port.
        ms, out = await _run(None)
        self.assertEqual(ms, [])
        self.assertIn("no audio in response", out)


class TestBenchServerProgram(unittest.IsolatedAsyncioTestCase):
    async def test_program_forwarded_to_measure_once(self):
        calls: list[str | None] = []

        async def fake_measure_once(*args, **kwargs):
            calls.append(kwargs.get("program", args[9] if len(args) > 9 else None))
            return Measurement(server="127.0.0.1:1", mode=args[2], text=args[3])

        with patch("wyoming_bench.tts.bench.measure_once", side_effect=fake_measure_once):
            with contextlib.redirect_stdout(io.StringIO()):
                await bench_server(
                    "127.0.0.1",
                    1,
                    ["a"],
                    [MODE_NON_STREAMING],
                    None,
                    None,
                    1,
                    0,
                    False,
                    0.0,
                    1.0,
                    1.0,
                    False,
                    1.0,
                    None,
                    program="piper",
                )
        self.assertTrue(calls)
        self.assertTrue(all(p == "piper" for p in calls))

    async def test_streaming_skip_uses_selected_program_flag(self):
        # tts-prog (the first program) advertises no streaming, but kokoro
        # does: the skip decision must follow the selected program, not the
        # first advertised one.
        info = make_info_two_tts()
        calls: list[str | None] = []

        async def fake_measure_once(*args, **kwargs):
            calls.append(kwargs.get("program", args[9] if len(args) > 9 else None))
            return Measurement(server="127.0.0.1:1", mode=args[2], text=args[3])

        kwargs = dict(
            host="127.0.0.1",
            port=1,
            texts=["a"],
            modes=[MODE_STREAMING],
            voice=None,
            text_format=None,
            rounds=1,
            warmup=0,
            verbose=False,
            chunk_delay=0.0,
            connect_timeout=1.0,
            read_timeout=1.0,
            unique=False,
            probe_timeout=1.0,
            info=info,
        )
        with patch("wyoming_bench.tts.bench.measure_once", side_effect=fake_measure_once):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                ms = await bench_server(**kwargs, program="kokoro")
        # kokoro advertises streaming, so the probe runs (and fails against the
        # fake: no audio) instead of skipping as "not advertised".
        self.assertEqual(ms, [])
        self.assertEqual(calls, ["kokoro"])
        self.assertNotIn("not advertised by server", out.getvalue())
        self.assertIn("no audio in response", out.getvalue())

        calls.clear()
        with patch("wyoming_bench.tts.bench.measure_once", side_effect=fake_measure_once):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                ms = await bench_server(**kwargs, program="tts-prog")
        # tts-prog advertises no streaming: skipped immediately, no probe.
        self.assertEqual(ms, [])
        self.assertEqual(calls, [])
        self.assertIn("not advertised by server", out.getvalue())


class TestBenchServerUnreachable(unittest.IsolatedAsyncioTestCase):
    async def test_skips_remaining_runs_after_connection_failure(self):
        calls: list[str] = []

        async def fake_measure_once(*args):
            mode, text = args[2], args[3]
            calls.append(f"{mode}:{text}")
            m = Measurement(server="127.0.0.1:1", mode=mode, text=text)
            m.error = "unavailable: ConnectionRefusedError: [Errno 111] Connection refused"
            m.connection_failed = True
            return m

        out = io.StringIO()
        with patch("wyoming_bench.tts.bench.measure_once", side_effect=fake_measure_once):
            with contextlib.redirect_stdout(out):
                ms = await bench_server(
                    "127.0.0.1",
                    1,
                    ["a", "b", "c"],
                    [MODE_NON_STREAMING, MODE_STREAMING],
                    None,
                    None,
                    3,
                    1,
                    False,
                    0.0,
                    1.0,
                    1.0,
                    False,
                    1.0,
                    None,
                )
        # The first warmup already proved the server is down: the streaming
        # probe and all rounds must not be attempted.
        self.assertEqual(calls, ["non_streaming:a"])
        self.assertEqual(ms, [])
        self.assertIn("unreachable", out.getvalue())


if __name__ == "__main__":
    unittest.main()
