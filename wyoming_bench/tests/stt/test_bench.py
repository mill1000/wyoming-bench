"""Tests for wyoming_bench.stt.bench_stt_server streaming-skip via advertised info.

A dead port (127.0.0.1:1) is used throughout: it proves the "not advertised"
path makes no connection at all, and it makes the behavioral probe fail fast
when the flag says streaming is supported (or unknown).
"""

from __future__ import annotations

import contextlib
import io
import unittest

from wyoming.asr import Transcribe

from wyoming_bench.const import MODE_STREAMING
from wyoming_bench.stt.bench import bench_stt_server
from wyoming_bench.tests.stt.test_client import _make_audio
from wyoming_bench.tests.test_info import make_info


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


if __name__ == "__main__":
    unittest.main()
