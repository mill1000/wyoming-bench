"""Tests for wyoming_bench.tts.bench_server streaming-skip via advertised info.

A dead port (127.0.0.1:1) is used throughout: it proves the "not advertised"
path makes no connection at all, and it makes the behavioral probe fail fast
when the flag says streaming is supported (or unknown).
"""

from __future__ import annotations

import contextlib
import io
import unittest

from wyoming_bench.const import MODE_STREAMING
from wyoming_bench.tests.test_info import make_info
from wyoming_bench.tts.bench import bench_server


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


if __name__ == "__main__":
    unittest.main()
