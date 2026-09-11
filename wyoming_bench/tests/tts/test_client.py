"""Async tests for wyoming_tts_bench.tts.client.synthesize_audio against a mock TTS."""

from __future__ import annotations

import asyncio
import unittest

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import async_read_event, async_write_event
from wyoming.tts import SynthesizeVoice

from wyoming_bench.const import MODE_NON_STREAMING
from wyoming_bench.tts.client import synthesize_audio

PCM = b"\x00\x01" * 512  # 512 bytes = 256 int16 mono samples


async def _mock_tts_handler(reader, writer):
    try:
        await async_read_event(reader)  # synthesize
        await async_write_event(AudioStart(rate=16000, width=2, channels=1, timestamp=0).event(), writer)
        await async_write_event(
            AudioChunk(rate=16000, width=2, channels=1, audio=PCM, timestamp=None).event(), writer
        )
        await async_write_event(AudioStop(timestamp=None).event(), writer)
    except (ConnectionResetError, asyncio.IncompleteReadError):
        pass
    finally:
        try:
            writer.close()
        except Exception:  # noqa: BLE001
            pass
        await writer.wait_closed()


class TestSynthesizeAudio(unittest.IsolatedAsyncioTestCase):
    async def test_non_streaming(self):
        server = await asyncio.start_server(_mock_tts_handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            pcm, rate, width, channels = await synthesize_audio(
                "127.0.0.1",
                port,
                MODE_NON_STREAMING,
                "hello world",
                SynthesizeVoice(name="alice"),
                None,
                5.0,
                5.0,
            )
        finally:
            server.close()
            await server.wait_closed()
        self.assertEqual(pcm, PCM)
        self.assertEqual((rate, width, channels), (16000, 2, 1))


if __name__ == "__main__":
    unittest.main()
