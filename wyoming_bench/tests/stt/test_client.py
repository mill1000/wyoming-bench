"""Async tests for wyoming_tts_bench.stt.client against a mock STT server."""

from __future__ import annotations

import asyncio
import unittest
from pathlib import Path

from wyoming.asr import (
    Transcribe,
    Transcript,
    TranscriptChunk,
    TranscriptStart,
    TranscriptStop,
)
from wyoming.audio import AudioChunk, AudioStop
from wyoming.error import Error
from wyoming.event import async_read_event, async_write_event

from wyoming_bench.const import MODE_NON_STREAMING, MODE_STREAMING
from wyoming_bench.stt.client import measure_stt_once
from wyoming_bench.stt.corpus import AudioInput

REFERENCE = "hello world"


def _make_audio(reference: str = REFERENCE, frames: int = 1600) -> AudioInput:
    rate, width, channels = 16000, 2, 1
    pcm = b"\x00\x01" * (frames * channels)
    return AudioInput(
        sample_id="sample",
        reference=reference,
        rate=rate,
        width=width,
        channels=channels,
        pcm=pcm,
        audio_path=Path("sample.wav"),
    )


async def _mock_stt_handler(reader, writer, reply):
    try:
        await async_read_event(reader)  # transcribe
        await async_read_event(reader)  # audio-start
        while True:  # audio-chunk* then audio-stop
            ev = await async_read_event(reader)
            if ev is None or AudioStop.is_type(ev.type):
                break
        for event in reply:
            await async_write_event(event, writer)
    except (ConnectionResetError, asyncio.IncompleteReadError):
        pass
    finally:
        try:
            writer.close()
        except Exception:  # noqa: BLE001
            pass
        await writer.wait_closed()


class TestMeasureSttOnce(unittest.IsolatedAsyncioTestCase):
    async def _run(self, reply, mode):
        server = await asyncio.start_server(lambda r, w: _mock_stt_handler(r, w, reply), "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            return await measure_stt_once("127.0.0.1", port, mode, _make_audio(), Transcribe(), 512, 5.0, 5.0)
        finally:
            server.close()
            await server.wait_closed()

    async def test_non_streaming_exact(self):
        m = await self._run([Transcript(text=REFERENCE).event()], MODE_NON_STREAMING)
        self.assertTrue(m.ok, m.error)
        self.assertEqual(m.hypothesis, REFERENCE)
        self.assertEqual(m.reference, REFERENCE)
        self.assertEqual(m.wer, 0.0)
        self.assertTrue(m.exact)
        self.assertIsNone(m.t_first)  # non-streaming has no first-chunk timing

    async def test_streaming_exact(self):
        reply = [
            TranscriptStart().event(),
            TranscriptChunk(text="hello").event(),
            TranscriptChunk(text=" world").event(),
            Transcript(text=REFERENCE).event(),
            TranscriptStop().event(),
        ]
        m = await self._run(reply, MODE_STREAMING)
        self.assertTrue(m.ok, m.error)
        self.assertEqual(m.hypothesis, REFERENCE)
        self.assertGreaterEqual(m.n_transcript_chunks, 1)
        self.assertIsNotNone(m.t_first)
        self.assertEqual(m.wer, 0.0)
        self.assertTrue(m.exact)

    async def test_accuracy_reflects_mismatch(self):
        m = await self._run([Transcript(text="goodbye moon").event()], MODE_NON_STREAMING)
        self.assertTrue(m.ok, m.error)
        self.assertEqual(m.hypothesis, "goodbye moon")
        self.assertGreater(m.wer, 0.0)
        self.assertFalse(m.exact)

    async def test_server_error(self):
        m = await self._run([Error(text="boom").event()], MODE_NON_STREAMING)
        self.assertFalse(m.ok)
        self.assertIn("boom", m.error)

    async def test_trailing_silence_appended(self):
        # With trailing_silence > 0, the PCM sent on the wire is longer than the
        # input by exactly the silence length (silence_samples * width * channels).
        captured: dict[str, int] = {"bytes": 0}

        async def handler(reader, writer):
            try:
                await async_read_event(reader)  # transcribe
                await async_read_event(reader)  # audio-start
                while True:  # audio-chunk* then audio-stop
                    ev = await async_read_event(reader)
                    if ev is None or AudioStop.is_type(ev.type):
                        break
                    captured["bytes"] += len(AudioChunk.from_event(ev).audio)
                await async_write_event(Transcript(text=REFERENCE).event(), writer)
            except (ConnectionResetError, asyncio.IncompleteReadError):
                pass
            finally:
                try:
                    writer.close()
                except Exception:  # noqa: BLE001
                    pass
                await writer.wait_closed()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        audio = _make_audio()
        try:
            await measure_stt_once(
                "127.0.0.1", port, MODE_NON_STREAMING, audio, Transcribe(), 512, 5.0, 5.0, 0.0, 0.5
            )
        finally:
            server.close()
            await server.wait_closed()

        expected_silence = int(round(0.5 * audio.rate * audio.width * audio.channels))
        self.assertEqual(captured["bytes"], len(audio.pcm) + expected_silence)


if __name__ == "__main__":
    unittest.main()
