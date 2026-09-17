"""Async tests for wyoming_tts_bench.tts.client.synthesize_audio against a mock TTS."""

from __future__ import annotations

import asyncio
import unittest

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import async_read_event, async_write_event
from wyoming.tts import SynthesizeVoice

from wyoming_bench.const import MODE_NON_STREAMING
from wyoming_bench.tts.client import measure_once, synthesize_audio

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


async def _send_audio(writer) -> None:
    await async_write_event(AudioStart(rate=16000, width=2, channels=1, timestamp=0).event(), writer)
    await async_write_event(
        AudioChunk(rate=16000, width=2, channels=1, audio=PCM, timestamp=None).event(), writer
    )
    await async_write_event(AudioStop(timestamp=None).event(), writer)


async def _mock_tts_handler_recording(reader, writer, events: list[str]):
    """Answer synthesize with fixed audio, recording the event types received."""
    try:
        while True:
            event = await async_read_event(reader)
            if event is None:
                break
            events.append(event.type)
            if event.type == "synthesize":
                await _send_audio(writer)
                break
    except (ConnectionResetError, asyncio.IncompleteReadError):
        pass
    finally:
        try:
            writer.close()
        except Exception:  # noqa: BLE001
            pass
        await writer.wait_closed()


async def _mock_tts_handler_requiring_program(reader, writer, events: list[str], expected: str):
    """Synthesize only if select-program picked *expected*; error otherwise."""
    selected: str | None = None
    try:
        while True:
            event = await async_read_event(reader)
            if event is None:
                break
            events.append(event.type)
            if event.type == "select-program":
                selected = event.data["name"]
            elif event.type == "synthesize":
                if selected == expected:
                    await _send_audio(writer)
                else:
                    await async_write_event(
                        Error(code="tts_unknown_program", text=f"program {selected!r} not found").event(),
                        writer,
                    )
                break
    except (ConnectionResetError, asyncio.IncompleteReadError):
        pass
    finally:
        try:
            writer.close()
        except Exception:  # noqa: BLE001
            pass
        await writer.wait_closed()


class TestMeasureOnce(unittest.IsolatedAsyncioTestCase):
    async def test_unreachable_sets_connection_failed(self):
        # Nothing listens on 127.0.0.1:1: the measurement must fail fast and
        # flag the connection failure so callers skip the rest of the batch.
        m = await measure_once("127.0.0.1", 1, MODE_NON_STREAMING, "hello", None, None, 1.0, 1.0)
        self.assertFalse(m.ok)
        self.assertTrue(m.connection_failed)
        self.assertIn("unavailable", m.error)
        self.assertIsNone(m.ttft_s)


class TestMeasureOnceProgram(unittest.IsolatedAsyncioTestCase):
    """select-program is sent before synthesis, and the server sees it."""

    async def _with_server(self, handler, *handler_args):
        server = await asyncio.start_server(lambda r, w: handler(r, w, *handler_args), "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            yield port
        finally:
            server.close()
            await server.wait_closed()

    async def test_program_sent_before_synthesize(self):
        events: list[str] = []
        async for port in self._with_server(_mock_tts_handler_recording, events):
            m = await measure_once(
                "127.0.0.1", port, MODE_NON_STREAMING, "hi", None, None, 5.0, 5.0, 0.0, "piper"
            )
        self.assertTrue(m.ok, m.error)
        self.assertEqual(events, ["select-program", "synthesize"])

    async def test_no_program_sends_no_select_event(self):
        events: list[str] = []
        async for port in self._with_server(_mock_tts_handler_recording, events):
            m = await measure_once("127.0.0.1", port, MODE_NON_STREAMING, "hi", None, None, 5.0, 5.0)
        self.assertTrue(m.ok, m.error)
        self.assertEqual(events, ["synthesize"])

    async def test_server_selects_the_requested_program(self):
        events: list[str] = []
        async for port in self._with_server(_mock_tts_handler_requiring_program, events, "piper"):
            ok = await measure_once(
                "127.0.0.1", port, MODE_NON_STREAMING, "hi", None, None, 5.0, 5.0, 0.0, "piper"
            )
        self.assertTrue(ok.ok, ok.error)
        self.assertEqual(events[0], "select-program")

    async def test_wrong_program_is_rejected_by_server(self):
        events: list[str] = []
        async for port in self._with_server(_mock_tts_handler_requiring_program, events, "piper"):
            bad = await measure_once(
                "127.0.0.1", port, MODE_NON_STREAMING, "hi", None, None, 5.0, 5.0, 0.0, "kokoro"
            )
        self.assertFalse(bad.ok)
        self.assertIn("tts_unknown_program", bad.error)


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

    async def test_program_sent_before_synthesize(self):
        events: list[str] = []
        server = await asyncio.start_server(
            lambda r, w: _mock_tts_handler_recording(r, w, events), "127.0.0.1", 0
        )
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
                "piper",
            )
        finally:
            server.close()
            await server.wait_closed()
        self.assertEqual(pcm, PCM)
        self.assertEqual(events, ["select-program", "synthesize"])

    async def test_program_reached_by_program_aware_server(self):
        events: list[str] = []
        server = await asyncio.start_server(
            lambda r, w: _mock_tts_handler_requiring_program(r, w, events, "piper"), "127.0.0.1", 0
        )
        port = server.sockets[0].getsockname()[1]
        try:
            pcm, _, _, _ = await synthesize_audio(
                "127.0.0.1", port, MODE_NON_STREAMING, "hello", None, None, 5.0, 5.0, "piper"
            )
        finally:
            server.close()
            await server.wait_closed()
        self.assertEqual(pcm, PCM)


if __name__ == "__main__":
    unittest.main()
