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
from wyoming_bench.stt.client import (
    TranscriptionError,
    measure_stt_once,
    transcribe_audio,
)
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


async def _mock_stt_handler_recording(reader, writer, events: list[str]):
    """Transcribe with fixed text, recording the event types received."""
    try:
        while True:
            event = await async_read_event(reader)
            if event is None:
                break
            events.append(event.type)
            if event.type == "transcribe":
                await async_read_event(reader)  # audio-start
                while True:  # audio-chunk* then audio-stop
                    ev = await async_read_event(reader)
                    if ev is None or AudioStop.is_type(ev.type):
                        break
                await async_write_event(Transcript(text=REFERENCE).event(), writer)
                break
    except (ConnectionResetError, asyncio.IncompleteReadError):
        pass
    finally:
        try:
            writer.close()
        except Exception:  # noqa: BLE001
            pass
        await writer.wait_closed()


async def _mock_stt_handler_requiring_program(reader, writer, events: list[str], expected: str):
    """Transcribe only if select-program picked *expected*; error otherwise."""
    selected: str | None = None
    try:
        while True:
            event = await async_read_event(reader)
            if event is None:
                break
            events.append(event.type)
            if event.type == "select-program":
                selected = event.data["name"]
            elif event.type == "transcribe":
                await async_read_event(reader)  # audio-start
                while True:  # audio-chunk* then audio-stop
                    ev = await async_read_event(reader)
                    if ev is None or AudioStop.is_type(ev.type):
                        break
                if selected == expected:
                    await async_write_event(Transcript(text=REFERENCE).event(), writer)
                else:
                    await async_write_event(
                        Error(code="asr_unknown_program", text=f"program {selected!r} not found").event(),
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


class TestMeasureSttOnce(unittest.IsolatedAsyncioTestCase):
    async def test_unreachable_sets_connection_failed(self):
        # Nothing listens on 127.0.0.1:1: the measurement must fail fast and
        # flag the connection failure so callers skip the rest of the batch.
        m = await measure_stt_once(
            "127.0.0.1", 1, MODE_NON_STREAMING, _make_audio(), Transcribe(), 512, 1.0, 1.0
        )
        self.assertFalse(m.ok)
        self.assertTrue(m.connection_failed)
        self.assertIn("unavailable", m.error)

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


class TestMeasureSttOnceProgram(unittest.IsolatedAsyncioTestCase):
    """select-program is sent before transcribe, and the server sees it."""

    async def _with_server(self, handler, *handler_args):
        server = await asyncio.start_server(lambda r, w: handler(r, w, *handler_args), "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            yield port
        finally:
            server.close()
            await server.wait_closed()

    async def _run(self, port, program=None):
        return await measure_stt_once(
            "127.0.0.1",
            port,
            MODE_NON_STREAMING,
            _make_audio(),
            Transcribe(),
            512,
            5.0,
            5.0,
            0.0,
            0.0,
            program,
        )

    async def test_program_sent_before_transcribe(self):
        events: list[str] = []
        async for port in self._with_server(_mock_stt_handler_recording, events):
            m = await self._run(port, "asr-prog")
        self.assertTrue(m.ok, m.error)
        self.assertEqual(events, ["select-program", "transcribe"])

    async def test_no_program_sends_no_select_event(self):
        events: list[str] = []
        async for port in self._with_server(_mock_stt_handler_recording, events):
            m = await self._run(port)
        self.assertTrue(m.ok, m.error)
        self.assertEqual(events, ["transcribe"])

    async def test_server_selects_the_requested_program(self):
        events: list[str] = []
        async for port in self._with_server(_mock_stt_handler_requiring_program, events, "asr-prog"):
            m = await self._run(port, "asr-prog")
        self.assertTrue(m.ok, m.error)
        self.assertEqual(events[0], "select-program")

    async def test_wrong_program_is_rejected_by_server(self):
        events: list[str] = []
        async for port in self._with_server(_mock_stt_handler_requiring_program, events, "asr-prog"):
            m = await self._run(port, "other")
        self.assertFalse(m.ok)
        self.assertIn("asr_unknown_program", m.error)


class TestTranscribeAudio(unittest.IsolatedAsyncioTestCase):
    """transcribe_audio (used by seed) returns the final text or raises."""

    async def _run(self, reply):
        server = await asyncio.start_server(lambda r, w: _mock_stt_handler(r, w, reply), "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            return await transcribe_audio("127.0.0.1", port, _make_audio(), Transcribe(), 512, 5.0, 5.0)
        finally:
            server.close()
            await server.wait_closed()

    async def test_returns_final_text(self):
        self.assertEqual(await self._run([Transcript(text=REFERENCE).event()]), REFERENCE)

    async def test_server_error_raises(self):
        with self.assertRaises(TranscriptionError) as ctx:
            await self._run([Error(text="boom").event()])
        self.assertIn("boom", str(ctx.exception))

    async def test_empty_transcript_raises(self):
        with self.assertRaises(TranscriptionError) as ctx:
            await self._run([Transcript(text="").event()])
        self.assertIn("empty", str(ctx.exception))

    async def test_unreachable_raises(self):
        # Nothing listens on 127.0.0.1:1: the connection failure must propagate
        # so the seed runner can report a per-sample failure.
        with self.assertRaises(OSError):
            await transcribe_audio("127.0.0.1", 1, _make_audio(), Transcribe(), 512, 1.0, 1.0)

    async def test_program_sent_before_transcribe(self):
        events: list[str] = []
        server = await asyncio.start_server(
            lambda r, w: _mock_stt_handler_recording(r, w, events), "127.0.0.1", 0
        )
        port = server.sockets[0].getsockname()[1]
        try:
            text = await transcribe_audio(
                "127.0.0.1", port, _make_audio(), Transcribe(), 512, 5.0, 5.0, "asr-prog"
            )
        finally:
            server.close()
            await server.wait_closed()
        self.assertEqual(text, REFERENCE)
        self.assertEqual(events, ["select-program", "transcribe"])

    async def test_program_reached_by_program_aware_server(self):
        events: list[str] = []
        server = await asyncio.start_server(
            lambda r, w: _mock_stt_handler_requiring_program(r, w, events, "asr-prog"), "127.0.0.1", 0
        )
        port = server.sockets[0].getsockname()[1]
        try:
            text = await transcribe_audio(
                "127.0.0.1", port, _make_audio(), Transcribe(), 512, 5.0, 5.0, "asr-prog"
            )
        finally:
            server.close()
            await server.wait_closed()
        self.assertEqual(text, REFERENCE)
        self.assertEqual(events[0], "select-program")


if __name__ == "__main__":
    unittest.main()
