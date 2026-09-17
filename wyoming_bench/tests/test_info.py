"""Tests for wyoming_bench.info (describe/info service discovery)."""

from __future__ import annotations

import asyncio
import unittest

from wyoming.event import async_read_event, async_write_event
from wyoming.info import (
    AsrModel,
    AsrProgram,
    Attribution,
    Describe,
    Info,
    TtsProgram,
    TtsVoice,
)

from wyoming_bench.info import (
    advertised_streaming,
    available_services,
    describe_services,
    fetch_info,
    program_name,
    server_label,
)

_ATTR = Attribution(name="acme", url="https://example.com")


def make_info(asr: bool = True, tts: bool = True) -> Info:
    """Build an Info with one ASR program and/or one TTS program."""
    asr_programs = []
    if asr:
        asr_programs.append(
            AsrProgram(
                name="asr-prog",
                attribution=_ATTR,
                installed=True,
                description="ASR",
                version="1.0",
                models=[
                    AsrModel(
                        name="base",
                        attribution=_ATTR,
                        installed=True,
                        description=None,
                        version=None,
                        languages=["en"],
                    )
                ],
                supports_transcript_streaming=True,
            )
        )
    tts_programs = []
    if tts:
        tts_programs.append(
            TtsProgram(
                name="tts-prog",
                attribution=_ATTR,
                installed=True,
                description="TTS",
                version="1.0",
                voices=[
                    TtsVoice(
                        name="alice",
                        attribution=_ATTR,
                        installed=True,
                        description=None,
                        version=None,
                        languages=["en"],
                    )
                ],
                supports_synthesize_streaming=False,
            )
        )
    return Info(asr=asr_programs, tts=tts_programs)


def make_info_two_tts() -> Info:
    """Info with two TTS programs that differ in streaming support."""
    info = make_info()  # first program: tts-prog, no streaming
    info.tts.append(
        TtsProgram(
            name="kokoro",
            attribution=_ATTR,
            installed=True,
            description="TTS",
            version="1.0",
            voices=[],
            supports_synthesize_streaming=True,
        )
    )
    return info


def make_info_two_asr() -> Info:
    """Info with two ASR programs that differ in streaming support."""
    info = make_info()
    info.asr[0].supports_transcript_streaming = False  # first program: asr-prog, no streaming
    info.asr.append(
        AsrProgram(
            name="whisper",
            attribution=_ATTR,
            installed=True,
            description="ASR",
            version="1.0",
            models=[],
            supports_transcript_streaming=True,
        )
    )
    return info


async def _info_handler(reader, writer, info: Info):
    """Answer ``describe`` with *info*; stay connected until the client leaves."""
    try:
        while True:
            event = await async_read_event(reader)
            if event is None:
                break
            if Describe.is_type(event.type):
                await async_write_event(info.event(), writer)
    except (ConnectionResetError, asyncio.IncompleteReadError, asyncio.CancelledError):
        pass
    finally:
        try:
            writer.close()
        except Exception:  # noqa: BLE001
            pass
        await writer.wait_closed()


async def _silent_handler(reader, writer):
    """Hold the connection without ever answering (simulate an unresponsive server)."""
    try:
        await asyncio.sleep(30)
    finally:
        try:
            writer.close()
        except Exception:  # noqa: BLE001
            pass
        await writer.wait_closed()


async def _close_handler(reader, writer):
    """Drop the connection immediately (simulate a non-wyoming listener)."""
    try:
        writer.close()
    finally:
        await writer.wait_closed()


class TestFetchInfo(unittest.IsolatedAsyncioTestCase):
    async def _with_server(self, handler, *handler_args):
        server = await asyncio.start_server(lambda r, w: handler(r, w, *handler_args), "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            yield port
        finally:
            server.close()
            # close_clients() only exists on 3.12+; on earlier versions rely on the
            # client-side disconnect to let the handler close its own connection.
            if hasattr(server, "close_clients"):
                server.close_clients()  # don't wait on handlers that never close the connection
            await server.wait_closed()

    async def test_returns_advertised_services(self):
        async for port in self._with_server(_info_handler, make_info()):
            info = await fetch_info("127.0.0.1", port, 5.0)
        self.assertIsNotNone(info)
        self.assertEqual(available_services(info), ["asr", "tts"])
        self.assertEqual(info.asr[0].name, "asr-prog")
        self.assertEqual(info.tts[0].name, "tts-prog")

    async def test_asr_only(self):
        async for port in self._with_server(_info_handler, make_info(tts=False)):
            info = await fetch_info("127.0.0.1", port, 5.0)
        self.assertIsNotNone(info)
        self.assertEqual(available_services(info), ["asr"])

    async def test_silent_server_returns_none(self):
        async for port in self._with_server(_silent_handler):
            info = await fetch_info("127.0.0.1", port, 0.5)
        self.assertIsNone(info)

    async def test_closed_connection_returns_none(self):
        async for port in self._with_server(_close_handler):
            info = await fetch_info("127.0.0.1", port, 5.0)
        self.assertIsNone(info)

    async def test_unreachable_raises(self):
        # Nothing listens on this port: the connection failure must propagate
        # (server unavailable), not be reported as "unknown".
        with self.assertRaises(ConnectionRefusedError):
            await fetch_info("127.0.0.1", 1, 2.0)


class TestAdvertisedStreaming(unittest.TestCase):
    def test_tts_default_no_streaming(self):
        self.assertIs(advertised_streaming(make_info(), "tts"), False)

    def test_tts_streaming_when_set(self):
        info = make_info()
        info.tts[0].supports_synthesize_streaming = True
        self.assertIs(advertised_streaming(info, "tts"), True)

    def test_asr_default_streaming(self):
        self.assertIs(advertised_streaming(make_info(), "asr"), True)

    def test_asr_no_streaming_when_unset(self):
        info = make_info()
        info.asr[0].supports_transcript_streaming = False
        self.assertIs(advertised_streaming(info, "asr"), False)

    def test_no_programs_returns_none(self):
        self.assertIsNone(advertised_streaming(Info(), "tts"))
        self.assertIsNone(advertised_streaming(Info(), "asr"))

    def test_service_absent_returns_none(self):
        self.assertIsNone(advertised_streaming(make_info(asr=False), "asr"))
        self.assertIsNone(advertised_streaming(make_info(tts=False), "tts"))

    def test_none_info_returns_none(self):
        self.assertIsNone(advertised_streaming(None, "tts"))
        self.assertIsNone(advertised_streaming(None, "asr"))

    def test_named_program_reads_its_own_flag(self):
        info = make_info_two_tts()
        self.assertIs(advertised_streaming(info, "tts", "tts-prog"), False)
        self.assertIs(advertised_streaming(info, "tts", "kokoro"), True)

    def test_asr_named_program_reads_its_own_flag(self):
        info = make_info_two_asr()
        self.assertIs(advertised_streaming(info, "asr", "asr-prog"), False)
        self.assertIs(advertised_streaming(info, "asr", "whisper"), True)

    def test_named_program_match_is_case_insensitive(self):
        info = make_info_two_tts()
        self.assertIs(advertised_streaming(info, "tts", "Kokoro"), True)
        self.assertIs(advertised_streaming(info, "tts", "TTS-PROG"), False)

    def test_unknown_program_name_returns_none(self):
        self.assertIsNone(advertised_streaming(make_info_two_tts(), "tts", "nope"))
        self.assertIsNone(advertised_streaming(make_info_two_asr(), "asr", "nope"))


class TestProgramName(unittest.TestCase):
    def test_returns_first_program_name(self):
        self.assertEqual(program_name(make_info(), "tts"), "tts-prog")
        self.assertEqual(program_name(make_info(), "asr"), "asr-prog")

    def test_service_absent_returns_none(self):
        self.assertIsNone(program_name(make_info(asr=False), "asr"))
        self.assertIsNone(program_name(make_info(tts=False), "tts"))

    def test_empty_programs_returns_none(self):
        self.assertIsNone(program_name(Info(), "tts"))
        self.assertIsNone(program_name(Info(), "asr"))

    def test_none_info_returns_none(self):
        self.assertIsNone(program_name(None, "tts"))
        self.assertIsNone(program_name(None, "asr"))


class TestServerLabel(unittest.TestCase):
    def test_label_includes_program_name(self):
        self.assertEqual(server_label("10.0.0.1", 10700, make_info(), "tts"), "10.0.0.1:10700 (tts-prog)")
        self.assertEqual(server_label("10.0.0.1", 10700, make_info(), "asr"), "10.0.0.1:10700 (asr-prog)")

    def test_label_without_info_is_bare(self):
        self.assertEqual(server_label("10.0.0.1", 10700, None, "tts"), "10.0.0.1:10700")

    def test_label_without_program_is_bare(self):
        self.assertEqual(server_label("10.0.0.1", 10700, make_info(tts=False), "tts"), "10.0.0.1:10700")

    def test_label_uses_explicit_program(self):
        self.assertEqual(
            server_label("10.0.0.1", 10700, make_info_two_tts(), "tts", "kokoro"),
            "10.0.0.1:10700 (kokoro)",
        )


class TestDescribeServices(unittest.TestCase):
    def test_both_services(self):
        lines = describe_services(make_info())
        self.assertIn("  asr  asr-prog (transcript streaming)", lines)
        self.assertIn("       model base [en]", lines)
        self.assertIn("  tts  tts-prog (no streaming)", lines)
        self.assertIn("       voice alice [en]", lines)

    def test_no_services(self):
        lines = describe_services(Info())
        self.assertIn("  asr  (none)", lines)
        self.assertIn("  tts  (none)", lines)
        self.assertEqual(len(lines), 2)


if __name__ == "__main__":
    unittest.main()
