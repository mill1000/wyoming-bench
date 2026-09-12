"""Tests for the parsing helpers in wyoming_bench.cli."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from wyoming.asr import Transcribe
from wyoming.tts import SynthesizeVoice

from wyoming_bench import cli
from wyoming_bench.cli import (
    DEFAULT_PORT,
    build_parser,
    load_texts,
    parse_server,
    parse_stt_config,
    parse_tts_config,
)
from wyoming_bench.const import MODE_NON_STREAMING
from wyoming_bench.stt.corpus import AudioInput
from wyoming_bench.tests.test_info import _info_handler, make_info


def _fake_sample() -> AudioInput:
    return AudioInput(
        sample_id="s1",
        reference="hello world",
        rate=16000,
        width=2,
        channels=1,
        pcm=b"\x00\x01" * 100,
        audio_path=Path("s1.wav"),
    )


class TestParseServer(unittest.TestCase):
    def test_host_only(self):
        self.assertEqual(parse_server("localhost"), ("localhost", DEFAULT_PORT))

    def test_host_port(self):
        self.assertEqual(parse_server("localhost:8000"), ("localhost", 8000))

    def test_missing_host(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_server(":8000")

    def test_bad_port(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_server("host:notaport")

    def test_port_zero(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_server("host:0")

    def test_port_too_large(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_server("host:70000")

    def test_port_bounds_ok(self):
        self.assertEqual(parse_server("host:1"), ("host", 1))
        self.assertEqual(parse_server("host:65535"), ("host", 65535))


class TestParseTtsConfig(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(parse_tts_config("{}"), (None, None))

    def test_voice_string(self):
        voice, fmt = parse_tts_config('{"voice": "alice"}')
        self.assertIsInstance(voice, SynthesizeVoice)
        self.assertEqual(voice.name, "alice")
        self.assertIsNone(fmt)

    def test_voice_object_and_format(self):
        voice, fmt = parse_tts_config('{"voice": {"name": "bob", "language": "en"}, "text_format": "ssml"}')
        self.assertEqual(voice.name, "bob")
        self.assertEqual(voice.language, "en")
        self.assertEqual(fmt, "ssml")

    def test_non_object(self):
        with self.assertRaises(SystemExit):
            parse_tts_config('"just a string"')

    def test_list(self):
        with self.assertRaises(SystemExit):
            parse_tts_config("[1, 2]")

    def test_voice_wrong_type(self):
        with self.assertRaises(SystemExit):
            parse_tts_config('{"voice": 123}')

    def test_format_wrong_type(self):
        with self.assertRaises(SystemExit):
            parse_tts_config('{"text_format": 123}')


class TestParseSttConfig(unittest.TestCase):
    def test_empty(self):
        cfg = parse_stt_config("{}")
        self.assertIsInstance(cfg, Transcribe)
        self.assertIsNone(cfg.name)

    def test_name_language(self):
        cfg = parse_stt_config('{"name": "model", "language": "en"}')
        self.assertEqual(cfg.name, "model")
        self.assertEqual(cfg.language, "en")

    def test_vad_and_lists(self):
        cfg = parse_stt_config(
            '{"vad_sensitivity": "high", "transcript_names": ["a"], "transcript_terms": ["b"]}'
        )
        self.assertEqual(cfg.vad_sensitivity, "high")
        self.assertEqual(cfg.transcript_names, ["a"])
        self.assertEqual(cfg.transcript_terms, ["b"])

    def test_name_wrong_type(self):
        with self.assertRaises(SystemExit):
            parse_stt_config('{"name": 123}')

    def test_vad_wrong_type(self):
        with self.assertRaises(SystemExit):
            parse_stt_config('{"vad_sensitivity": 3}')

    def test_names_wrong_type(self):
        with self.assertRaises(SystemExit):
            parse_stt_config('{"transcript_names": "not-a-list"}')

    def test_names_nonstring_items(self):
        with self.assertRaises(SystemExit):
            parse_stt_config('{"transcript_names": [1, 2]}')

    def test_non_object(self):
        with self.assertRaises(SystemExit):
            parse_stt_config('"nope"')


class TestLoadTexts(unittest.TestCase):
    def _ns(self, texts=None, texts_file=None, texts_dir=None):
        return argparse.Namespace(texts=texts, texts_file=texts_file, texts_dir=texts_dir)

    def test_from_texts(self):
        self.assertEqual(load_texts(self._ns(texts=["a", "b"])), ["a", "b"])

    def test_from_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("one\n\ntwo\n")
            path = f.name
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))
        self.assertEqual(load_texts(self._ns(texts_file=path)), ["one", "two"])

    def test_file_then_texts(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("one\n")
            path = f.name
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))
        self.assertEqual(load_texts(self._ns(texts=["two"], texts_file=path)), ["one", "two"])

    def test_default(self):
        self.assertEqual(load_texts(self._ns()), list(cli.DEFAULT_TEXTS))

    def test_from_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "b.txt").write_text("beta\n", encoding="utf-8")
            (d / "a.txt").write_text("alpha\n", encoding="utf-8")
            self.assertEqual(load_texts(self._ns(texts_dir=d)), ["alpha", "beta"])

    def test_dir_skips_non_txt_and_blank(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "a.txt").write_text("alpha\n", encoding="utf-8")
            (d / "notes.md").write_text("ignored\n", encoding="utf-8")
            (d / "empty.txt").write_text("   \n", encoding="utf-8")
            self.assertEqual(load_texts(self._ns(texts_dir=d)), ["alpha"])

    def test_dir_missing(self):
        with self.assertRaises(SystemExit):
            load_texts(self._ns(texts_dir="/no/such/dir"))

    def test_dir_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "notes.md").write_text("x\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                load_texts(self._ns(texts_dir=tmp))

    def test_dir_then_file_then_texts(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.txt").write_text("alpha\n", encoding="utf-8")
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
                f.write("one\n")
                path = f.name
            self.addCleanup(lambda: Path(path).unlink(missing_ok=True))
            self.assertEqual(
                load_texts(self._ns(texts=["two"], texts_file=path, texts_dir=tmp)),
                ["alpha", "one", "two"],
            )


class TestBuildParser(unittest.TestCase):
    def test_tts(self):
        args = build_parser().parse_args(["tts", "localhost"])
        self.assertEqual(args.task, "tts")
        self.assertEqual(args.servers, [("localhost", DEFAULT_PORT)])

    def test_tts_multiple_servers(self):
        args = build_parser().parse_args(["tts", "a:10700", "b:10701"])
        self.assertEqual(args.servers, [("a", 10700), ("b", 10701)])

    def test_tts_requires_server(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["tts"])

    def test_tts_texts_dir(self):
        args = build_parser().parse_args(["tts", "localhost", "--texts-dir", "/tmp/d"])
        self.assertEqual(args.texts_dir, "/tmp/d")

    def test_tts_unique_default_on(self):
        args = build_parser().parse_args(["tts", "localhost"])
        self.assertTrue(args.unique)

    def test_tts_unique_off(self):
        args = build_parser().parse_args(["tts", "localhost", "--no-unique"])
        self.assertFalse(args.unique)

    def test_tts_unique_explicit_on(self):
        args = build_parser().parse_args(["tts", "localhost", "--unique"])
        self.assertTrue(args.unique)

    def test_stt_requires_corpus(self):
        args = build_parser().parse_args(["stt", "localhost", "--corpus", "/tmp/c"])
        self.assertEqual(args.task, "stt")
        self.assertEqual(args.corpus, "/tmp/c")

    def test_stt_requires_server(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["stt", "--corpus", "/tmp/c"])

    def test_gen_corpus(self):
        args = build_parser().parse_args(["gen-corpus", "localhost", "--out", "/tmp/o"])
        self.assertEqual(args.task, "gen-corpus")
        self.assertEqual(args.out, "/tmp/o")

    def test_info(self):
        args = build_parser().parse_args(["info", "localhost:1234"])
        self.assertEqual(args.task, "info")
        self.assertEqual(args.servers, [("localhost", 1234)])
        self.assertEqual(args.timeout, 5.0)

    def test_info_requires_server(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["info"])

    def test_missing_task(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args([])

    def test_version_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            build_parser().parse_args(["--version"])
        self.assertEqual(ctx.exception.code, 0)


class TestCollectServers(unittest.TestCase):
    def test_dedupe_preserves_order(self):
        ns = argparse.Namespace(servers=[("a", 1), ("b", 2), ("a", 1)])
        self.assertEqual(cli._collect_servers(ns), [("a", 1), ("b", 2)])

    def test_single(self):
        ns = argparse.Namespace(servers=[("a", 1)])
        self.assertEqual(cli._collect_servers(ns), [("a", 1)])


class TestResolveProbeTimeout(unittest.TestCase):
    def test_default(self):
        self.assertEqual(cli._resolve_probe_timeout(argparse.Namespace(probe_timeout=None), 8.0), 8.0)

    def test_override(self):
        self.assertEqual(cli._resolve_probe_timeout(argparse.Namespace(probe_timeout=5.0), 8.0), 5.0)


class TestPreflight(unittest.IsolatedAsyncioTestCase):
    """The describe preflight must skip servers lacking the requested service."""

    async def _serve(self, info):
        return await asyncio.start_server(lambda r, w: _info_handler(r, w, info), "127.0.0.1", 0)

    async def test_stt_skips_server_without_asr(self):
        server = await self._serve(make_info(asr=False))
        port = server.sockets[0].getsockname()[1]
        out = io.StringIO()
        try:
            args = argparse.Namespace(
                rounds=1,
                warmup=0,
                verbose=False,
                chunk_samples=1024,
                chunk_delay=0.0,
                trailing_silence=0.0,
                timeout=2.0,
            )
            with contextlib.redirect_stdout(out):
                rc = await cli._async_main_stt(
                    args,
                    [("127.0.0.1", port)],
                    [_fake_sample()],
                    [MODE_NON_STREAMING],
                    Transcribe(),
                    2.0,
                )
        finally:
            server.close()
            await server.wait_closed()
        self.assertEqual(rc, 1)
        self.assertIn("[preflight]", out.getvalue())
        self.assertIn("skipping", out.getvalue())

    async def test_tts_skips_server_without_tts(self):
        server = await self._serve(make_info(tts=False))
        port = server.sockets[0].getsockname()[1]
        out = io.StringIO()
        try:
            args = argparse.Namespace(
                rounds=1, warmup=0, verbose=False, chunk_delay=0.0, timeout=2.0, unique=False
            )
            with contextlib.redirect_stdout(out):
                rc = await cli._async_main_tts(
                    args, [("127.0.0.1", port)], ["hello"], [MODE_NON_STREAMING], None, None, 2.0
                )
        finally:
            server.close()
            await server.wait_closed()
        self.assertEqual(rc, 1)
        self.assertIn("[preflight]", out.getvalue())
        self.assertIn("skipping", out.getvalue())

    async def test_tts_unreachable_server_skipped_fast(self):
        # A down server (nothing on 127.0.0.1:1) must be skipped by the
        # preflight: no benchmark runs are attempted, and the exit code is 1.
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            args = argparse.Namespace(
                rounds=3, warmup=1, verbose=False, chunk_delay=0.0, timeout=2.0, unique=False
            )
            rc = await cli._async_main_tts(
                args, [("127.0.0.1", 1)], ["hello"], [MODE_NON_STREAMING], None, None, 2.0
            )
        self.assertEqual(rc, 1)
        self.assertIn("[preflight]", out.getvalue())
        self.assertIn("unavailable", out.getvalue())
        self.assertIn("skipping server", out.getvalue())
        self.assertNotIn("FAIL", out.getvalue())  # no benchmark run was attempted

    async def test_stt_proceeds_when_asr_advertised(self):
        # Preflight must not over-block: with asr advertised the real benchmark
        # runs (and fails against the mock, which only answers describe).
        server = await self._serve(make_info())
        port = server.sockets[0].getsockname()[1]
        out = io.StringIO()
        try:
            args = argparse.Namespace(
                rounds=1,
                warmup=0,
                verbose=True,
                chunk_samples=1024,
                chunk_delay=0.0,
                trailing_silence=0.0,
                timeout=1.0,
            )
            with contextlib.redirect_stdout(out):
                rc = await cli._async_main_stt(
                    args,
                    [("127.0.0.1", port)],
                    [_fake_sample()],
                    [MODE_NON_STREAMING],
                    Transcribe(),
                    1.0,
                )
        finally:
            server.close()
            await server.wait_closed()
        self.assertEqual(rc, 1)
        self.assertNotIn("[preflight]", out.getvalue())
        self.assertIn("FAIL", out.getvalue())


class TestMain(unittest.TestCase):
    def test_version(self):
        with self.assertRaises(SystemExit) as ctx:
            cli.main(["--version"])
        self.assertEqual(ctx.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
