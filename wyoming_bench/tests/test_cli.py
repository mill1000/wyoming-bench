"""Tests for the parsing helpers in wyoming_bench.cli."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioStop
from wyoming.error import Error
from wyoming.event import async_read_event, async_write_event
from wyoming.info import Describe
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
from wyoming_bench.const import MODE_NON_STREAMING, PROGRAM_ALL, PROGRAM_ANY
from wyoming_bench.stt.corpus import AudioInput, load_recordings
from wyoming_bench.tests.stt.test_corpus import _write_wav
from wyoming_bench.tests.test_info import (
    _info_handler,
    make_info,
    make_info_two_asr,
    make_info_two_tts,
)


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


class TestParsePrograms(unittest.TestCase):
    """The comma-separated --programs value is split, trimmed, and normalized."""

    def test_single_name(self):
        self.assertEqual(cli._parse_programs("piper"), ["piper"])

    def test_comma_separated(self):
        self.assertEqual(cli._parse_programs("piper,kokoro"), ["piper", "kokoro"])

    def test_whitespace_trimmed(self):
        self.assertEqual(cli._parse_programs(" piper , kokoro "), ["piper", "kokoro"])

    def test_dedupe_preserves_order(self):
        self.assertEqual(cli._parse_programs("piper,kokoro,piper"), ["piper", "kokoro"])

    def test_all_alone(self):
        self.assertEqual(cli._parse_programs("all"), [PROGRAM_ALL])

    def test_any_alone(self):
        self.assertEqual(cli._parse_programs("any"), [PROGRAM_ANY])

    def test_names_with_any(self):
        self.assertEqual(cli._parse_programs("piper,any"), ["piper", PROGRAM_ANY])

    def test_trailing_comma_rejected(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            cli._parse_programs("piper,")

    def test_empty_value_rejected(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            cli._parse_programs(",")

    def test_all_mixed_with_names_rejected(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            cli._parse_programs("all,piper")

    def test_all_mixed_with_any_rejected(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            cli._parse_programs("all,any")


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
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
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

    def test_tts_programs_default_none(self):
        args = build_parser().parse_args(["tts", "localhost"])
        self.assertIsNone(args.programs)

    def test_tts_programs_names(self):
        args = build_parser().parse_args(["tts", "localhost", "--programs", "piper,kokoro"])
        self.assertEqual(args.programs, ["piper", "kokoro"])

    def test_tts_programs_all(self):
        args = build_parser().parse_args(["tts", "localhost", "--programs", "all"])
        self.assertEqual(args.programs, [PROGRAM_ALL])

    def test_tts_programs_empty_name_rejected(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            build_parser().parse_args(["tts", "localhost", "--programs", "piper,"])

    def test_tts_programs_all_mixed_rejected(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            build_parser().parse_args(["tts", "localhost", "--programs", "all,piper"])

    def test_gen_corpus_program(self):
        args = build_parser().parse_args(
            ["generate-corpus", "localhost", "--out", "/tmp/o", "--program", "piper"]
        )
        self.assertEqual(args.program, "piper")

    def test_stt_programs_default_none(self):
        args = build_parser().parse_args(["stt", "localhost", "--corpus", "./c"])
        self.assertIsNone(args.programs)

    def test_stt_programs_names(self):
        args = build_parser().parse_args(
            ["stt", "localhost", "--corpus", "./c", "--programs", "whisper,paraformer"]
        )
        self.assertEqual(args.programs, ["whisper", "paraformer"])

    def test_stt_programs_all(self):
        args = build_parser().parse_args(["stt", "localhost", "--corpus", "./c", "--programs", "all"])
        self.assertEqual(args.programs, [PROGRAM_ALL])

    def test_tts_programs_any(self):
        args = build_parser().parse_args(["tts", "localhost", "--programs", "any"])
        self.assertEqual(args.programs, [PROGRAM_ANY])

    def test_tts_programs_names_with_any(self):
        args = build_parser().parse_args(["tts", "localhost", "--programs", "piper,any"])
        self.assertEqual(args.programs, ["piper", PROGRAM_ANY])

    def test_tts_programs_all_any_rejected(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            build_parser().parse_args(["tts", "localhost", "--programs", "all,any"])

    def test_stt_programs_any(self):
        args = build_parser().parse_args(["stt", "localhost", "--corpus", "./c", "--programs", "any"])
        self.assertEqual(args.programs, [PROGRAM_ANY])

    def test_seed_program(self):
        args = build_parser().parse_args(
            ["seed-corpus", "localhost", "--corpus", "./c", "--program", "whisper"]
        )
        self.assertEqual(args.program, "whisper")

    def test_stt_requires_corpus(self):
        args = build_parser().parse_args(["stt", "localhost", "--corpus", "/tmp/c"])
        self.assertEqual(args.task, "stt")
        self.assertEqual(args.corpus, "/tmp/c")

    def test_stt_requires_server(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            build_parser().parse_args(["stt", "--corpus", "/tmp/c"])

    def test_gen_corpus(self):
        args = build_parser().parse_args(["generate-corpus", "localhost", "--out", "/tmp/o"])
        self.assertEqual(args.task, "generate-corpus")
        self.assertEqual(args.out, "/tmp/o")

    def test_seed(self):
        args = build_parser().parse_args(["seed-corpus", "localhost", "--corpus", "/tmp/c"])
        self.assertEqual(args.task, "seed-corpus")
        self.assertEqual(args.corpus, "/tmp/c")
        self.assertFalse(args.overwrite)

    def test_seed_requires_corpus(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            build_parser().parse_args(["seed-corpus", "localhost"])

    def test_seed_requires_server(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            build_parser().parse_args(["seed-corpus", "--corpus", "/tmp/c"])

    def test_seed_overwrite(self):
        args = build_parser().parse_args(["seed-corpus", "localhost", "--corpus", "/tmp/c", "--overwrite"])
        self.assertTrue(args.overwrite)

    def test_info(self):
        args = build_parser().parse_args(["info", "localhost:1234"])
        self.assertEqual(args.task, "info")
        self.assertEqual(args.servers, [("localhost", 1234)])
        self.assertEqual(args.timeout, 5.0)

    def test_info_requires_server(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            build_parser().parse_args(["info"])

    def test_missing_task(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            build_parser().parse_args([])

    def test_version_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx, contextlib.redirect_stdout(io.StringIO()):
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
    """Probes respect --timeout, capped at the 8s probe budget."""

    def test_capped_at_default(self):
        self.assertEqual(cli._resolve_probe_timeout(argparse.Namespace(timeout=60.0), 8.0), 8.0)

    def test_follows_lower_timeout(self):
        self.assertEqual(cli._resolve_probe_timeout(argparse.Namespace(timeout=3.0), 8.0), 3.0)


class TestNormalizePrograms(unittest.TestCase):
    def test_none_is_empty(self):
        self.assertEqual(cli.normalize_programs(None), [])

    def test_dedupe_preserves_order(self):
        self.assertEqual(cli.normalize_programs(["a", "b", "a"]), ["a", "b"])

    def test_all_alone_ok(self):
        self.assertEqual(cli.normalize_programs(["all"]), ["all"])

    def test_all_mixed_with_names_rejected(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            cli.normalize_programs(["all", "piper"])

    def test_any_mixed_with_names_ok(self):
        self.assertEqual(cli.normalize_programs(["piper", "any"]), ["piper", "any"])

    def test_all_mixed_with_any_rejected(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            cli.normalize_programs(["all", "any"])


class TestProgramsForServer(unittest.TestCase):
    """--programs resolution against one server's advertised info."""

    def test_default_selects_server_default(self):
        selected, missing = cli._programs_for_server([], make_info(), "tts")
        self.assertEqual(selected, [None])
        self.assertEqual(missing, [])

    def test_all_expands_advertised_programs(self):
        selected, missing = cli._programs_for_server(["all"], make_info_two_tts(), "tts")
        self.assertEqual(selected, ["tts-prog", "kokoro"])
        self.assertEqual(missing, [])

    def test_all_without_info_falls_back_to_default(self):
        selected, missing = cli._programs_for_server(["all"], None, "tts")
        self.assertEqual(selected, [None])
        self.assertEqual(missing, [])

    def test_named_programs_filtered_by_advertised(self):
        selected, missing = cli._programs_for_server(
            ["kokoro", "nope", "tts-prog"], make_info_two_tts(), "tts"
        )
        self.assertEqual(selected, ["kokoro", "tts-prog"])
        self.assertEqual(missing, ["nope"])

    def test_all_named_programs_missing(self):
        selected, missing = cli._programs_for_server(["nope"], make_info(), "tts")
        self.assertEqual(selected, [])
        self.assertEqual(missing, ["nope"])

    def test_named_programs_without_info_passed_through(self):
        selected, missing = cli._programs_for_server(["piper"], None, "tts")
        self.assertEqual(selected, ["piper"])
        self.assertEqual(missing, [])

    def test_asr_default_selects_server_default(self):
        selected, missing = cli._programs_for_server([], make_info(), "asr")
        self.assertEqual(selected, [None])
        self.assertEqual(missing, [])

    def test_asr_all_expands_advertised_programs(self):
        selected, missing = cli._programs_for_server(["all"], make_info_two_asr(), "asr")
        self.assertEqual(selected, ["asr-prog", "whisper"])
        self.assertEqual(missing, [])

    def test_asr_named_programs_filtered_by_advertised(self):
        selected, missing = cli._programs_for_server(
            ["whisper", "nope", "asr-prog"], make_info_two_asr(), "asr"
        )
        self.assertEqual(selected, ["whisper", "asr-prog"])
        self.assertEqual(missing, ["nope"])

    def test_match_is_case_insensitive_and_returns_canonical_name(self):
        selected, missing = cli._programs_for_server(
            ["Kokoro", "TTS-PROG", "Nope"], make_info_two_tts(), "tts"
        )
        self.assertEqual(selected, ["kokoro", "tts-prog"])
        self.assertEqual(missing, ["Nope"])

    def test_asr_match_is_case_insensitive_and_returns_canonical_name(self):
        selected, missing = cli._programs_for_server(["WHISPer"], make_info_two_asr(), "asr")
        self.assertEqual(selected, ["whisper"])
        self.assertEqual(missing, [])

    def test_any_alone_selects_server_default(self):
        selected, missing = cli._programs_for_server([PROGRAM_ANY], make_info(), "tts")
        self.assertEqual(selected, [None])
        self.assertEqual(missing, [])

    def test_any_falls_back_when_no_name_matches(self):
        selected, missing = cli._programs_for_server(["nope", PROGRAM_ANY], make_info(), "tts")
        self.assertEqual(selected, [None])
        self.assertEqual(missing, ["nope"])

    def test_any_ignored_when_a_name_matches(self):
        selected, missing = cli._programs_for_server(["kokoro", PROGRAM_ANY], make_info_two_tts(), "tts")
        self.assertEqual(selected, ["kokoro"])
        self.assertEqual(missing, [])

    def test_any_without_info_selects_server_default(self):
        selected, missing = cli._programs_for_server([PROGRAM_ANY], None, "tts")
        self.assertEqual(selected, [None])
        self.assertEqual(missing, [])

    def test_names_with_any_without_info_send_names(self):
        selected, missing = cli._programs_for_server(["piper", PROGRAM_ANY], None, "tts")
        self.assertEqual(selected, ["piper"])
        self.assertEqual(missing, [])


class TestProbeSummary(unittest.TestCase):
    """The advertised-only probe summary lists each server and its programs."""

    def _render(self, plan, service="tts") -> str:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cli._print_probe_summary(plan, service)
        return out.getvalue()

    def _entry(self, host, port, info, proceed, selected, missing=()):
        return cli._ServerPlan(host, port, info, proceed, list(selected), list(missing))

    def _line(self, out, name):
        return next(l for l in out.splitlines() if name in l)

    def test_lists_all_advertised_not_just_selected(self):
        info = make_info_two_tts()  # tts-prog: no streaming, kokoro: streaming
        # only kokoro is selected, but every advertised program is listed
        plan = [self._entry("a", 10200, info, True, ["kokoro"])]
        out = self._render(plan)
        self.assertIn("capabilities (advertised):", out)
        self.assertIn("a:10200", out)
        self.assertTrue(self._line(out, "tts-prog").rstrip().endswith("non-streaming"))
        self.assertTrue(self._line(out, "kokoro").rstrip().endswith("non-streaming, streaming"))

    def test_best_effort_shows_requested_as_unknown(self):
        # connected but no describe: advertised programs can't be enumerated, so the
        # requested ones are listed with an unknown capability
        plan = [self._entry("c", 10202, None, True, ["piper"])]
        out = self._render(plan)
        self.assertIn("c:10202", out)
        self.assertTrue(self._line(out, "piper").rstrip().endswith("unknown"))

    def test_lists_unreachable_and_all_advertised(self):
        info = make_info_two_tts()
        plan = [
            # down server: a single unreachable line, no programs
            self._entry("down", 40000, None, False, [None]),
            # reachable: all advertised programs; a requested-but-missing one is NOT shown
            self._entry("ok", 40001, info, True, ["kokoro"], ["nope"]),
        ]
        out = self._render(plan)
        self.assertIn("down:40000 - unreachable", out)
        self.assertIn("ok:40001", out)
        # the server's advertised programs are listed ...
        self.assertIn("tts-prog", out)
        self.assertIn("kokoro", out)
        # ... but not the requested program it does not advertise
        self.assertNotIn("nope", out)

    def test_programs_indented_under_their_server(self):
        info = make_info_two_tts()
        plan = [self._entry("a", 10200, info, True, ["tts-prog", "kokoro"])]
        out = self._render(plan)
        lines = out.splitlines()
        self.assertIn("  a:10200", lines)  # server on its own line, no capability
        self.assertTrue(any(l.startswith("    tts-prog") for l in lines))
        self.assertTrue(any(l.startswith("    kokoro") for l in lines))

    def test_nothing_printed_for_empty_plan(self):
        self.assertEqual(self._render([]), "")


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
                    [],
                )
        finally:
            server.close()
            await server.wait_closed()
        self.assertEqual(rc, 1)
        self.assertIn("[preflight]", out.getvalue())
        self.assertIn("skipping", out.getvalue())

    async def test_stt_skips_when_no_program_advertised(self):
        # A server that advertises none of the requested programs is warned
        # and skipped without any benchmark run, with exit code 1.
        server = await self._serve(make_info())
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
                    ["nope1", "nope2"],
                )
        finally:
            server.close()
            await server.wait_closed()
        self.assertEqual(rc, 1)
        self.assertIn("[programs]", out.getvalue())
        self.assertIn("skipping: nope1, nope2 not advertised", out.getvalue())
        self.assertNotIn("FAIL", out.getvalue())

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
                    args,
                    [("127.0.0.1", port)],
                    ["hello"],
                    [MODE_NON_STREAMING],
                    None,
                    None,
                    2.0,
                    [],
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
                args, [("127.0.0.1", 1)], ["hello"], [MODE_NON_STREAMING], None, None, 2.0, []
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
                    [],
                )
        finally:
            server.close()
            await server.wait_closed()
        self.assertEqual(rc, 1)
        self.assertNotIn("[preflight]", out.getvalue())
        self.assertIn("FAIL", out.getvalue())


class TestProgramSelection(unittest.IsolatedAsyncioTestCase):
    """The global --programs list is intersected with each server's programs."""

    async def _serve(self, info):
        return await asyncio.start_server(lambda r, w: _info_handler(r, w, info), "127.0.0.1", 0)

    def _stt_args(self) -> argparse.Namespace:
        # timeout/probe kept short: the mock only answers describe, so every
        # benchmark run fails on its first read timeout; these tests assert on
        # selection and labels, not on timeout behavior.
        return argparse.Namespace(
            rounds=1,
            warmup=0,
            verbose=False,
            chunk_samples=1024,
            chunk_delay=0.0,
            trailing_silence=0.0,
            timeout=0.1,
        )

    def _tts_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            rounds=1, warmup=0, verbose=False, chunk_delay=0.0, timeout=0.1, unique=False
        )

    async def test_stt_programs_intersect_per_server(self):
        # Server 1 advertises asr-prog only; server 2 advertises asr-prog and
        # whisper. The shared --programs list is intersected per server, with
        # no warning because every server matches at least one program.
        s1 = await self._serve(make_info())
        s2 = await self._serve(make_info_two_asr())
        p1 = s1.sockets[0].getsockname()[1]
        p2 = s2.sockets[0].getsockname()[1]
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                await cli._async_main_stt(
                    self._stt_args(),
                    [("127.0.0.1", p1), ("127.0.0.1", p2)],
                    [_fake_sample()],
                    [MODE_NON_STREAMING],
                    Transcribe(),
                    0.1,
                    ["asr-prog", "whisper"],
                )
        finally:
            s1.close()
            await s1.wait_closed()
            s2.close()
            await s2.wait_closed()
        report = out.getvalue()
        self.assertIn("programs: ['asr-prog', 'whisper']", report)
        self.assertIn(f"=== 127.0.0.1:{p1} (asr-prog) ===", report)
        self.assertIn(f"=== 127.0.0.1:{p2} (asr-prog) ===", report)
        self.assertIn(f"=== 127.0.0.1:{p2} (whisper) ===", report)
        self.assertNotIn("[programs]", report)

    async def test_stt_partial_match_benchmarks_intersection_without_warning(self):
        # The server advertises asr-prog and whisper; the request names
        # whisper plus a name the server lacks: benchmark whisper only, and
        # emit no warning because at least one requested program matched.
        server = await self._serve(make_info_two_asr())
        port = server.sockets[0].getsockname()[1]
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                await cli._async_main_stt(
                    self._stt_args(),
                    [("127.0.0.1", port)],
                    [_fake_sample()],
                    [MODE_NON_STREAMING],
                    Transcribe(),
                    0.1,
                    ["whisper", "nope"],
                )
        finally:
            server.close()
            await server.wait_closed()
        report = out.getvalue()
        self.assertIn(f"=== 127.0.0.1:{port} (whisper) ===", report)
        self.assertNotIn(f"=== 127.0.0.1:{port} (asr-prog) ===", report)
        self.assertNotIn("[programs]", report)

    async def test_tts_programs_intersect_per_server(self):
        # Server 1 advertises tts-prog only; server 2 advertises tts-prog and
        # kokoro. The shared --programs list is intersected per server.
        s1 = await self._serve(make_info())
        s2 = await self._serve(make_info_two_tts())
        p1 = s1.sockets[0].getsockname()[1]
        p2 = s2.sockets[0].getsockname()[1]
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                await cli._async_main_tts(
                    self._tts_args(),
                    [("127.0.0.1", p1), ("127.0.0.1", p2)],
                    ["hello"],
                    [MODE_NON_STREAMING],
                    None,
                    None,
                    0.1,
                    ["tts-prog", "kokoro"],
                )
        finally:
            s1.close()
            await s1.wait_closed()
            s2.close()
            await s2.wait_closed()
        report = out.getvalue()
        self.assertIn("programs: ['tts-prog', 'kokoro']", report)
        self.assertIn(f"=== 127.0.0.1:{p1} (tts-prog) ===", report)
        self.assertIn(f"=== 127.0.0.1:{p2} (tts-prog) ===", report)
        self.assertIn(f"=== 127.0.0.1:{p2} (kokoro) ===", report)
        self.assertNotIn("[programs]", report)

    async def test_programs_all_expands_every_advertised_program(self):
        server = await self._serve(make_info_two_asr())
        port = server.sockets[0].getsockname()[1]
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                await cli._async_main_stt(
                    self._stt_args(),
                    [("127.0.0.1", port)],
                    [_fake_sample()],
                    [MODE_NON_STREAMING],
                    Transcribe(),
                    0.1,
                    [PROGRAM_ALL],
                )
        finally:
            server.close()
            await server.wait_closed()
        report = out.getvalue()
        self.assertIn("programs: ['all']", report)
        self.assertIn(f"=== 127.0.0.1:{port} (asr-prog) ===", report)
        self.assertIn(f"=== 127.0.0.1:{port} (whisper) ===", report)
        self.assertNotIn("[programs]", report)

    async def test_programs_any_falls_back_to_server_default(self):
        # Server 1 advertises whisper (plus asr-prog); server 2 advertises
        # only asr-prog, which is not in the request. With 'any' in the list,
        # server 2 benchmarks its default program instead of being skipped.
        s1 = await self._serve(make_info_two_asr())
        s2 = await self._serve(make_info())
        p1 = s1.sockets[0].getsockname()[1]
        p2 = s2.sockets[0].getsockname()[1]
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                await cli._async_main_stt(
                    self._stt_args(),
                    [("127.0.0.1", p1), ("127.0.0.1", p2)],
                    [_fake_sample()],
                    [MODE_NON_STREAMING],
                    Transcribe(),
                    0.1,
                    ["whisper", PROGRAM_ANY],
                )
        finally:
            s1.close()
            await s1.wait_closed()
            s2.close()
            await s2.wait_closed()
        report = out.getvalue()
        self.assertIn(f"=== 127.0.0.1:{p1} (whisper) ===", report)
        self.assertIn(f"=== 127.0.0.1:{p2} (asr-prog) ===", report)
        self.assertNotIn("[programs]", report)


async def _seed_mock_handler(reader, writer, info, text):
    """Answer ``describe`` with *info*; transcribe each audio cycle with *text*.

    If *text* is ``None`` the transcription gets a server ``error`` instead.
    """
    try:
        while True:
            event = await async_read_event(reader)
            if event is None:
                break
            if Describe.is_type(event.type):
                await async_write_event(info.event(), writer)
            elif Transcribe.is_type(event.type):
                await async_read_event(reader)  # audio-start
                while True:  # audio-chunk* then audio-stop
                    chunk = await async_read_event(reader)
                    if chunk is None or AudioStop.is_type(chunk.type):
                        break
                if text is None:
                    await async_write_event(Error(text="boom").event(), writer)
                else:
                    await async_write_event(Transcript(text=text).event(), writer)
    except (ConnectionResetError, asyncio.IncompleteReadError):
        pass
    finally:
        try:
            writer.close()
        except Exception:  # noqa: BLE001
            pass
        await writer.wait_closed()


class TestSeedRunner(unittest.IsolatedAsyncioTestCase):
    """seed writes one .txt per recording, skipping transcripts that exist."""

    def _args(self, corpus_dir: Path, overwrite: bool = False):
        return argparse.Namespace(
            corpus=str(corpus_dir),
            chunk_samples=512,
            trailing_silence=0.0,
            timeout=2.0,
            overwrite=overwrite,
        )

    async def _seed(self, corpus_dir, text, overwrite=False, program=None):
        server = await asyncio.start_server(
            lambda r, w: _seed_mock_handler(r, w, make_info(asr=True, tts=False), text),
            "127.0.0.1",
            0,
        )
        port = server.sockets[0].getsockname()[1]
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                return (
                    await cli._async_seed_corpus(
                        self._args(corpus_dir, overwrite),
                        [("127.0.0.1", port)],
                        load_recordings(corpus_dir),
                        Transcribe(),
                        2.0,
                        program,
                    ),
                    out.getvalue(),
                )
        finally:
            server.close()
            await server.wait_closed()

    async def test_writes_transcripts_and_skips_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp)
            _write_wav(corpus / "rec_a.wav")
            _write_wav(corpus / "rec_b.wav")
            (corpus / "rec_b.txt").write_text("already reviewed\n", encoding="utf-8")

            rc, out = await self._seed(corpus, "hello world")

            self.assertEqual(rc, 0)
            self.assertEqual((corpus / "rec_a.txt").read_text(encoding="utf-8").strip(), "hello world")
            # The reviewed transcript is untouched.
            self.assertEqual((corpus / "rec_b.txt").read_text(encoding="utf-8").strip(), "already reviewed")
            self.assertIn("[skip] rec_b", out)
            self.assertIn("Wrote 1 transcript(s)", out)
            self.assertIn("1 skipped", out)

    async def test_overwrite_replaces_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp)
            _write_wav(corpus / "rec_a.wav")
            (corpus / "rec_a.txt").write_text("stale\n", encoding="utf-8")

            rc, out = await self._seed(corpus, "fresh transcript", overwrite=True)

            self.assertEqual(rc, 0)
            self.assertEqual((corpus / "rec_a.txt").read_text(encoding="utf-8").strip(), "fresh transcript")
            self.assertNotIn("[skip]", out)

    async def test_failed_transcription_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp)
            _write_wav(corpus / "rec_a.wav")

            rc, out = await self._seed(corpus, None)  # server answers with an error

            self.assertEqual(rc, 1)
            self.assertFalse((corpus / "rec_a.txt").exists())
            self.assertIn("[fail] rec_a", out)
            self.assertIn("boom", out)

    async def test_unreachable_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp)
            _write_wav(corpus / "rec_a.wav")

            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = await cli._async_seed_corpus(
                    self._args(corpus),
                    [("127.0.0.1", 1)],  # nothing listens here
                    load_recordings(corpus),
                    Transcribe(),
                    2.0,
                    None,
                )

            self.assertEqual(rc, 1)
            self.assertFalse((corpus / "rec_a.txt").exists())
            self.assertIn("[preflight]", out.getvalue())
            self.assertIn("skipping server", out.getvalue())

    async def test_program_not_advertised_fails(self):
        # The mock advertises only "asr-prog": requesting another name must
        # fail the preflight and write no transcripts.
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp)
            _write_wav(corpus / "rec_a.wav")

            rc, out = await self._seed(corpus, "hello world", program="nope")

            self.assertEqual(rc, 1)
            self.assertFalse((corpus / "rec_a.txt").exists())
            self.assertIn("[preflight]", out)
            self.assertIn("does not advertise STT (ASR) program", out)

    async def test_program_reaches_server(self):
        # select-program must be sent on the transcription connection, before
        # the transcribe event (after the preflight's describe on its own
        # connection).
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp)
            _write_wav(corpus / "rec_a.wav")
            seen: list[str] = []

            async def handler(reader, writer):
                try:
                    while True:
                        event = await async_read_event(reader)
                        if event is None:
                            break
                        seen.append(event.type)
                        if Describe.is_type(event.type):
                            await async_write_event(make_info(asr=True, tts=False).event(), writer)
                        elif event.type == "transcribe":
                            await async_read_event(reader)  # audio-start
                            while True:  # audio-chunk* then audio-stop
                                ev = await async_read_event(reader)
                                if ev is None or AudioStop.is_type(ev.type):
                                    break
                            await async_write_event(Transcript(text="hello world").event(), writer)
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
            try:
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    rc = await cli._async_seed_corpus(
                        self._args(corpus),
                        [("127.0.0.1", port)],
                        load_recordings(corpus),
                        Transcribe(),
                        2.0,
                        "asr-prog",
                    )
            finally:
                server.close()
                await server.wait_closed()

            self.assertEqual(rc, 0)
            self.assertEqual((corpus / "rec_a.txt").read_text(encoding="utf-8").strip(), "hello world")
            self.assertEqual(seen, ["describe", "select-program", "transcribe"])


if __name__ == "__main__":
    unittest.main()
