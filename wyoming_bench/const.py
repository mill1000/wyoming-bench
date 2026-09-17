"""Shared run-mode identifiers used by the TTS and STT benchmarks."""

MODE_NON_STREAMING = "non_streaming"
MODE_STREAMING = "streaming"

# Special value for --programs: benchmark every program the server advertises.
PROGRAM_ALL = "all"

# Special value for --programs: fall back to the server's first program when
# none of the other requested names are advertised by that server.
PROGRAM_ANY = "any"
