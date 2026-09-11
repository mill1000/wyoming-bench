"""wyoming_bench — benchmark Wyoming-protocol TTS and STT servers."""

from importlib.metadata import PackageNotFoundError, version

try:
    # Single source of truth: the version setuptools-scm derives from git tags
    # at install/build time (see pyproject.toml).
    __version__ = version("wyoming-bench")
except PackageNotFoundError:
    __version__ = "0.0.0+local"
