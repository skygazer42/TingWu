import pytest
import sys
from pathlib import Path
import types
import importlib.util
import builtins

sys.path.insert(0, str(Path(__file__).parent))


def _ensure_optional_dependency_stubs_installed() -> None:
    """Install lightweight stubs for optional heavy deps.

    The production code imports some optional dependencies (e.g. FunASR, numba)
    at module import time. Most unit tests mock the ASR backend and don't need
    the real libraries installed.
    """

    if "funasr" not in sys.modules and importlib.util.find_spec("funasr") is None:
        funasr_stub = types.ModuleType("funasr")

        class DummyAutoModel:
            def __init__(self, *args, **kwargs):
                pass

            def generate(self, **kwargs):
                return []

        funasr_stub.AutoModel = DummyAutoModel
        sys.modules["funasr"] = funasr_stub

    if "numba" not in sys.modules and importlib.util.find_spec("numba") is None:
        numba_stub = types.ModuleType("numba")

        def njit(*args, **kwargs):
            if args and callable(args[0]) and len(args) == 1 and not kwargs:
                return args[0]

            def decorator(func):
                return func

            return decorator

        numba_stub.njit = njit
        sys.modules["numba"] = numba_stub

    # `aiofiles` is a small runtime dependency, but may be missing in minimal
    # unit-test environments. Many tests patch out `process_audio_file` so we
    # only need a stub that supports `aiofiles.open(...).__aenter__/write`.
    if "aiofiles" not in sys.modules and importlib.util.find_spec("aiofiles") is None:
        aiofiles_stub = types.ModuleType("aiofiles")

        class _AsyncFile:
            def __init__(self, f):
                self._f = f

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                try:
                    self._f.close()
                except Exception:
                    pass

            async def write(self, data):
                return self._f.write(data)

            async def read(self, *args, **kwargs):
                return self._f.read(*args, **kwargs)

        def open(file, mode="r", *args, **kwargs):  # noqa: A001 - matches aiofiles API
            f = builtins.open(file, mode, *args, **kwargs)
            return _AsyncFile(f)

        aiofiles_stub.open = open  # type: ignore[attr-defined]
        sys.modules["aiofiles"] = aiofiles_stub

    # `ffmpeg-python` is used for media decoding. Unit tests generally patch the
    # decode path, so provide a minimal stub when the library isn't installed.
    if "ffmpeg" not in sys.modules and importlib.util.find_spec("ffmpeg") is None:
        ffmpeg_stub = types.ModuleType("ffmpeg")

        class Error(Exception):
            def __init__(self, *args, stderr: bytes | None = None, **kwargs):
                super().__init__(*args)
                self.stderr = stderr

        class _FFmpegNode:
            def output(self, *args, **kwargs):
                return self

            def run(self, *args, **kwargs):
                raise Error("ffmpeg-python stub: install ffmpeg-python to decode media", stderr=b"")

        def input(*args, **kwargs):  # noqa: A001 - matches ffmpeg-python API
            return _FFmpegNode()

        ffmpeg_stub.Error = Error  # type: ignore[attr-defined]
        ffmpeg_stub.input = input  # type: ignore[attr-defined]
        sys.modules["ffmpeg"] = ffmpeg_stub


_ensure_optional_dependency_stubs_installed()
