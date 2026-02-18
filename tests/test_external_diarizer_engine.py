import asyncio
import importlib.util
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


def _ensure_optional_dependency_stubs_installed() -> None:
    # These modules are imported at module-import time by the production code,
    # but aren't needed for these unit tests (we mock the backend).
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


_ensure_optional_dependency_stubs_installed()

import src.core.engine as engine_mod


@pytest.fixture
def mock_model_manager():
    with patch.object(engine_mod, "model_manager") as mock_mm:
        backend = MagicMock()
        backend.get_info.return_value = {"name": "Qwen3Remote", "type": "qwen3"}
        backend.supports_speaker = False
        backend.supports_hotwords = False
        backend.supports_streaming = False
        backend.transcribe.side_effect = [
            {"text": "第一段", "sentence_info": []},
            {"text": "第二段", "sentence_info": []},
        ]
        mock_mm.backend = backend
        yield mock_mm


def test_transcribe_async_with_external_diarizer_builds_speaker_turns(mock_model_manager, monkeypatch):
    monkeypatch.setattr(engine_mod.settings, "speaker_unsupported_behavior", "ignore", raising=False)
    monkeypatch.setattr(engine_mod.settings, "speaker_external_diarizer_enable", True, raising=False)
    monkeypatch.setattr(
        engine_mod.settings, "speaker_external_diarizer_base_url", "http://diar:8000", raising=False
    )

    async def fake_fetch(*args, **kwargs):
        return [
            {"spk": 0, "start": 0, "end": 1000},
            {"spk": 1, "start": 1000, "end": 2000},
        ]

    # 2 seconds of PCM16LE @16kHz mono
    audio_bytes = b"\x00" * (2 * 16000 * 2)

    with patch.object(engine_mod, "fetch_diarizer_segments", new=fake_fetch):
        engine = engine_mod.TranscriptionEngine()
        out = asyncio.run(
            engine.transcribe_async(
                audio_bytes,
                with_speaker=True,
                apply_hotword=False,
                apply_llm=False,
                asr_options={"speaker": {"label_style": "numeric"}},
            )
        )

    assert out["sentences"][0]["speaker"] == "说话人1"
    assert out["sentences"][0]["speaker_id"] == 0
    assert out["sentences"][0]["text"] == "第一段"
    assert out["sentences"][1]["speaker"] == "说话人2"
    assert out["sentences"][1]["text"] == "第二段"

    assert out.get("speaker_turns")
    assert out.get("transcript")
    assert "说话人1" in out["transcript"]

    assert mock_model_manager.backend.transcribe.call_count == 2
    for c in mock_model_manager.backend.transcribe.call_args_list:
        assert c.kwargs.get("with_speaker") is False

