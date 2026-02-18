from unittest.mock import Mock


def test_engine_lazy_import(monkeypatch):
    fake = Mock()
    monkeypatch.setitem(__import__("sys").modules, "pyannote", fake)

    from src.diarizer_service.engine import DiarizerEngine

    e = DiarizerEngine(model_id="x", device="cuda")
    # Should not crash on init without loading
    assert e is not None

