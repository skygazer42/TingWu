import importlib


def test_diarizer_routes_engine_reads_speaker_bounds(monkeypatch):
    monkeypatch.setenv("DIARIZER_MODEL", "pyannote/speaker-diarization-3.1")
    monkeypatch.setenv("DEVICE", "cuda")
    monkeypatch.setenv("DIARIZER_NUM_SPEAKERS", "3")
    monkeypatch.setenv("DIARIZER_MIN_SPEAKERS", "2")
    monkeypatch.setenv("DIARIZER_MAX_SPEAKERS", "5")

    import src.diarizer_service.routes as routes_mod

    routes_mod = importlib.reload(routes_mod)

    assert routes_mod.engine.num_speakers == 3
    assert routes_mod.engine.min_speakers == 2
    assert routes_mod.engine.max_speakers == 5

