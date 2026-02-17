def test_settings_parse_external_diarizer_env(monkeypatch):
    monkeypatch.setenv("SPEAKER_EXTERNAL_DIARIZER_ENABLE", "true")
    monkeypatch.setenv("SPEAKER_EXTERNAL_DIARIZER_BASE_URL", "http://diarizer:8000")
    monkeypatch.setenv("SPEAKER_EXTERNAL_DIARIZER_TIMEOUT_S", "12.5")
    from src.config import Settings
    s = Settings()
    assert s.speaker_external_diarizer_enable is True
    assert s.speaker_external_diarizer_base_url == "http://diarizer:8000"
    assert s.speaker_external_diarizer_timeout_s == 12.5

