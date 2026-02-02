def test_settings_has_remote_asr_fields():
    from src.config import settings

    assert hasattr(settings, "qwen3_asr_base_url")
    assert hasattr(settings, "vibevoice_asr_base_url")
    assert hasattr(settings, "router_long_audio_threshold_s")

