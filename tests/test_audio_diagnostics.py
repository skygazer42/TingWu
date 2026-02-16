import numpy as np

from src.core.audio.preprocessor import AudioPreprocessor


def test_audio_info_includes_dc_offset_and_clipping_ratio():
    sr = 16000
    processor = AudioPreprocessor()

    # DC offset of +0.2, no clipping
    audio = np.full((sr,), 0.2, dtype=np.float32)
    info = processor.get_audio_info(audio, sample_rate=sr)

    assert "dc_offset" in info
    assert abs(info["dc_offset"] - 0.2) < 1e-4

    assert "clipping_ratio" in info
    assert info["clipping_ratio"] == 0.0


def test_audio_info_clipping_ratio_detects_clipped_samples():
    sr = 16000
    processor = AudioPreprocessor()

    audio = np.full((1000,), 0.1, dtype=np.float32)
    audio[:10] = 1.0
    audio[10:20] = -1.0

    info = processor.get_audio_info(audio, sample_rate=sr)
    assert info["clipping_ratio"] == 20 / 1000

