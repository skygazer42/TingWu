import numpy as np

from src.core.audio.preprocessor import AudioPreprocessor


def test_audio_preprocessor_removes_dc_offset():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = 0.1 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32) + 0.2

    processor = AudioPreprocessor(
        normalize_enable=False,
        trim_silence_enable=False,
        denoise_enable=False,
        adaptive_enable=False,
    )

    out = processor.process(audio, sample_rate=sr, validate=True)
    assert abs(float(np.mean(out))) < 1e-3

