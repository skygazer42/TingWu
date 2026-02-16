import numpy as np

from src.core.audio.preprocessor import AudioPreprocessor


def _sine(freq_hz: float, duration_s: float, sr: int, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(round(duration_s * sr)), dtype=np.float32) / float(sr)
    return (amp * np.sin(2.0 * np.pi * float(freq_hz) * t)).astype(np.float32)


def test_highpass_attenuates_low_frequency_energy():
    sr = 16000
    audio = _sine(50.0, 2.0, sr, amp=0.5)

    pre = AudioPreprocessor(
        normalize_enable=False,
        trim_silence_enable=False,
        denoise_enable=False,
        adaptive_enable=False,
        remove_dc_offset=False,
        highpass_enable=True,
        highpass_cutoff_hz=200.0,
    )

    out = pre.process(audio, sample_rate=sr, validate=False)
    before_rms = float(np.sqrt(np.mean(audio ** 2)))
    after_rms = float(np.sqrt(np.mean(out ** 2)))

    assert after_rms < before_rms * 0.4


def test_highpass_keeps_high_frequency_energy_reasonable():
    sr = 16000
    audio = _sine(1000.0, 2.0, sr, amp=0.5)

    pre = AudioPreprocessor(
        normalize_enable=False,
        trim_silence_enable=False,
        denoise_enable=False,
        adaptive_enable=False,
        remove_dc_offset=False,
        highpass_enable=True,
        highpass_cutoff_hz=200.0,
    )

    out = pre.process(audio, sample_rate=sr, validate=False)
    before_rms = float(np.sqrt(np.mean(audio ** 2)))
    after_rms = float(np.sqrt(np.mean(out ** 2)))

    assert after_rms > before_rms * 0.8


def test_soft_limiter_reduces_clipping_ratio():
    # Build a heavily clipped waveform (flat at +/-1.0).
    audio = np.zeros((16000,), dtype=np.float32)
    audio[::2] = 1.0
    audio[1::2] = -1.0

    pre = AudioPreprocessor(
        normalize_enable=False,
        trim_silence_enable=False,
        denoise_enable=False,
        adaptive_enable=False,
        remove_dc_offset=False,
        soft_limit_enable=True,
        soft_limit_target=0.98,
        soft_limit_knee=3.0,
    )

    out = pre.process(audio, sample_rate=16000, validate=False)
    before_clip = float(np.mean(np.abs(audio) >= 0.999))
    after_clip = float(np.mean(np.abs(out) >= 0.999))

    assert before_clip > 0.9
    assert after_clip < 0.01

