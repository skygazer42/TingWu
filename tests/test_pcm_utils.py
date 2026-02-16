import io
import wave

import numpy as np
import pytest


def _pcm():
    try:
        from src.core.audio.pcm import (
            is_wav_bytes,
            pcm16le_bytes_to_float32,
            float32_to_pcm16le_bytes,
            wav_bytes_to_float32,
        )
    except Exception as e:
        pytest.fail(f"pcm utils not available: {e}")
    return is_wav_bytes, pcm16le_bytes_to_float32, float32_to_pcm16le_bytes, wav_bytes_to_float32


def test_pcm16le_roundtrip_is_reasonable():
    _, pcm16le_bytes_to_float32, float32_to_pcm16le_bytes, _ = _pcm()

    audio = np.array([-1.0, -0.5, 0.0, 0.5, 0.999], dtype=np.float32)
    b = float32_to_pcm16le_bytes(audio)
    audio2 = pcm16le_bytes_to_float32(b)

    assert audio2.dtype == np.float32
    assert audio2.shape == audio.shape
    # Quantization error is expected; ensure it's small.
    assert np.max(np.abs(audio2 - audio)) < 1e-3


def test_wav_bytes_detection_and_decode():
    is_wav_bytes, _, _, wav_bytes_to_float32 = _pcm()

    sr = 16000
    frames = np.array([0, 1000, -1000, 32767, -32768], dtype=np.int16).tobytes()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(frames)

    wav = buf.getvalue()

    assert is_wav_bytes(wav) is True

    audio, decoded_sr = wav_bytes_to_float32(wav)
    assert decoded_sr == sr
    assert audio.dtype == np.float32
    assert audio.shape == (5,)
    assert np.allclose(audio[0], 0.0, atol=1e-6)
    assert audio[1] > 0
    assert audio[2] < 0

