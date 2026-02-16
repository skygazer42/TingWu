import numpy as np

from src.core.audio.chunker import AudioChunker


def _sec_to_samples(seconds: float, sample_rate: int) -> int:
    return int(round(seconds * sample_rate))


def test_audio_chunker_prefers_latest_silence_near_target_end():
    sr = 16000

    # Build a 12s signal with two silence regions in the search range [7.5s, 10.0s].
    # We want the splitter to pick the *later* silence (closer to 10s), not the first one.
    speech = lambda s: np.full((_sec_to_samples(s, sr),), 0.1, dtype=np.float32)
    silence = lambda s: np.zeros((_sec_to_samples(s, sr),), dtype=np.float32)

    audio = np.concatenate(
        [
            speech(8.0),
            silence(0.6),   # ~8.3s midpoint (earlier candidate)
            speech(0.8),
            silence(0.6),   # ~9.7s midpoint (later candidate, should be chosen)
            speech(2.0),
        ]
    )

    chunker = AudioChunker(
        max_chunk_duration=10.0,
        min_chunk_duration=5.0,
        overlap_duration=0.5,
        silence_threshold_db=-40.0,
        min_silence_duration=0.3,
    )

    silence_points = chunker._find_silence_points(audio, sample_rate=sr)
    assert any(_sec_to_samples(8.0, sr) <= p <= _sec_to_samples(9.0, sr) for p in silence_points)
    assert any(_sec_to_samples(9.3, sr) <= p <= _sec_to_samples(10.0, sr) for p in silence_points)

    chunks = chunker.split(audio, sample_rate=sr)
    assert len(chunks) >= 2

    # Second chunk start is derived from best_split - overlap.
    # If we chose the *early* silence (~8.3s), second chunk would start ~7.8s.
    # If we chose the *late* silence (~9.7s), second chunk starts ~9.2s (or later).
    second_start = chunks[1][1]
    assert second_start >= _sec_to_samples(8.8, sr)

