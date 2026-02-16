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


def test_audio_chunker_time_strategy_splits_by_duration_without_silence():
    sr = 16000
    audio = np.full((_sec_to_samples(25.0, sr),), 0.1, dtype=np.float32)

    chunker = AudioChunker(
        max_chunk_duration=10.0,
        min_chunk_duration=5.0,
        overlap_duration=2.0,
        silence_threshold_db=-40.0,
        min_silence_duration=0.3,
        strategy="time",
    )

    chunks = chunker.split(audio, sample_rate=sr)
    assert len(chunks) == 3

    # Time strategy should split around target_end (10s), and the next chunk starts at (10s - overlap).
    assert chunks[0][1] == 0
    assert chunks[1][1] == _sec_to_samples(8.0, sr)


def test_audio_chunker_prefers_longer_silence_over_later_shorter_silence():
    sr = 16000

    speech = lambda s: np.full((_sec_to_samples(s, sr),), 0.1, dtype=np.float32)
    silence = lambda s: np.zeros((_sec_to_samples(s, sr),), dtype=np.float32)

    # Two silence regions exist within the search range near the target end (10s):
    # - a longer silence earlier (should be chosen)
    # - a shorter silence later (should not be chosen)
    audio = np.concatenate(
        [
            speech(8.0),
            silence(1.2),   # long silence, midpoint ~8.6s (prefer)
            speech(0.4),
            silence(0.4),   # short silence, midpoint ~9.8s
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

    chunks = chunker.split(audio, sample_rate=sr)
    assert len(chunks) >= 2

    # If the long silence midpoint (~8.6s) is chosen as split, next chunk starts at ~8.1s.
    second_start = chunks[1][1]
    assert second_start < _sec_to_samples(9.0, sr)
