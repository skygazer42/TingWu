import math

from src.core.speaker.external_diarizer_normalize import normalize_segments


def test_normalize_segments_sorts_clamps_and_drops_invalid():
    raw = [
        {"spk": 1, "start": 2000, "end": 1000},  # invalid (end < start) -> drop
        {"spk": 0, "start": -5, "end": 10},      # clamp start to 0
        {"spk": 0, "start": 10, "end": 20},
    ]
    segs = normalize_segments(raw, duration_ms=15)
    assert segs == [
        {"spk": 0, "start": 0, "end": 10},
        {"spk": 0, "start": 10, "end": 15},
    ]


def test_normalize_segments_drops_segments_missing_required_keys():
    raw = [
        {"spk": 7, "start": 10, "end": 20},
        {"start": 0, "end": 10},   # missing spk -> drop
        {"spk": 1, "end": 30},     # missing start -> drop
        {"spk": 2, "start": 40},   # missing end -> drop
    ]
    segs = normalize_segments(raw, duration_ms=100)
    assert segs == [{"spk": 7, "start": 10, "end": 20}]


def test_normalize_segments_bad_duration_does_not_wipe_valid_segments():
    raw = [
        {"spk": 0, "start": 0, "end": 5},
        {"spk": 0, "start": 10, "end": 20},
    ]
    segs = normalize_segments(raw, duration_ms="wat")
    assert segs == [
        {"spk": 0, "start": 0, "end": 5},
        {"spk": 0, "start": 10, "end": 20},
    ]


def test_normalize_segments_overflow_values_do_not_raise_and_are_dropped():
    raw = [{"spk": 0, "start": math.inf, "end": 20}]
    segs = normalize_segments(raw, duration_ms=None)
    assert segs == []
