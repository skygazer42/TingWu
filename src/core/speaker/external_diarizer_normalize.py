from __future__ import annotations

from typing import Any, Iterable, List

from src.core.speaker.external_diarizer_types import (
    ExternalDiarizerSegment,
    ExternalDiarizerSegmentLike,
)


def _try_coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _coerce_int(value: Any, *, default: int) -> int:
    coerced = _try_coerce_int(value)
    if coerced is None:
        return default
    return coerced


def normalize_segments(
    raw_segments: Iterable[ExternalDiarizerSegmentLike] | None,
    duration_ms: int | None,
) -> List[ExternalDiarizerSegment]:
    """Normalize external diarizer segments.

    Normalization rules (best-effort):
    - Coerce `spk`, `start`, `end` to ints (drop segment if any missing/invalid).
    - Clamp to [0, duration_ms] when duration is known and valid.
    - Drop segments with end <= start.
    - Sort by (start, end, spk).
    """
    if not raw_segments:
        return []

    duration_ms_int: int | None
    if duration_ms is None:
        duration_ms_int = None
    else:
        duration_ms_int = _try_coerce_int(duration_ms)
        if duration_ms_int is not None and duration_ms_int < 0:
            duration_ms_int = 0

    normalized: List[ExternalDiarizerSegment] = []

    for seg in raw_segments:
        if seg is None:
            continue

        try:
            spk_raw = seg.get("spk")
            start_raw = seg.get("start")
            end_raw = seg.get("end")
        except AttributeError:
            continue

        spk = _try_coerce_int(spk_raw)
        start = _try_coerce_int(start_raw)
        end = _try_coerce_int(end_raw)

        if spk is None or start is None or end is None:
            continue

        if start < 0:
            start = 0
        if end < 0:
            end = 0

        if duration_ms_int is not None:
            if start > duration_ms_int:
                start = duration_ms_int
            if end > duration_ms_int:
                end = duration_ms_int

        if end <= start:
            continue

        normalized.append({"spk": spk, "start": start, "end": end})

    normalized.sort(key=lambda s: (s["start"], s["end"], s["spk"]))
    return normalized
