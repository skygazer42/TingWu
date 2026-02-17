from __future__ import annotations

from typing import Any, List

import httpx

from src.core.speaker.external_diarizer_types import ExternalDiarizerSegmentLike


async def fetch_diarizer_segments(
    *,
    base_url: str,
    wav_bytes: bytes,
    timeout_s: float,
) -> List[ExternalDiarizerSegmentLike]:
    """Fetch raw diarization segments from an external diarizer service.

    This client returns the response segments verbatim (best-effort validation only).
    Normalization is handled elsewhere.
    """
    base_url = str(base_url or "").rstrip("/")
    if not base_url:
        raise ValueError("base_url must be non-empty")

    files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.post(f"{base_url}/api/v1/diarize", files=files)
        resp.raise_for_status()
        obj = resp.json()

    if not isinstance(obj, dict):
        raise ValueError("external diarizer response JSON must be an object")

    segments = obj.get("segments")
    if not isinstance(segments, list):
        raise ValueError("external diarizer response JSON must contain a list field: segments")

    return segments
