from __future__ import annotations

import io
import os
import wave
from typing import Set

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.diarizer_service.engine import DiarizerEngine
from src.diarizer_service.schemas import DiarizeResponse


router = APIRouter(prefix="/api/v1", tags=["diarizer"])


def _env_str(key: str, default: str) -> str:
    v = os.getenv(key)
    if v is None:
        return default
    v = str(v).strip()
    return v if v else default


def _env_int(key: str) -> int | None:
    v = os.getenv(key)
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


# NOTE: keep this lightweight at import time (no heavy ML deps).
engine = DiarizerEngine(
    model_id=_env_str("DIARIZER_MODEL", "pyannote/speaker-diarization-3.1"),
    device=_env_str("DEVICE", "cuda"),
    hf_token=(os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")),
    num_speakers=_env_int("DIARIZER_NUM_SPEAKERS"),
    min_speakers=_env_int("DIARIZER_MIN_SPEAKERS"),
    max_speakers=_env_int("DIARIZER_MAX_SPEAKERS"),
)


@router.post("/diarize", response_model=DiarizeResponse)
async def diarize(file: UploadFile = File(...)) -> DiarizeResponse:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="missing audio file")

    # Validate WAV container and compute duration quickly.
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            frames = wf.getnframes()
            sr = wf.getframerate() or 1
            duration_ms = int(frames * 1000 / sr)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid wav: {e}")

    try:
        segments = engine.diarize(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"diarization failed: {e}")

    speaker_ids: Set[int] = set()
    for seg in segments or []:
        if not isinstance(seg, dict):
            continue
        try:
            speaker_ids.add(int(seg.get("spk")))
        except Exception:
            continue

    return DiarizeResponse(
        segments=segments or [],
        duration_ms=duration_ms,
        speakers=len(speaker_ids),
    )
