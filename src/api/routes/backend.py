"""Backend info API (capabilities discovery).

Used by the frontend for multi-container / multi-port deployments to probe:
- which backend is configured for this TingWu instance
- whether speaker diarization is supported
"""

import logging

from fastapi import APIRouter

from src.api.schemas import BackendCapabilities, BackendInfoResponse
from src.config import settings
from src.core.engine import model_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["backend"])


def _cap_bool(value: object) -> bool:
    # Many backends implement supports_* as bool properties. In tests we often use
    # MagicMock, which should not accidentally evaluate to True.
    return value if isinstance(value, bool) else False


@router.get("/backend", response_model=BackendInfoResponse)
async def get_backend_info() -> BackendInfoResponse:
    backend = model_manager.backend

    info = {}
    try:
        raw_info = backend.get_info()
        if isinstance(raw_info, dict):
            info = raw_info
    except Exception as e:
        logger.warning(f"Failed to read backend.get_info(): {e}")

    capabilities = BackendCapabilities(
        supports_speaker=_cap_bool(getattr(backend, "supports_speaker", False)),
        supports_streaming=_cap_bool(getattr(backend, "supports_streaming", False)),
        supports_hotwords=_cap_bool(getattr(backend, "supports_hotwords", False)),
    )

    return BackendInfoResponse(
        backend=settings.asr_backend,
        info=info,
        capabilities=capabilities,
        speaker_unsupported_behavior=settings.speaker_unsupported_behavior_effective,
    )

