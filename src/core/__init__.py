"""Core module.

Keep imports lightweight: avoid importing the full engine at package import time.
This helps unit tests import small helpers (e.g. text merge utilities) without
requiring the full runtime dependency stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["TranscriptionEngine", "transcription_engine"]

if TYPE_CHECKING:
    from src.core.engine import TranscriptionEngine, transcription_engine


def __getattr__(name: str) -> Any:
    if name in ("TranscriptionEngine", "transcription_engine"):
        from src.core.engine import TranscriptionEngine, transcription_engine

        return TranscriptionEngine if name == "TranscriptionEngine" else transcription_engine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + __all__))
