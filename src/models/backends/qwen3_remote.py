"""Remote Qwen3-ASR backend (vLLM OpenAI-compatible transcription API)."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx

from src.models.backends.base import ASRBackend
from src.models.backends.remote_utils import audio_input_to_wav_bytes

logger = logging.getLogger(__name__)


class Qwen3RemoteBackend(ASRBackend):
    """Call a remote Qwen3-ASR server via `/v1/audio/transcriptions`."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout_s: float = 60.0,
    ):
        self.base_url = (base_url or "").rstrip("/")
        self.model = model
        self.api_key = api_key or ""
        self.timeout_s = float(timeout_s)
        self._client: Optional[httpx.Client] = None

    def load(self) -> None:
        # Lazy init so tests can patch httpx.Client.post without needing a real server.
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout_s)

    @property
    def supports_streaming(self) -> bool:
        return False

    @property
    def supports_speaker(self) -> bool:
        return False

    def transcribe(self, audio_input, hotwords: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        self.load()
        assert self._client is not None

        wav_bytes, _duration_s = audio_input_to_wav_bytes(audio_input)

        url = f"{self.base_url}/v1/audio/transcriptions"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # OpenAI transcription API uses multipart form fields:
        # - file: audio bytes
        # - model: served model id/name
        # - prompt: optional context (we pass hotwords)
        data = {"model": self.model}
        if hotwords:
            data["prompt"] = hotwords

        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}

        resp = self._client.post(url, data=data, files=files, headers=headers)
        resp.raise_for_status()

        payload = resp.json()
        text = ""
        if isinstance(payload, dict):
            text = str(payload.get("text") or payload.get("transcript") or "")
        else:
            text = str(payload)

        return {
            "text": text,
            "sentence_info": [],
        }

