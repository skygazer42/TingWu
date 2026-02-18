from __future__ import annotations

import io
import logging
import threading
import wave
from typing import Dict, List, Optional


logger = logging.getLogger(__name__)


class DiarizerEngine:
    """External diarization engine (pyannote), with lazy imports.

    This module must stay importable without heavyweight ML deps so unit tests for
    the main TingWu service can run without installing `pyannote.audio`.
    """

    def __init__(
        self,
        *,
        model_id: str,
        device: str = "cuda",
        hf_token: Optional[str] = None,
    ) -> None:
        self.model_id = str(model_id or "").strip()
        if not self.model_id:
            raise ValueError("model_id must be non-empty")

        self.device = str(device or "cpu").strip() or "cpu"
        self.hf_token = str(hf_token).strip() if hf_token else None

        self._load_lock = threading.Lock()
        self._loaded = False
        self._pipeline = None

    def load(self) -> None:
        if self._loaded:
            return

        with self._load_lock:
            if self._loaded:
                return

            # Lazy heavy imports.
            from pyannote.audio import Pipeline  # type: ignore[import-not-found]

            import torch

            pipeline = Pipeline.from_pretrained(self.model_id, use_auth_token=self.hf_token)
            if pipeline is None:
                raise RuntimeError("failed to load diarization pipeline")

            # Best-effort: move to requested device; fall back silently.
            try:
                pipeline.to(torch.device(self.device))
            except Exception:
                try:
                    pipeline.to(self.device)
                except Exception as e:
                    logger.warning(f"Failed to move diarizer pipeline to device={self.device!r}: {e}")

            self._pipeline = pipeline
            self._loaded = True

    def diarize(self, wav_bytes: bytes) -> List[Dict[str, int]]:
        """Run diarization and return raw segments [{spk,start,end}, ...] in ms."""
        if not wav_bytes:
            return []

        self.load()
        if self._pipeline is None:
            raise RuntimeError("diarizer pipeline not loaded")

        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sample_rate = wf.getframerate() or 0
            nframes = wf.getnframes()
            pcm = wf.readframes(nframes)

        if channels != 1:
            raise ValueError("Only mono WAV is supported")
        if sampwidth != 2:
            raise ValueError("Only 16-bit PCM WAV is supported")
        if sample_rate <= 0:
            raise ValueError("Invalid WAV sample_rate")

        # Convert PCM16LE to float32 waveform tensor expected by pyannote.
        import numpy as np
        import torch

        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32, copy=False)
        if audio.size == 0:
            return []
        audio = audio / 32768.0
        waveform = torch.from_numpy(audio).unsqueeze(0)

        diarization = self._pipeline({"waveform": waveform, "sample_rate": sample_rate})

        items = []
        try:
            it = diarization.itertracks(yield_label=True)
        except Exception:
            it = []

        for segment, _track, label in it:
            try:
                start_s = float(segment.start)
                end_s = float(segment.end)
            except Exception:
                continue
            items.append((start_s, end_s, str(label)))

        items.sort(key=lambda x: (x[0], x[1], x[2]))

        speaker_mapping: Dict[str, int] = {}
        out: List[Dict[str, int]] = []
        for start_s, end_s, label in items:
            if label not in speaker_mapping:
                speaker_mapping[label] = len(speaker_mapping)
            spk_id = speaker_mapping[label]

            start_ms = int(start_s * 1000)
            end_ms = int(end_s * 1000)
            if start_ms < 0:
                start_ms = 0
            if end_ms <= start_ms:
                continue

            out.append({"spk": spk_id, "start": start_ms, "end": end_ms})

        return out

