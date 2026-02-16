"""API 依赖注入"""
import os
from pathlib import Path
from typing import AsyncGenerator, Optional, Any, Dict
import numpy as np
import aiofiles
import ffmpeg
from fastapi import UploadFile, HTTPException

from src.config import settings
from src.core.audio import AudioPreprocessor


# 全局预处理器实例
_audio_preprocessor = None


def get_audio_preprocessor() -> AudioPreprocessor:
    """获取音频预处理器单例"""
    global _audio_preprocessor
    if _audio_preprocessor is None:
        _audio_preprocessor = AudioPreprocessor(
            target_db=settings.audio_normalize_target_db,
            silence_threshold_db=settings.audio_silence_threshold_db,
            normalize_enable=settings.audio_normalize_enable,
            trim_silence_enable=settings.audio_trim_silence_enable,
            denoise_enable=settings.audio_denoise_enable,
            denoise_prop=settings.audio_denoise_prop,
            denoise_backend=settings.audio_denoise_backend,
            vocal_separate_enable=settings.audio_vocal_separate_enable,
            vocal_separate_model=settings.audio_vocal_separate_model,
            device=settings.device,
            adaptive_enable=settings.audio_adaptive_preprocess,
            snr_threshold=settings.audio_snr_threshold,
        )
    return _audio_preprocessor


def _build_request_preprocessor(preprocess_options: Optional[Dict[str, Any]]) -> AudioPreprocessor:
    """Build a request-scoped AudioPreprocessor from settings + overrides.

    This must not mutate global `settings` or the singleton preprocessor instance.
    """
    if not preprocess_options:
        return get_audio_preprocessor()

    # Defaults from Settings.
    cfg = {
        "target_db": settings.audio_normalize_target_db,
        "silence_threshold_db": settings.audio_silence_threshold_db,
        "min_silence_ms": 500,
        "normalize_enable": settings.audio_normalize_enable,
        "trim_silence_enable": settings.audio_trim_silence_enable,
        "denoise_enable": settings.audio_denoise_enable,
        "denoise_prop": settings.audio_denoise_prop,
        "denoise_backend": settings.audio_denoise_backend,
        "vocal_separate_enable": settings.audio_vocal_separate_enable,
        "vocal_separate_model": settings.audio_vocal_separate_model,
        "device": settings.device,
        "adaptive_enable": settings.audio_adaptive_preprocess,
        "snr_threshold": settings.audio_snr_threshold,
        # New per-request toggle (defaults to True to match current behavior).
        "remove_dc_offset": True,
        # Accuracy-first filters (disabled by default unless explicitly enabled).
        "highpass_enable": False,
        "highpass_cutoff_hz": 80.0,
        "soft_limit_enable": False,
        "soft_limit_target": 0.98,
        "soft_limit_knee": 2.0,
    }

    # Map overrides (API keys match AudioPreprocessor kwargs).
    for k, v in preprocess_options.items():
        if k in cfg:
            cfg[k] = v

    return AudioPreprocessor(**cfg)


async def process_audio_file(
    file: UploadFile,
    *,
    preprocess_options: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[bytes, None]:
    """处理上传的音频文件，转换为 16kHz PCM"""
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    temp_path = settings.uploads_dir / f"temp_{os.urandom(8).hex()}{suffix}"

    try:
        async with aiofiles.open(temp_path, 'wb') as f:
            content = await file.read()
            await f.write(content)

        try:
            audio_bytes, _ = (
                ffmpeg
                .input(str(temp_path), threads=0)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=16000)
                .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
            )

            # Apply audio preprocessing (request-scoped overrides supported).
            preprocessor = _build_request_preprocessor(preprocess_options)

            # If preprocessing is effectively disabled, skip extra float conversion.
            should_process = (
                getattr(preprocessor, "remove_dc_offset", True)
                or getattr(preprocessor, "highpass_enable", False)
                or getattr(preprocessor, "soft_limit_enable", False)
                or preprocessor.normalize_enable
                or preprocessor.trim_silence_enable
                or preprocessor.denoise_enable
                or preprocessor.vocal_separate_enable
            )
            if should_process:
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                audio_array = preprocessor.process(audio_array, sample_rate=16000, validate=False)
                audio_bytes = (audio_array * 32768.0).astype(np.int16).tobytes()

            yield audio_bytes
        except ffmpeg.Error as e:
            raise HTTPException(
                status_code=400,
                detail=f"音频处理失败: {e.stderr.decode() if e.stderr else str(e)}"
            )

    finally:
        if temp_path.exists():
            temp_path.unlink()
