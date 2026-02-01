"""API 依赖注入"""
import os
from pathlib import Path
from typing import AsyncGenerator
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
        )
    return _audio_preprocessor


async def process_audio_file(file: UploadFile) -> AsyncGenerator[bytes, None]:
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

            # 应用音频预处理
            if (settings.audio_normalize_enable or settings.audio_trim_silence_enable or
                settings.audio_denoise_enable or settings.audio_vocal_separate_enable):
                preprocessor = get_audio_preprocessor()
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
