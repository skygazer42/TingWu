"""API 依赖注入"""
import os
from pathlib import Path
from typing import AsyncGenerator
import aiofiles
import ffmpeg
from fastapi import UploadFile, HTTPException

from src.config import settings


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
            yield audio_bytes
        except ffmpeg.Error as e:
            raise HTTPException(
                status_code=400,
                detail=f"音频处理失败: {e.stderr.decode() if e.stderr else str(e)}"
            )

    finally:
        if temp_path.exists():
            temp_path.unlink()
