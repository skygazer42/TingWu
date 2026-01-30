"""转写 API 路由"""
import logging
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional

from src.api.schemas import TranscribeResponse, SentenceInfo
from src.api.dependencies import process_audio_file
from src.core.engine import transcription_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["transcribe"])


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    file: UploadFile = File(..., description="音频文件"),
    with_speaker: bool = Form(default=False, description="是否进行说话人识别"),
    apply_hotword: bool = Form(default=True, description="是否应用热词纠错"),
    apply_llm: bool = Form(default=False, description="是否应用LLM润色"),
    llm_role: str = Form(default="default", description="LLM角色 (default/translator/code)"),
    hotwords: Optional[str] = Form(default=None, description="额外热词 (空格分隔)"),
):
    """
    上传音频文件进行转写

    支持的音频格式: wav, mp3, m4a, flac, ogg 等
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="请上传音频文件")

    try:
        async for audio_bytes in process_audio_file(file):
            result = await transcription_engine.transcribe_async(
                audio_bytes,
                with_speaker=with_speaker,
                apply_hotword=apply_hotword,
                apply_llm=apply_llm,
                llm_role=llm_role,
                hotwords=hotwords,
            )

            return TranscribeResponse(
                code=0,
                text=result["text"],
                sentences=[SentenceInfo(**s) for s in result["sentences"]],
                transcript=result.get("transcript"),
                raw_text=result.get("raw_text"),
            )

    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"转写失败: {str(e)}")
