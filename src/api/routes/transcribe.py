"""转写 API 路由"""
import asyncio
import logging
from typing import Optional, List
from fastapi import APIRouter, File, UploadFile, Form, HTTPException

from src.api.schemas import (
    TranscribeResponse, SentenceInfo,
    BatchTranscribeResponse, BatchTranscribeItem
)
from src.api.dependencies import process_audio_file
from src.core.engine import transcription_engine
from src.utils.service_metrics import metrics

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

    metrics.increment_requests()

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

            # 更新指标
            audio_duration = len(audio_bytes) / 2 / 16000  # 16bit, 16kHz
            metrics.add_audio_duration(audio_duration)
            metrics.increment_success()

            return TranscribeResponse(
                code=0,
                text=result["text"],
                sentences=[SentenceInfo(**s) for s in result["sentences"]],
                transcript=result.get("transcript"),
                raw_text=result.get("raw_text"),
            )

    except Exception as e:
        metrics.increment_failure()
        logger.error(f"Transcription failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"转写失败: {str(e)}")


@router.post("/transcribe/batch", response_model=BatchTranscribeResponse)
async def transcribe_batch(
    files: List[UploadFile] = File(..., description="多个音频文件"),
    with_speaker: bool = Form(default=False, description="是否进行说话人识别"),
    apply_hotword: bool = Form(default=True, description="是否应用热词纠错"),
    apply_llm: bool = Form(default=False, description="是否应用LLM润色"),
    llm_role: str = Form(default="default", description="LLM角色"),
    hotwords: Optional[str] = Form(default=None, description="额外热词"),
    max_concurrent: int = Form(default=3, description="最大并发数"),
):
    """
    批量上传音频文件进行转写

    支持同时上传多个文件，并行处理。
    """
    if not files:
        raise HTTPException(status_code=400, detail="请上传至少一个音频文件")

    metrics.increment_requests()

    results: List[BatchTranscribeItem] = []
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_single_file(index: int, file: UploadFile) -> BatchTranscribeItem:
        """处理单个文件"""
        async with semaphore:
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

                    # 更新指标
                    audio_duration = len(audio_bytes) / 2 / 16000
                    metrics.add_audio_duration(audio_duration)

                    return BatchTranscribeItem(
                        index=index,
                        filename=file.filename or f"file_{index}",
                        success=True,
                        result=TranscribeResponse(
                            code=0,
                            text=result["text"],
                            sentences=[SentenceInfo(**s) for s in result["sentences"]],
                            transcript=result.get("transcript"),
                            raw_text=result.get("raw_text"),
                        ),
                    )
            except Exception as e:
                logger.error(f"Batch item {index} failed: {e}")
                return BatchTranscribeItem(
                    index=index,
                    filename=file.filename or f"file_{index}",
                    success=False,
                    error=str(e),
                )

    # 并行处理所有文件
    tasks = [process_single_file(i, f) for i, f in enumerate(files)]
    results = await asyncio.gather(*tasks)

    # 统计结果
    success_count = sum(1 for r in results if r.success)
    failed_count = len(results) - success_count

    if success_count > 0:
        metrics.increment_success()
    if failed_count > 0:
        metrics.increment_failure()

    return BatchTranscribeResponse(
        code=0 if failed_count == 0 else 1,
        total=len(files),
        success_count=success_count,
        failed_count=failed_count,
        results=results,
    )
