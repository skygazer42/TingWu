# coding: utf-8
"""配置管理 API"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/config", tags=["config"])


class ConfigUpdateRequest(BaseModel):
    """配置更新请求"""
    updates: Dict[str, Any]


class ConfigResponse(BaseModel):
    """配置响应"""
    config: Dict[str, Any]


# 允许运行时修改的配置项
MUTABLE_CONFIG_KEYS = {
    # 纠错相关
    "text_correct_enable",
    "text_correct_backend",
    "correction_pipeline",
    "confidence_threshold",
    "confidence_fallback",
    # 热词相关
    "hotwords_threshold",
    "hotword_injection_enable",
    "hotword_injection_max",
    # LLM 相关
    "llm_enable",
    "llm_role",
    "llm_fulltext_enable",
    "llm_batch_size",
    "llm_context_sentences",
    # 文本后处理
    "filler_remove_enable",
    "filler_aggressive",
    "qj2bj_enable",
    "itn_enable",
    "itn_erhua_remove",
    "spacing_cjk_ascii_enable",
    "zh_convert_enable",
    "zh_convert_locale",
    "punc_convert_enable",
    "punc_restore_enable",
    "punc_merge_enable",
    # 音频预处理
    "audio_normalize_enable",
    "audio_denoise_enable",
    "audio_denoise_backend",
    "audio_vocal_separate_enable",
    "audio_trim_silence_enable",
}


def get_current_config() -> Dict[str, Any]:
    """获取当前可变配置"""
    return {
        key: getattr(settings, key)
        for key in MUTABLE_CONFIG_KEYS
        if hasattr(settings, key)
    }


@router.get("", response_model=ConfigResponse)
async def get_config():
    """获取当前配置

    返回所有可运行时修改的配置项及其当前值。
    """
    return ConfigResponse(config=get_current_config())


@router.get("/all")
async def get_all_config():
    """获取完整配置（包括只读项）

    返回所有配置项，包括服务启动后不可修改的项。
    """
    config = {}
    for key in dir(settings):
        if not key.startswith('_') and not callable(getattr(settings, key)):
            value = getattr(settings, key)
            # 转换 Path 为字符串
            if hasattr(value, '__fspath__'):
                value = str(value)
            config[key] = value
    return {"config": config, "mutable_keys": list(MUTABLE_CONFIG_KEYS)}


@router.post("", response_model=ConfigResponse)
async def update_config(request: ConfigUpdateRequest):
    """更新配置

    运行时更新配置项。仅支持 MUTABLE_CONFIG_KEYS 中的配置项。

    Args:
        request: 包含要更新的配置键值对

    Returns:
        更新后的配置
    """
    updates = request.updates
    updated = []
    rejected = []

    for key, value in updates.items():
        if key not in MUTABLE_CONFIG_KEYS:
            rejected.append(key)
            continue

        if not hasattr(settings, key):
            rejected.append(key)
            continue

        try:
            # 类型检查
            current_value = getattr(settings, key)
            if current_value is not None and type(value) != type(current_value):
                # 尝试类型转换
                value = type(current_value)(value)

            setattr(settings, key, value)
            updated.append(key)
            logger.info(f"Config updated: {key} = {value}")
        except Exception as e:
            logger.warning(f"Failed to update {key}: {e}")
            rejected.append(key)

    if rejected:
        logger.warning(f"Rejected config updates: {rejected}")

    # 如果更新了纠错相关配置，需要重新初始化引擎
    correction_keys = {"text_correct_enable", "text_correct_backend", "correction_pipeline"}
    if correction_keys & set(updated):
        try:
            from src.core.engine import transcription_engine
            # 重新创建后处理器
            from src.core.text_processor import TextPostProcessor
            transcription_engine.post_processor = TextPostProcessor.from_config(settings)
            logger.info("Transcription engine post-processor reloaded")
        except Exception as e:
            logger.error(f"Failed to reload engine: {e}")

    return ConfigResponse(config=get_current_config())


@router.post("/reload")
async def reload_engine():
    """重新加载引擎

    重新初始化转写引擎的各组件，应用最新配置。
    """
    try:
        from src.core.engine import transcription_engine
        from src.core.text_processor import TextPostProcessor

        # 重新加载后处理器
        transcription_engine.post_processor = TextPostProcessor.from_config(settings)

        # 重新加载热词
        transcription_engine.load_all()

        logger.info("Transcription engine reloaded")
        return {"status": "success", "message": "Engine reloaded"}
    except Exception as e:
        logger.error(f"Failed to reload engine: {e}")
        raise HTTPException(status_code=500, detail=str(e))
