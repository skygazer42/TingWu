"""API 请求/响应模式"""
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class SentenceInfo(BaseModel):
    """句子信息"""
    text: str = Field(..., description="句子文本")
    start: int = Field(..., description="开始时间 (毫秒)")
    end: int = Field(..., description="结束时间 (毫秒)")
    speaker: Optional[str] = Field(default=None, description="说话人标签")
    speaker_id: Optional[int] = Field(default=None, description="说话人 ID")


class SpeakerTurn(BaseModel):
    """说话人 turn/段落（合并后的说话人连续发言）"""
    speaker: str = Field(..., description="说话人标签")
    speaker_id: int = Field(..., description="说话人 ID")
    start: int = Field(..., description="开始时间 (毫秒)")
    end: int = Field(..., description="结束时间 (毫秒)")
    text: str = Field(..., description="该 turn 的文本")
    sentence_count: int = Field(default=1, description="包含句子数")


class TranscribeResponse(BaseModel):
    """转写响应"""
    code: int = Field(default=0, description="状态码 (0=成功)")
    text: str = Field(..., description="完整转写文本")
    text_accu: Optional[str] = Field(
        default=None,
        description="精确拼接文本（长音频分块去重更严格，适合回忆/会议转录）",
    )
    sentences: List[SentenceInfo] = Field(default=[], description="分句信息")
    speaker_turns: Optional[List[SpeakerTurn]] = Field(default=None, description="说话人 turn/段落")
    transcript: Optional[str] = Field(default=None, description="格式化转写稿")
    raw_text: Optional[str] = Field(default=None, description="原始文本 (未纠错)")


class BatchTranscribeItem(BaseModel):
    """批量转写结果项"""
    index: int = Field(..., description="文件索引")
    filename: str = Field(..., description="文件名")
    success: bool = Field(default=True, description="是否成功")
    result: Optional[TranscribeResponse] = Field(default=None, description="转写结果")
    error: Optional[str] = Field(default=None, description="错误信息")


class BatchTranscribeResponse(BaseModel):
    """批量转写响应"""
    code: int = Field(default=0, description="状态码")
    total: int = Field(..., description="总文件数")
    success_count: int = Field(..., description="成功数")
    failed_count: int = Field(..., description="失败数")
    results: List[BatchTranscribeItem] = Field(default=[], description="各文件结果")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态")
    version: str = Field(..., description="服务版本")


class MetricsResponse(BaseModel):
    """指标响应"""
    uptime_seconds: float = Field(..., description="服务运行时间")
    total_requests: int = Field(..., description="总请求数")
    successful_requests: int = Field(..., description="成功请求数")
    failed_requests: int = Field(..., description="失败请求数")
    total_audio_seconds: float = Field(..., description="处理音频总时长")
    avg_rtf: float = Field(..., description="平均实时因子")
    llm_cache_stats: Dict[str, Any] = Field(default={}, description="LLM 缓存统计")


class BackendCapabilities(BaseModel):
    """后端能力声明（用于前端探测/提示）"""
    supports_speaker: bool = Field(..., description="是否支持说话人识别/分离输出")
    supports_streaming: bool = Field(..., description="是否支持流式转写")
    supports_hotwords: bool = Field(..., description="是否支持热词（注入或后处理）")


class BackendInfoResponse(BaseModel):
    """后端信息（用于多端口/多模型部署场景的能力探测）"""
    backend: str = Field(..., description="配置的后端类型（ASR_BACKEND）")
    info: Dict[str, Any] = Field(default_factory=dict, description="backend.get_info() 输出（安全元信息）")
    capabilities: BackendCapabilities = Field(..., description="后端能力")
    speaker_unsupported_behavior: Literal["error", "fallback", "ignore"] = Field(
        ...,
        description="当 with_speaker=true 但后端不支持时的行为",
    )

