"""API 请求/响应模式"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SentenceInfo(BaseModel):
    """句子信息"""
    text: str = Field(..., description="句子文本")
    start: int = Field(..., description="开始时间 (毫秒)")
    end: int = Field(..., description="结束时间 (毫秒)")
    speaker: Optional[str] = Field(default=None, description="说话人标签")
    speaker_id: Optional[int] = Field(default=None, description="说话人 ID")


class TranscribeResponse(BaseModel):
    """转写响应"""
    code: int = Field(default=0, description="状态码 (0=成功)")
    text: str = Field(..., description="完整转写文本")
    sentences: List[SentenceInfo] = Field(default=[], description="分句信息")
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

