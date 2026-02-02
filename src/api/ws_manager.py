"""WebSocket 连接管理"""
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from fastapi import WebSocket

from src.config import settings
from src.core.text_processor.stream_merger import StreamTextMerger
from src.core.llm.cancel_token import CancelToken
from src.utils.service_metrics import metrics

logger = logging.getLogger(__name__)


@dataclass
class ConnectionState:
    """WebSocket 连接状态"""
    is_speaking: bool = False
    asr_cache: Dict[str, Any] = field(default_factory=dict)
    vad_cache: Dict[str, Any] = field(default_factory=dict)
    chunk_interval: int = 10
    mode: str = "2pass"
    hotwords: Optional[str] = None
    text_merger: StreamTextMerger = field(default_factory=lambda: StreamTextMerger(
        overlap_chars=settings.stream_dedup_overlap,
        error_tolerance=settings.stream_dedup_tolerance,
    ))

    # 自适应分块相关
    chunk_size: int = field(default_factory=lambda: settings.ws_chunk_size)
    latency_samples: list = field(default_factory=list)
    adaptive_chunk_enable: bool = True

    # LLM 取消令牌
    cancel_token: CancelToken = field(default_factory=CancelToken)

    def reset(self):
        """重置状态"""
        self.is_speaking = False
        self.asr_cache = {}
        self.vad_cache = {}
        self.text_merger.reset()
        # 重置取消令牌
        self.cancel_token.reset()

    def update_latency(self, latency_ms: float):
        """更新延迟样本并自适应调整分块大小

        Args:
            latency_ms: 最近一次处理的延迟(毫秒)
        """
        if not self.adaptive_chunk_enable:
            return

        # 保留最近 10 个样本
        self.latency_samples.append(latency_ms)
        if len(self.latency_samples) > 10:
            self.latency_samples.pop(0)

        if len(self.latency_samples) < 3:
            return

        avg_latency = sum(self.latency_samples) / len(self.latency_samples)

        # 根据平均延迟调整分块大小
        # 延迟高 -> 减小分块  延迟低 -> 增大分块
        min_chunk = 4800   # 300ms @ 16kHz
        max_chunk = 19200  # 1200ms @ 16kHz

        if avg_latency > 500:  # 延迟 > 500ms，减小分块
            self.chunk_size = max(min_chunk, int(self.chunk_size * 0.8))
        elif avg_latency < 200:  # 延迟 < 200ms，增大分块
            self.chunk_size = min(max_chunk, int(self.chunk_size * 1.2))


class WebSocketManager:
    """WebSocket 连接管理器"""

    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}
        self.states: Dict[str, ConnectionState] = {}

    def connect(self, websocket: WebSocket, connection_id: str):
        """添加新连接"""
        self.connections[connection_id] = websocket
        self.states[connection_id] = ConnectionState()
        metrics.ws_connect()
        logger.info(f"WebSocket connected: {connection_id}")

    def disconnect(self, connection_id: str):
        """移除连接"""
        if connection_id in self.connections:
            del self.connections[connection_id]
        if connection_id in self.states:
            del self.states[connection_id]
        metrics.ws_disconnect()
        logger.info(f"WebSocket disconnected: {connection_id}")

    def get_state(self, connection_id: str) -> Optional[ConnectionState]:
        """获取连接状态"""
        return self.states.get(connection_id)

    async def send_json(self, connection_id: str, data: Dict[str, Any]):
        """发送 JSON 消息"""
        websocket = self.connections.get(connection_id)
        if websocket:
            try:
                await websocket.send_json(data)
            except Exception as e:
                logger.error(f"Failed to send message to {connection_id}: {e}")


ws_manager = WebSocketManager()
