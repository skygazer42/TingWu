"""WebSocket 连接管理"""
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from fastapi import WebSocket

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

    def reset(self):
        """重置状态"""
        self.is_speaking = False
        self.asr_cache = {}
        self.vad_cache = {}


class WebSocketManager:
    """WebSocket 连接管理器"""

    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}
        self.states: Dict[str, ConnectionState] = {}

    def connect(self, websocket: WebSocket, connection_id: str):
        """添加新连接"""
        self.connections[connection_id] = websocket
        self.states[connection_id] = ConnectionState()
        logger.info(f"WebSocket connected: {connection_id}")

    def disconnect(self, connection_id: str):
        """移除连接"""
        if connection_id in self.connections:
            del self.connections[connection_id]
        if connection_id in self.states:
            del self.states[connection_id]
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
