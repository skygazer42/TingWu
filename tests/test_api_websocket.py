import pytest
from unittest.mock import patch, Mock, MagicMock

def test_ws_connection_state():
    """测试 WebSocket 连接状态管理"""
    from src.api.ws_manager import ConnectionState

    state = ConnectionState()
    assert state.is_speaking == False
    assert state.asr_cache == {}

    state.is_speaking = True
    assert state.is_speaking == True

def test_ws_manager_add_remove():
    """测试连接管理器添加/移除"""
    from src.api.ws_manager import WebSocketManager

    manager = WebSocketManager()
    mock_ws = Mock()

    manager.connect(mock_ws, "test-id")
    assert "test-id" in manager.connections

    manager.disconnect("test-id")
    assert "test-id" not in manager.connections

def test_ws_manager_get_state():
    """测试获取连接状态"""
    from src.api.ws_manager import WebSocketManager

    manager = WebSocketManager()
    mock_ws = Mock()

    manager.connect(mock_ws, "test-id")
    state = manager.get_state("test-id")

    assert state is not None
    assert state.is_speaking == False

def test_connection_state_reset():
    """测试状态重置"""
    from src.api.ws_manager import ConnectionState

    state = ConnectionState()
    state.is_speaking = True
    state.asr_cache = {"key": "value"}

    state.reset()

    assert state.is_speaking == False
    assert state.asr_cache == {}

def test_connection_state_defaults():
    """测试默认值"""
    from src.api.ws_manager import ConnectionState

    state = ConnectionState()

    assert state.mode == "2pass"
    assert state.chunk_interval == 10
    assert state.hotwords is None
