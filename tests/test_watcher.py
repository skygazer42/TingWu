"""测试热词文件监视器"""
import pytest
import tempfile
import time
import threading
from pathlib import Path
from unittest.mock import Mock, MagicMock

# 检查 watchdog 是否可用
try:
    import watchdog
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False


@pytest.fixture
def temp_watch_dir():
    """创建临时监控目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestHotwordWatcher:
    """热词文件监视器测试"""

    @pytest.mark.skipif(not WATCHDOG_AVAILABLE, reason="watchdog not installed")
    def test_watcher_initialization(self, temp_watch_dir):
        """测试监视器初始化"""
        from src.core.hotword.watcher import HotwordWatcher

        callback = Mock()
        watcher = HotwordWatcher(
            watch_dir=str(temp_watch_dir),
            on_hotwords_change=callback,
        )
        assert watcher._callbacks.get("hotwords.txt") == callback

    @pytest.mark.skipif(not WATCHDOG_AVAILABLE, reason="watchdog not installed")
    def test_watcher_start_stop(self, temp_watch_dir):
        """测试监视器启动和停止"""
        from src.core.hotword.watcher import HotwordWatcher

        watcher = HotwordWatcher(
            watch_dir=str(temp_watch_dir),
            on_hotwords_change=Mock(),
        )
        watcher.start()
        assert watcher._observer is not None
        watcher.stop()
        assert watcher._observer is None

    @pytest.mark.skipif(not WATCHDOG_AVAILABLE, reason="watchdog not installed")
    def test_watcher_context_manager(self, temp_watch_dir):
        """测试上下文管理器"""
        from src.core.hotword.watcher import HotwordWatcher

        with HotwordWatcher(
            watch_dir=str(temp_watch_dir),
            on_hotwords_change=Mock(),
        ) as watcher:
            assert watcher._observer is not None

    @pytest.mark.skipif(not WATCHDOG_AVAILABLE, reason="watchdog not installed")
    def test_file_handler_debounce(self, temp_watch_dir):
        """测试防抖机制"""
        from src.core.hotword.watcher import HotwordFileHandler

        callback = Mock()
        handler = HotwordFileHandler(
            watched_files={"hotwords.txt"},
            callback=callback,
            debounce_delay=0.1,  # 短延迟用于测试
        )

        # 模拟多次文件变化
        for _ in range(5):
            handler._handle_event(str(temp_watch_dir / "hotwords.txt"))

        # 等待防抖
        time.sleep(0.2)

        # 应该只触发一次回调
        assert callback.call_count == 1

    @pytest.mark.skipif(not WATCHDOG_AVAILABLE, reason="watchdog not installed")
    def test_file_handler_filters_unwatched(self, temp_watch_dir):
        """测试过滤未监控文件"""
        from src.core.hotword.watcher import HotwordFileHandler

        callback = Mock()
        handler = HotwordFileHandler(
            watched_files={"hotwords.txt"},
            callback=callback,
            debounce_delay=0.1,
        )

        # 触发未监控的文件
        handler._handle_event(str(temp_watch_dir / "other.txt"))

        time.sleep(0.2)

        # 不应触发回调
        assert callback.call_count == 0


class TestGlobalWatcher:
    """全局监视器测试"""

    @pytest.mark.skipif(not WATCHDOG_AVAILABLE, reason="watchdog not installed")
    def test_setup_and_stop(self, temp_watch_dir):
        """测试全局设置和停止"""
        from src.core.hotword.watcher import (
            setup_hotword_watcher,
            stop_hotword_watcher,
            get_hotword_watcher,
        )

        watcher = setup_hotword_watcher(
            watch_dir=str(temp_watch_dir),
            on_hotwords_change=Mock(),
        )
        assert get_hotword_watcher() is watcher
        assert watcher._observer is not None

        stop_hotword_watcher()
        assert get_hotword_watcher() is None
