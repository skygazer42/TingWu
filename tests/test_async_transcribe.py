"""测试异步转写 API 和任务管理器"""
import pytest
import time
from unittest.mock import patch, MagicMock


class TestTaskManager:
    """任务管理器测试"""

    def test_submit_and_get_result(self):
        """测试提交和获取结果"""
        from src.core.task_manager import TaskManager, TaskStatus

        manager = TaskManager()
        manager.register_handler("test", lambda payload: {"result": payload["value"] * 2})
        manager.start()

        task_id = manager.submit("test", {"value": 21})
        assert task_id is not None

        # 等待处理
        for _ in range(50):
            result = manager.get_result(task_id, delete=False)
            if result and result.status == TaskStatus.COMPLETED:
                break
            time.sleep(0.1)

        result = manager.get_result(task_id)
        assert result.status == TaskStatus.COMPLETED
        assert result.result == {"result": 42}
        manager.stop()

    def test_failed_task(self):
        """测试失败任务"""
        from src.core.task_manager import TaskManager, TaskStatus

        def fail_handler(payload):
            raise ValueError("test error")

        manager = TaskManager()
        manager.register_handler("fail", fail_handler)
        manager.start()

        task_id = manager.submit("fail", {})

        for _ in range(50):
            result = manager.get_result(task_id, delete=False)
            if result and result.status == TaskStatus.FAILED:
                break
            time.sleep(0.1)

        result = manager.get_result(task_id)
        assert result.status == TaskStatus.FAILED
        assert "test error" in result.error
        manager.stop()

    def test_get_nonexistent_task(self):
        """测试获取不存在的任务"""
        from src.core.task_manager import TaskManager

        manager = TaskManager()
        assert manager.get_result("nonexistent") is None

    def test_get_status(self):
        """测试获取任务状态"""
        from src.core.task_manager import TaskManager, TaskStatus

        manager = TaskManager()
        manager.register_handler("slow", lambda p: time.sleep(0.1) or {"ok": True})
        manager.start()

        task_id = manager.submit("slow", {})
        # 刚提交应该是 PENDING 或 PROCESSING
        status = manager.get_status(task_id)
        assert status in (TaskStatus.PENDING, TaskStatus.PROCESSING)

        # 等待完成
        for _ in range(50):
            if manager.get_status(task_id) == TaskStatus.COMPLETED:
                break
            time.sleep(0.1)

        assert manager.get_status(task_id) == TaskStatus.COMPLETED
        manager.stop()

    def test_delete_on_get(self):
        """测试获取后删除"""
        from src.core.task_manager import TaskManager, TaskStatus

        manager = TaskManager()
        manager.register_handler("quick", lambda p: {"done": True})
        manager.start()

        task_id = manager.submit("quick", {})

        for _ in range(50):
            r = manager.get_result(task_id, delete=False)
            if r and r.status == TaskStatus.COMPLETED:
                break
            time.sleep(0.1)

        # 第一次获取并删除
        result = manager.get_result(task_id, delete=True)
        assert result is not None

        # 第二次应该为 None
        assert manager.get_result(task_id) is None
        manager.stop()


class TestAsyncTranscribeHelpers:
    """异步转写辅助函数测试"""

    def test_ms_to_srt_time(self):
        """测试时间格式转换"""
        from src.api.routes.async_transcribe import ms_to_srt_time

        assert ms_to_srt_time(0) == "00:00:00.000"
        assert ms_to_srt_time(1500) == "00:00:01.500"
        assert ms_to_srt_time(61000) == "00:01:01.000"
        assert ms_to_srt_time(3661500) == "01:01:01.500"

    def test_ms_to_srt_time_edge_cases(self):
        """测试时间格式转换边界情况"""
        from src.api.routes.async_transcribe import ms_to_srt_time

        assert ms_to_srt_time(999) == "00:00:00.999"
        assert ms_to_srt_time(3600000) == "01:00:00.000"
