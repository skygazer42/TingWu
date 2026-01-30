"""异步任务管理器 - 参考 FunASR_API

支持异步任务队列处理，适用于 URL 音频转写等耗时操作。
"""
import asyncio
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from queue import Queue
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"      # 等待处理
    PROCESSING = "processing"  # 处理中
    COMPLETED = "completed"   # 完成
    FAILED = "failed"        # 失败


@dataclass
class TaskResult:
    """任务结果"""
    task_id: str
    status: TaskStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class TaskItem:
    """任务项"""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)


class TaskManager:
    """
    异步任务管理器

    使用内存队列 + 线程池处理异步任务，
    结果存储在内存中（可扩展为 Redis/MySQL）。
    """

    def __init__(self, max_results: int = 1000, result_ttl: int = 3600):
        """
        初始化任务管理器

        Args:
            max_results: 最大结果缓存数
            result_ttl: 结果存活时间（秒）
        """
        self._queue: Queue[TaskItem] = Queue()
        self._results: Dict[str, TaskResult] = {}
        self._handlers: Dict[str, Callable] = {}
        self._max_results = max_results
        self._result_ttl = result_ttl
        self._lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False

    def register_handler(self, task_type: str, handler: Callable):
        """
        注册任务处理器

        Args:
            task_type: 任务类型
            handler: 处理函数，签名为 (payload: dict) -> dict
        """
        self._handlers[task_type] = handler
        logger.info(f"Registered task handler: {task_type}")

    def submit(self, task_type: str, payload: Dict[str, Any]) -> str:
        """
        提交任务

        Args:
            task_type: 任务类型
            payload: 任务数据

        Returns:
            任务 ID
        """
        task_id = uuid.uuid4().hex
        task = TaskItem(
            task_id=task_id,
            task_type=task_type,
            payload=payload
        )

        # 初始化结果
        with self._lock:
            self._results[task_id] = TaskResult(
                task_id=task_id,
                status=TaskStatus.PENDING,
                created_at=task.created_at
            )

        self._queue.put(task)
        logger.info(f"Task submitted: {task_id} ({task_type})")
        return task_id

    def get_result(self, task_id: str, delete: bool = True) -> Optional[TaskResult]:
        """
        获取任务结果

        Args:
            task_id: 任务 ID
            delete: 获取后是否删除

        Returns:
            任务结果，不存在则返回 None
        """
        with self._lock:
            result = self._results.get(task_id)
            if result and delete and result.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                del self._results[task_id]
            return result

    def get_status(self, task_id: str) -> Optional[TaskStatus]:
        """获取任务状态"""
        result = self.get_result(task_id, delete=False)
        return result.status if result else None

    def start(self):
        """启动任务处理器"""
        if self._running:
            return

        self._running = True
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
        logger.info("Task manager started")

    def stop(self):
        """停止任务处理器"""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        logger.info("Task manager stopped")

    def _worker(self):
        """后台工作线程"""
        while self._running:
            try:
                # 阻塞获取任务，超时 1 秒
                try:
                    task = self._queue.get(timeout=1)
                except:
                    continue

                self._process_task(task)
                self._cleanup_old_results()

            except Exception as e:
                logger.error(f"Worker error: {e}")

    def _process_task(self, task: TaskItem):
        """处理单个任务"""
        task_id = task.task_id
        logger.info(f"Processing task: {task_id} ({task.task_type})")

        # 更新状态为处理中
        with self._lock:
            if task_id in self._results:
                self._results[task_id].status = TaskStatus.PROCESSING

        try:
            handler = self._handlers.get(task.task_type)
            if not handler:
                raise ValueError(f"Unknown task type: {task.task_type}")

            result = handler(task.payload)

            # 更新结果
            with self._lock:
                if task_id in self._results:
                    self._results[task_id].status = TaskStatus.COMPLETED
                    self._results[task_id].completed_at = datetime.now()
                    self._results[task_id].result = result

            logger.info(f"Task completed: {task_id}")

        except Exception as e:
            logger.error(f"Task failed: {task_id} - {e}")
            with self._lock:
                if task_id in self._results:
                    self._results[task_id].status = TaskStatus.FAILED
                    self._results[task_id].completed_at = datetime.now()
                    self._results[task_id].error = str(e)

    def _cleanup_old_results(self):
        """清理过期结果"""
        now = datetime.now()
        with self._lock:
            to_delete = []
            for task_id, result in self._results.items():
                if result.completed_at:
                    age = (now - result.completed_at).total_seconds()
                    if age > self._result_ttl:
                        to_delete.append(task_id)

            # 超过最大数量时清理最旧的
            if len(self._results) > self._max_results:
                sorted_results = sorted(
                    self._results.items(),
                    key=lambda x: x[1].created_at
                )
                to_delete.extend([r[0] for r in sorted_results[:len(self._results) - self._max_results]])

            for task_id in set(to_delete):
                del self._results[task_id]


# 全局任务管理器
task_manager = TaskManager()
