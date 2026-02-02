"""LLM 生成取消令牌"""
import asyncio


class CancelToken:
    """LLM 生成取消令牌

    用于在流式生成过程中取消 LLM 请求。
    """

    def __init__(self):
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def reset(self):
        self._cancelled = False
