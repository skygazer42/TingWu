"""LLM 模块"""
from src.core.llm.client import LLMClient, LLMMessage
from src.core.llm.prompt_builder import PromptBuilder, DEFAULT_SYSTEM_PROMPT
from src.core.llm.roles import get_role, RoleRegistry

__all__ = [
    'LLMClient', 'LLMMessage', 'PromptBuilder', 'DEFAULT_SYSTEM_PROMPT',
    'get_role', 'RoleRegistry',
]
