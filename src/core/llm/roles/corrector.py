# coding: utf-8
"""纠错角色 - 专用于语音识别后处理纠错"""

from src.core.llm.roles.base import Role, RoleRegistry


@RoleRegistry.register
class CorrectorRole(Role):
    """纠错专家角色"""

    name = "corrector"
    description = "语音识别文本纠错专家"

    @property
    def system_prompt(self) -> str:
        return (
            "你是一个语音识别后处理专家。\n"
            "任务：修正语音识别文本中的错误。\n"
            "规则：\n"
            "- 仅修正明显的同音字/形近字错误\n"
            "- 不改变原意，不添加/删除内容\n"
            "- 保持原始格式和标点\n"
            "- 对不确定的地方保持原样\n"
            "- 直接输出修正后的文本，不要解释\n"
        )

    def format_user_input(self, text: str) -> str:
        return f"请修正以下语音识别文本中的错误：\n{text}"
