"""测试 LLM 角色系统"""
import pytest


class TestRoleSystem:
    """角色系统测试"""

    def test_default_role(self):
        """测试默认角色"""
        from src.core.llm.roles import get_role, DefaultRole

        role = get_role("default")
        assert role is not None
        assert isinstance(role, DefaultRole)
        assert role.name == "default"
        assert "复读机" in role.system_prompt

    def test_translator_role(self):
        """测试翻译角色"""
        from src.core.llm.roles import get_role, TranslatorRole

        role = get_role("translator")
        assert role is not None
        assert isinstance(role, TranslatorRole)
        assert role.name == "translator"
        assert "翻译" in role.system_prompt

    def test_code_role(self):
        """测试代码角色"""
        from src.core.llm.roles import get_role, CodeRole

        role = get_role("code")
        assert role is not None
        assert isinstance(role, CodeRole)
        assert role.name == "code"
        assert "代码" in role.system_prompt

    def test_fallback_to_default(self):
        """测试未知角色回退到默认"""
        from src.core.llm.roles import get_role, DefaultRole

        role = get_role("unknown_role")
        assert role is not None
        assert isinstance(role, DefaultRole)

    def test_role_registry(self):
        """测试角色注册表"""
        from src.core.llm.roles import RoleRegistry

        roles = RoleRegistry.list_roles()
        assert "default" in roles
        assert "translator" in roles
        assert "code" in roles

    def test_role_format_user_input(self):
        """测试角色格式化用户输入"""
        from src.core.llm.roles import get_role

        default_role = get_role("default")
        assert "用户输入：" in default_role.format_user_input("测试")

        translator_role = get_role("translator")
        assert "请翻译：" in translator_role.format_user_input("测试")

        code_role = get_role("code")
        assert "代码输入：" in code_role.format_user_input("测试")


class TestRoleIntegration:
    """角色集成测试"""

    def test_role_with_prompt_builder(self):
        """测试角色与提示词构建器集成"""
        from src.core.llm.roles import get_role
        from src.core.llm.prompt_builder import PromptBuilder

        role = get_role("translator")
        builder = PromptBuilder(system_prompt=role.system_prompt)

        messages = builder.build("Hello World")
        assert len(messages) >= 2
        assert messages[0]["role"] == "system"
        assert "翻译" in messages[0]["content"]
