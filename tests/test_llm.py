"""Tests for LLM module"""
import pytest
from src.core.llm.client import LLMClient, LLMMessage
from src.core.llm.prompt_builder import PromptBuilder, DEFAULT_SYSTEM_PROMPT


class TestLLMMessage:
    def test_message_creation(self):
        msg = LLMMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"


class TestLLMClient:
    def test_initialization_ollama(self):
        client = LLMClient(base_url="http://localhost:11434", model="qwen2.5:7b")
        assert client.backend == "ollama"
        assert client.model == "qwen2.5:7b"

    def test_initialization_openai(self):
        client = LLMClient(base_url="https://api.openai.com", model="gpt-4")
        assert client.backend == "openai"
        assert client.model == "gpt-4"

    def test_default_params(self):
        client = LLMClient()
        assert client.temperature == 0.7
        assert client.max_tokens == 4096
        assert client.timeout == 120.0


class TestPromptBuilder:
    def test_basic_build(self):
        builder = PromptBuilder()
        messages = builder.build("你好世界")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "用户输入：你好世界" in messages[1]["content"]

    def test_build_with_hotwords(self):
        builder = PromptBuilder()
        messages = builder.build(
            "科大迅飞",
            hotwords=["科大讯飞", "麦当劳"]
        )
        content = messages[1]["content"]
        assert "热词列表：" in content
        assert "科大讯飞" in content
        assert "麦当劳" in content

    def test_build_with_rectify_context(self):
        builder = PromptBuilder()
        rectify = "纠错历史：\n- 买当劳 => 麦当劳"
        messages = builder.build("买当劳", rectify_context=rectify)
        content = messages[1]["content"]
        assert "纠错历史：" in content
        assert "买当劳 => 麦当劳" in content

    def test_build_with_all_context(self):
        builder = PromptBuilder()
        messages = builder.build(
            "科大迅飞买当劳",
            hotwords=["科大讯飞", "麦当劳"],
            rectify_context="纠错历史：\n- 买当劳 => 麦当劳"
        )
        content = messages[1]["content"]
        assert "热词列表：" in content
        assert "纠错历史：" in content
        assert "用户输入：" in content

    def test_history_management(self):
        builder = PromptBuilder()
        builder.add_to_history("input1", "output1")
        builder.add_to_history("input2", "output2")

        messages = builder.build("input3", include_history=True)
        # system + 2 history pairs + current user
        assert len(messages) == 6

    def test_clear_history(self):
        builder = PromptBuilder()
        builder.add_to_history("input1", "output1")
        builder.clear_history()

        messages = builder.build("input2")
        assert len(messages) == 2  # system + user only

    def test_no_history(self):
        builder = PromptBuilder()
        builder.add_to_history("input1", "output1")

        messages = builder.build("input2", include_history=False)
        assert len(messages) == 2  # system + user only

    def test_custom_system_prompt(self):
        builder = PromptBuilder(system_prompt="自定义提示词")
        messages = builder.build("test")
        assert messages[0]["content"] == "自定义提示词"

    def test_default_system_prompt(self):
        assert "高级智能复读机" in DEFAULT_SYSTEM_PROMPT
        assert "语气词" in DEFAULT_SYSTEM_PROMPT
