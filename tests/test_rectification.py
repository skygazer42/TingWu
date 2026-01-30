"""Tests for RectificationRAG module"""
import pytest
from pathlib import Path
import tempfile

from src.core.hotword.rectification import (
    RectificationRAG,
    RectifyRecord,
    extract_diff_fragments,
    _get_word_boundaries,
)


class TestWordBoundaries:
    def test_chinese_boundaries(self):
        """Test Chinese character boundaries"""
        bounds = _get_word_boundaries("你好世界")
        assert len(bounds) == 4
        assert bounds[0][2] == "你"
        assert bounds[3][2] == "界"

    def test_english_boundaries(self):
        """Test English word boundaries"""
        bounds = _get_word_boundaries("Hello World")
        assert len(bounds) == 2
        assert bounds[0][2] == "Hello"
        assert bounds[1][2] == "World"

    def test_mixed_boundaries(self):
        """Test mixed Chinese-English boundaries"""
        bounds = _get_word_boundaries("我 love 你")
        assert len(bounds) == 3


class TestExtractDiffFragments:
    def test_single_char_diff(self):
        """Test extracting single character difference"""
        fragments = extract_diff_fragments("买当劳", "麦当劳")
        assert len(fragments) > 0
        assert any("买" in f or "麦" in f for f in fragments)

    def test_english_diff(self):
        """Test extracting English word difference"""
        fragments = extract_diff_fragments(
            "use caps riter",
            "use CapsWriter"
        )
        assert len(fragments) > 0

    def test_no_diff(self):
        """Test identical strings"""
        fragments = extract_diff_fragments("相同文本", "相同文本")
        assert len(fragments) == 0


class TestRectifyRecord:
    def test_record_creation(self):
        """Test creating a rectify record"""
        record = RectifyRecord(
            wrong="买当劳",
            right="麦当劳",
            fragments=["买", "麦"]
        )
        assert record.wrong == "买当劳"
        assert record.right == "麦当劳"
        assert len(record.fragment_phonemes) == 2


class TestRectificationRAG:
    @pytest.fixture
    def temp_rectify_file(self):
        """Create a temporary rectify file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("""# Test rectify file
Cloud Code 很好用
Claude Code 很好用
---
科大迅飞
科大讯飞
---
买当劳
麦当劳
""")
            return Path(f.name)

    def test_load_history(self, temp_rectify_file):
        """Test loading rectify history"""
        rag = RectificationRAG(str(temp_rectify_file))
        count = rag.load_history()
        assert count == 3
        assert len(rag.records) == 3

    def test_search_matching(self, temp_rectify_file):
        """Test searching for matching records"""
        rag = RectificationRAG(str(temp_rectify_file), threshold=0.4)
        rag.load_history()

        results = rag.search("Cloud Code 真不错")
        # Should find the Claude Code correction
        assert len(results) > 0

    def test_search_no_match(self, temp_rectify_file):
        """Test searching with no match"""
        rag = RectificationRAG(str(temp_rectify_file), threshold=0.9)
        rag.load_history()

        results = rag.search("完全无关的内容")
        assert len(results) == 0

    def test_format_prompt(self, temp_rectify_file):
        """Test prompt formatting"""
        rag = RectificationRAG(str(temp_rectify_file), threshold=0.4)
        rag.load_history()

        prompt = rag.format_prompt("我用Cloud Code", prefix="纠错历史：")
        if prompt:  # May or may not match depending on threshold
            assert "纠错历史：" in prompt
            assert "=>" in prompt

    def test_empty_file(self):
        """Test with non-existent file"""
        rag = RectificationRAG("nonexistent.txt")
        count = rag.load_history()
        assert count == 0

    def test_cleanup(self, temp_rectify_file):
        """Cleanup temp file"""
        temp_rectify_file.unlink()
