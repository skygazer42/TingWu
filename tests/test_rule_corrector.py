"""Tests for RuleCorrector module"""
import pytest
import tempfile
from pathlib import Path

from src.core.hotword.rule_corrector import RuleCorrector


@pytest.fixture
def corrector():
    """Create a RuleCorrector with common rules"""
    c = RuleCorrector()
    c.update_rules("""
        毫安时 = mAh
        伏特 = V
        赫兹 = Hz
        摄氏度 = °C
    """)
    return c


class TestRuleCorrector:
    def test_initialization(self):
        """Test basic initialization"""
        c = RuleCorrector()
        assert len(c.patterns) == 0

    def test_update_rules(self, corrector):
        """Test updating rules"""
        assert len(corrector.patterns) == 4

    def test_update_rules_with_comments(self):
        """Test that comments are ignored"""
        c = RuleCorrector()
        count = c.update_rules("""
            # This is a comment
            毫安时 = mAh
            # Another comment
            伏特 = V
        """)
        assert count == 2

    def test_substitute_unit(self, corrector):
        """Test unit substitution"""
        result = corrector.substitute("这款手机有5000毫安时的电池")
        assert "5000mAh" in result

    def test_substitute_multiple(self, corrector):
        """Test multiple substitutions"""
        result = corrector.substitute("电压12伏特，频率50赫兹")
        assert "12V" in result
        assert "50Hz" in result

    def test_substitute_no_match(self, corrector):
        """Test text with no matching rules"""
        original = "今天天气不错"
        result = corrector.substitute(original)
        assert result == original

    def test_substitute_empty(self, corrector):
        """Test empty input"""
        assert corrector.substitute("") == ""
        assert corrector.substitute(None) == ""

    def test_regex_pattern(self):
        """Test regex pattern matching"""
        c = RuleCorrector()
        c.update_rules(r"(\d+)\s*度 = \1°")
        result = c.substitute("温度是25度")
        assert "25°" in result

    def test_substitute_with_info(self, corrector):
        """Test substitute with replacement info"""
        text, replacements = corrector.substitute_with_info("电池5000毫安时")
        assert "5000mAh" in text
        assert len(replacements) > 0
        assert replacements[0][0] == "毫安时"
        assert replacements[0][1] == "mAh"

    def test_load_rules_file(self):
        """Test loading rules from file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("测试 = TEST\n")
            f.write("示例 = EXAMPLE\n")
            temp_path = Path(f.name)

        try:
            c = RuleCorrector()
            count = c.load_rules_file(str(temp_path))
            assert count == 2

            result = c.substitute("这是一个测试示例")
            assert "TEST" in result
            assert "EXAMPLE" in result
        finally:
            temp_path.unlink()

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file"""
        c = RuleCorrector()
        count = c.load_rules_file("nonexistent.txt")
        assert count == 0

    def test_invalid_regex(self):
        """Test handling of invalid regex patterns"""
        c = RuleCorrector()
        c.update_rules(r"[invalid = replacement")  # Invalid regex
        # Should not raise, just skip the invalid pattern
        result = c.substitute("some text")
        assert result == "some text"
