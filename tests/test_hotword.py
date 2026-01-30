import pytest
from src.core.hotword.corrector import PhonemeCorrector

@pytest.fixture
def corrector():
    c = PhonemeCorrector(threshold=0.8, similar_threshold=0.6)
    c.update_hotwords("Claude\nBilibili\n麦当劳\n肯德基\nFunASR")
    return c

def test_chinese_correction(corrector):
    """测试中文热词纠错"""
    result = corrector.correct("我想去吃买当劳")
    assert "麦当劳" in result.text

def test_english_correction(corrector):
    """测试英文热词纠错"""
    result = corrector.correct("Hello klaude")
    assert "Claude" in result.text

def test_similar_phoneme_matching(corrector):
    """测试相似音素匹配 (得->德)"""
    result = corrector.correct("肯得鸡很好吃")
    assert "肯德基" in result.text

def test_no_false_positive(corrector):
    """测试不误纠正"""
    result = corrector.correct("今天天气不错")
    assert result.text == "今天天气不错"

def test_correction_result_structure(corrector):
    """测试返回结构"""
    result = corrector.correct("我要去买当劳")
    assert hasattr(result, 'text')
    assert hasattr(result, 'matches')
    assert hasattr(result, 'similars')

def test_bilibili_correction(corrector):
    """测试 Bilibili 纠错"""
    result = corrector.correct("喜欢刷bili bili")
    # Should recognize bilibili pattern
    assert "Bilibili" in result.text or "bilibili" in result.text.lower()

def test_update_hotwords():
    """测试更新热词"""
    c = PhonemeCorrector()
    count = c.update_hotwords("测试\n热词")
    assert count == 2

def test_empty_input(corrector):
    """测试空输入"""
    result = corrector.correct("")
    assert result.text == ""
