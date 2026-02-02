"""Tests for AccuRAG module"""
import pytest
from src.core.hotword.phoneme import get_phoneme_info
from src.core.hotword.rag_accu import AccuRAG


@pytest.fixture
def accu_rag():
    """Create AccuRAG with test hotwords"""
    rag = AccuRAG(threshold=0.6)
    hotwords = {
        "撒贝宁": get_phoneme_info("撒贝宁"),
        "康辉": get_phoneme_info("康辉"),
        "东方财富": get_phoneme_info("东方财富"),
        "科大讯飞": get_phoneme_info("科大讯飞"),
        "麦当劳": get_phoneme_info("麦当劳"),
        "CapsWriter": get_phoneme_info("CapsWriter"),
    }
    rag.update_hotwords(hotwords)
    return rag


def test_accu_rag_initialization():
    """Test AccuRAG initialization"""
    rag = AccuRAG(threshold=0.7)
    assert rag.threshold == 0.7
    assert len(rag.hotwords) == 0


def test_accu_rag_update_hotwords(accu_rag):
    """Test updating hotwords"""
    assert len(accu_rag.hotwords) == 6


def test_accu_rag_exact_match(accu_rag):
    """Test exact match search"""
    results = accu_rag.search_from_text("科大讯飞的语音识别")
    assert len(results) > 0
    assert results[0][0] == "科大讯飞"
    assert results[0][1] > 0.9  # High score for exact match


def test_accu_rag_similar_match(accu_rag):
    """Test similar phoneme matching"""
    # 撒贝你 vs 撒贝宁 (前后鼻音)
    results = accu_rag.search_from_text("撒贝你主持节目")
    assert len(results) > 0
    top_match = next((r for r in results if r[0] == "撒贝宁"), None)
    assert top_match is not None
    assert top_match[1] > 0.7  # Good score for similar match


def test_accu_rag_candidate_filter(accu_rag):
    """Test searching with candidate filter"""
    results = accu_rag.search(
        get_phoneme_info("科大迅飞"),
        candidate_hws=["科大讯飞", "撒贝宁"],
        top_k=5
    )
    # Should only find among candidates
    for hw, score, _, _ in results:
        assert hw in ["科大讯飞", "撒贝宁"]


def test_accu_rag_threshold(accu_rag):
    """Test threshold filtering"""
    # High threshold should filter more
    accu_rag.threshold = 0.95
    results = accu_rag.search_from_text("完全不相关的文本")
    assert len(results) == 0


def test_accu_rag_no_threshold(accu_rag):
    """Test search without threshold"""
    results = accu_rag.search(
        get_phoneme_info("今天天气"),
        apply_threshold=False,
        top_k=3
    )
    # Should return results even below threshold
    assert len(results) <= 3


def test_accu_rag_english_hotword(accu_rag):
    """Test English hotword matching"""
    results = accu_rag.search_from_text("use caps riter to type")
    # Should find CapsWriter with partial match
    top_match = next((r for r in results if r[0] == "CapsWriter"), None)
    assert top_match is not None
    assert top_match[1] > 0.5


def test_accu_rag_position_info(accu_rag):
    """Test that position info is returned"""
    results = accu_rag.search_from_text("我想去麦当劳吃饭")
    assert len(results) > 0
    hw, score, start, end = results[0]
    assert hw == "麦当劳"
    assert start >= 0
    assert end > start
