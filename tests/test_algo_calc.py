"""Tests for algo_calc module"""
import pytest
from src.core.hotword.phoneme import Phoneme, get_phoneme_info
from src.core.hotword.algo_calc import (
    lcs_length,
    get_phoneme_cost,
    find_best_match,
    fuzzy_substring_score,
    fuzzy_substring_search_constrained,
)


class TestLcsLength:
    def test_identical_strings(self):
        assert lcs_length("hello", "hello") == 5

    def test_empty_string(self):
        assert lcs_length("", "hello") == 0
        assert lcs_length("hello", "") == 0

    def test_partial_match(self):
        assert lcs_length("abc", "adc") == 2
        assert lcs_length("caps", "capswriter") == 4

    def test_no_match(self):
        assert lcs_length("abc", "xyz") == 0


class TestGetPhonemeCost:
    def test_identical_phonemes(self):
        p1 = Phoneme("zh", "zh", is_word_start=True)
        p2 = Phoneme("zh", "zh", is_word_start=True)
        assert get_phoneme_cost(p1, p2) == 0.0

    def test_similar_chinese_phonemes(self):
        # 前后鼻音
        p1 = Phoneme("an", "zh")
        p2 = Phoneme("ang", "zh")
        assert get_phoneme_cost(p1, p2) == 0.5

        # 平翘舌
        p1 = Phoneme("z", "zh")
        p2 = Phoneme("zh", "zh")
        assert get_phoneme_cost(p1, p2) == 0.5

    def test_different_languages(self):
        p1 = Phoneme("test", "en")
        p2 = Phoneme("test", "zh")
        assert get_phoneme_cost(p1, p2) == 1.0

    def test_english_similarity(self):
        p1 = Phoneme("caps", "en")
        p2 = Phoneme("cap", "en")
        cost = get_phoneme_cost(p1, p2)
        assert 0.0 < cost < 1.0  # Partial match


class TestFindBestMatch:
    def test_exact_match(self):
        main = get_phoneme_info("你好世界")
        sub = get_phoneme_info("世界")
        score, start, end = find_best_match(main, sub)
        assert score > 0.9

    def test_similar_match(self):
        main = get_phoneme_info("撒贝你主持节目")
        sub = get_phoneme_info("撒贝宁")
        score, start, end = find_best_match(main, sub)
        assert score > 0.7  # Should find similar match

    def test_no_match(self):
        main = get_phoneme_info("今天天气")
        sub = get_phoneme_info("再见")
        score, start, end = find_best_match(main, sub)
        assert score < 0.5


class TestFuzzySubstringScore:
    def test_exact_substring(self):
        hw = [p.info for p in get_phoneme_info("麦当劳")]
        inp = [p.info for p in get_phoneme_info("我想去麦当劳")]
        score = fuzzy_substring_score(hw, inp)
        assert score > 0.9

    def test_similar_substring(self):
        hw = [p.info for p in get_phoneme_info("麦当劳")]
        inp = [p.info for p in get_phoneme_info("我想去买当劳")]  # 麦->买
        score = fuzzy_substring_score(hw, inp)
        assert score > 0.6


class TestFuzzySubstringSearchConstrained:
    def test_find_match_with_boundary(self):
        hw = [p.info for p in get_phoneme_info("科大讯飞")]
        inp = [p.info for p in get_phoneme_info("科大迅飞的语音识别")]  # 讯->迅
        results = fuzzy_substring_search_constrained(hw, inp, threshold=0.6)
        assert len(results) > 0
        assert results[0][0] > 0.7  # High score

    def test_no_match_below_threshold(self):
        hw = [p.info for p in get_phoneme_info("完全不同")]
        inp = [p.info for p in get_phoneme_info("今天天气")]
        results = fuzzy_substring_search_constrained(hw, inp, threshold=0.8)
        assert len(results) == 0
