"""热词纠错器 - 两阶段检索优化版

基于 CapsWriter-Offline 的两阶段检索策略：
1. FastRAG 粗筛：使用倒排索引 + Numba JIT 快速过滤候选
2. 精确计算：使用模糊音权重 + 边界约束进行精确匹配
"""
import os
import threading
from typing import List, Dict, Tuple, NamedTuple, Optional

from src.core.hotword.phoneme import Phoneme, get_phoneme_info, SIMILAR_PHONEMES
from src.core.hotword.rag import FastRAG
from src.core.hotword.algo_calc import fuzzy_substring_search_constrained


class MatchResult(NamedTuple):
    """匹配结果"""
    start: int      # 字符起始位置
    end: int        # 字符结束位置
    score: float    # 匹配分数
    hotword: str    # 热词
    original: str   # 原始文本片段


class CorrectionResult(NamedTuple):
    """纠错结果"""
    text: str                                   # 纠错后的文本
    matches: List[Tuple[str, str, float]]       # [(原词, 热词, 分数), ...]
    similars: List[Tuple[str, str, float]]      # [(原词, 热词, 分数), ...]


class PhonemeCorrector:
    """
    基于音素的热词纠错器 - 两阶段检索

    使用两阶段检索策略：
    1. FastRAG 粗筛：使用倒排索引快速过滤候选
    2. 精确计算：使用 fuzzy_substring_search_constrained 进行边界约束匹配

    优势：
    - FastRAG 倒排索引减少 90% 计算量
    - 边界约束搜索保证只在词边界匹配
    - 模糊音权重提高相似音匹配精度
    """

    def __init__(self, threshold: float = 0.8, similar_threshold: float = None, cache_size: int = 1000):
        """
        初始化纠错器

        Args:
            threshold: 替换阈值 (高于此分数才替换)
            similar_threshold: 相似度阈值 (高于此分数加入相似列表)
            cache_size: 缓存大小 (最大缓存条目数)
        """
        self.threshold = threshold
        self.similar_threshold = similar_threshold if similar_threshold is not None else (threshold - 0.2)
        self.hotwords: Dict[str, List[Phoneme]] = {}
        self.fast_rag = FastRAG(threshold=min(self.threshold, self.similar_threshold) - 0.1)
        self._lock = threading.Lock()
        self._cache: Dict[str, CorrectionResult] = {}
        self._cache_size = cache_size

    def update_hotwords(self, text: str) -> int:
        """从文本更新热词"""
        lines = [l.strip() for l in text.splitlines()
                 if l.strip() and not l.strip().startswith('#')]

        new_hotwords = {}
        for hw in lines:
            phs = get_phoneme_info(hw)
            if phs:
                new_hotwords[hw] = phs

        with self._lock:
            self.hotwords = new_hotwords
            self.fast_rag = FastRAG(threshold=min(self.threshold, self.similar_threshold) - 0.1)
            self.fast_rag.add_hotwords(new_hotwords)
            self._cache.clear()  # 热词变化时清空缓存

        return len(new_hotwords)

    def load_hotwords_file(self, path: str) -> int:
        """从文件加载热词"""
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return self.update_hotwords(f.read())
        return 0

    def correct(self, text: str, top_k: int = 10) -> CorrectionResult:
        """
        执行热词纠错

        Args:
            text: 输入文本
            top_k: 返回相似热词的最大数量

        Returns:
            CorrectionResult(纠错后文本, 匹配列表, 相似列表)
        """
        if not text or not self.hotwords:
            return CorrectionResult(text or "", [], [])

        # 缓存检查
        if text in self._cache:
            return self._cache[text]

        input_phs = get_phoneme_info(text)
        if not input_phs:
            return CorrectionResult(text, [], [])

        with self._lock:
            # 阶段1: FastRAG 粗筛
            fast_results = self.fast_rag.search(input_phs, top_k=100)

            # 阶段2: 精确匹配 (带边界约束)
            input_processed = [p.info for p in input_phs]
            matches, similars = self._find_matches(text, fast_results, input_processed)

        # 阶段3: 冲突解决与替换
        new_text, final_matches = self._resolve_and_replace(text, matches)

        result = CorrectionResult(
            new_text,
            final_matches,
            similars[:top_k]
        )

        # 写入缓存
        if len(self._cache) < self._cache_size:
            self._cache[text] = result

        return result

    def _find_matches(
        self,
        text: str,
        fast_results: List[Tuple[str, float]],
        input_processed: List[Tuple]
    ) -> Tuple[List[MatchResult], List[Tuple[str, str, float]]]:
        """
        精细匹配逻辑：边界约束的模糊搜索

        Args:
            text: 原始文本
            fast_results: FastRAG 粗筛结果 [(热词, 分数), ...]
            input_processed: 输入音素 info 元组列表

        Returns:
            (matches, similars)
        """
        matches = []
        similars = []
        search_threshold = min(self.threshold, self.similar_threshold) - 0.1

        for hw, fast_score in fast_results:
            hw_phonemes = self.hotwords[hw]
            hw_compare = [p.info for p in hw_phonemes]

            # 使用边界约束搜索
            found_segments = fuzzy_substring_search_constrained(
                hw_compare, input_processed, threshold=search_threshold
            )

            for score, start_phon_idx, end_phon_idx in found_segments:
                # 从 input_processed 获取字符位置
                char_start = input_processed[start_phon_idx][5]
                char_end = input_processed[end_phon_idx - 1][6]
                original = text[char_start:char_end]

                res = MatchResult(char_start, char_end, score, hw, original)

                # 分类到 matches 和 similars
                if score >= self.threshold:
                    matches.append(res)

                if score >= self.similar_threshold:
                    similars.append((original, hw, round(score, 3)))

        # 相似列表去重与排序
        seen_hw = set()
        final_similars = []
        similars.sort(key=lambda x: (x[2], len(x[1])), reverse=True)

        for original, hw, score in similars:
            if hw not in seen_hw:
                final_similars.append((original, hw, score))
                seen_hw.add(hw)

        return matches, final_similars

    def _resolve_and_replace(
        self,
        text: str,
        matches: List[MatchResult]
    ) -> Tuple[str, List[Tuple[str, str, float]]]:
        """
        冲突解决与文本替换

        Args:
            text: 原始文本
            matches: 匹配结果列表

        Returns:
            (替换后文本, [(原词, 热词, 分数), ...])
        """
        # 分数优先 > 长度优先
        matches.sort(key=lambda x: (x.score, x.end - x.start), reverse=True)

        final_matches = []
        occupied_ranges = []

        for m in matches:
            if m.score < self.threshold:
                continue

            # 检查重叠
            is_overlap = False
            for r_start, r_end in occupied_ranges:
                if not (m.end <= r_start or m.start >= r_end):
                    is_overlap = True
                    break

            if not is_overlap:
                # 检查是否真的有变化
                if text[m.start:m.end] != m.hotword:
                    final_matches.append(m)
                occupied_ranges.append((m.start, m.end))

        # 执行替换 (从后向前，避免索引偏移)
        final_matches.sort(key=lambda x: x.start, reverse=True)
        result_list = list(text)
        for m in final_matches:
            result_list[m.start:m.end] = list(m.hotword)

        # 构建返回结果
        result_info = [
            (m.original, m.hotword, round(m.score, 3))
            for m in sorted(final_matches, key=lambda x: x.start)
        ]

        return "".join(result_list), result_info
