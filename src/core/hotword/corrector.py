"""热词纠错器"""
import os
import threading
from typing import List, Dict, Tuple, NamedTuple, Optional

from src.core.hotword.phoneme import Phoneme, get_phoneme_info, SIMILAR_PHONEMES
from src.core.hotword.rag import FastRAG


class MatchResult(NamedTuple):
    start: int
    end: int
    score: float
    hotword: str


class CorrectionResult(NamedTuple):
    text: str
    matches: List[Tuple[str, float]]
    similars: List[Tuple[str, float]]


class PhonemeCorrector:
    """基于音素的热词纠错器"""

    def __init__(self, threshold: float = 0.8, similar_threshold: float = None):
        self.threshold = threshold
        self.similar_threshold = similar_threshold or (threshold - 0.2)
        self.hotwords: Dict[str, List[Phoneme]] = {}
        self.fast_rag = FastRAG(threshold=min(self.threshold, self.similar_threshold) - 0.1)
        self._lock = threading.Lock()

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

        return len(new_hotwords)

    def load_hotwords_file(self, path: str) -> int:
        """从文件加载热词"""
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return self.update_hotwords(f.read())
        return 0

    def correct(self, text: str, top_k: int = 10) -> CorrectionResult:
        """执行热词纠错"""
        if not text:
            return CorrectionResult("", [], [])

        input_phs = get_phoneme_info(text)
        if not input_phs:
            return CorrectionResult(text, [], [])

        with self._lock:
            fast_results = self.fast_rag.search(input_phs, top_k=100)
            processed = [p.info for p in input_phs]
            matches, similars = self._find_matches(fast_results, processed)

        new_text, final_matches, _ = self._resolve_and_replace(text, matches)
        return CorrectionResult(
            new_text,
            final_matches,
            [(m.hotword, m.score) for m in similars[:top_k]]
        )

    def _find_matches(self, fast_results, input_processed):
        """查找匹配的热词"""
        matches = []
        similars = []
        input_len = len(input_processed)

        for hw, _ in fast_results:
            hw_phs = self.hotwords[hw]
            hw_compare = [p.info[:5] for p in hw_phs]
            target_len = len(hw_compare)

            if target_len > input_len:
                continue

            for i in range(input_len - target_len + 1):
                seg = input_processed[i:i + target_len]

                # 检查词边界
                if not seg[0][2]:  # is_word_start
                    continue

                score = self._fuzzy_score(hw_compare, seg)
                m = MatchResult(seg[0][5], seg[-1][6], score, hw)
                similars.append(m)

                if score >= self.threshold:
                    matches.append(m)

        seen = set()
        sorted_sims = sorted(similars, key=lambda x: x.score, reverse=True)
        sims_final = [m for m in sorted_sims
                      if m.score >= self.similar_threshold
                      and not (m.hotword in seen or seen.add(m.hotword))]

        return matches, sims_final

    def _fuzzy_score(self, target, source) -> float:
        """计算模糊匹配得分"""
        n = len(target)
        if n == 0:
            return 0.0

        total_cost = 0.0
        for t, s in zip(target, source):
            if t[0] != s[0]:  # value 不同
                if t[1] == s[1] == 'zh':  # 都是中文
                    pair = {t[0], s[0]}
                    found = False
                    for sim_set in SIMILAR_PHONEMES:
                        if pair.issubset(sim_set):
                            total_cost += 0.5
                            found = True
                            break
                    if not found:
                        total_cost += 1.0
                else:
                    total_cost += 1.0

        return 1.0 - (total_cost / n)

    def _resolve_and_replace(self, text, matches):
        """解决冲突并替换文本"""
        matches.sort(key=lambda x: (x.score, x.end - x.start), reverse=True)

        final = []
        all_info = []
        occupied = []
        seen = set()

        for m in matches:
            if (m.hotword, m.score) not in seen:
                all_info.append((m.hotword, m.score))
                seen.add((m.hotword, m.score))

            if m.score < self.threshold:
                continue

            overlaps = any(not (m.end <= rs or m.start >= re) for rs, re in occupied)
            if overlaps:
                continue

            if text[m.start:m.end] != m.hotword:
                final.append(m)
            occupied.append((m.start, m.end))

        result = list(text)
        final.sort(key=lambda x: x.start, reverse=True)
        for m in final:
            result[m.start:m.end] = list(m.hotword)

        return "".join(result), [(m.hotword, m.score) for m in sorted(final, key=lambda x: x.start)], all_info
