"""纠错历史 RAG - 基于 CapsWriter-Offline

检索用户自定义的纠错历史 (错句 => 正句)，作为 LLM 的背景知识。

工作原理：
1. 从 "错句 => 正句" 中提取被修改的片段（改前、改后）
2. 用这些片段作为检索词，与语音识别结果匹配
3. 返回得分最高的前 n 条完整纠错记录喂给 LLM

文件格式 (hot-rectify.txt)：
    用 --- 分隔的多行内容，每段第一行是错误文本，第二行是正确文本。
    忽略以 # 开头的注释和空行。
"""
import threading
from pathlib import Path
from typing import List, Tuple, Optional
from difflib import SequenceMatcher
from dataclasses import dataclass

from src.core.hotword.phoneme import Phoneme, get_phoneme_info
from src.core.hotword.algo_calc import fuzzy_substring_distance


@dataclass
class Fragment:
    """纠错片段"""
    text: str
    source_text: str
    start: int
    end: int


class RectifyRecord:
    """单条纠错记录"""
    def __init__(self, wrong: str, right: str, fragments: List[str]):
        self.wrong = wrong
        self.right = right
        self.fragments = fragments
        # 预计算每个片段的音素序列
        self.fragment_phonemes = {
            f: get_phoneme_info(f) for f in fragments
        }

    def __repr__(self):
        return f"RectifyRecord('{self.wrong}' => '{self.right}', fragments={self.fragments})"


def _get_word_boundaries(text: str) -> List[Tuple[int, int, str]]:
    """
    获取文本中所有单词的边界
    Returns: [(start, end, word), ...]
    """
    boundaries = []
    i, n = 0, len(text)
    while i < n:
        if not (text[i].isalnum() or '\u4e00' <= text[i] <= '\u9fff'):
            i += 1
            continue
        start = i
        if '\u4e00' <= text[i] <= '\u9fff':
            i += 1
        elif text[i].isalnum():
            last_was_lower = text[i].islower()
            while i < n and text[i].isalnum():
                if text[i].isupper() and last_was_lower and i > start:
                    break
                last_was_lower = text[i].islower()
                i += 1
        boundaries.append((start, i, text[start:i]))
    return boundaries


def _expand_by_words(text: str, start: int, end: int, expand_count: int = 1) -> Tuple[int, int]:
    """按单词数量向左右扩展片段"""
    bounds = _get_word_boundaries(text)
    start_idx = next((i for i, b in enumerate(bounds) if b[0] == start), None)
    end_idx = next((i + 1 for i, b in enumerate(bounds) if b[1] == end), None)

    if start_idx is None or end_idx is None:
        return start, end

    new_start = bounds[max(0, start_idx - expand_count)][0]
    new_end = bounds[min(len(bounds), end_idx + expand_count) - 1][1]
    return new_start, new_end


def extract_diff_fragments(
    wrong: str,
    right: str,
    zh_min_phonemes: int = 4,
    expand_words: int = 1
) -> List[str]:
    """
    提取两个句子之间的差异片段（包括错误版本和正确版本）
    基于单词序列进行精准提取
    """
    wrong_bounds = _get_word_boundaries(wrong)
    right_bounds = _get_word_boundaries(right)

    matcher = SequenceMatcher(
        None,
        [b[2] for b in wrong_bounds],
        [b[2] for b in right_bounds]
    )
    fragments: List[Fragment] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ('replace', 'delete') and i2 > i1:
            frag_text = wrong[wrong_bounds[i1][0]:wrong_bounds[i2-1][1]]
            if frag_text:
                fragments.append(Fragment(frag_text, wrong, wrong_bounds[i1][0], wrong_bounds[i2-1][1]))
        if tag in ('replace', 'insert') and j2 > j1:
            frag_text = right[right_bounds[j1][0]:right_bounds[j2-1][1]]
            if frag_text:
                fragments.append(Fragment(frag_text, right, right_bounds[j1][0], right_bounds[j2-1][1]))

    # 智能过滤和扩展
    result = []
    for frag in fragments:
        phonemes = get_phoneme_info(frag.text)
        if not phonemes:
            continue

        if any(p.lang != 'zh' for p in phonemes) or len(phonemes) >= zh_min_phonemes:
            result.append(frag.text)
        else:
            # 中文片段太短，扩展单词
            exp_start, exp_end = _expand_by_words(
                frag.source_text, frag.start, frag.end, expand_words
            )
            expanded = frag.source_text[exp_start:exp_end]
            result.append(expanded if expanded else frag.text)

    return list(dict.fromkeys(result))  # 去重并保持顺序


class RectificationRAG:
    """
    纠错历史 RAG 检索器

    加载 'hot-rectify.txt'，通过 RAG 检索相似的差异片段，
    返回对应的完整纠错记录作为 Prompt 上下文。
    """

    def __init__(self, rectify_file: str = 'hot-rectify.txt', threshold: float = 0.5):
        self.rectify_file = Path(rectify_file)
        self.threshold = threshold
        self.records: List[RectifyRecord] = []
        self._lock = threading.Lock()

    def load_history(self) -> int:
        """
        加载纠错历史

        Returns:
            加载的纠错记录数量
        """
        if not self.rectify_file.exists():
            return 0

        content = self.rectify_file.read_text(encoding='utf-8')
        new_records = []

        blocks = content.split('---')
        for block in blocks:
            lines = block.strip().split('\n')
            valid_lines = [
                line.strip() for line in lines
                if line.strip() and not line.strip().startswith('#')
            ]

            if len(valid_lines) >= 2:
                wrong = valid_lines[0]
                right = valid_lines[1]

                if wrong and right:
                    fragments = extract_diff_fragments(wrong, right)
                    if not fragments:
                        fragments = [wrong]

                    record = RectifyRecord(wrong, right, fragments)
                    new_records.append(record)

        with self._lock:
            self.records = new_records

        return len(new_records)

    def _score_record(
        self,
        input_phonemes: List[Phoneme],
        record: RectifyRecord
    ) -> Tuple[float, List[dict]]:
        """计算单条记录与输入音素序列的匹配得分"""
        fragment_details = []
        for fragment, frag_phonemes in record.fragment_phonemes.items():
            if not frag_phonemes:
                continue

            input_info = [p.info for p in input_phonemes]
            frag_info = [p.info for p in frag_phonemes]

            min_dist = fuzzy_substring_distance(frag_info, input_info)
            score = 1.0 - (min_dist / len(frag_info))

            fragment_details.append({
                'fragment': fragment,
                'score': round(score, 3),
                'phonemes': len(frag_info)
            })

        if not fragment_details:
            return 0.0, []

        fragment_details.sort(key=lambda x: x['score'], reverse=True)
        return fragment_details[0]['score'], fragment_details

    def search(self, text: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """
        检索相关的纠错历史

        Args:
            text: 输入文本
            top_k: 最大结果数

        Returns:
            [(错误文本, 正确文本, 分数), ...]
        """
        if not text or not self.records:
            return []

        input_phonemes = get_phoneme_info(text)
        if not input_phonemes:
            return []

        with self._lock:
            records = self.records[:]

        matches = []
        for record in records:
            best_score, _ = self._score_record(input_phonemes, record)
            if best_score >= self.threshold:
                matches.append((record.wrong, record.right, round(best_score, 3)))

        matches.sort(key=lambda x: x[2], reverse=True)
        return matches[:top_k]

    def format_prompt(self, text: str, top_k: int = 5, prefix: str = "纠错历史：") -> str:
        """
        生成提示词

        Args:
            text: 输入文本
            top_k: 最大结果数
            prefix: 提示词前缀

        Returns:
            包含纠错历史的提示词段落，无匹配则返回空字符串
        """
        if not self.records:
            return ""

        results = self.search(text, top_k=top_k)
        if not results:
            return ""

        lines = [prefix]
        for wrong, right, score in results:
            lines.append(f"- {wrong} => {right}")

        return "\n".join(lines)
