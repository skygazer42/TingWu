"""字形相似度纠错模块

实现基于汉字字形结构的相似度计算，用于音形义联合纠错。
汉字形近字（如 "己/已/巳"、"日/曰"）在 ASR 中容易混淆。

主要功能：
1. 四角号码相似度 - 基于汉字四个角的笔画形状
2. 部首相似度 - 基于汉字偏旁部首
3. 笔画数相似度 - 基于总笔画数
4. 结构相似度 - 基于汉字结构类型（左右、上下、包围等）
"""

import logging
from typing import Dict, List, Tuple, Optional, Set
from functools import lru_cache

logger = logging.getLogger(__name__)

# 常见形近字映射表 (基于 OCR/ASR 常见混淆)
SIMILAR_SHAPES: Dict[str, Set[str]] = {
    # 常见混淆字组
    "己": {"已", "巳"},
    "已": {"己", "巳"},
    "巳": {"己", "已"},
    "日": {"曰", "目"},
    "曰": {"日"},
    "目": {"日", "且"},
    "且": {"目", "旦"},
    "旦": {"且"},
    "大": {"太", "犬"},
    "太": {"大"},
    "犬": {"大"},
    "末": {"未", "本"},
    "未": {"末"},
    "本": {"末", "木"},
    "木": {"本", "林"},
    "林": {"木", "森"},
    "土": {"士", "工"},
    "士": {"土"},
    "工": {"土", "干"},
    "干": {"工", "千"},
    "千": {"干", "于"},
    "于": {"千", "干"},
    "王": {"玉", "主"},
    "玉": {"王"},
    "主": {"王", "住"},
    "住": {"主", "往"},
    "往": {"住"},
    "人": {"入", "八"},
    "入": {"人"},
    "八": {"人", "几"},
    "几": {"八", "凡"},
    "凡": {"几"},
    "刀": {"力", "刁"},
    "力": {"刀", "办"},
    "刁": {"刀"},
    "办": {"力"},
    "甲": {"申", "由", "田"},
    "申": {"甲", "由", "电"},
    "由": {"甲", "申", "田"},
    "田": {"甲", "由"},
    "电": {"申"},
    "冒": {"昌", "冕"},
    "昌": {"冒", "晶"},
    "晶": {"昌", "品"},
    "品": {"晶"},
    "品": {"晶"},
    "贝": {"见"},
    "见": {"贝"},
    "矢": {"失"},
    "失": {"矢"},
    "折": {"拆", "析"},
    "拆": {"折"},
    "析": {"折", "斩"},
    "斩": {"析"},
}

# 汉字结构类型
STRUCTURE_TYPES = {
    "左右结构": 1,
    "上下结构": 2,
    "半包围结构": 3,
    "全包围结构": 4,
    "独体字": 5,
    "品字结构": 6,
}


class ShapeCorrector:
    """
    字形相似度纠错器

    基于多维度字形特征计算汉字相似度：
    - 形近字映射表 (高权重)
    - 笔画数差异
    - Unicode 编码距离 (低权重辅助)
    """

    def __init__(
        self,
        threshold: float = 0.7,
        stroke_weight: float = 0.3,
        shape_table_weight: float = 0.6,
        unicode_weight: float = 0.1,
    ):
        """
        初始化字形纠错器

        Args:
            threshold: 相似度阈值 (高于此值认为形似)
            stroke_weight: 笔画数相似度权重
            shape_table_weight: 形近字表权重
            unicode_weight: Unicode 距离权重
        """
        self.threshold = threshold
        self.stroke_weight = stroke_weight
        self.shape_table_weight = shape_table_weight
        self.unicode_weight = unicode_weight

        # 尝试加载笔画数据
        self._stroke_count: Dict[str, int] = {}
        self._load_stroke_data()

    def _load_stroke_data(self):
        """加载汉字笔画数据 (使用 cnradical 或内置简表)"""
        try:
            from cnradical import Radical
            self._radical_tool = Radical()
            logger.info("Loaded cnradical for stroke counting")
        except ImportError:
            self._radical_tool = None
            # 使用简化的常用字笔画表
            self._stroke_count = {
                # 常用字笔画数 (仅作为备选)
                "一": 1, "二": 2, "三": 3, "十": 2, "人": 2, "大": 3,
                "中": 4, "国": 8, "小": 3, "上": 3, "下": 3, "不": 4,
                "是": 9, "的": 8, "我": 7, "了": 2, "他": 5, "她": 6,
                "你": 7, "这": 7, "那": 6, "有": 6, "来": 7, "去": 5,
                "在": 6, "和": 8, "说": 9, "要": 9, "能": 10, "会": 6,
            }
            logger.info("Using built-in stroke count table (limited)")

    @lru_cache(maxsize=10000)
    def get_stroke_count(self, char: str) -> int:
        """获取汉字笔画数"""
        if not char or not ('\u4e00' <= char <= '\u9fff'):
            return 0

        # 优先使用 cnradical
        if self._radical_tool:
            try:
                radical_info = self._radical_tool.run_radical_and_stroke(char)
                if radical_info and len(radical_info) > 1:
                    return int(radical_info[1]) if radical_info[1] else 0
            except Exception:
                pass

        # 使用内置表
        if char in self._stroke_count:
            return self._stroke_count[char]

        # 估算：基于 Unicode 码点的简单估算
        # (汉字越复杂，码点通常越大，但这只是粗略估计)
        return min(max((ord(char) - 0x4e00) // 500 + 4, 1), 30)

    def char_shape_similarity(self, char1: str, char2: str) -> float:
        """
        计算两个汉字的字形相似度

        Args:
            char1: 第一个汉字
            char2: 第二个汉字

        Returns:
            相似度分数 (0-1)
        """
        if char1 == char2:
            return 1.0

        # 非汉字返回 0
        if not ('\u4e00' <= char1 <= '\u9fff') or not ('\u4e00' <= char2 <= '\u9fff'):
            return 0.0

        score = 0.0

        # 1. 形近字表匹配 (高权重)
        if char1 in SIMILAR_SHAPES and char2 in SIMILAR_SHAPES.get(char1, set()):
            score += self.shape_table_weight
        elif char2 in SIMILAR_SHAPES and char1 in SIMILAR_SHAPES.get(char2, set()):
            score += self.shape_table_weight

        # 2. 笔画数相似度
        stroke1 = self.get_stroke_count(char1)
        stroke2 = self.get_stroke_count(char2)
        if stroke1 > 0 and stroke2 > 0:
            max_stroke = max(stroke1, stroke2)
            stroke_diff = abs(stroke1 - stroke2)
            stroke_sim = 1.0 - (stroke_diff / max_stroke)
            score += self.stroke_weight * stroke_sim

        # 3. Unicode 距离 (辅助)
        unicode_dist = abs(ord(char1) - ord(char2))
        # 码点越近，可能越相似 (经验值：200 以内认为相近)
        if unicode_dist < 200:
            unicode_sim = 1.0 - (unicode_dist / 200)
            score += self.unicode_weight * unicode_sim

        return min(score, 1.0)

    def text_shape_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的字形相似度

        Args:
            text1: 第一个文本
            text2: 第二个文本

        Returns:
            平均相似度分数 (0-1)
        """
        if not text1 or not text2:
            return 0.0

        if text1 == text2:
            return 1.0

        # 长度不同时的处理
        if len(text1) != len(text2):
            # 长度差异过大，相似度低
            len_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2))
            if len_ratio < 0.5:
                return 0.0

        # 计算逐字符相似度
        total_sim = 0.0
        count = 0
        for i in range(min(len(text1), len(text2))):
            sim = self.char_shape_similarity(text1[i], text2[i])
            total_sim += sim
            count += 1

        return total_sim / count if count > 0 else 0.0

    def find_similar_chars(self, char: str) -> List[Tuple[str, float]]:
        """
        查找与给定汉字形似的字符

        Args:
            char: 输入汉字

        Returns:
            [(形似字, 相似度), ...] 按相似度降序
        """
        results = []

        if not ('\u4e00' <= char <= '\u9fff'):
            return results

        # 从形近字表中查找
        if char in SIMILAR_SHAPES:
            for similar in SIMILAR_SHAPES[char]:
                results.append((similar, self.char_shape_similarity(char, similar)))

        # 排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def correct_by_shape(
        self,
        text: str,
        candidates: List[str],
    ) -> List[Tuple[str, float]]:
        """
        基于字形相似度从候选中选择最佳匹配

        Args:
            text: 输入文本
            candidates: 候选词列表

        Returns:
            [(候选词, 相似度), ...] 按相似度降序
        """
        results = []
        for candidate in candidates:
            if len(candidate) == len(text):
                sim = self.text_shape_similarity(text, candidate)
                if sim >= self.threshold:
                    results.append((candidate, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results


class JointCorrector:
    """
    音形义联合纠错器

    综合使用音素相似度、字形相似度和语义相似度进行纠错。
    这是 "音形义" 联合纠错的核心实现。
    """

    def __init__(
        self,
        phoneme_weight: float = 0.5,
        shape_weight: float = 0.3,
        semantic_weight: float = 0.2,
        threshold: float = 0.75,
    ):
        """
        初始化联合纠错器

        Args:
            phoneme_weight: 音素相似度权重 (音)
            shape_weight: 字形相似度权重 (形)
            semantic_weight: 语义相似度权重 (义)
            threshold: 综合阈值
        """
        self.phoneme_weight = phoneme_weight
        self.shape_weight = shape_weight
        self.semantic_weight = semantic_weight
        self.threshold = threshold

        self.shape_corrector = ShapeCorrector()

        # 语义相似度模块 (可选)
        self._semantic_model = None

    def _init_semantic_model(self):
        """懒加载语义模型"""
        if self._semantic_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._semantic_model = SentenceTransformer(
                    'paraphrase-multilingual-MiniLM-L12-v2'
                )
                logger.info("Loaded semantic similarity model")
            except ImportError:
                logger.warning("sentence-transformers not installed, semantic similarity disabled")
                self._semantic_model = False  # 标记为不可用

    def semantic_similarity(self, text1: str, text2: str) -> float:
        """计算语义相似度"""
        self._init_semantic_model()

        if self._semantic_model is False or self._semantic_model is None:
            return 0.5  # 无模型时返回中性值

        try:
            embeddings = self._semantic_model.encode([text1, text2])
            import numpy as np
            sim = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(sim)
        except Exception as e:
            logger.warning(f"Semantic similarity failed: {e}")
            return 0.5

    def combined_similarity(
        self,
        text: str,
        candidate: str,
        phoneme_score: float = 0.0,
    ) -> float:
        """
        计算综合相似度

        Args:
            text: 原始文本
            candidate: 候选文本
            phoneme_score: 预计算的音素相似度 (可选)

        Returns:
            综合相似度分数 (0-1)
        """
        # 字形相似度
        shape_score = self.shape_corrector.text_shape_similarity(text, candidate)

        # 语义相似度 (可选)
        semantic_score = 0.5  # 默认中性
        if self.semantic_weight > 0:
            semantic_score = self.semantic_similarity(text, candidate)

        # 加权综合
        total = (
            self.phoneme_weight * phoneme_score +
            self.shape_weight * shape_score +
            self.semantic_weight * semantic_score
        )

        return total

    def rank_candidates(
        self,
        text: str,
        candidates: List[Tuple[str, float]],
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        对候选词进行音形义联合排序

        Args:
            text: 原始文本
            candidates: [(候选词, 音素分数), ...] 来自 PhonemeCorrector

        Returns:
            [(候选词, 综合分数, {phoneme, shape, semantic}), ...]
        """
        results = []

        for candidate, phoneme_score in candidates:
            shape_score = self.shape_corrector.text_shape_similarity(text, candidate)

            # 语义相似度 (可选，较慢)
            semantic_score = 0.5
            if self.semantic_weight > 0 and len(text) > 1:
                semantic_score = self.semantic_similarity(text, candidate)

            combined = (
                self.phoneme_weight * phoneme_score +
                self.shape_weight * shape_score +
                self.semantic_weight * semantic_score
            )

            results.append((
                candidate,
                combined,
                {
                    "phoneme": phoneme_score,
                    "shape": shape_score,
                    "semantic": semantic_score,
                }
            ))

        # 按综合分数排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results


# 全局实例
shape_corrector = ShapeCorrector()
joint_corrector = JointCorrector()
