# coding: utf-8
"""
口语填充词移除模块

移除 ASR 输出中的常见中文口语填充词，提升文本质量。

支持两种模式:
- 保守模式(默认): 只移除明确的填充词，保留可能有意义的用法
- 激进模式: 更广泛地移除填充词，可能会误删有意义的词

用法示例:
    from src.core.text_processor import FillerRemover

    remover = FillerRemover()
    text = remover.remove("呃那个就是说AI很厉害")
    print(text)  # "AI很厉害"
"""

__all__ = ['FillerRemover']

import re
from typing import List, Optional, Set


# 语气填充词 (单字)
FILLER_INTERJECTIONS = {
    '呃', '啊', '嗯', '哦', '噢', '哼', '唔', '额', '诶', '欸',
}

# 口头禅填充词 (多字)
FILLER_PHRASES = [
    '就是说', '就是那个', '那个那个', '然后呢', '然后吧', '所以说', '所以呢',
    '那个', '就是', '然后', '这个', '那么', '其实', '反正',
]

# 重复强调词 (激进模式)
FILLER_REPETITIONS = [
    '对对对', '好好好', '是是是', '嗯嗯嗯', '啊啊啊',
    '对对', '好好', '是是', '嗯嗯',
]

# 句首填充词模式
SENTENCE_START_FILLERS = [
    '那', '嗯', '啊', '哦', '诶', '欸',
]


class FillerRemover:
    """口语填充词移除器

    Attributes:
        aggressive: 是否使用激进模式
        custom_fillers: 自定义填充词列表
    """

    def __init__(
        self,
        aggressive: bool = False,
        custom_fillers: Optional[List[str]] = None,
    ):
        """初始化填充词移除器

        Args:
            aggressive: 激进模式，更广泛地移除填充词
            custom_fillers: 自定义填充词列表
        """
        self.aggressive = aggressive
        self.custom_fillers = custom_fillers or []

        # 构建填充词集合
        self._build_patterns()

    def _build_patterns(self):
        """构建匹配模式"""
        # 多字填充词 (按长度降序，优先匹配长词)
        phrases = list(FILLER_PHRASES)
        if self.aggressive:
            phrases.extend(FILLER_REPETITIONS)
        phrases.extend(self.custom_fillers)
        phrases = sorted(set(phrases), key=len, reverse=True)

        # 单字语气词
        interjections = set(FILLER_INTERJECTIONS)

        # 构建正则模式
        # 多字填充词模式
        if phrases:
            phrase_pattern = '|'.join(re.escape(p) for p in phrases)
            self._phrase_regex = re.compile(f'({phrase_pattern})')
        else:
            self._phrase_regex = None

        # 句首单字语气词模式 (只在句首或标点后移除)
        interjection_chars = ''.join(interjections)
        # 匹配: 开头的语气词 或 标点后的语气词
        self._start_filler_regex = re.compile(
            rf'^[{re.escape(interjection_chars)}]+|(?<=[，。？！；：,.?!;:])[{re.escape(interjection_chars)}]+'
        )

        # 激进模式: 移除独立的单字语气词 (前后是标点或空格)
        if self.aggressive:
            self._standalone_filler_regex = re.compile(
                rf'(?<=[，。？！；：,.?!;:\s])[{re.escape(interjection_chars)}](?=[，。？！；：,.?!;:\s])'
            )
        else:
            self._standalone_filler_regex = None

    def remove(self, text: str) -> str:
        """移除填充词

        Args:
            text: 输入文本

        Returns:
            移除填充词后的文本
        """
        if not text:
            return text

        result = text

        # 1. 移除多字填充词
        if self._phrase_regex:
            result = self._phrase_regex.sub('', result)

        # 2. 移除句首/标点后的单字语气词
        result = self._start_filler_regex.sub('', result)

        # 3. 激进模式: 移除独立的单字语气词
        if self._standalone_filler_regex:
            result = self._standalone_filler_regex.sub('', result)

        # 4. 清理多余空格
        result = re.sub(r'\s+', ' ', result).strip()

        # 5. 清理标点前的空格
        result = re.sub(r'\s+([，。？！；：,.?!;:])', r'\1', result)

        return result

    def remove_with_info(self, text: str) -> tuple:
        """移除填充词并返回详细信息

        Args:
            text: 输入文本

        Returns:
            (处理后文本, 移除的填充词列表)
        """
        if not text:
            return text, []

        removed = []

        # 记录移除的多字填充词
        if self._phrase_regex:
            matches = self._phrase_regex.findall(text)
            removed.extend(matches)

        # 执行移除
        result = self.remove(text)

        return result, removed


def remove_fillers(text: str, aggressive: bool = False) -> str:
    """便捷函数: 移除填充词

    Args:
        text: 输入文本
        aggressive: 是否使用激进模式

    Returns:
        移除填充词后的文本
    """
    remover = FillerRemover(aggressive=aggressive)
    return remover.remove(text)


if __name__ == "__main__":
    # 测试用例
    test_cases = [
        "呃那个就是说AI很厉害",
        "嗯，今天天气真好",
        "那个那个，我想说的是",
        "对对对，就是这样",
        "然后呢，我们继续",
        "他说呃我不知道",  # 句中的"呃"不应移除
    ]

    print("=== 保守模式 ===")
    remover = FillerRemover(aggressive=False)
    for text in test_cases:
        result = remover.remove(text)
        print(f"{text!r} -> {result!r}")

    print("\n=== 激进模式 ===")
    remover_aggressive = FillerRemover(aggressive=True)
    for text in test_cases:
        result = remover_aggressive.remove(text)
        print(f"{text!r} -> {result!r}")
