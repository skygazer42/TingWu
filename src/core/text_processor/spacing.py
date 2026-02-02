# coding: utf-8
"""
中英文自动间距模块

在中文和英文/数字之间自动添加空格，提升可读性。

用法示例:
    from src.core.text_processor import SpacingProcessor

    processor = SpacingProcessor()
    text = processor.add_spacing("人工智能AI技术")
    print(text)  # "人工智能 AI 技术"
"""

__all__ = ['SpacingProcessor', 'add_cjk_ascii_spacing']

import re


# CJK 字符范围
CJK_RANGES = (
    '\u4e00-\u9fff'     # CJK Unified Ideographs
    '\u3400-\u4dbf'     # CJK Unified Ideographs Extension A
    '\uf900-\ufaff'     # CJK Compatibility Ideographs
    '\u3000-\u303f'     # CJK Symbols and Punctuation (部分)
)

# ASCII 字母数字
ASCII_ALNUM = r'a-zA-Z0-9'

# 模式: CJK 后面跟着 ASCII 字母/数字
PATTERN_CJK_THEN_ASCII = re.compile(
    rf'([{CJK_RANGES}])([{ASCII_ALNUM}])'
)

# 模式: ASCII 字母/数字后面跟着 CJK
PATTERN_ASCII_THEN_CJK = re.compile(
    rf'([{ASCII_ALNUM}])([{CJK_RANGES}])'
)


def add_cjk_ascii_spacing(text: str) -> str:
    """
    在中文和英文/数字之间添加空格

    Args:
        text: 输入文本

    Returns:
        添加空格后的文本

    Examples:
        >>> add_cjk_ascii_spacing("人工智能AI技术")
        '人工智能 AI 技术'
        >>> add_cjk_ascii_spacing("Python3编程")
        'Python3 编程'
        >>> add_cjk_ascii_spacing("2025年报告")
        '2025 年报告'
    """
    if not text:
        return text

    # 先处理 CJK -> ASCII
    result = PATTERN_CJK_THEN_ASCII.sub(r'\1 \2', text)
    # 再处理 ASCII -> CJK
    result = PATTERN_ASCII_THEN_CJK.sub(r'\1 \2', result)

    # 清理多余空格 (保留单个空格)
    result = re.sub(r' +', ' ', result)

    return result


class SpacingProcessor:
    """
    中英文间距处理器

    在中文和英文/数字之间自动添加空格，符合排版规范。

    用法:
        processor = SpacingProcessor()
        result = processor.add_spacing("AI技术")
        # result: "AI 技术"
    """

    def __init__(self):
        """初始化间距处理器"""
        pass

    def add_spacing(self, text: str) -> str:
        """
        添加中英文间距

        Args:
            text: 输入文本

        Returns:
            处理后的文本
        """
        return add_cjk_ascii_spacing(text)

    def process(self, text: str) -> str:
        """
        处理文本 (add_spacing 的别名)

        Args:
            text: 输入文本

        Returns:
            处理后的文本
        """
        return self.add_spacing(text)


if __name__ == "__main__":
    processor = SpacingProcessor()

    test_cases = [
        "人工智能AI技术",
        "Python3编程语言",
        "2025年度报告",
        "使用TensorFlow和PyTorch进行深度学习",
        "iPhone15发布会",
        "这是一个test测试",
        "100%正确率",
        "A和B的比较",
        "已有 空格 的文本",  # 不应重复添加
    ]

    print("=== 中英文间距测试 ===")
    for text in test_cases:
        result = processor.add_spacing(text)
        print(f"{text!r} -> {result!r}")
