"""
标点符号转换工具

移植自 CapsWriter-Offline，支持全角标点与半角标点的转换。
"""

__all__ = ['PunctuationConverter', 'convert_full_to_half', 'convert_half_to_full']


# 全角标点到半角标点的映射
FULL_TO_HALF = {
    '，': ', ',
    '。': '. ',
    '？': '? ',
    '！': '! ',
    '：': ': ',
    '；': '; ',
    '（': '(',
    '）': ')',
    '【': '[',
    '】': ']',
    '「': '"',
    '」': '"',
    '『': "'",
    '』': "'",
    '"': '"',
    '"': '"',
    "'": "'",
    "'": "'",
}

# 半角标点到全角标点的映射（用于反向转换）
HALF_TO_FULL = {
    ',': '，',
    '.': '。',
    '?': '？',
    '!': '！',
    ':': '：',
    ';': '；',
    '(': '（',
    ')': '）',
    '[': '【',
    ']': '】',
    '"': '"',
    "'": "'",
}


def convert_full_to_half(text: str, add_space: bool = True) -> str:
    """
    将全角标点转换为半角标点

    Args:
        text: 待转换的文本
        add_space: 是否在标点后添加空格（默认True）

    Returns:
        转换后的文本
    """
    if not text:
        return text

    mapping = FULL_TO_HALF if add_space else {
        k: v.strip() for k, v in FULL_TO_HALF.items()
    }

    result = text
    for full, half in mapping.items():
        result = result.replace(full, half)
    return result


def convert_half_to_full(text: str) -> str:
    """
    将半角标点转换为全角标点

    Args:
        text: 待转换的文本

    Returns:
        转换后的文本
    """
    if not text:
        return text

    result = text
    for half, full in HALF_TO_FULL.items():
        result = result.replace(half, full)
    return result


class PunctuationConverter:
    """
    标点符号转换器

    支持:
    - 全角 → 半角转换
    - 半角 → 全角转换
    - 可选是否在标点后添加空格

    用法:
        converter = PunctuationConverter()
        result = converter.to_half("你好，世界！")
        # result: "你好, 世界! "
    """

    def __init__(self, add_space: bool = True):
        """
        初始化转换器

        Args:
            add_space: 转换为半角时是否在标点后添加空格
        """
        self.add_space = add_space

    def to_half(self, text: str) -> str:
        """
        全角标点转半角

        Args:
            text: 输入文本

        Returns:
            转换后的文本
        """
        return convert_full_to_half(text, self.add_space)

    def to_full(self, text: str) -> str:
        """
        半角标点转全角

        Args:
            text: 输入文本

        Returns:
            转换后的文本
        """
        return convert_half_to_full(text)

    def convert(self, text: str, to_half: bool = True) -> str:
        """
        转换标点

        Args:
            text: 输入文本
            to_half: True=转半角, False=转全角

        Returns:
            转换后的文本
        """
        if to_half:
            return self.to_half(text)
        else:
            return self.to_full(text)


if __name__ == '__main__':
    converter = PunctuationConverter()
    print("全角 -> 半角:")
    print(converter.to_half("你好，世界！这是一个测试。"))
    print("\n半角 -> 全角:")
    print(converter.to_full("Hello, World! This is a test."))
