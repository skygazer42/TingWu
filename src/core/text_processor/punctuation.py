"""
标点符号转换和全角字符归一化工具

支持:
- 全角标点与半角标点的转换
- 全角 ASCII 字符归一化 (QJ2BJ: 全角 → 半角)
"""

__all__ = [
    'PunctuationConverter',
    'FullwidthNormalizer',
    'convert_full_to_half',
    'convert_half_to_full',
    'normalize_fullwidth',
    'merge_punctuation',
]

import re


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
    ''': "'",
    ''': "'",
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
    "'": '\u2018',  # Left single quotation mark '
}

# 全角 ASCII 字符映射表 (QJ2BJ)
# 全角 ASCII: U+FF01 (！) 到 U+FF5E (~)，对应半角 0x21 到 0x7E
# 全角空格: U+3000 → 普通空格
FULLWIDTH_ASCII = {chr(0xFF01 + i): chr(0x21 + i) for i in range(94)}
FULLWIDTH_ASCII['\u3000'] = ' '  # 全角空格

# 中文标点集合
CHINESE_PUNCTUATION = set('，。？！：；、''""（）【】《》')
# 英文标点集合
ENGLISH_PUNCTUATION = set(',.?!:;\'\"()[]<>')
# 所有标点
ALL_PUNCTUATION = CHINESE_PUNCTUATION | ENGLISH_PUNCTUATION | {' '}

# 重复标点模式
_DUPLICATE_PUNC_PATTERN = re.compile(r'([，。？！,\.?!])\1+')
# 混合标点模式 (如 "。." 或 ",，")
_MIXED_PUNC_PATTERN = re.compile(r'([，。？！])([,\.?!])|([,\.?!])([，。？！])')


def merge_punctuation(text: str, prefer_chinese: bool = True) -> str:
    """
    合并重复和混合标点

    处理:
    - 移除连续重复标点: "。。" → "。", ",," → ","
    - 修复中英标点混用: "。." → "。", ",，" → "，"
    - 句末标点检查 (可选)

    Args:
        text: 输入文本
        prefer_chinese: 混合标点时优先使用中文标点

    Returns:
        处理后的文本
    """
    if not text:
        return text

    # 移除重复标点
    text = _DUPLICATE_PUNC_PATTERN.sub(r'\1', text)

    # 处理混合标点
    def _merge_mixed(match):
        groups = match.groups()
        # 有 4 个捕获组，前 2 个是 (中文)(英文)，后 2 个是 (英文)(中文)
        if groups[0] and groups[1]:
            # 中文+英文，保留中文
            return groups[0] if prefer_chinese else groups[1]
        elif groups[2] and groups[3]:
            # 英文+中文，保留中文
            return groups[3] if prefer_chinese else groups[2]
        return match.group(0)

    text = _MIXED_PUNC_PATTERN.sub(_merge_mixed, text)

    # 移除多余空格
    text = re.sub(r' {2,}', ' ', text)

    return text


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


def normalize_fullwidth(
    text: str,
    include_letters: bool = True,
    include_digits: bool = True,
    include_space: bool = True,
) -> str:
    """
    将全角 ASCII 字符转换为半角 (QJ2BJ)

    Args:
        text: 待转换的文本
        include_letters: 是否转换全角字母 (A-Z, a-z)
        include_digits: 是否转换全角数字 (0-9)
        include_space: 是否转换全角空格

    Returns:
        转换后的文本
    """
    if not text:
        return text

    result = []
    for char in text:
        code = ord(char)

        # 全角空格
        if char == '\u3000':
            if include_space:
                result.append(' ')
            else:
                result.append(char)
            continue

        # 全角 ASCII 范围: U+FF01 到 U+FF5E
        if 0xFF01 <= code <= 0xFF5E:
            half_code = code - 0xFF01 + 0x21
            half_char = chr(half_code)

            # 检查是否需要转换
            if half_char.isalpha() and not include_letters:
                result.append(char)
            elif half_char.isdigit() and not include_digits:
                result.append(char)
            else:
                result.append(half_char)
        else:
            result.append(char)

    return ''.join(result)


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


class FullwidthNormalizer:
    """
    全角 ASCII 字符归一化器 (QJ2BJ)

    将全角 ASCII 字符 (A-Z, a-z, 0-9, 符号, 空格) 转换为半角。

    用法:
        normalizer = FullwidthNormalizer()
        result = normalizer.normalize("ＡＢＣＤ１２３")
        # result: "ABCD123"
    """

    def __init__(
        self,
        include_letters: bool = True,
        include_digits: bool = True,
        include_space: bool = True,
    ):
        """
        初始化归一化器

        Args:
            include_letters: 是否转换全角字母
            include_digits: 是否转换全角数字
            include_space: 是否转换全角空格
        """
        self.include_letters = include_letters
        self.include_digits = include_digits
        self.include_space = include_space

    def normalize(self, text: str) -> str:
        """
        归一化全角字符

        Args:
            text: 输入文本

        Returns:
            归一化后的文本
        """
        return normalize_fullwidth(
            text,
            include_letters=self.include_letters,
            include_digits=self.include_digits,
            include_space=self.include_space,
        )


if __name__ == '__main__':
    converter = PunctuationConverter()
    print("=== 标点转换 ===")
    print("全角 -> 半角:")
    print(converter.to_half("你好，世界！这是一个测试。"))
    print("\n半角 -> 全角:")
    print(converter.to_full("Hello, World! This is a test."))

    print("\n\n=== 全角ASCII归一化 (QJ2BJ) ===")
    normalizer = FullwidthNormalizer()
    print(f"ＡＢＣＤ１２３４ -> {normalizer.normalize('ＡＢＣＤ１２３４')}")
    print(f"Ｈｅｌｌｏ　Ｗｏｒｌｄ -> {normalizer.normalize('Ｈｅｌｌｏ　Ｗｏｒｌｄ')}")
    print(f"全角符号！＠＃ -> {normalizer.normalize('全角符号！＠＃')}")
