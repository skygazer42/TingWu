"""
统一文本后处理器

整合 ITN、繁简转换、标点转换等功能。
"""

__all__ = ['TextPostProcessor']

from typing import Optional
from dataclasses import dataclass

from .chinese_itn import ChineseITN
from .zh_convert import ZhConverter
from .punctuation import PunctuationConverter


@dataclass
class PostProcessorSettings:
    """后处理配置"""
    itn_enable: bool = True
    zh_convert_enable: bool = False
    zh_convert_locale: str = "zh-hans"
    punc_convert_enable: bool = False
    punc_add_space: bool = True


class TextPostProcessor:
    """
    统一文本后处理器

    整合多个后处理功能，按固定顺序执行：
    1. 中文数字格式化 (ITN)
    2. 繁简转换
    3. 标点转换

    用法:
        settings = PostProcessorSettings(itn_enable=True)
        processor = TextPostProcessor(settings)
        result = processor.process("今天是二零二五年一月三十日")
        # result: "今天是2025年1月30日"
    """

    def __init__(self, settings: Optional[PostProcessorSettings] = None):
        """
        初始化后处理器

        Args:
            settings: 后处理配置，为 None 时使用默认配置
        """
        if settings is None:
            settings = PostProcessorSettings()

        self.settings = settings

        # 按需初始化各组件
        self.itn = ChineseITN() if settings.itn_enable else None
        self.zh_converter = ZhConverter() if settings.zh_convert_enable else None
        self.punc_converter = (
            PunctuationConverter(add_space=settings.punc_add_space)
            if settings.punc_convert_enable
            else None
        )

    def process(self, text: str) -> str:
        """
        执行文本后处理

        处理顺序:
        1. ITN - 中文数字转阿拉伯数字
        2. 繁简转换
        3. 标点转换

        Args:
            text: 输入文本

        Returns:
            处理后的文本
        """
        if not text:
            return text

        # 1. 中文数字格式化
        if self.itn:
            text = self.itn.convert(text)

        # 2. 繁简转换
        if self.zh_converter:
            text = self.zh_converter.convert(text, self.settings.zh_convert_locale)

        # 3. 标点转换
        if self.punc_converter:
            text = self.punc_converter.to_half(text)

        return text

    def process_itn(self, text: str) -> str:
        """仅执行 ITN 转换"""
        if not text or not self.itn:
            return text
        return self.itn.convert(text)

    def process_zh_convert(self, text: str, locale: Optional[str] = None) -> str:
        """仅执行繁简转换"""
        if not text or not self.zh_converter:
            return text
        locale = locale or self.settings.zh_convert_locale
        return self.zh_converter.convert(text, locale)

    def process_punctuation(self, text: str, to_half: bool = True) -> str:
        """仅执行标点转换"""
        if not text or not self.punc_converter:
            return text
        return self.punc_converter.to_half(text) if to_half else self.punc_converter.to_full(text)

    @classmethod
    def from_config(cls, config) -> 'TextPostProcessor':
        """
        从配置对象创建后处理器

        Args:
            config: 包含后处理配置的对象 (如 Settings)

        Returns:
            TextPostProcessor 实例
        """
        settings = PostProcessorSettings(
            itn_enable=getattr(config, 'itn_enable', True),
            zh_convert_enable=getattr(config, 'zh_convert_enable', False),
            zh_convert_locale=getattr(config, 'zh_convert_locale', 'zh-hans'),
            punc_convert_enable=getattr(config, 'punc_convert_enable', False),
            punc_add_space=getattr(config, 'punc_add_space', True),
        )
        return cls(settings)


if __name__ == '__main__':
    # 测试
    settings = PostProcessorSettings(
        itn_enable=True,
        zh_convert_enable=False,
        punc_convert_enable=True,
    )
    processor = TextPostProcessor(settings)

    test_cases = [
        "今天是二零二五年一月三十日",
        "价格是三百五十元",
        "你好，世界！这是一个测试。",
        "百分之五十的人同意",
    ]

    for text in test_cases:
        result = processor.process(text)
        print(f"{text} → {result}")
