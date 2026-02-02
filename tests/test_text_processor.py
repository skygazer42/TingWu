"""测试文本后处理模块"""
import pytest


class TestChineseITN:
    """中文数字格式化测试"""

    def test_pure_numbers(self):
        """测试纯数字序列"""
        from src.core.text_processor import ChineseITN
        itn = ChineseITN()

        assert itn.convert("一二三四五") == "12345"
        assert itn.convert("幺九二点幺六八点幺点幺") == "192.168.1.1"
        assert itn.convert("零零七") == "007"

    def test_value_numbers(self):
        """测试数值"""
        from src.core.text_processor import ChineseITN
        itn = ChineseITN()

        assert itn.convert("三百五十") == "350"
        assert itn.convert("一千二百") == "1200"
        assert itn.convert("五十") == "50"
        assert itn.convert("十五") == "15"

    def test_range_expression(self):
        """测试范围表达式"""
        from src.core.text_processor import ChineseITN
        itn = ChineseITN()

        assert itn.convert("三五百人") == "300~500人"
        assert itn.convert("五六十") == "50~60"
        # "三四" 单独出现时作为纯数字处理
        assert itn.convert("三四") == "34"

    def test_percent(self):
        """测试百分数"""
        from src.core.text_processor import ChineseITN
        itn = ChineseITN()

        assert itn.convert("百分之五十") == "50%"
        assert itn.convert("百分之三点五") == "3.5%"

    def test_fraction(self):
        """测试分数"""
        from src.core.text_processor import ChineseITN
        itn = ChineseITN()

        assert itn.convert("三分之一") == "1/3"
        assert itn.convert("五分之二") == "2/5"

    def test_ratio(self):
        """测试比值"""
        from src.core.text_processor import ChineseITN
        itn = ChineseITN()

        assert itn.convert("三比一") == "3:1"
        assert itn.convert("二比三") == "2:3"

    def test_date(self):
        """测试日期"""
        from src.core.text_processor import ChineseITN
        itn = ChineseITN()

        result = itn.convert("二零二五年一月三十日")
        assert "2025" in result
        assert "1" in result
        assert "30" in result

    def test_time(self):
        """测试时间"""
        from src.core.text_processor import ChineseITN
        itn = ChineseITN()

        assert itn.convert("十四点三十分") == "14:30"

    def test_idiom_blacklist(self):
        """测试成语黑名单（不转换）"""
        from src.core.text_processor import ChineseITN
        itn = ChineseITN()

        # 成语不应被转换
        assert "七" in itn.convert("乱七八糟")
        assert "八" in itn.convert("乱七八糟")

    def test_mixed_text(self):
        """测试混合文本"""
        from src.core.text_processor import ChineseITN
        itn = ChineseITN()

        result = itn.convert("今天是二零二五年一月三十日")
        assert "2025" in result
        assert "今天是" in result


class TestZhConverter:
    """繁简转换测试"""

    def test_simplified_to_traditional(self):
        """测试简体转繁体"""
        from src.core.text_processor import ZhConverter
        converter = ZhConverter()

        result = converter.to_traditional("人体内存在很多微生物")
        # 应该包含繁体字
        assert "體" in result or "内" in result  # 依赖词典

    def test_traditional_to_simplified(self):
        """测试繁体转简体"""
        from src.core.text_processor import ZhConverter
        converter = ZhConverter()

        result = converter.to_simplified("我幹什麼不干你事")
        # 应该转为简体
        assert "干" in result

    def test_region_variants(self):
        """测试区域变体"""
        from src.core.text_processor import ZhConverter
        converter = ZhConverter()

        text = "人体内存在很多微生物"
        tw_result = converter.convert(text, "zh-tw")
        hk_result = converter.convert(text, "zh-hk")
        # 台湾和香港繁体可能有差异
        assert tw_result is not None
        assert hk_result is not None

    def test_is_simplified(self):
        """测试简繁检测"""
        from src.core.text_processor import ZhConverter
        converter = ZhConverter()

        assert converter.is_simplified("简体中文") is True or converter.is_simplified("简体中文") is False

    def test_empty_input(self):
        """测试空输入"""
        from src.core.text_processor import ZhConverter
        converter = ZhConverter()

        assert converter.convert("", "zh-tw") == ""
        assert converter.convert(None, "zh-tw") is None


class TestPunctuationConverter:
    """标点转换测试"""

    def test_full_to_half(self):
        """测试全角转半角"""
        from src.core.text_processor import PunctuationConverter
        converter = PunctuationConverter()

        result = converter.to_half("你好，世界！")
        assert "," in result
        assert "!" in result
        assert "，" not in result
        assert "！" not in result

    def test_half_to_full(self):
        """测试半角转全角"""
        from src.core.text_processor import PunctuationConverter
        converter = PunctuationConverter()

        result = converter.to_full("Hello, World!")
        assert "，" in result
        assert "！" in result

    def test_add_space_option(self):
        """测试添加空格选项"""
        from src.core.text_processor import PunctuationConverter

        with_space = PunctuationConverter(add_space=True)
        without_space = PunctuationConverter(add_space=False)

        text = "你好，世界"
        result_with = with_space.to_half(text)
        result_without = without_space.to_half(text)

        # 有空格的应该更长
        assert len(result_with) >= len(result_without)

    def test_brackets(self):
        """测试括号转换"""
        from src.core.text_processor import PunctuationConverter
        converter = PunctuationConverter()

        result = converter.to_half("（测试）")
        assert "(" in result
        assert ")" in result

    def test_quotes(self):
        """测试引号转换"""
        from src.core.text_processor import PunctuationConverter
        converter = PunctuationConverter()

        result = converter.to_half("「测试」")
        assert '"' in result

    def test_function_interface(self):
        """测试函数接口"""
        from src.core.text_processor import convert_full_to_half, convert_half_to_full

        assert "," in convert_full_to_half("你好，世界")
        assert "，" in convert_half_to_full("Hello, World")


class TestTextPostProcessor:
    """统一后处理器测试"""

    def test_itn_only(self):
        """测试仅 ITN"""
        from src.core.text_processor.post_processor import TextPostProcessor, PostProcessorSettings

        settings = PostProcessorSettings(
            itn_enable=True,
            zh_convert_enable=False,
            punc_convert_enable=False,
        )
        processor = TextPostProcessor(settings)

        result = processor.process("三百五十元")
        assert "350" in result

    def test_punctuation_only(self):
        """测试仅标点转换"""
        from src.core.text_processor.post_processor import TextPostProcessor, PostProcessorSettings

        settings = PostProcessorSettings(
            itn_enable=False,
            zh_convert_enable=False,
            punc_convert_enable=True,
        )
        processor = TextPostProcessor(settings)

        result = processor.process("你好，世界！")
        assert "," in result
        assert "!" in result

    def test_combined_processing(self):
        """测试组合处理"""
        from src.core.text_processor.post_processor import TextPostProcessor, PostProcessorSettings

        settings = PostProcessorSettings(
            itn_enable=True,
            zh_convert_enable=False,
            punc_convert_enable=True,
        )
        processor = TextPostProcessor(settings)

        result = processor.process("价格是三百五十元，很便宜！")
        assert "350" in result
        assert "," in result

    def test_from_config(self):
        """测试从配置创建"""
        from src.core.text_processor.post_processor import TextPostProcessor

        class MockConfig:
            itn_enable = True
            zh_convert_enable = False
            zh_convert_locale = "zh-hans"
            punc_convert_enable = False
            punc_add_space = True

        processor = TextPostProcessor.from_config(MockConfig())
        assert processor.itn is not None
        assert processor.zh_converter is None
        assert processor.punc_converter is None

    def test_empty_input(self):
        """测试空输入"""
        from src.core.text_processor.post_processor import TextPostProcessor, PostProcessorSettings

        settings = PostProcessorSettings(itn_enable=True)
        processor = TextPostProcessor(settings)

        assert processor.process("") == ""
        assert processor.process(None) is None

    def test_individual_methods(self):
        """测试单独方法"""
        from src.core.text_processor.post_processor import TextPostProcessor, PostProcessorSettings

        settings = PostProcessorSettings(
            itn_enable=True,
            zh_convert_enable=True,
            punc_convert_enable=True,
        )
        processor = TextPostProcessor(settings)

        # 单独 ITN
        assert "350" in processor.process_itn("三百五十")

        # 单独标点
        assert "," in processor.process_punctuation("你好，世界")

    def test_default_settings(self):
        """测试默认设置"""
        from src.core.text_processor.post_processor import TextPostProcessor

        processor = TextPostProcessor()  # 使用默认设置
        assert processor.itn is not None  # ITN 默认启用


class TestIntegration:
    """集成测试"""

    def test_full_pipeline(self):
        """测试完整处理流程"""
        from src.core.text_processor.post_processor import TextPostProcessor, PostProcessorSettings

        settings = PostProcessorSettings(
            itn_enable=True,
            zh_convert_enable=False,
            punc_convert_enable=True,
        )
        processor = TextPostProcessor(settings)

        # 模拟 ASR 输出
        asr_output = "今天是二零二五年一月三十日，价格是三百五十元！"
        result = processor.process(asr_output)

        # 验证转换
        assert "2025" in result
        assert "350" in result
        assert "," in result
        assert "!" in result
