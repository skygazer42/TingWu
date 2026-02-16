"""集成测试"""
import pytest
from unittest.mock import patch, Mock, MagicMock


def test_full_pipeline():
    """测试完整转写流程"""
    import src.core.engine as engine_mod

    with patch.object(engine_mod, "model_manager") as mock_mm:
        mock_backend = MagicMock()
        mock_backend.supports_speaker = True
        mock_backend.transcribe.return_value = {
            "text": "买当劳很好吃",
            "sentence_info": [
                {"text": "买当劳很好吃", "start": 0, "end": 1500, "spk": 0}
            ]
        }
        mock_mm.backend = mock_backend

        from src.core.engine import TranscriptionEngine

        engine = TranscriptionEngine()
        engine.update_hotwords(["麦当劳", "肯德基"])

        result = engine.transcribe(
            b"fake_audio",
            with_speaker=True,
            apply_hotword=True
        )

        # 验证热词纠错生效
        assert "麦当劳" in result["text"]
        # 验证说话人标注
        assert result["sentences"][0]["speaker"] == "说话人甲"


def test_multi_speaker_pipeline():
    """测试多说话人流程"""
    import src.core.engine as engine_mod

    with patch.object(engine_mod, "model_manager") as mock_mm:
        mock_backend = MagicMock()
        mock_backend.supports_speaker = True
        mock_backend.transcribe.return_value = {
            "text": "你好世界你也好",
            "sentence_info": [
                {"text": "你好世界", "start": 0, "end": 1000, "spk": 0},
                {"text": "你也好", "start": 1200, "end": 2000, "spk": 1},
            ]
        }
        mock_mm.backend = mock_backend

        from src.core.engine import TranscriptionEngine

        engine = TranscriptionEngine()
        result = engine.transcribe(b"fake_audio", with_speaker=True)

        assert len(result["sentences"]) == 2
        assert result["sentences"][0]["speaker"] == "说话人甲"
        assert result["sentences"][1]["speaker"] == "说话人乙"
        assert "transcript" in result


def test_hotword_correction_accuracy():
    """测试热词纠错准确率"""
    from src.core.hotword import PhonemeCorrector

    corrector = PhonemeCorrector(threshold=0.8)
    corrector.update_hotwords("Claude\nBilibili\n麦当劳\n肯德基\nFunASR")

    test_cases = [
        ("Hello klaude", "Claude"),
        ("我要去买当劳", "麦当劳"),
        ("肯得鸡真好吃", "肯德基"),
    ]

    for input_text, expected_word in test_cases:
        result = corrector.correct(input_text)
        assert expected_word in result.text, f"Failed: '{input_text}' -> '{result.text}', expected '{expected_word}'"


def test_hotword_no_false_positives():
    """测试不误纠正"""
    from src.core.hotword import PhonemeCorrector

    corrector = PhonemeCorrector(threshold=0.8)
    corrector.update_hotwords("Claude\n麦当劳")

    safe_texts = [
        "今天天气不错",
        "我要去上班",
        "Python是一种编程语言",
    ]

    for text in safe_texts:
        result = corrector.correct(text)
        assert result.text == text, f"False positive: '{text}' -> '{result.text}'"


# ============================================================
# Phase 4 集成测试
# ============================================================

def test_text_corrector_integration():
    """测试 TextCorrector 集成"""
    try:
        from src.core.text_processor.text_corrector import TextCorrector
        corrector = TextCorrector(backend="kenlm")
        # 仅测试初始化，实际纠错需要 pycorrector 安装
        assert corrector._backend == "kenlm"
    except ImportError:
        pytest.skip("pycorrector not installed")


def test_punctuation_merge():
    """测试标点合并"""
    from src.core.text_processor.punctuation import merge_punctuation

    test_cases = [
        ("你好。。", "你好。"),
        ("你好,,", "你好,"),
        ("你好。.", "你好。"),
        ("多余  空格", "多余 空格"),
    ]

    for input_text, expected in test_cases:
        result = merge_punctuation(input_text)
        assert result == expected, f"Expected '{expected}', got '{result}'"


def test_llm_corrector_role():
    """测试 LLM 纠错角色"""
    from src.core.llm.roles import get_role

    role = get_role("corrector")
    assert role is not None
    assert role.name == "corrector"
    assert "修正" in role.system_prompt or "纠错" in role.system_prompt


def test_hotword_cache():
    """测试热词缓存"""
    from src.core.hotword import PhonemeCorrector

    corrector = PhonemeCorrector(threshold=0.8, cache_size=100)
    corrector.update_hotwords("测试热词")

    # 第一次调用
    result1 = corrector.correct("这是测试热词")

    # 第二次调用（应该命中缓存）
    result2 = corrector.correct("这是测试热词")

    assert result1.text == result2.text

    # 验证更新热词时清空缓存
    assert len(corrector._cache) > 0
    corrector.update_hotwords("热词B")
    assert len(corrector._cache) == 0


def test_audio_preprocessor_denoise_fallback():
    """测试音频预处理器降噪降级"""
    import numpy as np
    from src.core.audio import AudioPreprocessor

    processor = AudioPreprocessor(denoise_enable=True, denoise_backend="deepfilter")

    # 即使 DeepFilterNet 不可用，也应该优雅降级
    audio = np.random.randn(16000).astype(np.float32) * 0.1
    result = processor.denoise(audio, 16000)

    # 应该返回有效音频
    assert len(result) == len(audio)


def test_config_api_keys():
    """测试配置 API 键"""
    from src.api.routes.config import MUTABLE_CONFIG_KEYS, get_current_config

    config = get_current_config()
    assert len(config) > 0
    assert all(key in MUTABLE_CONFIG_KEYS for key in config.keys())


def test_prompt_builder_token_estimation():
    """测试 PromptBuilder token 估算"""
    from src.core.llm.prompt_builder import PromptBuilder

    builder = PromptBuilder(max_tokens=4096)

    # 测试中文估算
    cn_text = "这是一段中文测试文本"
    tokens = builder._estimate_tokens(cn_text)
    assert tokens > 0

    # 测试英文估算
    en_text = "This is an English test text"
    tokens = builder._estimate_tokens(en_text)
    assert tokens > 0


def test_correction_pipeline_config():
    """测试纠错管线配置"""
    from src.config import settings

    # 验证默认管线配置格式
    pipeline = settings.correction_pipeline.split(',')
    assert len(pipeline) > 0
    assert "post_process" in pipeline
