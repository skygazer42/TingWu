import pytest
from unittest.mock import patch, Mock, MagicMock

@pytest.fixture
def mock_model_manager():
    with patch('src.core.engine.model_manager') as mock_mm:
        # Engine now uses model_manager.backend.transcribe(...) as the primary path.
        mock_backend = MagicMock()
        mock_backend.supports_speaker = True
        mock_backend.transcribe.return_value = {
            "text": "买当劳很好吃",
            "sentence_info": [
                {"text": "买当劳很好吃", "start": 0, "end": 1500, "spk": 0}
            ]
        }
        mock_mm.backend = mock_backend
        yield mock_mm

def test_engine_initialization(mock_model_manager):
    """测试引擎初始化"""
    from src.core.engine import TranscriptionEngine
    engine = TranscriptionEngine()
    assert engine is not None
    assert engine.corrector is not None
    assert engine.speaker_labeler is not None

def test_transcribe_basic(mock_model_manager):
    """测试基本转写"""
    from src.core.engine import TranscriptionEngine
    engine = TranscriptionEngine()

    result = engine.transcribe(b"fake_audio")

    assert result is not None
    assert "text" in result
    assert "sentences" in result

def test_transcribe_with_hotword(mock_model_manager):
    """测试带热词纠错的转写"""
    from src.core.engine import TranscriptionEngine
    engine = TranscriptionEngine()
    engine.update_hotwords(["麦当劳", "肯德基"])

    result = engine.transcribe(b"fake_audio", apply_hotword=True)

    # 热词纠错应该将 "买当劳" 替换为 "麦当劳"
    assert "麦当劳" in result["text"]

def test_transcribe_with_speaker(mock_model_manager):
    """测试带说话人识别的转写"""
    from src.core.engine import TranscriptionEngine
    engine = TranscriptionEngine()

    result = engine.transcribe(b"fake_audio", with_speaker=True)

    assert len(result["sentences"]) == 1
    assert "speaker" in result["sentences"][0]
    assert result["sentences"][0]["speaker"] == "说话人甲"

def test_update_hotwords(mock_model_manager):
    """测试更新热词"""
    from src.core.engine import TranscriptionEngine
    engine = TranscriptionEngine()

    engine.update_hotwords(["Claude", "Bilibili"])

    assert engine._hotwords_loaded == True
    assert len(engine.corrector.hotwords) == 2

def test_transcribe_generates_transcript(mock_model_manager):
    """测试生成格式化转写稿"""
    from src.core.engine import TranscriptionEngine
    engine = TranscriptionEngine()

    result = engine.transcribe(b"fake_audio", with_speaker=True)

    assert "transcript" in result
    assert "说话人甲" in result["transcript"]

def test_transcribe_without_hotword(mock_model_manager):
    """测试禁用热词纠错"""
    from src.core.engine import TranscriptionEngine
    engine = TranscriptionEngine()
    engine.update_hotwords(["麦当劳"])

    result = engine.transcribe(b"fake_audio", apply_hotword=False)

    # 禁用热词时应该保留原文
    assert result["text"] == "买当劳很好吃"
