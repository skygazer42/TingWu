import pytest
from unittest.mock import patch, Mock, MagicMock

def test_model_loader_initialization():
    """测试模型加载器可以正确初始化"""
    from src.models.asr_loader import ASRModelLoader
    loader = ASRModelLoader(device="cpu", ngpu=0)
    assert loader is not None
    assert loader.device == "cpu"
    assert loader.ngpu == 0

def test_model_loader_lazy_load():
    """测试模型懒加载机制 - 初始化时不加载模型"""
    from src.models.asr_loader import ASRModelLoader
    loader = ASRModelLoader(device="cpu", ngpu=0)
    assert loader._asr_model is None
    assert loader._asr_model_online is None
    assert loader._asr_model_with_spk is None

def test_model_loader_config():
    """测试模型名称配置正确"""
    from src.models.asr_loader import ASRModelLoader
    loader = ASRModelLoader(
        device="cpu", ngpu=0,
        asr_model="test-model",
        vad_model="test-vad"
    )
    assert loader._asr_model_name == "test-model"
    assert loader._vad_model_name == "test-vad"

@patch('src.models.asr_loader.AutoModel')
def test_transcribe_calls_generate(mock_auto_model):
    """测试 transcribe 调用模型 generate"""
    from src.models.asr_loader import ASRModelLoader

    mock_instance = MagicMock()
    mock_instance.generate.return_value = [{"text": "你好", "sentence_info": []}]
    mock_auto_model.return_value = mock_instance

    loader = ASRModelLoader(device="cpu", ngpu=0)
    result = loader.transcribe(b"fake_audio")

    assert result["text"] == "你好"
    mock_instance.generate.assert_called_once()

@patch('src.models.asr_loader.AutoModel')
def test_transcribe_with_speaker(mock_auto_model):
    """测试带说话人的 transcribe"""
    from src.models.asr_loader import ASRModelLoader

    mock_instance = MagicMock()
    mock_instance.generate.return_value = [{
        "text": "你好世界",
        "sentence_info": [{"text": "你好", "spk": 0}]
    }]
    mock_auto_model.return_value = mock_instance

    loader = ASRModelLoader(device="cpu", ngpu=0)
    result = loader.transcribe(b"fake_audio", with_speaker=True)

    assert result["text"] == "你好世界"

def test_model_manager_singleton():
    """测试 ModelManager 单例模式"""
    from src.models.model_manager import ModelManager
    m1 = ModelManager()
    m2 = ModelManager()
    assert m1 is m2
