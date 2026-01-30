import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock, MagicMock, AsyncMock
import io

@pytest.fixture
def client():
    with patch('src.core.engine.model_manager') as mock_mm:
        mock_loader = MagicMock()
        mock_mm.loader = mock_loader
        mock_loader.transcribe.return_value = {
            "text": "你好世界",
            "sentence_info": [{"text": "你好世界", "start": 0, "end": 1000}]
        }

        from src.main import app
        with TestClient(app) as c:
            yield c

def test_health_check(client):
    """测试健康检查接口"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_root_endpoint(client):
    """测试根路径"""
    response = client.get("/")
    assert response.status_code == 200
    assert "name" in response.json()

def test_transcribe_endpoint(client):
    """测试转写接口"""
    with patch('src.core.engine.transcription_engine') as mock_engine, \
         patch('src.api.routes.transcribe.process_audio_file') as mock_process:
        mock_engine.transcribe.return_value = {
            "text": "你好世界",
            "sentences": [{"text": "你好世界", "start": 0, "end": 1000}],
            "raw_text": "你好世界"
        }
        mock_engine._hotwords_loaded = False

        async def fake_process(file):
            yield b"\x00" * 32000
        mock_process.side_effect = fake_process

        audio_content = b"fake_audio_content_wav_header"
        files = {"file": ("test.wav", io.BytesIO(audio_content), "audio/wav")}

        response = client.post("/api/v1/transcribe", files=files)

        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "sentences" in data

def test_transcribe_with_speaker(client):
    """测试带说话人的转写"""
    with patch('src.core.engine.transcription_engine') as mock_engine, \
         patch('src.api.routes.transcribe.process_audio_file') as mock_process:
        mock_engine.transcribe.return_value = {
            "text": "你好",
            "sentences": [{"text": "你好", "start": 0, "end": 500, "speaker": "说话人甲", "speaker_id": 0}],
            "transcript": "[00:00 - 00:00] 说话人甲: 你好",
            "raw_text": "你好"
        }
        mock_engine._hotwords_loaded = False

        async def fake_process(file):
            yield b"\x00" * 16000
        mock_process.side_effect = fake_process

        files = {"file": ("test.wav", io.BytesIO(b"fake"), "audio/wav")}
        data = {"with_speaker": "true"}

        response = client.post("/api/v1/transcribe", files=files, data=data)

        assert response.status_code == 200
        result = response.json()
        assert "transcript" in result

def test_transcribe_no_file(client):
    """测试无文件上传"""
    response = client.post("/api/v1/transcribe")
    assert response.status_code == 422  # Validation error
