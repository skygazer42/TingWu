import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock, MagicMock, AsyncMock
import io

@pytest.fixture
def client():
    with patch('src.core.engine.model_manager') as mock_mm:
        # App lifespan triggers `transcription_engine.warmup()`, which uses
        # `model_manager.backend`. Mock backend to avoid loading real models.
        mock_backend = MagicMock()
        mock_backend.get_info.return_value = {"name": "MockBackend", "type": "mock"}
        mock_backend.warmup.return_value = None
        mock_backend.supports_speaker = True
        mock_backend.transcribe.return_value = {
            "text": "你好世界",
            "sentence_info": [{"text": "你好世界", "start": 0, "end": 1000}]
        }
        mock_mm.backend = mock_backend

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
    # Route uses `await transcription_engine.transcribe_async(...)`, so patch that
    # symbol where it is imported/used (src.api.routes.transcribe).
    with patch('src.api.routes.transcribe.transcription_engine.transcribe_async', new_callable=AsyncMock) as mock_transcribe_async, \
         patch('src.api.routes.transcribe.process_audio_file') as mock_process:
        mock_transcribe_async.return_value = {
            "text": "你好世界",
            "sentences": [{"text": "你好世界", "start": 0, "end": 1000}],
            "raw_text": "你好世界"
        }

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
    with patch('src.api.routes.transcribe.transcription_engine.transcribe_async', new_callable=AsyncMock) as mock_transcribe_async, \
         patch('src.api.routes.transcribe.process_audio_file') as mock_process:
        mock_transcribe_async.return_value = {
            "text": "你好",
            "sentences": [{"text": "你好", "start": 0, "end": 500, "speaker": "说话人甲", "speaker_id": 0}],
            "transcript": "[00:00 - 00:00] 说话人甲: 你好",
            "raw_text": "你好"
        }

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
