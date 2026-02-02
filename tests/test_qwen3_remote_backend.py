from unittest.mock import patch


def test_qwen3_remote_backend_calls_transcriptions_endpoint():
    from src.models.backends.qwen3_remote import Qwen3RemoteBackend

    backend = Qwen3RemoteBackend(
        base_url="http://fake",
        model="Qwen/Qwen3-ASR-1.7B",
        api_key="EMPTY",
        timeout_s=1.0,
    )

    class Resp:
        status_code = 200

        def json(self):
            return {"text": "ok"}

        def raise_for_status(self):
            return None

    with patch("httpx.Client.post", return_value=Resp()) as post:
        out = backend.transcribe(b"\x00\x00" * 16000, hotwords="foo bar")
        assert out["text"] == "ok"
        assert "sentence_info" in out
        assert post.called

