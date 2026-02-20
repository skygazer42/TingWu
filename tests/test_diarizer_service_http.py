import io
import wave

from fastapi.testclient import TestClient


def _make_wav_bytes(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    nframes = int(duration_s * sample_rate)
    pcm16le = b"\x00\x00" * nframes
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16le)
    return buf.getvalue()


def test_diarizer_health_and_diarize_returns_segments(monkeypatch):
    from src.diarizer_service.app import app
    import src.diarizer_service.routes as routes_mod

    monkeypatch.setattr(
        routes_mod.engine,
        "diarize",
        lambda _wav: [
            {"spk": 0, "start": 0, "end": 500},
            {"spk": 1, "start": 500, "end": 1000},
        ],
    )

    with TestClient(app) as c:
        assert c.get("/health").status_code == 200

        wav_bytes = _make_wav_bytes(duration_s=1.0, sample_rate=16000)
        resp = c.post("/api/v1/diarize", files={"file": ("a.wav", wav_bytes, "audio/wav")})
        assert resp.status_code == 200
        data = resp.json()
        assert data["segments"] == [
            {"spk": 0, "start": 0, "end": 500},
            {"spk": 1, "start": 500, "end": 1000},
        ]
        assert data["duration_ms"] == 1000
        assert data["speakers"] == 2
