import asyncio
import pytest
from unittest.mock import patch

import httpx

from src.core.speaker.external_diarizer_client import fetch_diarizer_segments


def test_fetch_diarizer_segments_parses_segments():
    class FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"segments": [{"spk": 0, "start": 0, "end": 1000}]}

    async def fake_post(self, url, data=None, files=None):
        assert url.endswith("/api/v1/diarize")
        return FakeResp()

    with patch.object(httpx.AsyncClient, "post", new=fake_post):
        segs = asyncio.run(
            fetch_diarizer_segments(
                base_url="http://diarizer:8000",
                wav_bytes=b"RIFF....",
                timeout_s=5.0,
            )
        )

    assert segs == [{"spk": 0, "start": 0, "end": 1000}]
