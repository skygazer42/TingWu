from src.core.audio.chunker import AudioChunker


def test_audio_chunker_merge_results_dedupes_noise_prefix_overlap():
    chunker = AudioChunker()

    merged = chunker.merge_results(
        [
            {
                "start_sample": 0,
                "end_sample": 16000,
                "success": True,
                "result": {"text": "今天天气真", "sentences": []},
            },
            {
                "start_sample": 16000,
                "end_sample": 32000,
                "success": True,
                "result": {"text": "嗯天气真好啊", "sentences": []},
            },
        ],
        sample_rate=16000,
        overlap_chars=20,
    )

    assert merged["text"] == "今天天气真好啊"

