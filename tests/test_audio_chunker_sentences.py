from src.core.audio.chunker import AudioChunker


def test_audio_chunker_merge_results_dedupes_overlap_sentences_by_time_and_text():
    sr = 16000
    chunker = AudioChunker()

    # Two chunks overlap by 1s: [0s..10s] and [9s..18s]
    merged = chunker.merge_results(
        [
            {
                "start_sample": 0,
                "end_sample": 10 * sr,
                "success": True,
                "result": {
                    "text": "今天天气真好啊",
                    "sentences": [{"text": "今天天气真好啊", "start": 0, "end": 10_000}],
                },
            },
            {
                "start_sample": 9 * sr,
                "end_sample": 18 * sr,
                "success": True,
                "result": {
                    "text": "好啊我们出发",
                    "sentences": [
                        # Fully inside overlap region (<= 1s from chunk start) and duplicated by text.
                        {"text": "好啊", "start": 0, "end": 700},
                        # New content beyond overlap.
                        {"text": "我们出发", "start": 700, "end": 4_000},
                    ],
                },
            },
        ],
        sample_rate=sr,
        overlap_chars=20,
    )

    assert merged["text"] == "今天天气真好啊我们出发"

    # Sentence "好啊" from the second chunk should be dropped.
    assert [s["text"] for s in merged["sentences"]] == ["今天天气真好啊", "我们出发"]

    # Timestamps should be offset to global time.
    assert merged["sentences"][0]["start"] == 0
    assert merged["sentences"][0]["end"] == 10_000
    assert merged["sentences"][1]["start"] == 9_000 + 700
    assert merged["sentences"][1]["end"] == 9_000 + 4_000

