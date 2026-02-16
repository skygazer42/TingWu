import pytest


def _merge_by_text(prev_text: str, new_text: str, **kwargs) -> str:
    try:
        from src.core.text_processor.text_merge import merge_by_text
    except Exception as e:
        pytest.fail(f"merge_by_text not available: {e}")
    return merge_by_text(prev_text, new_text, **kwargs)


def test_merge_by_text_perfect_overlap():
    assert _merge_by_text("今天天气真", "天气真好啊", overlap_chars=20, error_tolerance=0) == "今天天气真好啊"


def test_merge_by_text_skips_noise_prefix():
    assert _merge_by_text("今天天气真", "嗯天气真好啊", overlap_chars=20, error_tolerance=0) == "今天天气真好啊"


def test_merge_by_text_discards_prev_drift_tail_and_new_prefix():
    assert (
        _merge_by_text("今天天气真的", "好的天气真好啊", overlap_chars=20, error_tolerance=0)
        == "今天天气真好啊"
    )


def test_merge_by_text_handles_punctuation_boundaries():
    assert (
        _merge_by_text("你好，世界。", "。世界真好", overlap_chars=20, error_tolerance=0)
        == "你好，世界真好"
    )


def test_merge_by_text_fallback_concat_when_no_overlap():
    assert _merge_by_text("你好", "世界", overlap_chars=20, error_tolerance=0) == "你好世界"

