import pytest

from src.api.asr_options import parse_asr_options


def test_asr_options_chunking_strategy_normalizes():
    opts = parse_asr_options('{"chunking":{"strategy":" TIME "}}')
    assert opts is not None
    assert opts["chunking"]["strategy"] == "time"


def test_asr_options_chunking_strategy_rejects_unknown():
    with pytest.raises(ValueError, match="chunking\\.strategy"):
        parse_asr_options('{"chunking":{"strategy":"vad"}}')


def test_asr_options_preprocess_highpass_and_limiter_keys_allowed():
    opts = parse_asr_options(
        '{"preprocess":{"highpass_enable":true,"highpass_cutoff_hz":200,"soft_limit_enable":true,"soft_limit_target":0.98,"soft_limit_knee":3.0}}'
    )
    assert opts is not None
    assert opts["preprocess"]["highpass_enable"] is True
    assert opts["preprocess"]["highpass_cutoff_hz"] == 200
    assert opts["preprocess"]["soft_limit_enable"] is True


def test_asr_options_preprocess_highpass_cutoff_must_be_positive():
    with pytest.raises(ValueError, match="highpass_cutoff_hz"):
        parse_asr_options('{"preprocess":{"highpass_enable":true,"highpass_cutoff_hz":-1}}')
