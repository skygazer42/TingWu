import pytest
from src.core.speaker.diarization import SpeakerLabeler

def test_speaker_labeling():
    """测试说话人标注"""
    labeler = SpeakerLabeler()
    sentences = [
        {"text": "你好", "start": 0, "end": 1000, "spk": 0},
        {"text": "你好呀", "start": 1200, "end": 2000, "spk": 1},
        {"text": "今天天气不错", "start": 2500, "end": 4000, "spk": 0},
    ]

    result = labeler.label_speakers(sentences)

    assert len(result) == 3
    assert result[0]["speaker"] == "说话人甲"
    assert result[1]["speaker"] == "说话人乙"
    assert result[2]["speaker"] == "说话人甲"

def test_speaker_labels_cycle():
    """测试多说话人标签"""
    labeler = SpeakerLabeler()
    sentences = [
        {"text": "A", "spk": 0},
        {"text": "B", "spk": 1},
        {"text": "C", "spk": 2},
        {"text": "D", "spk": 3},
    ]

    result = labeler.label_speakers(sentences)

    assert result[0]["speaker"] == "说话人甲"
    assert result[1]["speaker"] == "说话人乙"
    assert result[2]["speaker"] == "说话人丙"
    assert result[3]["speaker"] == "说话人丁"

def test_format_transcript():
    """测试转写稿格式化"""
    labeler = SpeakerLabeler()
    sentences = [
        {"text": "你好", "start": 0, "end": 1000, "speaker": "说话人甲"},
        {"text": "你好呀", "start": 1200, "end": 2000, "speaker": "说话人乙"},
    ]

    transcript = labeler.format_transcript(sentences)

    assert "说话人甲" in transcript
    assert "说话人乙" in transcript
    assert "你好" in transcript

def test_speaker_id_mapping():
    """测试说话人ID映射"""
    labeler = SpeakerLabeler()
    sentences = [
        {"text": "A", "spk": 5},  # 非顺序 ID
        {"text": "B", "spk": 2},
        {"text": "C", "spk": 5},  # 重复 ID
    ]

    result = labeler.label_speakers(sentences)

    # 第一个出现的 spk=5 应该是甲
    assert result[0]["speaker"] == "说话人甲"
    # 第二个出现的 spk=2 应该是乙
    assert result[1]["speaker"] == "说话人乙"
    # spk=5 再次出现还是甲
    assert result[2]["speaker"] == "说话人甲"

def test_missing_speaker():
    """测试缺失说话人ID"""
    labeler = SpeakerLabeler()
    sentences = [
        {"text": "A"},  # 没有 spk 字段
        {"text": "B", "spk": 0},
    ]

    result = labeler.label_speakers(sentences)

    assert result[0]["speaker"] == "未知"
    assert result[1]["speaker"] == "说话人甲"

def test_time_formatting():
    """测试时间格式化"""
    labeler = SpeakerLabeler()

    assert labeler._format_time(0) == "00:00"
    assert labeler._format_time(65000) == "01:05"
    assert labeler._format_time(3661000) == "01:01:01"
