"""说话人标注模块"""
from typing import List, Dict, Any, Optional

SPEAKER_LABELS_ZH = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛"]


class SpeakerLabeler:
    """说话人标注器"""

    def __init__(self, label_prefix: str = "说话人"):
        self.label_prefix = label_prefix
        self.labels = SPEAKER_LABELS_ZH

    def _get_speaker_label(self, spk_id: int) -> str:
        """获取说话人标签"""
        if spk_id < len(self.labels):
            return f"{self.label_prefix}{self.labels[spk_id]}"
        return f"{self.label_prefix}{spk_id + 1}"

    def label_speakers(
        self,
        sentences: List[Dict[str, Any]],
        spk_key: str = "spk"
    ) -> List[Dict[str, Any]]:
        """为句子添加说话人标签"""
        result = []
        spk_mapping = {}  # 原始 ID -> 顺序 ID

        for sent in sentences:
            sent_copy = dict(sent)
            spk_id = sent.get(spk_key)

            if spk_id is not None:
                if spk_id not in spk_mapping:
                    spk_mapping[spk_id] = len(spk_mapping)

                mapped_id = spk_mapping[spk_id]
                sent_copy["speaker"] = self._get_speaker_label(mapped_id)
                sent_copy["speaker_id"] = mapped_id
            else:
                sent_copy["speaker"] = "未知"
                sent_copy["speaker_id"] = -1

            result.append(sent_copy)

        return result

    def format_transcript(
        self,
        sentences: List[Dict[str, Any]],
        include_timestamp: bool = True
    ) -> str:
        """格式化为转写文本"""
        lines = []
        for sent in sentences:
            speaker = sent.get("speaker", "未知")
            text = sent.get("text", "")

            if include_timestamp:
                start = sent.get("start", 0)
                end = sent.get("end", 0)
                timestamp = f"[{self._format_time(start)} - {self._format_time(end)}]"
                lines.append(f"{timestamp} {speaker}: {text}")
            else:
                lines.append(f"{speaker}: {text}")

        return "\n".join(lines)

    @staticmethod
    def _format_time(ms: int) -> str:
        """格式化毫秒为时间字符串"""
        seconds = ms // 1000
        minutes = seconds // 60
        hours = minutes // 60

        if hours > 0:
            return f"{hours:02d}:{minutes % 60:02d}:{seconds % 60:02d}"
        return f"{minutes:02d}:{seconds % 60:02d}"
