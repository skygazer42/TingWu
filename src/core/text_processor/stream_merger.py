"""
流式文本去重器

处理 WebSocket 流式识别产生的重叠文本:
- 在线 ASR 缓存重置时的重叠
- 在线 + 离线两阶段级联的重叠

算法参考 CapsWriter-Offline text_merge.py
"""

from typing import Tuple, Optional

__all__ = ['StreamTextMerger']


def levenshtein_distance(s1: str, s2: str) -> int:
    """计算编辑距离 (空间优化版本)"""
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, 1):
        curr_row = [i]
        for j, c2 in enumerate(s2, 1):
            if c1 == c2:
                curr_row.append(prev_row[j - 1])
            else:
                curr_row.append(min(prev_row[j], curr_row[j - 1], prev_row[j - 1]) + 1)
        prev_row = curr_row

    return prev_row[-1]


class StreamTextMerger:
    """
    流式文本合并器

    合并策略:
    1. 精确匹配: 检查 old[-overlap:] == new[:overlap]
    2. 模糊匹配: 允许一定编辑距离容差

    用法:
        merger = StreamTextMerger()

        # 流式接收文本
        delta1 = merger.merge("今天天气")      # "今天天气"
        delta2 = merger.merge("天气很好")      # "很好" (去重)
        delta3 = merger.merge("很好明天见")    # "明天见" (去重)

        # 获取完整文本
        full_text = merger.get_full_text()     # "今天天气很好明天见"
    """

    def __init__(
        self,
        overlap_chars: int = 5,
        error_tolerance: int = 1,
        max_overlap_check: int = 20,
    ):
        """
        初始化合并器

        Args:
            overlap_chars: 检查重叠的字符数
            error_tolerance: 模糊匹配允许的编辑距离
            max_overlap_check: 最大重叠检查长度
        """
        self.overlap_chars = overlap_chars
        self.error_tolerance = error_tolerance
        self.max_overlap_check = max_overlap_check
        self.buffer = ""

    def reset(self):
        """重置缓冲区"""
        self.buffer = ""

    def get_full_text(self) -> str:
        """获取完整合并文本"""
        return self.buffer

    def merge(self, new_text: str) -> str:
        """
        合并新文本，返回去重后的增量

        Args:
            new_text: 新接收的文本

        Returns:
            去重后应该追加的增量文本
        """
        if not new_text:
            return ""

        if not self.buffer:
            self.buffer = new_text
            return new_text

        # 查找重叠
        overlap_len, is_exact = self._find_overlap(self.buffer, new_text)

        if overlap_len > 0:
            # 有重叠，只追加新的部分
            delta = new_text[overlap_len:]
            self.buffer += delta
            return delta
        else:
            # 无重叠，完整追加
            self.buffer += new_text
            return new_text

    def _find_overlap(self, old_text: str, new_text: str) -> Tuple[int, bool]:
        """
        查找重叠长度

        Args:
            old_text: 现有文本
            new_text: 新文本

        Returns:
            (重叠长度, 是否精确匹配)
        """
        max_check = min(
            len(old_text),
            len(new_text),
            self.max_overlap_check
        )

        if max_check < 2:
            return 0, False

        # Phase 1: 精确匹配
        for overlap in range(min(max_check, self.overlap_chars), 0, -1):
            old_suffix = old_text[-overlap:]
            new_prefix = new_text[:overlap]
            if old_suffix == new_prefix:
                return overlap, True

        # Phase 2: 模糊匹配 (允许容差)
        if self.error_tolerance > 0:
            for overlap in range(self.overlap_chars, 1, -1):
                if overlap > len(old_text) or overlap > len(new_text):
                    continue

                old_suffix = old_text[-overlap:]
                new_prefix = new_text[:overlap]
                distance = levenshtein_distance(old_suffix, new_prefix)

                if distance <= self.error_tolerance:
                    return overlap, False

        return 0, False

    def merge_final(self, final_text: str) -> str:
        """
        合并最终文本 (通常是离线识别结果)

        最终文本会替换之前的在线识别结果，但尝试保留已确认的前缀。

        Args:
            final_text: 最终识别结果

        Returns:
            应该显示的完整文本
        """
        if not self.buffer:
            self.buffer = final_text
            return final_text

        # 比较在线和离线结果，取更长的公共前缀
        common_len = 0
        for i, (c1, c2) in enumerate(zip(self.buffer, final_text)):
            if c1 == c2:
                common_len = i + 1
            else:
                break

        # 如果离线结果更可靠，使用离线结果
        # 但保留在线识别已确认的部分
        if len(final_text) >= len(self.buffer) * 0.8:
            self.buffer = final_text
        else:
            # 在线结果更长，可能是离线漏识别
            # 合并: 取离线结果 + 在线多出的部分
            if common_len > 0:
                extra = self.buffer[common_len:]
                if extra and not final_text.endswith(extra):
                    self.buffer = final_text + extra
                else:
                    self.buffer = final_text
            else:
                self.buffer = final_text

        return self.buffer


if __name__ == '__main__':
    print("=== StreamTextMerger 测试 ===")

    merger = StreamTextMerger(overlap_chars=5, error_tolerance=1)

    # 测试 1: 基本重叠
    print("\n测试 1: 基本重叠")
    test_sequences = [
        "今天天气",
        "天气很好",
        "很好明天见",
    ]

    for text in test_sequences:
        delta = merger.merge(text)
        print(f"  输入: '{text}' -> 增量: '{delta}' | 完整: '{merger.get_full_text()}'")

    print(f"  期望: '今天天气很好明天见'")
    print(f"  实际: '{merger.get_full_text()}'")

    # 测试 2: 模糊匹配
    print("\n测试 2: 模糊匹配 (1字差异)")
    merger.reset()
    merger.merge("人工智能")
    delta = merger.merge("智慧很厉害")  # "智慧" vs "智能"，差1字
    print(f"  输入: '人工智能' + '智慧很厉害'")
    print(f"  增量: '{delta}'")
    print(f"  完整: '{merger.get_full_text()}'")

    # 测试 3: 无重叠
    print("\n测试 3: 无重叠")
    merger.reset()
    merger.merge("你好")
    delta = merger.merge("世界")
    print(f"  输入: '你好' + '世界'")
    print(f"  增量: '{delta}'")
    print(f"  完整: '{merger.get_full_text()}'")

    # 测试 4: 最终文本合并
    print("\n测试 4: 最终文本合并 (在线 vs 离线)")
    merger.reset()
    merger.merge("今天天汽")  # 在线识别 (有错字)
    final = merger.merge_final("今天天气很好")  # 离线识别 (正确)
    print(f"  在线: '今天天汽'")
    print(f"  离线: '今天天气很好'")
    print(f"  合并: '{final}'")

    print("\n✓ 测试完成")
