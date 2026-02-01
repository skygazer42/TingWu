"""
ASR 评估指标计算

提供:
- CER (Character Error Rate): 字符错误率
- WER (Word Error Rate): 词错误率
- Levenshtein 编辑距离
"""

import re
import unicodedata
from typing import List, Tuple

__all__ = [
    'levenshtein_distance',
    'normalize_for_cer',
    'normalize_for_wer',
    'calculate_cer',
    'calculate_wer',
    'calculate_cer_details',
]


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    计算两个字符串的 Levenshtein 编辑距离

    Args:
        s1: 第一个字符串 (通常是假设 hypothesis)
        s2: 第二个字符串 (通常是参考 reference)

    Returns:
        编辑距离 (插入、删除、替换的最小次数)
    """
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)

    # 使用空间优化的动态规划
    prev_row = list(range(len(s2) + 1))
    curr_row = [0] * (len(s2) + 1)

    for i, c1 in enumerate(s1, 1):
        curr_row[0] = i
        for j, c2 in enumerate(s2, 1):
            if c1 == c2:
                curr_row[j] = prev_row[j - 1]
            else:
                curr_row[j] = min(
                    prev_row[j] + 1,      # 删除
                    curr_row[j - 1] + 1,  # 插入
                    prev_row[j - 1] + 1   # 替换
                )
        prev_row, curr_row = curr_row, prev_row

    return prev_row[len(s2)]


def levenshtein_distance_with_ops(s1: str, s2: str) -> Tuple[int, int, int, int]:
    """
    计算编辑距离并返回操作计数

    Returns:
        (distance, substitutions, insertions, deletions)
    """
    m, n = len(s1), len(s2)
    if m == 0:
        return n, 0, n, 0
    if n == 0:
        return m, 0, 0, m

    # 完整的 DP 表用于回溯
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

    # 回溯计算操作数
    i, j = m, n
    subs, ins, dels = 0, 0, 0

    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i - 1] == s2[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ins += 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            break

    return dp[m][n], subs, ins, dels


def normalize_for_cer(text: str) -> str:
    """
    CER 计算前的文本规范化

    处理:
    - 移除空格和标点
    - Unicode NFKC 规范化
    - 统一全角/半角
    - 转小写 (对英文)

    Args:
        text: 输入文本

    Returns:
        规范化后的文本
    """
    if not text:
        return ""

    # Unicode NFKC 规范化 (全角→半角等)
    text = unicodedata.normalize('NFKC', text)

    # 转小写
    text = text.lower()

    # 移除标点和空格
    # 保留: 中文字符、英文字母、数字
    text = re.sub(r'[^\u4e00-\u9fff\u3400-\u4dbfa-z0-9]', '', text)

    return text


def normalize_for_wer(text: str) -> List[str]:
    """
    WER 计算前的文本规范化

    处理:
    - 中文按字分词
    - 英文按空格分词
    - 移除标点

    Args:
        text: 输入文本

    Returns:
        词列表
    """
    if not text:
        return []

    # Unicode NFKC 规范化
    text = unicodedata.normalize('NFKC', text)
    text = text.lower()

    # 分词: 中文按字，英文按空格
    words = []
    current_word = []
    prev_is_cjk = None

    for char in text:
        # 判断是否为 CJK 字符
        is_cjk = '\u4e00' <= char <= '\u9fff' or '\u3400' <= char <= '\u4dbf'
        is_alpha = char.isalpha() and not is_cjk
        is_digit = char.isdigit()

        if is_cjk:
            # CJK 字符单独成词
            if current_word:
                words.append(''.join(current_word))
                current_word = []
            words.append(char)
            prev_is_cjk = True
        elif is_alpha or is_digit:
            # 英文/数字连续
            if prev_is_cjk and current_word:
                words.append(''.join(current_word))
                current_word = []
            current_word.append(char)
            prev_is_cjk = False
        else:
            # 空格/标点分隔
            if current_word:
                words.append(''.join(current_word))
                current_word = []
            prev_is_cjk = None

    if current_word:
        words.append(''.join(current_word))

    return words


def calculate_cer(hypothesis: str, reference: str) -> float:
    """
    计算字符错误率 (Character Error Rate)

    CER = EditDistance(hyp, ref) / len(ref)

    Args:
        hypothesis: 识别结果
        reference: 参考文本

    Returns:
        CER 值 (0.0 ~ 1.0+，可能超过 1.0)
    """
    hyp = normalize_for_cer(hypothesis)
    ref = normalize_for_cer(reference)

    if not ref:
        return 0.0 if not hyp else 1.0

    distance = levenshtein_distance(hyp, ref)
    return distance / len(ref)


def calculate_cer_details(hypothesis: str, reference: str) -> dict:
    """
    计算 CER 并返回详细信息

    Args:
        hypothesis: 识别结果
        reference: 参考文本

    Returns:
        包含 CER、替换/插入/删除数的字典
    """
    hyp = normalize_for_cer(hypothesis)
    ref = normalize_for_cer(reference)

    if not ref:
        return {
            'cer': 0.0 if not hyp else 1.0,
            'distance': len(hyp),
            'substitutions': 0,
            'insertions': len(hyp),
            'deletions': 0,
            'reference_length': 0,
            'hypothesis_length': len(hyp),
        }

    distance, subs, ins, dels = levenshtein_distance_with_ops(hyp, ref)

    return {
        'cer': distance / len(ref),
        'distance': distance,
        'substitutions': subs,
        'insertions': ins,
        'deletions': dels,
        'reference_length': len(ref),
        'hypothesis_length': len(hyp),
    }


def calculate_wer(hypothesis: str, reference: str) -> float:
    """
    计算词错误率 (Word Error Rate)

    WER = EditDistance(hyp_words, ref_words) / len(ref_words)

    对于中文，每个字视为一个词。

    Args:
        hypothesis: 识别结果
        reference: 参考文本

    Returns:
        WER 值 (0.0 ~ 1.0+)
    """
    hyp_words = normalize_for_wer(hypothesis)
    ref_words = normalize_for_wer(reference)

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    # 将词列表转为字符串进行编辑距离计算
    # 使用特殊分隔符避免词边界问题
    sep = '\x00'
    hyp_str = sep.join(hyp_words)
    ref_str = sep.join(ref_words)

    # 计算词级别的编辑距离
    m, n = len(hyp_words), len(ref_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if hyp_words[i - 1] == ref_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

    return dp[m][n] / len(ref_words)


if __name__ == '__main__':
    # 测试
    print("=== CER 测试 ===")
    test_cases = [
        ("今天天气很好", "今天天气很好", 0.0),
        ("今天天汽很好", "今天天气很好", 0.167),  # 1/6
        ("今天天气", "今天天气很好", 0.333),      # 2/6
        ("今天天气很好啊", "今天天气很好", 0.167),  # 1/6
        ("", "今天天气很好", 1.0),
        ("今天天气很好", "", 0.0),
    ]

    for hyp, ref, expected in test_cases:
        cer = calculate_cer(hyp, ref)
        status = "✓" if abs(cer - expected) < 0.01 else "✗"
        print(f"{status} CER('{hyp}', '{ref}') = {cer:.3f} (expected ~{expected:.3f})")

    print("\n=== CER 详细信息 ===")
    details = calculate_cer_details("今天天汽很好", "今天天气很好")
    print(f"CER: {details['cer']:.3f}")
    print(f"Distance: {details['distance']}")
    print(f"Substitutions: {details['substitutions']}")
    print(f"Insertions: {details['insertions']}")
    print(f"Deletions: {details['deletions']}")

    print("\n=== WER 测试 ===")
    wer = calculate_wer("今天天气很好", "今天天气很好")
    print(f"WER (相同): {wer:.3f}")

    wer = calculate_wer("今天天汽很好", "今天天气很好")
    print(f"WER (1字错): {wer:.3f}")

    wer = calculate_wer("hello world", "hello world")
    print(f"WER (英文相同): {wer:.3f}")

    wer = calculate_wer("hello word", "hello world")
    print(f"WER (英文1词错): {wer:.3f}")
