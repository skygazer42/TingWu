"""RAG 核心算法模块 - 基于 CapsWriter-Offline

提供基于音素的模糊编辑距离计算功能：
- get_phoneme_cost: 音素匹配代价 (相似音0.5, 不同音1.0)
- lcs_length: 最长公共子序列 (用于英文匹配)
- find_best_match: 带词边界约束的模糊匹配
- fuzzy_substring_search_constrained: 边界约束搜索
"""
from typing import List, Tuple

from src.core.hotword.phoneme import Phoneme, SIMILAR_PHONEMES


def lcs_length(s1: str, s2: str) -> int:
    """
    计算两个字符串的最长公共子序列 (LCS) 长度

    时间复杂度: O(m*n)
    空间复杂度: O(min(m,n)) - 使用滚动数组优化
    """
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    m, n = len(s1), len(s2)
    if n == 0:
        return 0

    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev, curr = curr, prev

    return prev[n]


def get_phoneme_cost(p1: Phoneme, p2: Phoneme) -> float:
    """
    计算音素匹配代价（基于 Phoneme 对象的语言属性）

    返回值范围: 0.0 (完全匹配) ~ 1.0 (完全不匹配)

    规则:
    1. 完全相同（value + lang）-> 0.0
    2. 相似音素（前后鼻音、平翘舌等）-> 0.5（仅限中文音素）
    3. 英文单词之间 -> 使用 LCS 计算字符级相似度
    4. 不同语言音素混合 -> 1.0（完全不匹配）
    5. 其他 -> 1.0
    """
    # 不同语言，直接返回不匹配
    if p1.lang != p2.lang:
        return 1.0

    # 相同语言，比较 value
    if p1.value == p2.value:
        return 0.0

    # 中文音素：检查相似音素
    if p1.lang == 'zh' and p2.lang == 'zh':
        pair = {p1.value, p2.value}
        for s in SIMILAR_PHONEMES:
            if pair.issubset(s):
                return 0.5

    # 英文单词：使用 LCS 计算相似度
    if p1.lang == 'en' and p2.lang == 'en':
        lcs_len = lcs_length(p1.value, p2.value)
        max_len = max(len(p1.value), len(p2.value))
        if max_len > 0:
            return 1.0 - (lcs_len / max_len)

    return 1.0


def get_tuple_cost(t1: Tuple, t2: Tuple) -> float:
    """
    计算两个音素元组的匹配代价 (用于性能优化，避免创建 Phoneme 对象)

    参数:
        t1, t2: 音素 info 元组 (value, lang, is_word_start, is_word_end, is_tone, char_start, char_end)
    """
    # 不同语言，直接返回不匹配
    if t1[1] != t2[1]:
        return 1.0

    # 相同语言，比较 value
    if t1[0] == t2[0]:
        return 0.0

    # 中文相似音素判断
    if t1[1] == 'zh':
        # 声调差异给予较低代价
        if t1[4] and t2[4]:  # both are tones
            return 0.5
        pair = {t1[0], t2[0]}
        for s in SIMILAR_PHONEMES:
            if pair.issubset(s):
                return 0.5

    # 英文单词字符级相似度
    if t1[1] == 'en':
        lcs_len = lcs_length(t1[0], t2[0])
        max_len = max(len(t1[0]), len(t2[0]))
        if max_len > 0:
            return 1.0 - (lcs_len / max_len)

    return 1.0


def find_best_match(main_seq: List[Phoneme], sub_seq: List[Phoneme]) -> Tuple[float, int, int]:
    """
    寻找最佳模糊匹配位置（基于 Phoneme 对象，限制只能从字边界开始）

    Args:
        main_seq: 主音素序列（长）
        sub_seq: 子音素序列（短，热词）

    Returns:
        (score, start_index, end_index)
        score: 相似度 0-1
        start_index: 匹配在 main_seq 中的起始索引 (inclusive)
        end_index: 匹配在 main_seq 中的结束索引 (exclusive)
    """
    n = len(sub_seq)
    m = len(main_seq)
    if n == 0 or m == 0:
        return 0.0, 0, 0

    # 预计算字边界：只允许从这些位置开始匹配
    valid_starts = [j for j in range(m) if main_seq[j].is_word_start]

    # DP 矩阵: rows=n+1, cols=m+1
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]

    # 初始化第一行 (Start Constraints)
    for j in range(m + 1):
        dp[0][j] = 0.0 if j in valid_starts else float('inf')

    # 初始化第一列
    for i in range(1, n + 1):
        dp[i][0] = dp[i-1][0] + 1.0

    # 填充 DP 矩阵
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = get_phoneme_cost(sub_seq[i-1], main_seq[j-1])
            dp[i][j] = min(
                dp[i-1][j] + 1.0,     # Deletion
                dp[i][j-1] + 1.0,     # Insertion
                dp[i-1][j-1] + cost   # Match/Mismatch/Similar
            )

    # 找到最后一行最小值的位置，且起始位置必须在字边界
    min_dist = float('inf')
    end_pos = 0
    best_start = 0

    for j in range(1, m + 1):
        if dp[n][j] < min_dist:
            # 回溯找到起始位置
            curr_i, curr_j = n, j
            while curr_i > 0:
                cost = get_phoneme_cost(sub_seq[curr_i-1], main_seq[curr_j-1])

                if curr_j > 0 and abs(dp[curr_i][curr_j] - (dp[curr_i-1][curr_j-1] + cost)) < 1e-9:
                    curr_i -= 1
                    curr_j -= 1
                elif abs(dp[curr_i][curr_j] - (dp[curr_i-1][curr_j] + 1.0)) < 1e-9:
                    curr_i -= 1
                elif curr_j > 0 and abs(dp[curr_i][curr_j] - (dp[curr_i][curr_j-1] + 1.0)) < 1e-9:
                    curr_j -= 1
                else:
                    curr_i -= 1

            # 检查起始位置是否在字边界
            if curr_j in valid_starts:
                min_dist = dp[n][j]
                end_pos = j
                best_start = curr_j

    # 计算分数
    score = 1.0 - (min_dist / n) if n > 0 else 0.0
    return max(0.0, score), best_start, end_pos


def fuzzy_substring_distance(hw_info: List[Tuple], input_info: List[Tuple]) -> float:
    """
    计算子序列在主序列中的最小编辑距离（允许子序列匹配主序列的任意部分）
    使用滚动数组优化的动态规划实现

    参数:
        hw_info: 热词音素序列（info 元组列表）
        input_info: 输入音素序列（info 元组列表）
    """
    n = len(hw_info)
    m = len(input_info)
    if n == 0:
        return 0.0
    if m == 0:
        return float(n)

    prev = [0.0] * (m + 1)
    curr = [0.0] * (m + 1)

    for i in range(1, n + 1):
        curr[0] = float(i)
        for j in range(1, m + 1):
            cost = get_tuple_cost(hw_info[i-1], input_info[j-1])
            curr[j] = min(
                prev[j] + 1.0,
                curr[j-1] + 1.0,
                prev[j-1] + cost
            )
        prev, curr = curr, prev

    return min(prev)


def fuzzy_substring_score(hw_info: List[Tuple], input_info: List[Tuple]) -> float:
    """
    计算子序列在主序列中的相似度分数（0-1之间）
    """
    n = len(hw_info)
    if n == 0:
        return 0.0
    distance = fuzzy_substring_distance(hw_info, input_info)
    score = 1.0 - (distance / n)
    return max(0.0, min(1.0, score))


def fuzzy_substring_search_constrained(
    hw_info: List[Tuple],
    input_info: List[Tuple],
    threshold: float = 0.6
) -> List[Tuple[float, int, int]]:
    """
    在输入序列中搜索热词的最佳匹配片段（边界约束版）

    使用 DP 计算局部相似度，要求：
    1. 起始位置必须是原句的词起始 (is_word_start)
    2. 结束位置必须是原句的词结束 (is_word_end)

    参数:
        hw_info: 热词音素 info 元组列表 (value, lang, is_word_start, is_word_end, is_tone, char_start, char_end)
        input_info: 输入文本音素 info 元组列表
        threshold: 相似度阈值

    返回:
        List[(score, start_idx, end_idx)] - 匹配结果列表（按分数降序）
    """
    n = len(hw_info)
    m = len(input_info)
    if n == 0 or m == 0:
        return []

    # DP 矩阵: [n+1][m+1]
    dp = [[float('inf')] * (m + 1) for _ in range(n + 1)]
    path = [[(0, 0)] * (m + 1) for _ in range(n + 1)]

    # 初始化第一行：允许从任何词起始边界开始匹配
    for j in range(m + 1):
        if j == 0:
            dp[0][j] = 0.0
            path[0][j] = (0, j)
        elif j < m and input_info[j][2]:  # is_word_start
            dp[0][j] = 0.0
            path[0][j] = (0, j)

    # 填充 DP
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = get_tuple_cost(hw_info[i-1], input_info[j-1])

            dist_match = dp[i-1][j-1] + cost
            dist_del = dp[i-1][j] + 1.0
            dist_ins = dp[i][j-1] + 1.0

            min_dist = min(dist_match, dist_del, dist_ins)
            dp[i][j] = min_dist

            if min_dist == dist_match:
                path[i][j] = path[i-1][j-1]
            elif min_dist == dist_del:
                path[i][j] = path[i-1][j]
            else:
                path[i][j] = path[i][j-1]

    # 收集结果
    results = []

    for j in range(1, m + 1):
        # 约束：终点必须是词边界
        if not input_info[j-1][3]:  # is_word_end
            continue

        dist = dp[n][j]
        if dist >= n * 0.8:  # 距离太大，强制过滤
            continue

        score = 1.0 - (dist / n)
        if score >= threshold:
            start_idx = path[n][j][1]
            end_idx = j
            results.append((score, start_idx, end_idx))

    # 按得分降序
    results.sort(key=lambda x: x[0], reverse=True)

    # 区间排重
    used_ends = {}
    for score, s, e in results:
        if e not in used_ends or score > used_ends[e][0]:
            used_ends[e] = (score, s, e)

    return sorted(used_ends.values(), key=lambda x: x[0], reverse=True)
