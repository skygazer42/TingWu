# TingWu 优化计划 - 基于 CapsWriter-Offline 技术

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 CapsWriter-Offline 的核心优化技术移植到 TingWu，大幅提升语音识别准确率

**Architecture:** 采用多阶段纠错管道：FastRAG粗筛 → AccuRAG精筛 → 规则纠错 → 纠错历史RAG → LLM润色

**Tech Stack:** Python, pypinyin, numba, watchdog, httpx (async LLM client)

---

## 优化内容概览

| 功能 | CapsWriter | TingWu现状 | 优先级 |
|------|-----------|-----------|--------|
| AccuRAG精确匹配 | ✅ 带词边界约束 | ❌ 无 | P0 |
| 更多相似音素 | ✅ 15组 | ⚠️ 11组 | P0 |
| 纠错历史RAG | ✅ 智能片段提取 | ❌ 无 | P0 |
| 规则纠错器 | ✅ 正则替换 | ❌ 无 | P1 |
| LLM润色系统 | ✅ 完整 | ⚠️ 仅配置 | P1 |
| 热词文件监视 | ✅ watchdog | ❌ 无 | P2 |

---

## Phase 1: 核心算法升级 (AccuRAG + 相似音素)

### Task 1: 扩展相似音素集合

**Files:**
- Modify: `src/core/hotword/phoneme.py`

**Step 1: 更新 SIMILAR_PHONEMES**

将现有的11组扩展到15组，添加CapsWriter中的额外相似音组：

```python
SIMILAR_PHONEMES = [
    # 前后鼻音
    {'an', 'ang'}, {'en', 'eng'}, {'in', 'ing'},
    {'ian', 'iang'}, {'uan', 'uang'},
    # 平翘舌
    {'z', 'zh'}, {'c', 'ch'}, {'s', 'sh'},
    # 鼻音/边音
    {'l', 'n'},
    # 唇齿音/声门音
    {'f', 'h'},
    # 常见易混韵母
    {'ai', 'ei'}, {'o', 'uo'}, {'e', 'ie'},
    # 清浊音/送气不送气
    {'p', 'b'}, {'t', 'd'}, {'k', 'g'},
]
```

**Step 2: 运行测试验证**

```bash
pytest tests/test_hotword.py -v
```

**Step 3: Commit**

```bash
git add src/core/hotword/phoneme.py
git commit -m "feat(hotword): expand similar phonemes to 16 groups

Add more similar phoneme pairs from CapsWriter:
- o/uo, e/ie vowel pairs
- p/b, t/d, k/g consonant pairs

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 2: 实现精确匹配算法模块 (algo_calc.py)

**Files:**
- Create: `src/core/hotword/algo_calc.py`

**Step 1: 创建算法模块**

实现CapsWriter的核心算法：
- `get_phoneme_cost()`: 音素匹配代价计算 (相似音0.5, 不同音1.0)
- `lcs_length()`: 最长公共子序列 (用于英文匹配)
- `find_best_match()`: 带词边界约束的模糊匹配
- `fuzzy_substring_search_constrained()`: 边界约束搜索

```python
"""RAG 核心算法模块 - 移植自 CapsWriter-Offline"""
from typing import List, Tuple
from src.core.hotword.phoneme import Phoneme, SIMILAR_PHONEMES


def lcs_length(s1: str, s2: str) -> int:
    """计算两个字符串的最长公共子序列长度"""
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
    """计算音素匹配代价"""
    if p1.lang != p2.lang:
        return 1.0
    if p1.value == p2.value:
        return 0.0
    if p1.lang == 'zh':
        pair = {p1.value, p2.value}
        for s in SIMILAR_PHONEMES:
            if pair.issubset(s):
                return 0.5
    if p1.lang == 'en':
        lcs = lcs_length(p1.value, p2.value)
        max_len = max(len(p1.value), len(p2.value))
        return 1.0 - (lcs / max_len) if max_len > 0 else 1.0
    return 1.0


def find_best_match(main_seq: List[Phoneme], sub_seq: List[Phoneme]) -> Tuple[float, int, int]:
    """带词边界约束的模糊匹配"""
    n, m = len(sub_seq), len(main_seq)
    if n == 0 or m == 0:
        return 0.0, 0, 0

    valid_starts = [j for j in range(m) if main_seq[j].is_word_start]
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]

    for j in range(m + 1):
        dp[0][j] = 0.0 if j in valid_starts else float('inf')
    for i in range(1, n + 1):
        dp[i][0] = dp[i-1][0] + 1.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = get_phoneme_cost(sub_seq[i-1], main_seq[j-1])
            dp[i][j] = min(
                dp[i-1][j] + 1.0,
                dp[i][j-1] + 1.0,
                dp[i-1][j-1] + cost
            )

    min_dist = float('inf')
    end_pos = 0
    best_start = 0

    for j in range(1, m + 1):
        if dp[n][j] < min_dist:
            curr_i, curr_j = n, j
            while curr_i > 0:
                cost = get_phoneme_cost(sub_seq[curr_i-1], main_seq[curr_j-1])
                if curr_j > 0 and abs(dp[curr_i][curr_j] - (dp[curr_i-1][curr_j-1] + cost)) < 1e-9:
                    curr_i -= 1
                    curr_j -= 1
                elif abs(dp[curr_i][curr_j] - (dp[curr_i-1][curr_j] + 1.0)) < 1e-9:
                    curr_i -= 1
                elif curr_j > 0:
                    curr_j -= 1
                else:
                    curr_i -= 1
            if curr_j in valid_starts:
                min_dist = dp[n][j]
                end_pos = j
                best_start = curr_j

    score = 1.0 - (min_dist / n) if n > 0 else 0.0
    return max(0.0, score), best_start, end_pos
```

**Step 2: 添加测试**

Create `tests/test_algo_calc.py`

**Step 3: Commit**

---

### Task 3: 实现 AccuRAG 精确检索器

**Files:**
- Create: `src/core/hotword/rag_accu.py`

实现两阶段检索的第二阶段精确计算器。

---

### Task 4: 升级 PhonemeCorrector 使用两阶段检索

**Files:**
- Modify: `src/core/hotword/corrector.py`

将现有的简单匹配升级为：FastRAG粗筛 → AccuRAG精筛

---

## Phase 2: 纠错历史RAG系统

### Task 5: 实现纠错历史RAG

**Files:**
- Create: `src/core/hotword/rectification.py`

核心功能：
- 解析 `hot-rectify.txt` 格式 (错句 => 正句)
- 智能差异片段提取 (使用 SequenceMatcher)
- 词边界扩展策略
- 音素级相似度检索
- 生成LLM提示词上下文

---

### Task 6: 创建纠错历史数据文件

**Files:**
- Create: `data/hotwords/hot-rectify.txt`

---

## Phase 3: 规则纠错器

### Task 7: 实现规则纠错器

**Files:**
- Create: `src/core/hotword/rule_corrector.py`

基于正则表达式的精确替换，用于：
- 单位符号 (毫安时→mAh, 赫兹→Hz)
- 格式转换 (艾特xxx点yyy → @xxx.yyy)
- 固定术语

---

### Task 8: 创建规则文件

**Files:**
- Create: `data/hotwords/hot-rules.txt`

---

## Phase 4: LLM润色系统

### Task 9: 实现LLM客户端

**Files:**
- Create: `src/core/llm/client.py`

支持：
- Ollama (本地)
- OpenAI兼容接口
- 流式输出

---

### Task 10: 实现提示词构建器

**Files:**
- Create: `src/core/llm/prompt_builder.py`

构建包含上下文的提示词：
- 热词列表
- 纠错历史
- 系统提示词

---

### Task 11: 实现角色系统

**Files:**
- Create: `src/core/llm/roles/default.py`
- Create: `src/core/llm/roles/__init__.py`

---

### Task 12: 集成LLM到转写引擎

**Files:**
- Modify: `src/core/engine.py`
- Modify: `src/config.py`

---

## Phase 5: 热词文件监视

### Task 13: 实现文件监视器

**Files:**
- Create: `src/core/hotword/watcher.py`

使用 watchdog 监视热词文件变化，自动重载。

---

### Task 14: 更新配置和依赖

**Files:**
- Modify: `src/config.py`
- Modify: `requirements.txt`
- Modify: `pyproject.toml`

添加新配置项和依赖：
- watchdog
- httpx (async)

---

## Phase 6: 集成测试

### Task 15: 添加集成测试

**Files:**
- Modify: `tests/test_integration.py`

测试完整管道：
1. 热词纠错 (两阶段)
2. 规则纠错
3. 纠错历史
4. LLM润色 (mock)

---

### Task 16: 更新 README

**Files:**
- Modify: `README.md`

文档化新功能和配置项。

---

## 预期效果

1. **准确率提升**: 两阶段检索 + 更多相似音 → 减少误判
2. **性能保持**: FastRAG粗筛保证速度
3. **可扩展性**: 规则纠错处理固定模式
4. **持续学习**: 纠错历史记录用户修正
5. **智能润色**: LLM处理复杂语义错误
