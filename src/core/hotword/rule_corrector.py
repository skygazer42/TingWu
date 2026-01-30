"""规则纠错器 - 基于 CapsWriter-Offline

基于正则表达式的精确规则替换。
适用于固定格式的替换（单位、符号、格式等）。

使用方法示例：
```python
corrector = RuleCorrector()
corrector.update_rules('''
    毫安时  =  mAh
    伏特   =   V
    赫兹   =   Hz
    (艾特)\\s*(\\w+)\\s*(点)\\s*(\\w+)    =    @$2.$4
''')

corrector.substitute('这款手机有5000毫安时')  # 输出：这款手机有5000mAh
corrector.substitute('国内交流电50赫兹')      # 输出：国内交流电50Hz
```
"""
import re
import os
from pathlib import Path
from threading import Lock
from typing import Dict, List, Tuple


class RuleCorrector:
    """规则纠错器 - 基于正则表达式的精确替换"""

    def __init__(self):
        self.patterns: Dict[str, str] = {}
        self._lock = Lock()

    def update_rules(self, rule_text: str) -> int:
        """
        更新规则词典（线程安全）

        Args:
            rule_text: 规则文本，每行一条，格式为 "正则模式 = 替换文本"
                       支持 # 开头的注释行

        Returns:
            加载的规则数量
        """
        new_patterns = {}

        for line in rule_text.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split(' = ', 1)
            if len(parts) == 2:
                pattern = parts[0].strip()
                replacement = parts[1].strip()
                if pattern:
                    new_patterns[pattern] = replacement

        with self._lock:
            self.patterns = new_patterns

        return len(new_patterns)

    def load_rules_file(self, path: str) -> int:
        """
        从文件加载规则

        Args:
            path: 规则文件路径

        Returns:
            加载的规则数量
        """
        path = Path(path)
        if path.exists():
            content = path.read_text(encoding='utf-8')
            return self.update_rules(content)
        return 0

    def substitute(self, text: str) -> str:
        """
        执行规则替换

        Args:
            text: 原始文本

        Returns:
            替换后的文本
        """
        if not text or not self.patterns:
            return text or ""

        result = text

        with self._lock:
            patterns = self.patterns.copy()

        for pattern, replacement in patterns.items():
            try:
                result = re.sub(pattern, replacement, result)
            except re.error:
                # 忽略无效的正则表达式
                pass

        return result

    def substitute_with_info(self, text: str) -> Tuple[str, List[Tuple[str, str, str]]]:
        """
        执行规则替换并返回替换信息

        Args:
            text: 原始文本

        Returns:
            (替换后文本, [(原文, 替换后, 规则模式), ...])
        """
        if not text or not self.patterns:
            return text or "", []

        result = text
        replacements = []

        with self._lock:
            patterns = self.patterns.copy()

        for pattern, replacement in patterns.items():
            try:
                matches = list(re.finditer(pattern, result))
                if matches:
                    for match in matches:
                        original = match.group(0)
                        replaced = re.sub(pattern, replacement, original)
                        if original != replaced:
                            replacements.append((original, replaced, pattern))
                    result = re.sub(pattern, replacement, result)
            except re.error:
                pass

        return result, replacements
