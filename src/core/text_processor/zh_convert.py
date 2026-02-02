#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
中文简繁转换模块

移植自 CapsWriter-Offline (基于 zhconv 库)。
支持简体↔繁体转换，包括多种区域变体。

用法示例：
    from src.core.text_processor import ZhConverter

    converter = ZhConverter()
    print(converter.convert('人体内存在很多微生物', 'zh-tw'))
    # 人體內存在很多微生物

    print(converter.convert('我幹什麼不干你事。', 'zh-cn'))
    # 我干什么不干你事。

支持的区域:
    - zh-cn / zh-hans: 简体中文（大陆）
    - zh-tw: 繁体中文（台湾）
    - zh-hk: 繁体中文（香港）
    - zh-hant: 繁体中文（通用）
"""
__all__ = ['ZhConverter', 'convert', 'issimp']

import os
import re
import json
from pathlib import Path

# 区域回退顺序
LOCALES = {
    'zh-cn': ('zh-cn', 'zh-hans', 'zh-sg', 'zh'),
    'zh-hk': ('zh-hk', 'zh-hant', 'zh-tw', 'zh'),
    'zh-tw': ('zh-tw', 'zh-hant', 'zh-hk', 'zh'),
    'zh-sg': ('zh-sg', 'zh-hans', 'zh-cn', 'zh'),
    'zh-my': ('zh-my', 'zh-sg', 'zh-hans', 'zh-cn', 'zh'),
    'zh-mo': ('zh-mo', 'zh-hk', 'zh-hant', 'zh-tw', 'zh'),
    'zh-hant': ('zh-hant', 'zh-tw', 'zh-hk', 'zh'),
    'zh-hans': ('zh-hans', 'zh-cn', 'zh-sg', 'zh'),
    'zh': ('zh',)  # special value for no conversion
}

# 词典文件路径
_DEFAULT_DICT = "zhcdict.json"

# 全局缓存
_zhcdicts = None
_dict_zhcn = None
_dict_zhsg = None
_dict_zhtw = None
_dict_zhhk = None
_pfsdict = {}


def _get_dict_path():
    """获取词典文件路径"""
    # 先尝试 data/text_processor 目录
    module_dir = Path(__file__).parent.parent.parent.parent
    data_path = module_dir / "data" / "text_processor" / _DEFAULT_DICT
    if data_path.exists():
        return data_path
    # 再尝试模块目录
    module_path = Path(__file__).parent / _DEFAULT_DICT
    if module_path.exists():
        return module_path
    raise FileNotFoundError(f"Cannot find dictionary file: {_DEFAULT_DICT}")


def _loaddict(filename=None):
    """加载词典"""
    global _zhcdicts
    if _zhcdicts:
        return

    if filename is None:
        filepath = _get_dict_path()
    else:
        filepath = Path(filename)

    with open(filepath, 'rb') as f:
        _zhcdicts = json.loads(f.read().decode('utf-8'))

    _zhcdicts['SIMPONLY'] = frozenset(_zhcdicts['SIMPONLY'])
    _zhcdicts['TRADONLY'] = frozenset(_zhcdicts['TRADONLY'])


def _getdict(locale):
    """获取或生成特定区域的转换词典缓存"""
    global _zhcdicts, _dict_zhcn, _dict_zhsg, _dict_zhtw, _dict_zhhk, _pfsdict

    if _zhcdicts is None:
        _loaddict()

    if locale == 'zh-cn':
        if _dict_zhcn:
            got = _dict_zhcn
        else:
            _dict_zhcn = _zhcdicts['zh2Hans'].copy()
            _dict_zhcn.update(_zhcdicts['zh2CN'])
            got = _dict_zhcn
    elif locale == 'zh-tw':
        if _dict_zhtw:
            got = _dict_zhtw
        else:
            _dict_zhtw = _zhcdicts['zh2Hant'].copy()
            _dict_zhtw.update(_zhcdicts['zh2TW'])
            got = _dict_zhtw
    elif locale == 'zh-hk' or locale == 'zh-mo':
        if _dict_zhhk:
            got = _dict_zhhk
        else:
            _dict_zhhk = _zhcdicts['zh2Hant'].copy()
            _dict_zhhk.update(_zhcdicts['zh2HK'])
            got = _dict_zhhk
    elif locale == 'zh-sg' or locale == 'zh-my':
        if _dict_zhsg:
            got = _dict_zhsg
        else:
            _dict_zhsg = _zhcdicts['zh2Hans'].copy()
            _dict_zhsg.update(_zhcdicts['zh2SG'])
            got = _dict_zhsg
    elif locale == 'zh-hans':
        got = _zhcdicts['zh2Hans']
    elif locale == 'zh-hant':
        got = _zhcdicts['zh2Hant']
    else:
        got = {}

    if locale not in _pfsdict:
        _pfsdict[locale] = _getpfset(got)

    return got


def _getpfset(convdict):
    """生成前缀集合用于最长匹配"""
    pfset = []
    for word in convdict:
        for ch in range(len(word)):
            pfset.append(word[:ch+1])
    return frozenset(pfset)


def issimp(s, full=False):
    """
    检测文本是简体还是繁体中文

    Args:
        s: 输入文本
        full: 是否完整扫描（False=遇到即返回，True=统计所有字符）

    Returns:
        True: 简体中文
        False: 繁体中文
        None: 无法判断
    """
    if _zhcdicts is None:
        _loaddict()

    simp, trad = 0, 0
    if full:
        for ch in s:
            if ch in _zhcdicts['SIMPONLY']:
                simp += 1
            elif ch in _zhcdicts['TRADONLY']:
                trad += 1
        if simp > trad:
            return True
        elif simp < trad:
            return False
        else:
            return None
    else:
        for ch in s:
            if ch in _zhcdicts['SIMPONLY']:
                return True
            elif ch in _zhcdicts['TRADONLY']:
                return False
        return None


def convert(s, locale, update=None):
    """
    主转换函数

    Args:
        s: 输入文本
        locale: 目标区域 ('zh-hans', 'zh-hant', 'zh-cn', 'zh-tw', 'zh-hk' 等)
        update: 额外的转换规则 dict，如 {'from1': 'to1'}

    Returns:
        转换后的文本
    """
    if locale == 'zh' or locale not in LOCALES:
        return s

    zhdict = _getdict(locale)
    pfset = _pfsdict[locale]
    newset = set()

    if update:
        newset = set()
        for word in update:
            for ch in range(len(word)):
                newset.add(word[:ch+1])

    ch = []
    N = len(s)
    pos = 0

    while pos < N:
        i = pos
        frag = s[pos]
        maxword = None
        maxpos = 0

        while i < N and (frag in pfset or frag in newset):
            if update and frag in update:
                maxword = update[frag]
                maxpos = i
            elif frag in zhdict:
                maxword = zhdict[frag]
                maxpos = i
            i += 1
            frag = s[pos:i+1]

        if maxword is None:
            maxword = s[pos]
            pos += 1
        else:
            pos = maxpos + 1

        ch.append(maxword)

    return ''.join(ch)


class ZhConverter:
    """
    中文简繁转换器

    支持区域:
        - zh-cn / zh-hans: 简体中文（大陆）
        - zh-tw: 繁体中文（台湾）
        - zh-hk: 繁体中文（香港）
        - zh-hant: 繁体中文（通用）

    用法:
        converter = ZhConverter()
        result = converter.convert("人体内存在很多微生物", "zh-tw")
        # result: "人體內存在很多微生物"
    """

    def __init__(self, dict_path=None):
        """
        初始化转换器

        Args:
            dict_path: 自定义词典路径（可选）
        """
        if dict_path:
            _loaddict(dict_path)
        else:
            _loaddict()

    def convert(self, text: str, locale: str = 'zh-hans', update=None) -> str:
        """
        转换文本

        Args:
            text: 输入文本
            locale: 目标区域
            update: 额外的转换规则

        Returns:
            转换后的文本
        """
        if not text:
            return text
        return convert(text, locale, update)

    def to_simplified(self, text: str) -> str:
        """转换为简体中文"""
        return self.convert(text, 'zh-cn')

    def to_traditional(self, text: str, locale: str = 'zh-hant') -> str:
        """转换为繁体中文"""
        return self.convert(text, locale)

    def is_simplified(self, text: str) -> bool:
        """检测是否为简体中文"""
        result = issimp(text)
        return result is True

    def is_traditional(self, text: str) -> bool:
        """检测是否为繁体中文"""
        result = issimp(text)
        return result is False


if __name__ == '__main__':
    converter = ZhConverter()
    print("简体 -> 繁体（标准）:")
    print(converter.convert('人体内存在很多微生物', 'zh-hant'))
    print("\n简体 -> 繁体（台湾）:")
    print(converter.convert('人体内存在很多微生物', 'zh-tw'))
    print("\n繁体 -> 简体:")
    print(converter.convert('我幹什麼不干你事。', 'zh-cn'))
