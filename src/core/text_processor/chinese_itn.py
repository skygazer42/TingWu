# coding: utf-8
"""
中文数字转阿拉伯数字 (Chinese ITN - Inverse Text Normalization)

移植自 CapsWriter-Offline，用于把语音识别出的中文数字转为阿拉伯数字形式。

用法示例：
    from src.core.text_processor import ChineseITN
    itn = ChineseITN()

    res = itn.convert('幺九二点幺六八点幺点幺')
    print(res)  # 192.168.1.1

    res = itn.convert('三五百人')
    print(res)  # 300~500人
"""

__all__ = ['ChineseITN']

import re


# ============================================================
# 儿化移除
# ============================================================

# 儿化白名单 - 这些词本身含"儿"，不应移除
ERHUA_WHITELIST = {
    '儿女', '儿子', '儿孙', '儿童', '儿歌', '女儿', '婴儿', '幼儿',
    '胎儿', '孤儿', '孩儿', '健儿', '宠儿', '娇儿', '弃儿',
    '儿时', '儿媳', '少儿', '育儿', '产儿', '儿科', '儿戏',
    '小儿', '男儿', '儿郎', '生儿', '养儿', '侄儿',
}

# 儿化正则：匹配"X儿"模式（X为中文字符）
_ERHUA_PATTERN = re.compile(r'([\u4e00-\u9fff])儿')


def _erhua_replace(match):
    """儿化替换回调"""
    char_before = match.group(1)
    word = char_before + '儿'
    if word in ERHUA_WHITELIST:
        return word
    return char_before


def remove_erhua(text: str) -> str:
    """移除非词汇儿化

    例如：
    - 那边儿 → 那边
    - 一点儿 → 一点
    - 这儿那儿 → 这那
    - 女儿 → 女儿 (白名单保留)
    - 儿童 → 儿童 (不受影响，儿在词头)
    """
    if not text or '儿' not in text:
        return text

    # 先保护白名单中以"儿"开头的词
    protected = {}
    for word in sorted(ERHUA_WHITELIST, key=len, reverse=True):
        if word in text and word.startswith('儿'):
            placeholder = f'\x00{len(protected)}\x00'
            text = text.replace(word, placeholder)
            protected[placeholder] = word

    # 替换儿化
    text = _ERHUA_PATTERN.sub(_erhua_replace, text)

    # 还原被保护的词
    for placeholder, word in protected.items():
        text = text.replace(placeholder, word)

    return text


# ============================================================
# 第一部分：配置和映射表
# ============================================================

# 单位映射：中文单位 -> 映射后的单位（None表示保留原样）
UNIT_MAPPING = {
    '个': None, '只': None, '分': None, '万': None, '亿': None, '秒': None, '年': None,
    '月': None, '日': None, '天': None, '时': None, '钟': None, '人': None, '层': None,
    '楼': None, '倍': None, '块': None, '次': None,
    '克': 'g', '千克': 'kg',
    '米': '米', '千米': '千米', '千米每小时': 'km/h',
}

# 生成单位正则（按长度从长到短排序，确保先匹配长的）
_sorted_units = sorted(UNIT_MAPPING.keys(), key=len, reverse=True)
COMMON_UNITS = '|'.join(f'{u}' for u in _sorted_units)

# 中文数字映射表
NUM_MAPPER = {
    '零': '0',  '一': '1',  '幺': '1',  '二': '2',
    '两': '2',  '三': '3',  '四': '4',  '五': '5',
    '六': '6',  '七': '7',  '八': '8',  '九': '9',
    '点': '.',
}

# 中文数字对数值的映射
VALUE_MAPPER = {
    '零': 0,  '一': 1,  '二': 2,  '两': 2,  '三': 3,  '四': 4,  '五': 5,
    '六': 6,  '七': 7,  '八': 8,  '九': 9,  "十": 10,  "百": 100,  "千": 1000,  "万": 10000,  "亿": 100000000,
}

# 成语和习语黑名单
IDIOMS = '''
正经八百  五零二落 五零四散
五十步笑百步 乌七八糟 污七八糟 四百四病 思绪万千
十有八九 十之八九 三十而立 三十六策 三十六计 三十六行
三五成群 三百六十行 三六九等
七老八十 七零八落 七零八碎 七七八八 乱七八遭 乱七八糟 略知一二 零零星星 零七八碎
九九归一 二三其德 二三其意 无银三百两 八九不离十
百分之百 年三十 烂七八糟 一点一滴 路易十六 九三学社 五四运动 入木三分 三十六计
九九八十一 三七二十一
十二五 十三五 十四五 十五五 十六五 十七五 十八五
'''.split()

# 模糊表达黑名单（包含"几"的表达不转换）
FUZZY_REGEX = re.compile(r'几')


# ============================================================
# 第二部分：范围表达式处理
# ============================================================

def _chinese_digit_to_num(char):
    """将单个中文数字转为阿拉伯数字"""
    return VALUE_MAPPER.get(char, 0)

def _parse_tens(tens):
    """解析"十"或"X十"格式的数值"""
    return 10 if tens == '十' else _chinese_digit_to_num(tens[0]) * 10

# 范围表达式模式
_range_pattern_1 = re.compile(r'([二三四五六七八九])([二三四五六七八九])([十百千万亿])([万千百亿])?')
_range_pattern_2 = re.compile(r'(十|[一二三四五六七八九十]+[十百千万])([一二三四五六七八九])([一二三四五六七八九])([万千亿])?')
_range_pattern_3 = re.compile(r'^([一二三四五六七八九])([一二三四五六七八九])$')

def _convert_range_pattern_1(match):
    """转换模式1: 三五百 → 300~500, 五六十 → 50~60, 三四十万 → 30~40万"""
    groups = match.groups()
    d1, d2, unit = groups[0], groups[1], groups[2]
    suffix_unit = groups[3] if len(groups) > 3 and groups[3] else ''

    v1 = _chinese_digit_to_num(d1)
    v2 = _chinese_digit_to_num(d2)

    if unit == '十':
        v1, v2 = v1 * 10, v2 * 10
        return f"{v1}~{v2}{suffix_unit}"
    elif unit in ['万', '亿']:
        return f"{v1}~{v2}{unit}{suffix_unit}"
    elif unit == '千' and suffix_unit:
        return f"{v1}~{v2}{unit}{suffix_unit}"
    else:
        v1 = v1 * VALUE_MAPPER[unit]
        v2 = v2 * VALUE_MAPPER[unit]
        return f"{v1}~{v2}{suffix_unit}"

def _convert_range_pattern_2(match):
    """转换模式2: 十五六 → 15~16, 四十五六万 → 45~46万, 一百六七 → 160~170"""
    groups = match.groups()
    base_part, d1, d2 = groups[0], groups[1], groups[2]
    unit = groups[3] if len(groups) > 3 and groups[3] else ''

    last_char = base_part[-1]

    # 计算基数值
    if last_char == '十':
        base_value = 10 if len(base_part) == 1 else _chinese_digit_to_num(base_part[0]) * 10
    elif last_char in VALUE_MAPPER:
        num_part = base_part[:-1]
        base_value = _chinese_digit_to_num(num_part[0]) * VALUE_MAPPER[last_char] if num_part else VALUE_MAPPER[last_char]
    else:
        base_value = _parse_tens(base_part)

    num1 = _chinese_digit_to_num(d1)
    num2 = _chinese_digit_to_num(d2)
    multiplier = VALUE_MAPPER.get(last_char, 10) // 10

    return f"{base_value + num1 * multiplier}~{base_value + num2 * multiplier}{unit}"

def _convert_range_pattern_3(match):
    """转换模式3: 三四 → 3~4, 五六 → 5~6"""
    d1, d2 = match.groups()
    v1 = _chinese_digit_to_num(d1)
    v2 = _chinese_digit_to_num(d2)
    return f"{v1}~{v2}"

def _is_range_expression(text):
    """判断是否为范围表达式"""
    sorted_units = sorted(UNIT_MAPPING.keys(), key=len, reverse=True)
    unit_pattern = '|'.join(re.escape(u) for u in sorted_units)
    optional_unit = rf'(?:{unit_pattern})?'

    range_pattern = re.compile(rf'''(?x)
        (?<!点)
        (?:
            [二三四五六七八九]{{2}}(?:十|[百千万亿]){optional_unit}
            |
            [一二三四五六七八九]?十[一二三四五六七八九]{{2}}(?:[万千亿]|{optional_unit})
            |
            [一二三四五六七八九][百千][二三四五六七八九]{{2}}十
            |
            [一二三四五六七八九十]+[万千百][一二三四五六七八九]{{2}}{optional_unit}
        )
    ''')

    return range_pattern.search(text) is not None

def _convert_range_expression(text):
    """转换范围表达式"""
    stripped_text = text
    mapped_unit = ''

    numeric_units = {'万', '亿', '千', '百', '十'}
    sorted_units = sorted(UNIT_MAPPING.keys(), key=len, reverse=True)

    for unit_cn in sorted_units:
        if unit_cn in numeric_units:
            continue
        if text.endswith(unit_cn):
            stripped_text = text[:-len(unit_cn)]
            mapped_unit = UNIT_MAPPING[unit_cn]
            if mapped_unit is None:
                mapped_unit = unit_cn
            break

    match = _range_pattern_2.search(stripped_text)
    if match:
        return _convert_range_pattern_2(match) + mapped_unit

    match = _range_pattern_1.search(stripped_text)
    if match:
        return _convert_range_pattern_1(match) + mapped_unit

    match = _range_pattern_3.search(stripped_text)
    if match:
        return _convert_range_pattern_3(match) + mapped_unit

    return text


# ============================================================
# 第三部分：正则表达式模式定义
# ============================================================

# 用于去除末尾单位的正则
_unit_suffix_pattern = re.compile(rf'({COMMON_UNITS}|[a-zA-Z]+)$')

# 总模式，筛选出可能需要替换的内容
_pattern = re.compile(f"""(?ix)
([a-z]\s*)?
(
  (
    [几零幺一二两三四五六七八九十百千万点比]
    |[零一二三四五六七八九十][ ]
    |(?<=[一二两三四五六七八九十])[年月日号分]
    |(分之)
  )+
  (
    (?<=[一二两三四五六七八九十])([a-zA-Z年月日号]|{COMMON_UNITS})
    |(?<=[一二两三四五六七八九十]\s)[a-zA-Z]
  )?
  (?(1)
  |(?(5)
    |(
      [零幺一二两三四五六七八九十百千万亿点比]
      |(分之)
    )
  )+
  )
)
""")

# 纯数字序号
_pure_num = re.compile(f'[零幺一二三四五六七八九]+(点[零幺一二三四五六七八九]+)* *([a-zA-Z]|{COMMON_UNITS})?')

# 数值
_value_num = re.compile(f"十?(零?[一二两三四五六七八九十][十百千万]{{1,2}})*零?十?[一二三四五六七八九]?(点[零一二三四五六七八九]+)? *([a-zA-Z]|{COMMON_UNITS})?")

# 连续数值检测
_consecutive_tens = re.compile(rf'^((?:十[一二三四五六七八九])+)({COMMON_UNITS})?$')
_consecutive_hundreds = re.compile(rf'^((?:[一二三四五六七八九]百零?[一二三四五六七八九])+)({COMMON_UNITS})?$')

# 百分值
_percent_value = re.compile('(?<![一二三四五六七八九])(百分之)[零一二三四五六七八九十百千万]+(点)?(?(2)[零一二三四五六七八九]+)')

# 分数
_fraction_value = re.compile('([零一二三四五六七八九十百千万]+(点)?(?(2)[零一二三四五六七八九]+))分之([零一二三四五六七八九十百千万]+(点)?(?(4)[零一二三四五六七八九]+))')

# 比值
_ratio_value = re.compile('([零一二三四五六七八九十百千万]+(点)?(?(2)[零一二三四五六七八九]+))比([零一二三四五六七八九十百千万]+(点)?(?(4)[零一二三四五六七八九]+))')

# 时间
_time_value = re.compile("[零一二两三四五六七八九十]+点([零一二三四五六七八九十]+分)([零一二三四五六七八九十]+秒)?")

# 日期
_date_value = re.compile("([零一二三四五六七八九十]+年)?([一二三四五六七八九十]+月)?([一二三四五六七八九十]+[日号])?")


# ============================================================
# 第四部分：辅助函数
# ============================================================

def _strip_trailing_unit(text):
    """用正则去除末尾的单位"""
    match = _unit_suffix_pattern.search(text)
    if match:
        return text[:match.start()]
    return text

def _is_consecutive_value(text):
    """检测是否是连续数值结构"""
    return _consecutive_tens.match(text) or _consecutive_hundreds.match(text)

def _strip_unit(original):
    """把数字后面跟着的单位剥离开，并应用单位映射"""
    unit_pattern = re.compile(rf'({COMMON_UNITS})$')
    match = unit_pattern.search(original)

    if match:
        unit_cn = match.group(1)
        stripped = original[:match.start()]
        mapped_unit = UNIT_MAPPING.get(unit_cn)
        unit = mapped_unit if mapped_unit is not None else unit_cn
    else:
        stripped = original
        unit = ''

    if not unit and stripped:
        letter_match = re.search(r'[a-zA-Z]+$', stripped)
        if letter_match:
            unit = letter_match.group()
            stripped = stripped[:letter_match.start()]

    return stripped.strip(), unit


# ============================================================
# 第五部分：转换函数
# ============================================================

def _convert_pure_num(original, strict=False):
    """把中文数字转为对应的阿拉伯数字"""
    stripped, unit = _strip_unit(original)
    if stripped in ['一'] and not strict:
        return original
    converted = [NUM_MAPPER[c] for c in stripped]
    return ''.join(converted) + unit

def _convert_value_num(original):
    """把中文数值转为阿拉伯数字"""
    stripped, unit = _strip_unit(original)
    if '点' not in stripped:
        stripped += '点'
    int_part, decimal_part = stripped.split("点")
    if not int_part:
        return original

    # 计算整数部分的值
    value, temp, base = 0, 0, 1
    for c in int_part:
        if c == '十' :
            temp = 10 if temp==0 else VALUE_MAPPER[c]*temp
            base = 1
        elif c == '零':
            base = 1
        elif c in '一二两三四五六七八九':
            temp += VALUE_MAPPER[c]
        elif c in '万':
            value += temp
            value *= VALUE_MAPPER[c]
            base = VALUE_MAPPER[c] // 10
            temp = 0
        elif c in '百千':
            value += temp * VALUE_MAPPER[c]
            base = VALUE_MAPPER[c] // 10
            temp = 0
    value += temp * base
    final = str(value)

    # 小数部分
    decimal_str = _convert_pure_num(decimal_part, strict=True)
    if decimal_str:
        final += '.' + decimal_str
    final += unit

    return final

def _convert_fraction_value(original):
    """转换分数"""
    denominator, numerator = original.split('分之')
    return _convert_value_num(numerator) + '/' + _convert_value_num(denominator)

def _convert_percent_value(original):
    """转换百分数"""
    return _convert_value_num(original[3:]) + '%'

def _convert_ratio_value(original):
    """转换比值"""
    num1, num2 = original.split("比")
    return _convert_value_num(num1) + ':' + _convert_value_num(num2)

def _convert_time_value(original):
    """转换时间"""
    res = [x for x in re.split('[点分秒]', original) if x]
    final = ''
    hour = _convert_value_num(res[0])
    final += hour.zfill(2)
    minute = _convert_value_num(res[1])
    final += ':' + minute.zfill(2)
    if len(res) > 2:
        second = _convert_value_num(res[2])
        final += ':' + second.zfill(2)
    if len(res) > 3:
        final += '.' + _convert_pure_num(res[3])
    return final

def _convert_date_value(original):
    """转换日期"""
    final = ''
    if '年' in original:
        year, original = original.split('年')
        final += _convert_pure_num(year) + '年'
    if '月' in original:
        month, original = original.split('月')
        final += _convert_value_num(month) + '月'
    if '日' in original:
        day, original = original.split('日')
        final += _convert_value_num(day) + '日'
    elif '号' in original:
        day, original = original.split('号')
        final += _convert_value_num(day) + '号'
    return final

def _split_consecutive_value(text):
    """分割连续数值为空格分隔的阿拉伯数字"""
    unit = ''
    for c in COMMON_UNITS:
        if text.endswith(c):
            unit = c
            text = text[:-1]
            break

    if _consecutive_tens.match(text + unit):
        parts = re.findall(r'十[一二三四五六七八九]', text)
        nums = [_convert_value_num(p) for p in parts]
        return ' '.join(nums) + unit

    if _consecutive_hundreds.match(text + unit):
        parts = re.findall(r'[一二三四五六七八九]百零?[一二三四五六七八九]', text)
        nums = [_convert_value_num(p) for p in parts]
        return ' '.join(nums) + unit

    return text + unit


# ============================================================
# 第六部分：主替换逻辑
# ============================================================

def _replace(original):
    """主替换函数"""
    string = original.string
    l_pos, r_pos = original.regs[2]
    l_pos = max(l_pos-2, 0)
    head = original.group(1)
    original_text = original.group(2)
    original = original_text

    try:
        # 成语/习语检测
        if IDIOMS and any([string.find(idiom) in range(l_pos, r_pos) for idiom in IDIOMS]):
            final = original

        # 模糊表达检测
        elif FUZZY_REGEX.search(original):
            final = original

        # 范围表达式
        elif _is_range_expression(original):
            final = _convert_range_expression(original)

        # 时间
        elif _time_value.fullmatch(original):
            final = _convert_time_value(original)

        # 纯数字
        elif _pure_num.fullmatch(_strip_trailing_unit(original)):
            final = _convert_pure_num(original)

        # 连续数值
        elif _is_consecutive_value(original):
            final = _split_consecutive_value(original)

        # 数值
        elif _value_num.fullmatch(_strip_trailing_unit(original)):
            final = _convert_value_num(original)

        # 百分数
        elif _percent_value.fullmatch(original):
            final = _convert_percent_value(original)

        # 分数
        elif _fraction_value.fullmatch(original):
            final = _convert_fraction_value(original)

        # 比值
        elif _ratio_value.fullmatch(original):
            final = _convert_ratio_value(original)

        # 日期
        elif _date_value.fullmatch(original):
            final = _convert_date_value(original)

        else:
            final = original

        if head:
            final = head + final

    except Exception:
        final = original

    return final


# ============================================================
# 第七部分：主类
# ============================================================

class ChineseITN:
    """
    中文数字逆文本规范化

    将中文数字转换为阿拉伯数字，支持:
    - 纯数字: 一二三 → 123
    - 数值: 三百五十 → 350
    - 范围: 三五百人 → 300~500人
    - 百分比: 百分之五十 → 50%
    - 分数: 三分之一 → 1/3
    - 比值: 三比一 → 3:1
    - 时间: 十四点三十分 → 14:30
    - 日期: 二零二五年一月 → 2025年1月
    - 儿化移除: 那边儿 → 那边 (可选)

    用法:
        itn = ChineseITN()
        result = itn.convert("今天是二零二五年一月三十日")
        # result: "今天是2025年1月30日"
    """

    def __init__(self, erhua_remove: bool = False):
        self.erhua_remove = erhua_remove

    def convert(self, text: str) -> str:
        """
        将中文数字转换为阿拉伯数字

        Args:
            text: 输入文本

        Returns:
            转换后的文本
        """
        if not text:
            return text

        # 儿化移除 (在 ITN 之前)
        if self.erhua_remove:
            text = remove_erhua(text)

        return _pattern.sub(_replace, text)


# 便捷函数
def chinese_to_num(text: str) -> str:
    """便捷函数：将中文数字转换为阿拉伯数字"""
    return ChineseITN().convert(text)


if __name__ == "__main__":
    itn = ChineseITN()
    print(itn.convert('二零二五年十月'))
    print(itn.convert('乱七八糟'))
    print(itn.convert('三百五十人'))
    print(itn.convert('百分之五十'))
