"""音素处理模块 - 基于 CapsWriter-Offline"""
import re
from dataclasses import dataclass
from typing import List, Literal, Tuple

from pypinyin import pinyin, Style


@dataclass(frozen=True, slots=True)
class Phoneme:
    """音素数据类"""
    value: str
    lang: Literal['zh', 'en', 'num', 'other']
    is_word_start: bool = False
    is_word_end: bool = False
    char_start: int = 0
    char_end: int = 0

    @property
    def is_tone(self) -> bool:
        return self.value.isdigit()

    @property
    def info(self) -> Tuple:
        return (self.value, self.lang, self.is_word_start,
                self.is_word_end, self.is_tone, self.char_start, self.char_end)


# 相似音素集合 (用于模糊匹配) - 基于 CapsWriter-Offline
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


def get_phoneme_info(text: str, split_char: bool = True) -> List[Phoneme]:
    """提取文本的音素序列"""
    seq = []
    pos = 0

    while pos < len(text):
        c = text[pos]

        # 处理中文字符
        if '\u4e00' <= c <= '\u9fff':
            start = pos
            pos += 1
            while pos < len(text) and '\u4e00' <= text[pos] <= '\u9fff':
                pos += 1
            frag = text[start:pos]

            try:
                pi = pinyin(frag, style=Style.INITIALS, strict=False)
                pf = pinyin(frag, style=Style.FINALS, strict=False)
                pt = pinyin(frag, style=Style.TONE3, neutral_tone_with_five=True)

                for i in range(min(len(frag), len(pi), len(pf), len(pt))):
                    idx = start + i
                    init, fin, tone = pi[i][0], pf[i][0], pt[i][0]

                    if init:
                        seq.append(Phoneme(init, 'zh', is_word_start=True,
                                          char_start=idx, char_end=idx+1))
                    if fin:
                        seq.append(Phoneme(fin, 'zh', is_word_start=not init,
                                          char_start=idx, char_end=idx+1))
                    if tone and tone[-1].isdigit():
                        seq.append(Phoneme(tone[-1], 'zh', is_word_end=True,
                                          char_start=idx, char_end=idx+1))
            except Exception:
                for i, char in enumerate(frag):
                    seq.append(Phoneme(char, 'zh', is_word_start=True,
                                      is_word_end=True, char_start=start+i,
                                      char_end=start+i+1))

        # 处理英文/数字
        elif 'a' <= c.lower() <= 'z' or '0' <= c <= '9':
            start = pos
            pos += 1
            while pos < len(text):
                cur = text[pos]
                if not ('a' <= cur.lower() <= 'z' or '0' <= cur <= '9'):
                    break
                if (text[pos-1].islower() and cur.isupper()) or \
                   (text[pos-1].isalpha() and cur.isdigit()) or \
                   (text[pos-1].isdigit() and cur.isalpha()):
                    break
                pos += 1

            token = text[start:pos].lower()
            lang = 'num' if token.isdigit() else 'en'

            if split_char:
                for i, char in enumerate(token):
                    seq.append(Phoneme(char, lang, is_word_start=(i==0),
                                      is_word_end=(i==len(token)-1),
                                      char_start=start+i, char_end=start+i+1))
            else:
                seq.append(Phoneme(token, lang, is_word_start=True,
                                  is_word_end=True, char_start=start, char_end=pos))
        else:
            pos += 1

    return seq
