# coding: utf-8
"""
标点恢复模块

基于 FunASR ct-punc 模型的独立标点恢复，适用于标点不准确或缺失的场景。

用法:
    from src.core.text_processor.punctuation_restorer import PunctuationRestorer

    restorer = PunctuationRestorer(device="cpu")
    text_with_punc = restorer.restore("今天天气很好明天也不错")
    # "今天天气很好，明天也不错。"
"""

__all__ = ['PunctuationRestorer']

import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


class PunctuationRestorer:
    """基于 FunASR ct-punc 的标点恢复

    使用 FunASR 的 ct-punc 模型为无标点或标点不完整的文本添加标点。

    Args:
        model: 标点模型名称 (默认 ct-punc-c)
        device: 设备 ("cpu" 或 "cuda")
    """

    def __init__(self, model: str = "ct-punc-c", device: str = "cpu"):
        self._model_name = model
        self._device = device
        self._model = None

    def _init_model(self):
        """懒加载模型"""
        if self._model is not None:
            return True

        try:
            from funasr import AutoModel
            self._model = AutoModel(model=self._model_name, device=self._device)
            logger.info(f"PunctuationRestorer initialized (model={self._model_name}, device={self._device})")
            return True
        except ImportError as e:
            logger.error(f"FunASR not installed: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load punctuation model: {e}")
            return False

    def restore(self, text: str) -> str:
        """恢复标点

        Args:
            text: 输入文本（无标点或标点不完整）

        Returns:
            添加标点后的文本
        """
        if not text or not text.strip():
            return text

        if not self._init_model():
            return text

        try:
            result = self._model.generate(input=text)
            if result and len(result) > 0:
                restored = result[0].get('text', text)
                if restored:
                    logger.debug(f"Punctuation restored: {text!r} -> {restored!r}")
                    return restored
            return text
        except Exception as e:
            logger.warning(f"Punctuation restoration failed: {e}")
            return text

    def restore_batch(self, texts: List[str]) -> List[str]:
        """批量恢复标点

        Args:
            texts: 文本列表

        Returns:
            添加标点后的文本列表
        """
        if not texts:
            return texts

        if not self._init_model():
            return texts

        results = []
        for text in texts:
            results.append(self.restore(text))
        return results

    def is_available(self) -> bool:
        """检查标点模型是否可用"""
        try:
            from funasr import AutoModel
            return True
        except ImportError:
            return False
