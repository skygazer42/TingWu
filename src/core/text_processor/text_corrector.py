# coding: utf-8
"""
通用文本纠错模块

基于 pycorrector 提供中文文本纠错，支持 kenlm 和 macbert 后端。

用法:
    from src.core.text_processor.text_corrector import TextCorrector

    corrector = TextCorrector(backend="kenlm")
    corrected, errors = corrector.correct("今天的天气真不措")
    # corrected: "今天的天气真不错"
"""

__all__ = ['TextCorrector']

import logging
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)


class TextCorrector:
    """基于 pycorrector 的通用文本纠错

    支持两种后端:
    - kenlm: 基于统计语言模型，轻量快速
    - macbert: 基于 BERT，语义理解更强，需要 GPU

    Args:
        backend: 纠错后端 ("kenlm" 或 "macbert")
        device: 设备 ("cpu" 或 "cuda")，仅 macbert 使用
        model_name: macbert 模型名称
    """

    def __init__(
        self,
        backend: str = "kenlm",
        device: str = "cpu",
        model_name: str = "shibing624/macbert4csc-base-chinese",
    ):
        self._backend = backend
        self._device = device
        self._model_name = model_name
        self._corrector = None

    @property
    def corrector(self):
        """懒加载纠错器"""
        if self._corrector is None:
            self._corrector = self._init_corrector()
        return self._corrector

    def _init_corrector(self):
        """初始化纠错器"""
        try:
            if self._backend == "kenlm":
                from pycorrector import Corrector
                corrector = Corrector()
                logger.info("Initialized pycorrector (kenlm backend)")
                return corrector
            elif self._backend == "macbert":
                from pycorrector import MacBertCorrector
                corrector = MacBertCorrector(self._model_name)
                logger.info(f"Initialized pycorrector (macbert backend: {self._model_name})")
                return corrector
            else:
                raise ValueError(f"Unknown backend: {self._backend}")
        except ImportError as e:
            logger.error(f"Failed to import pycorrector: {e}. Install with: pip install pycorrector")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize pycorrector ({self._backend}): {e}")
            raise

    def correct(self, text: str) -> Tuple[str, List]:
        """纠正文本

        Args:
            text: 输入文本

        Returns:
            (纠正后文本, 错误列表)
            错误列表格式: [(错误词, 纠正词, 位置), ...]
        """
        if not text or not text.strip():
            return text, []

        try:
            result = self.corrector.correct(text)
            corrected = result.get('target', text)
            errors = result.get('errors', [])
            if errors:
                logger.debug(f"Text correction: {text!r} -> {corrected!r}, errors={errors}")
            return corrected, errors
        except Exception as e:
            logger.warning(f"Text correction failed: {e}")
            return text, []

    def correct_batch(self, texts: List[str]) -> List[Tuple[str, List]]:
        """批量纠正文本

        Args:
            texts: 文本列表

        Returns:
            [(纠正后文本, 错误列表), ...]
        """
        if not texts:
            return []

        try:
            results = self.corrector.correct_batch(texts)
            return [
                (r.get('target', t), r.get('errors', []))
                for t, r in zip(texts, results)
            ]
        except Exception as e:
            logger.warning(f"Batch text correction failed: {e}")
            return [(t, []) for t in texts]
