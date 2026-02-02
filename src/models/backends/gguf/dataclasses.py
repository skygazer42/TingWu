"""FunASR-GGUF 数据类型定义

移植自 CapsWriter-Offline
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np


@dataclass
class GGUFConfig:
    """GGUF 后端配置"""
    encoder_onnx_path: str
    ctc_onnx_path: str
    decoder_gguf_path: str
    tokens_path: str
    hotwords_path: Optional[str] = None
    n_predict: int = 512
    n_threads: Optional[int] = None
    n_threads_batch: Optional[int] = None
    n_ubatch: int = 512
    sample_rate: int = 16000


@dataclass
class CTCToken:
    """CTC 解码结果"""
    text: str
    start: float


@dataclass
class Timings:
    """各阶段耗时统计（秒）"""
    encode: float = 0.0
    ctc: float = 0.0
    prepare: float = 0.0
    inject: float = 0.0
    llm_generate: float = 0.0
    align: float = 0.0
    total: float = 0.0
