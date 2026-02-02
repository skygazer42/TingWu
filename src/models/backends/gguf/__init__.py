"""GGUF 量化模型后端

使用 ONNX (encoder/CTC) + llama.cpp (GGUF decoder) 进行语音识别。
移植自 CapsWriter-Offline。
"""

from .backend import GGUFBackend

__all__ = ['GGUFBackend']
