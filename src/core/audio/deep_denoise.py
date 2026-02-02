# coding: utf-8
"""
深度学习降噪模块
基于 DeepFilterNet 的深度学习语音增强，延迟 <20ms，质量高于频谱减法。

用法:
    from src.core.audio.deep_denoise import DeepDenoiser

    denoiser = DeepDenoiser(device="cpu")
    enhanced = denoiser.enhance(audio, sample_rate=16000)
"""

__all__ = ['DeepDenoiser']

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class DeepDenoiser:
    """基于 DeepFilterNet 的深度学习降噪

    DeepFilterNet 是一个低延迟、高质量的语音增强模型：
    - 延迟 <20ms，适合实时场景
    - 对非平稳噪声效果显著优于频谱减法
    - 支持 CPU/GPU 推理

    Args:
        device: 设备 ("cpu" 或 "cuda")
        post_filter: 是否启用后置滤波（提升质量但增加延迟）
    """

    def __init__(self, device: str = "cpu", post_filter: bool = True):
        self._device = device
        self._post_filter = post_filter
        self._model = None
        self._df_state = None
        self._sr_target = 48000  # DeepFilterNet 要求 48kHz

    def _init_model(self):
        """懒加载模型"""
        if self._model is not None:
            return True

        try:
            from df.enhance import init_df
            self._model, self._df_state, _ = init_df(post_filter=self._post_filter)
            logger.info(f"DeepFilterNet initialized (device={self._device}, post_filter={self._post_filter})")
            return True
        except ImportError as e:
            logger.error(f"DeepFilterNet not installed: {e}. Install with: pip install deepfilternet")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize DeepFilterNet: {e}")
            return False

    def _resample(self, audio: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
        """重采样"""
        if sr_from == sr_to:
            return audio
        try:
            import librosa
            return librosa.resample(audio, orig_sr=sr_from, target_sr=sr_to)
        except ImportError:
            # 简单线性插值作为后备
            ratio = sr_to / sr_from
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            return np.interp(indices, np.arange(len(audio)), audio)

    def enhance(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """增强音频

        Args:
            audio: 输入音频数据 (numpy array, float32, 单声道)
            sample_rate: 输入采样率

        Returns:
            增强后的音频（保持原采样率）
        """
        if len(audio) == 0:
            return audio

        if not self._init_model():
            return audio

        try:
            import torch
            from df.enhance import enhance

            # 保存原始采样率
            original_sr = sample_rate

            # 重采样到 48kHz (DeepFilterNet 要求)
            if sample_rate != self._sr_target:
                audio_48k = self._resample(audio, sample_rate, self._sr_target)
            else:
                audio_48k = audio

            # 转换为 torch tensor
            audio_tensor = torch.from_numpy(audio_48k.astype(np.float32))

            # 添加 batch 维度如果需要
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            # 增强
            enhanced = enhance(self._model, self._df_state, audio_tensor)

            # 转回 numpy
            if isinstance(enhanced, torch.Tensor):
                enhanced = enhanced.squeeze().cpu().numpy()

            # 重采样回原始采样率
            if original_sr != self._sr_target:
                enhanced = self._resample(enhanced, self._sr_target, original_sr)

            return enhanced.astype(audio.dtype)

        except Exception as e:
            logger.warning(f"DeepFilterNet enhancement failed: {e}")
            return audio

    def is_available(self) -> bool:
        """检查 DeepFilterNet 是否可用"""
        try:
            from df.enhance import init_df
            return True
        except ImportError:
            return False
