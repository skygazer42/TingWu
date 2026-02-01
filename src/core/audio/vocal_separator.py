# coding: utf-8
"""
人声分离模块

基于 Demucs 的音频源分离，提取人声轨道。
用于含背景音乐/环境声的音频预处理。

用法:
    from src.core.audio.vocal_separator import VocalSeparator

    separator = VocalSeparator(model_name="htdemucs")
    vocals = separator.separate_vocals(audio, sample_rate=16000)
"""

__all__ = ['VocalSeparator']

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class VocalSeparator:
    """基于 Demucs 的人声分离

    Demucs 是 Meta 开发的音频源分离模型，可将音频分离为：
    - drums, bass, other, vocals
    我们只提取 vocals 轨道。

    Args:
        model_name: Demucs 模型名称 (默认 htdemucs)
        device: 设备 ("cpu" 或 "cuda")
    """

    def __init__(self, model_name: str = "htdemucs", device: str = "cpu"):
        self._model_name = model_name
        self._device = device
        self._model = None

    def _init_model(self):
        """懒加载模型"""
        if self._model is not None:
            return True

        try:
            import torch
            from demucs.pretrained import get_model
            self._model = get_model(self._model_name)
            self._model.to(self._device)
            self._model.eval()
            logger.info(f"Demucs model '{self._model_name}' loaded (device={self._device})")
            return True
        except ImportError as e:
            logger.error(f"Demucs not installed: {e}. Install with: pip install demucs")
            return False
        except Exception as e:
            logger.error(f"Failed to load Demucs model: {e}")
            return False

    def _resample(self, audio: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
        """重采样"""
        if sr_from == sr_to:
            return audio
        try:
            import librosa
            return librosa.resample(audio, orig_sr=sr_from, target_sr=sr_to)
        except ImportError:
            ratio = sr_to / sr_from
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            return np.interp(indices, np.arange(len(audio)), audio)

    def separate_vocals(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """提取人声轨道

        Args:
            audio: 输入音频 (numpy array, float32, 单声道)
            sample_rate: 输入采样率

        Returns:
            人声音频（保持原采样率）
        """
        if len(audio) == 0:
            return audio

        if not self._init_model():
            return audio

        try:
            import torch
            from demucs.apply import apply_model

            original_sr = sample_rate
            model_sr = self._model.samplerate

            # 重采样到模型要求的采样率
            if sample_rate != model_sr:
                audio_resampled = self._resample(audio, sample_rate, model_sr)
            else:
                audio_resampled = audio

            # 转换为 tensor: (batch, channels, samples)
            wav = torch.from_numpy(audio_resampled.astype(np.float32))
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)  # (1, samples) = mono
            wav = wav.unsqueeze(0)  # (1, 1, samples) = batch

            # 如果模型期望立体声，复制声道
            if self._model.audio_channels == 2 and wav.shape[1] == 1:
                wav = wav.expand(-1, 2, -1)

            wav = wav.to(self._device)

            # 分离
            with torch.no_grad():
                sources = apply_model(self._model, wav)

            # 获取 vocals 索引 (通常是索引 3)
            source_names = self._model.sources
            vocals_idx = source_names.index('vocals') if 'vocals' in source_names else -1

            if vocals_idx < 0:
                logger.warning("'vocals' source not found in model, returning original audio")
                return audio

            # 提取人声: (batch, channels, samples) -> (samples,)
            vocals = sources[0, vocals_idx]
            if vocals.shape[0] > 1:
                vocals = vocals.mean(dim=0)  # 混合为单声道
            else:
                vocals = vocals.squeeze(0)

            vocals_np = vocals.cpu().numpy()

            # 重采样回原始采样率
            if original_sr != model_sr:
                vocals_np = self._resample(vocals_np, model_sr, original_sr)

            return vocals_np.astype(audio.dtype)

        except Exception as e:
            logger.warning(f"Vocal separation failed: {e}")
            return audio

    def is_available(self) -> bool:
        """检查 Demucs 是否可用"""
        try:
            from demucs.pretrained import get_model
            return True
        except ImportError:
            return False
