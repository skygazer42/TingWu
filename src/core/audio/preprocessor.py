"""
音频预处理器

提供:
- 降噪 (noise reduction)
- 音量归一化 (RMS normalization)
- 静音裁剪 (silence trimming)
- 格式验证
"""

import logging
import numpy as np
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

__all__ = ['AudioPreprocessor']


class AudioPreprocessor:
    """
    音频预处理器

    支持:
    - 基于频谱减法的降噪
    - RMS 音量归一化到目标电平
    - 首尾静音裁剪
    - 音频格式验证

    用法:
        processor = AudioPreprocessor(target_db=-20.0, denoise_enable=True)
        audio = processor.process(audio_data, sample_rate)
    """

    def __init__(
        self,
        target_db: float = -20.0,
        silence_threshold_db: float = -40.0,
        min_silence_ms: int = 500,
        normalize_enable: bool = True,
        trim_silence_enable: bool = False,
        denoise_enable: bool = False,
        denoise_prop: float = 0.8,
        denoise_backend: str = "noisereduce",
        vocal_separate_enable: bool = False,
        vocal_separate_model: str = "htdemucs",
        device: str = "cpu",
    ):
        """
        初始化预处理器

        Args:
            target_db: 目标音量电平 (dB)，默认 -20 dB
            silence_threshold_db: 静音判定阈值 (dB)，默认 -40 dB
            min_silence_ms: 最小静音时长 (毫秒)，默认 500ms
            normalize_enable: 是否启用音量归一化
            trim_silence_enable: 是否启用静音裁剪
            denoise_enable: 是否启用降噪
            denoise_prop: 降噪强度 (0-1)，默认 0.8
            denoise_backend: 降噪后端 ("noisereduce" 或 "deepfilter")
            vocal_separate_enable: 是否启用人声分离
            vocal_separate_model: Demucs 模型名称
            device: 设备 ("cpu" 或 "cuda")
        """
        self.target_db = target_db
        self.silence_threshold_db = silence_threshold_db
        self.min_silence_ms = min_silence_ms
        self.normalize_enable = normalize_enable
        self.trim_silence_enable = trim_silence_enable
        self.denoise_enable = denoise_enable
        self.denoise_prop = denoise_prop
        self.denoise_backend = denoise_backend
        self.vocal_separate_enable = vocal_separate_enable
        self.vocal_separate_model = vocal_separate_model
        self.device = device

        # 预计算阈值
        self.target_rms = self._db_to_amplitude(target_db)
        self.silence_threshold = self._db_to_amplitude(silence_threshold_db)

        # 懒加载 noisereduce
        self._nr = None

        # 懒加载 DeepFilterNet
        self._deep_denoiser = None

        # 懒加载人声分离
        self._vocal_separator = None

    @property
    def nr(self):
        """懒加载 noisereduce 模块"""
        if self._nr is None:
            try:
                import noisereduce as nr
                self._nr = nr
            except ImportError:
                logger.warning(
                    "noisereduce not installed, denoising disabled. "
                    "Install with: pip install noisereduce"
                )
                self.denoise_enable = False
                return None
        return self._nr

    @property
    def deep_denoiser(self):
        """懒加载 DeepFilterNet 降噪器"""
        if self._deep_denoiser is None:
            try:
                from src.core.audio.deep_denoise import DeepDenoiser
                self._deep_denoiser = DeepDenoiser(device=self.device)
                if not self._deep_denoiser.is_available():
                    logger.warning("DeepFilterNet not available, falling back to noisereduce")
                    self._deep_denoiser = None
            except Exception as e:
                logger.warning(f"Failed to load DeepFilterNet: {e}")
                self._deep_denoiser = None
        return self._deep_denoiser

    @property
    def vocal_separator(self):
        """懒加载人声分离器"""
        if self._vocal_separator is None:
            try:
                from src.core.audio.vocal_separator import VocalSeparator
                self._vocal_separator = VocalSeparator(
                    model_name=self.vocal_separate_model,
                    device=self.device
                )
                if not self._vocal_separator.is_available():
                    logger.warning("Demucs not available, vocal separation disabled")
                    self._vocal_separator = None
                    self.vocal_separate_enable = False
            except Exception as e:
                logger.warning(f"Failed to load vocal separator: {e}")
                self._vocal_separator = None
                self.vocal_separate_enable = False
        return self._vocal_separator

    @staticmethod
    def _db_to_amplitude(db: float) -> float:
        """dB 转振幅"""
        return 10 ** (db / 20)

    @staticmethod
    def _amplitude_to_db(amplitude: float) -> float:
        """振幅转 dB"""
        return 20 * np.log10(amplitude + 1e-10)

    def get_rms(self, audio: np.ndarray) -> float:
        """计算 RMS 值"""
        return np.sqrt(np.mean(audio ** 2))

    def get_rms_db(self, audio: np.ndarray) -> float:
        """计算 RMS dB 值"""
        rms = self.get_rms(audio)
        return self._amplitude_to_db(rms)

    def normalize_volume(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        RMS 音量归一化

        将音频归一化到目标 RMS 电平。

        Args:
            audio: 音频数据 (numpy array)
            sample_rate: 采样率 (未使用，保留接口)

        Returns:
            归一化后的音频
        """
        if len(audio) == 0:
            return audio

        # 计算当前 RMS
        current_rms = self.get_rms(audio)

        if current_rms < 1e-10:
            # 静音或接近静音，不处理
            return audio

        # 计算增益
        gain = self.target_rms / current_rms

        # 应用增益，限制最大增益避免过度放大噪声
        max_gain = 10.0  # 最大 20dB 增益
        gain = min(gain, max_gain)

        # 归一化
        normalized = audio * gain

        # 防止削波
        max_val = np.max(np.abs(normalized))
        if max_val > 0.99:
            normalized = normalized * (0.99 / max_val)

        return normalized.astype(audio.dtype)

    def denoise(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        降噪处理

        根据配置的 backend 使用不同的降噪算法：
        - noisereduce: 频谱减法，轻量快速
        - deepfilter: DeepFilterNet 深度学习，质量更高

        Args:
            audio: 音频数据 (numpy array)
            sample_rate: 采样率

        Returns:
            降噪后的音频
        """
        if len(audio) == 0:
            return audio

        # 使用 DeepFilterNet
        if self.denoise_backend == "deepfilter":
            if self.deep_denoiser is not None:
                return self.deep_denoiser.enhance(audio, sample_rate)
            logger.warning("DeepFilterNet not available, falling back to noisereduce")

        # 使用 noisereduce
        if self.nr is None:
            return audio

        try:
            denoised = self.nr.reduce_noise(
                y=audio,
                sr=sample_rate,
                stationary=False,  # 非平稳噪声
                prop_decrease=self.denoise_prop,
            )
            return denoised.astype(audio.dtype)
        except Exception as e:
            logger.warning(f"Denoising failed: {e}")
            return audio

    def trim_silence(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """
        裁剪首尾静音

        使用帧能量分析检测静音段。

        Args:
            audio: 音频数据
            sample_rate: 采样率

        Returns:
            裁剪后的音频
        """
        if len(audio) == 0:
            return audio

        # 帧参数
        frame_length_ms = 25  # 25ms 帧
        hop_length_ms = 10    # 10ms 步长
        frame_length = int(sample_rate * frame_length_ms / 1000)
        hop_length = int(sample_rate * hop_length_ms / 1000)

        # 最小静音帧数
        min_silence_frames = int(self.min_silence_ms / hop_length_ms)

        # 计算帧能量
        num_frames = (len(audio) - frame_length) // hop_length + 1
        if num_frames <= 0:
            return audio

        frame_energies = np.zeros(num_frames)
        for i in range(num_frames):
            start = i * hop_length
            end = start + frame_length
            frame = audio[start:end]
            frame_energies[i] = np.sqrt(np.mean(frame ** 2))

        # 查找非静音区域
        threshold = self.silence_threshold
        is_speech = frame_energies > threshold

        if not np.any(is_speech):
            # 全是静音
            return audio

        # 找到第一个和最后一个语音帧
        speech_indices = np.where(is_speech)[0]
        first_speech = speech_indices[0]
        last_speech = speech_indices[-1]

        # 转换为样本索引，保留一些边界
        margin_frames = 5  # 保留边界帧
        start_frame = max(0, first_speech - margin_frames)
        end_frame = min(num_frames - 1, last_speech + margin_frames)

        start_sample = start_frame * hop_length
        end_sample = min(len(audio), (end_frame + 1) * hop_length + frame_length)

        return audio[start_sample:end_sample]

    def validate(self, audio: np.ndarray, sample_rate: int) -> Tuple[bool, str]:
        """
        验证音频格式

        Args:
            audio: 音频数据
            sample_rate: 采样率

        Returns:
            (是否有效, 错误信息)
        """
        if audio is None:
            return False, "音频数据为空"

        if len(audio) == 0:
            return False, "音频长度为0"

        if sample_rate < 8000:
            return False, f"采样率过低: {sample_rate}Hz (最低 8000Hz)"

        if sample_rate > 48000:
            return False, f"采样率过高: {sample_rate}Hz (最高 48000Hz)"

        # 检查音频时长
        duration = len(audio) / sample_rate
        if duration < 0.1:
            return False, f"音频时长过短: {duration:.2f}s (最短 0.1s)"

        if duration > 3600:
            return False, f"音频时长过长: {duration:.2f}s (最长 3600s)"

        return True, ""

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        validate: bool = True,
    ) -> np.ndarray:
        """
        执行完整预处理流程

        处理顺序:
        1. 格式验证 (可选)
        2. 人声分离 (如果启用)
        3. 降噪 (如果启用)
        4. 静音裁剪 (如果启用)
        5. 音量归一化 (如果启用)

        Args:
            audio: 音频数据
            sample_rate: 采样率
            validate: 是否进行格式验证

        Returns:
            处理后的音频

        Raises:
            ValueError: 格式验证失败时抛出
        """
        if validate:
            is_valid, error = self.validate(audio, sample_rate)
            if not is_valid:
                raise ValueError(f"音频验证失败: {error}")

        # 1. 人声分离 (应在降噪之前)
        if self.vocal_separate_enable and self.vocal_separator is not None:
            audio = self.vocal_separator.separate_vocals(audio, sample_rate)

        # 2. 降噪 (应在其他处理之前)
        if self.denoise_enable:
            audio = self.denoise(audio, sample_rate)

        # 2. 静音裁剪
        if self.trim_silence_enable:
            audio = self.trim_silence(audio, sample_rate)

        # 3. 音量归一化
        if self.normalize_enable:
            audio = self.normalize_volume(audio, sample_rate)

        return audio

    def get_audio_info(self, audio: np.ndarray, sample_rate: int) -> dict:
        """
        获取音频信息

        Args:
            audio: 音频数据
            sample_rate: 采样率

        Returns:
            音频信息字典
        """
        if len(audio) == 0:
            return {
                'duration': 0.0,
                'samples': 0,
                'sample_rate': sample_rate,
                'rms_db': float('-inf'),
                'peak_db': float('-inf'),
            }

        duration = len(audio) / sample_rate
        rms = self.get_rms(audio)
        peak = np.max(np.abs(audio))

        return {
            'duration': duration,
            'samples': len(audio),
            'sample_rate': sample_rate,
            'rms_db': self._amplitude_to_db(rms),
            'peak_db': self._amplitude_to_db(peak),
        }


if __name__ == '__main__':
    # 测试
    print("=== AudioPreprocessor 测试 ===")

    # 创建测试音频
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    # 低音量正弦波 + 首尾静音
    silence_duration = 0.3
    silence_samples = int(sample_rate * silence_duration)
    signal = np.sin(2 * np.pi * 440 * t) * 0.01  # -40dB 左右
    audio = np.concatenate([
        np.zeros(silence_samples),
        signal,
        np.zeros(silence_samples),
    ]).astype(np.float32)

    processor = AudioPreprocessor(
        target_db=-20.0,
        normalize_enable=True,
        trim_silence_enable=True,
    )

    print(f"\n原始音频:")
    info = processor.get_audio_info(audio, sample_rate)
    print(f"  时长: {info['duration']:.2f}s")
    print(f"  RMS: {info['rms_db']:.1f} dB")
    print(f"  Peak: {info['peak_db']:.1f} dB")

    processed = processor.process(audio, sample_rate)

    print(f"\n处理后音频:")
    info = processor.get_audio_info(processed, sample_rate)
    print(f"  时长: {info['duration']:.2f}s")
    print(f"  RMS: {info['rms_db']:.1f} dB")
    print(f"  Peak: {info['peak_db']:.1f} dB")

    print("\n✓ 测试完成")
