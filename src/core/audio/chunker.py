"""
智能音频分块模块
基于 VAD 的智能分割，支持分块重叠处理和并行转写。
"""

import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import asyncio

logger = logging.getLogger(__name__)


class AudioChunker:
    """智能音频分块器

    功能:
    - 基于 VAD 的智能分割点检测
    - 分块重叠避免边界截断
    - 支持并行处理分块
    - 结果合并

    用法:
        chunker = AudioChunker(max_chunk_duration=60.0)
        chunks = chunker.split(audio, sample_rate)
        results = chunker.process_parallel(chunks, transcribe_func)
        final_text = chunker.merge_results(results)
    """

    def __init__(
        self,
        max_chunk_duration: float = 60.0,
        min_chunk_duration: float = 5.0,
        overlap_duration: float = 0.5,
        silence_threshold_db: float = -40.0,
        min_silence_duration: float = 0.3,
    ):
        """
        初始化分块器

        Args:
            max_chunk_duration: 最大分块时长(秒)
            min_chunk_duration: 最小分块时长(秒)
            overlap_duration: 分块重叠时长(秒)
            silence_threshold_db: 静音检测阈值(dB)
            min_silence_duration: 最小静音时长(秒)，用于检测分割点
        """
        self.max_chunk_duration = max_chunk_duration
        self.min_chunk_duration = min_chunk_duration
        self.overlap_duration = overlap_duration
        self.silence_threshold = 10 ** (silence_threshold_db / 20)
        self.min_silence_duration = min_silence_duration

    def _find_silence_points(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> List[int]:
        """查找静音分割点

        Args:
            audio: 音频数据
            sample_rate: 采样率

        Returns:
            静音点位置列表 (样本索引)
        """
        # 帧参数
        frame_length = int(sample_rate * 0.025)  # 25ms
        hop_length = int(sample_rate * 0.010)    # 10ms
        min_silence_frames = int(self.min_silence_duration / 0.010)

        # 计算帧能量
        num_frames = (len(audio) - frame_length) // hop_length + 1
        if num_frames <= 0:
            return []

        frame_energies = np.zeros(num_frames)
        for i in range(num_frames):
            start = i * hop_length
            end = start + frame_length
            frame = audio[start:end]
            frame_energies[i] = np.sqrt(np.mean(frame ** 2))

        # 找到静音区域
        is_silence = frame_energies < self.silence_threshold

        # 找到连续静音区域的中点
        silence_points = []
        silence_start = None

        for i, silent in enumerate(is_silence):
            if silent:
                if silence_start is None:
                    silence_start = i
            else:
                if silence_start is not None:
                    silence_length = i - silence_start
                    if silence_length >= min_silence_frames:
                        # 取静音区域的中点作为分割点
                        mid_frame = silence_start + silence_length // 2
                        mid_sample = mid_frame * hop_length
                        silence_points.append(mid_sample)
                    silence_start = None

        return silence_points

    def split(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> List[Tuple[np.ndarray, int, int]]:
        """分割长音频为多个块

        Args:
            audio: 音频数据
            sample_rate: 采样率

        Returns:
            分块列表 [(chunk_audio, start_sample, end_sample), ...]
        """
        duration = len(audio) / sample_rate

        # 如果音频较短，不需要分割
        if duration <= self.max_chunk_duration:
            return [(audio, 0, len(audio))]

        # 找到所有静音分割点
        silence_points = self._find_silence_points(audio, sample_rate)

        # 计算分块边界
        max_samples = int(self.max_chunk_duration * sample_rate)
        min_samples = int(self.min_chunk_duration * sample_rate)
        overlap_samples = int(self.overlap_duration * sample_rate)

        chunks = []
        current_start = 0

        while current_start < len(audio):
            # 目标结束位置
            target_end = min(current_start + max_samples, len(audio))

            if target_end >= len(audio):
                # 最后一个块
                chunks.append((audio[current_start:], current_start, len(audio)))
                break

            # 在目标结束位置附近查找最佳分割点
            best_split = target_end
            search_start = max(current_start + min_samples, target_end - max_samples // 4)

            for point in silence_points:
                if search_start <= point <= target_end:
                    best_split = point
                    break

            # 添加分块 (包含重叠)
            chunk_end = min(best_split + overlap_samples, len(audio))
            chunks.append((audio[current_start:chunk_end], current_start, chunk_end))

            # 下一个块的起始位置 (减去重叠)
            current_start = max(best_split - overlap_samples, current_start + min_samples)

        logger.info(f"Split audio ({duration:.1f}s) into {len(chunks)} chunks")
        return chunks

    def process_parallel(
        self,
        chunks: List[Tuple[np.ndarray, int, int]],
        transcribe_func,
        max_workers: int = 2,
    ) -> List[Dict[str, Any]]:
        """并行处理分块

        Args:
            chunks: 分块列表
            transcribe_func: 转写函数，接受音频数据返回结果字典
            max_workers: 最大工作线程数

        Returns:
            各分块的转写结果
        """
        def process_chunk(args):
            chunk_audio, start_sample, end_sample = args
            try:
                result = transcribe_func(chunk_audio)
                return {
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                    "success": True,
                    "result": result,
                }
            except Exception as e:
                logger.error(f"Chunk processing failed: {e}")
                return {
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                    "success": False,
                    "error": str(e),
                }

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_chunk, chunks))

        return results

    async def process_parallel_async(
        self,
        chunks: List[Tuple[np.ndarray, int, int]],
        transcribe_func,
        max_concurrent: int = 2,
    ) -> List[Dict[str, Any]]:
        """异步并行处理分块

        Args:
            chunks: 分块列表
            transcribe_func: 异步转写函数
            max_concurrent: 最大并发数

        Returns:
            各分块的转写结果
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_chunk(chunk_audio, start_sample, end_sample):
            async with semaphore:
                try:
                    result = await transcribe_func(chunk_audio)
                    return {
                        "start_sample": start_sample,
                        "end_sample": end_sample,
                        "success": True,
                        "result": result,
                    }
                except Exception as e:
                    logger.error(f"Chunk processing failed: {e}")
                    return {
                        "start_sample": start_sample,
                        "end_sample": end_sample,
                        "success": False,
                        "error": str(e),
                    }

        tasks = [
            process_chunk(chunk_audio, start, end)
            for chunk_audio, start, end in chunks
        ]
        return await asyncio.gather(*tasks)

    def merge_results(
        self,
        chunk_results: List[Dict[str, Any]],
        sample_rate: int = 16000,
        overlap_chars: int = 10,
    ) -> Dict[str, Any]:
        """合并分块转写结果

        Args:
            chunk_results: 分块结果列表
            sample_rate: 采样率
            overlap_chars: 重叠文本字符数 (用于去重)

        Returns:
            合并后的结果
        """
        # 按起始位置排序
        sorted_results = sorted(chunk_results, key=lambda x: x["start_sample"])

        texts = []
        all_sentences = []

        for i, result in enumerate(sorted_results):
            if not result["success"]:
                continue

            chunk_result = result.get("result", {})
            text = chunk_result.get("text", "")
            sentences = chunk_result.get("sentences", [])

            # 时间偏移
            time_offset_ms = int(result["start_sample"] / sample_rate * 1000)

            # 调整句子时间戳
            for sent in sentences:
                sent["start"] = sent.get("start", 0) + time_offset_ms
                sent["end"] = sent.get("end", 0) + time_offset_ms
                all_sentences.append(sent)

            # 去除重叠文本
            if i > 0 and texts and text and overlap_chars > 0:
                prev_text = texts[-1]
                # 查找重叠部分
                for j in range(min(overlap_chars * 2, len(prev_text)), 0, -1):
                    suffix = prev_text[-j:]
                    if text.startswith(suffix):
                        text = text[j:]
                        break

            if text:
                texts.append(text)

        return {
            "text": "".join(texts),
            "sentences": all_sentences,
        }


# 全局分块器实例
audio_chunker = AudioChunker()
