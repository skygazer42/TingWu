"""ONNX 后端 - 基于 funasr-onnx 的高性能推理"""
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

from .base import ASRBackend

logger = logging.getLogger(__name__)

# ONNX 模型 ID
ONNX_MODEL_PARAFORMER = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx"
ONNX_MODEL_VAD = "iic/speech_fsmn_vad_zh-cn-16k-common-onnx"
ONNX_MODEL_PUNC = "iic/punc_ct-transformer_cn-en-common-vocab471067-large-onnx"


class ONNXBackend(ASRBackend):
    """基于 funasr-onnx 的 ONNX Runtime 后端

    特点：
    - 使用 ONNX Runtime 进行推理，性能提升 2-10x
    - 支持 INT8 量化，进一步降低内存和延迟
    - 仅支持离线转写，不支持流式

    Requirements:
        pip install funasr-onnx onnxruntime
        # 或 GPU 版本
        pip install funasr-onnx onnxruntime-gpu
    """

    def __init__(
        self,
        device: str = "cpu",
        ncpu: int = 4,
        quantize: bool = True,
        model_dir: Optional[str] = None,
        vad_model_dir: Optional[str] = None,
        punc_model_dir: Optional[str] = None,
        **kwargs
    ):
        """初始化 ONNX 后端

        Args:
            device: 设备类型 ("cpu" 或 "cuda")
            ncpu: CPU 线程数
            quantize: 是否启用 INT8 量化
            model_dir: ASR 模型路径（默认自动下载）
            vad_model_dir: VAD 模型路径（默认自动下载）
            punc_model_dir: 标点模型路径（默认自动下载）
        """
        self.device = device
        self.ncpu = ncpu
        self.quantize = quantize
        self.model_dir = model_dir or ONNX_MODEL_PARAFORMER
        self.vad_model_dir = vad_model_dir or ONNX_MODEL_VAD
        self.punc_model_dir = punc_model_dir or ONNX_MODEL_PUNC

        self._model = None
        self._vad_model = None
        self._punc_model = None
        self._loaded = False

    def load(self) -> None:
        """加载 ONNX 模型"""
        if self._loaded:
            return

        try:
            from funasr_onnx import Paraformer, Fsmn_vad, CT_Transformer
        except ImportError:
            raise ImportError(
                "ONNX 后端需要安装 funasr-onnx: pip install funasr-onnx onnxruntime"
            )

        logger.info(f"Loading ONNX ASR model: {self.model_dir}")
        logger.info(f"Quantization: {self.quantize}, Device: {self.device}")

        # 加载 ASR 模型
        self._model = Paraformer(
            model_dir=self.model_dir,
            quantize=self.quantize,
        )

        # 加载 VAD 模型
        logger.info(f"Loading ONNX VAD model: {self.vad_model_dir}")
        self._vad_model = Fsmn_vad(
            model_dir=self.vad_model_dir,
            quantize=self.quantize,
        )

        # 加载标点模型
        logger.info(f"Loading ONNX Punctuation model: {self.punc_model_dir}")
        self._punc_model = CT_Transformer(
            model_dir=self.punc_model_dir,
            quantize=self.quantize,
        )

        self._loaded = True
        logger.info("ONNX backend loaded successfully")

    def _ensure_loaded(self):
        """确保模型已加载"""
        if not self._loaded:
            self.load()

    @property
    def supports_streaming(self) -> bool:
        """ONNX 后端不支持流式"""
        return False

    @property
    def supports_speaker(self) -> bool:
        """ONNX 后端暂不支持说话人识别"""
        return False

    def transcribe(
        self,
        audio_input,
        hotwords: Optional[str] = None,
        with_speaker: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """执行转写

        Args:
            audio_input: 音频输入（文件路径或字节）
            hotwords: 热词（ONNX 后端不直接支持，需通过后处理纠正）
            with_speaker: 忽略（ONNX 不支持说话人）
            **kwargs: 其他参数

        Returns:
            转写结果字典，包含 text 和 sentence_info
        """
        self._ensure_loaded()

        if with_speaker:
            logger.warning("ONNX backend does not support speaker diarization, ignoring with_speaker=True")

        if hotwords:
            logger.debug("ONNX backend: hotwords will be processed via post-processing pipeline")

        try:
            # 直接对完整音频进行 ASR
            # funasr_onnx Paraformer 返回: [{'preds': ('text', [chars])}]
            asr_result = self._model(audio_input)

            if not asr_result or len(asr_result) == 0:
                return {"text": "", "sentence_info": []}

            # 提取原始文本
            item = asr_result[0]
            if isinstance(item, dict) and "preds" in item:
                raw_text = item["preds"][0] if isinstance(item["preds"], tuple) else str(item["preds"])
            elif isinstance(item, dict) and "text" in item:
                raw_text = item["text"]
            else:
                raw_text = str(item) if item else ""

            if not raw_text:
                return {"text": "", "sentence_info": []}

            # 添加标点
            # funasr_onnx CT_Transformer 返回: ('text_with_punc', [punc_codes])
            try:
                punc_result = self._punc_model(raw_text)
                if isinstance(punc_result, tuple) and len(punc_result) >= 1:
                    text = punc_result[0]
                elif isinstance(punc_result, list) and len(punc_result) > 0:
                    text = punc_result[0] if isinstance(punc_result[0], str) else raw_text
                else:
                    text = raw_text
            except Exception as e:
                logger.warning(f"Punctuation model failed, using raw text: {e}")
                text = raw_text

            return {
                "text": text,
                "sentence_info": [{"text": text, "start": 0, "end": 0}] if text else [],
            }

        except Exception as e:
            logger.error(f"ONNX transcription failed: {e}")
            raise

    def unload(self) -> None:
        """释放模型资源"""
        self._model = None
        self._vad_model = None
        self._punc_model = None
        self._loaded = False
        logger.info("ONNX backend models unloaded")

    def get_info(self) -> Dict[str, Any]:
        """获取后端信息"""
        return {
            "name": "ONNXBackend",
            "type": "onnx",
            "device": self.device,
            "quantize": self.quantize,
            "model_dir": self.model_dir,
            "vad_model_dir": self.vad_model_dir,
            "punc_model_dir": self.punc_model_dir,
            "supports_streaming": False,
            "supports_hotwords": True,  # 通过后处理支持
            "supports_speaker": False,
        }
