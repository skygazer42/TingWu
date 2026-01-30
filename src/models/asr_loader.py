"""FunASR 模型加载器 - 懒加载模式"""
import logging
from typing import Optional, Dict, Any
from funasr import AutoModel

logger = logging.getLogger(__name__)

class ASRModelLoader:
    """ASR 模型加载器，支持懒加载和模型组合"""

    def __init__(
        self,
        device: str = "cuda",
        ngpu: int = 1,
        ncpu: int = 4,
        asr_model: str = "paraformer-zh",
        asr_model_online: str = "paraformer-zh-streaming",
        vad_model: str = "fsmn-vad",
        punc_model: str = "ct-punc-c",
        spk_model: Optional[str] = "cam++",
    ):
        self.device = device
        self.ngpu = ngpu
        self.ncpu = ncpu

        self._asr_model_name = asr_model
        self._asr_model_online_name = asr_model_online
        self._vad_model_name = vad_model
        self._punc_model_name = punc_model
        self._spk_model_name = spk_model

        self._asr_model: Optional[AutoModel] = None
        self._asr_model_online: Optional[AutoModel] = None
        self._asr_model_with_spk: Optional[AutoModel] = None

    @property
    def asr_model(self) -> AutoModel:
        """获取离线 ASR 模型 (VAD + ASR + Punc)"""
        if self._asr_model is None:
            logger.info(f"Loading offline ASR model: {self._asr_model_name}")
            self._asr_model = AutoModel(
                model=self._asr_model_name,
                vad_model=self._vad_model_name,
                punc_model=self._punc_model_name,
                device=self.device,
                ngpu=self.ngpu,
                ncpu=self.ncpu,
                disable_pbar=True,
                disable_log=True,
            )
            logger.info("Offline ASR model loaded successfully")
        return self._asr_model

    @property
    def asr_model_online(self) -> AutoModel:
        """获取在线流式 ASR 模型"""
        if self._asr_model_online is None:
            logger.info(f"Loading online ASR model: {self._asr_model_online_name}")
            self._asr_model_online = AutoModel(
                model=self._asr_model_online_name,
                device=self.device,
                ngpu=self.ngpu,
                ncpu=self.ncpu,
                disable_pbar=True,
                disable_log=True,
            )
            logger.info("Online ASR model loaded successfully")
        return self._asr_model_online

    @property
    def asr_model_with_spk(self) -> AutoModel:
        """获取带说话人识别的 ASR 模型"""
        if self._asr_model_with_spk is None:
            logger.info("Loading ASR model with speaker diarization")
            self._asr_model_with_spk = AutoModel(
                model=self._asr_model_name,
                vad_model=self._vad_model_name,
                punc_model=self._punc_model_name,
                spk_model=self._spk_model_name,
                device=self.device,
                ngpu=self.ngpu,
                ncpu=self.ncpu,
                disable_pbar=True,
                disable_log=True,
            )
            logger.info("ASR model with speaker loaded successfully")
        return self._asr_model_with_spk

    def transcribe(
        self,
        audio_input,
        hotwords: Optional[str] = None,
        with_speaker: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """执行转写"""
        params = {
            "input": audio_input,
            "sentence_timestamp": True,
            "batch_size_s": 300,
        }
        if hotwords:
            params["hotword"] = hotwords
        params.update(kwargs)

        model = self.asr_model_with_spk if with_speaker else self.asr_model
        result = model.generate(**params)
        return result[0] if result else {}
