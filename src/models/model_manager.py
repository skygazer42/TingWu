"""全局模型管理器"""
from typing import Optional
from src.config import settings
from src.models.asr_loader import ASRModelLoader

class ModelManager:
    """模型管理器单例"""
    _instance: Optional['ModelManager'] = None
    _loader: Optional[ASRModelLoader] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def loader(self) -> ASRModelLoader:
        if self._loader is None:
            self._loader = ASRModelLoader(
                device=settings.device,
                ngpu=settings.ngpu,
                ncpu=settings.ncpu,
                asr_model=settings.asr_model,
                asr_model_online=settings.asr_model_online,
                vad_model=settings.vad_model,
                punc_model=settings.punc_model,
                spk_model=settings.spk_model,
            )
        return self._loader

    def preload_models(self, with_speaker: bool = True):
        """预加载模型"""
        _ = self.loader.asr_model
        _ = self.loader.asr_model_online
        if with_speaker:
            _ = self.loader.asr_model_with_spk

model_manager = ModelManager()
