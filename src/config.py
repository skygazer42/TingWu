import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional, Literal

class Settings(BaseSettings):
    """应用配置"""
    # 服务配置
    app_name: str = "TingWu Speech Service"
    version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # 路径配置
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = base_dir / "data"
    models_dir: Path = data_dir / "models"
    hotwords_dir: Path = data_dir / "hotwords"
    uploads_dir: Path = data_dir / "uploads"
    outputs_dir: Path = data_dir / "outputs"

    # FunASR 模型配置
    asr_model: str = "paraformer-zh"
    asr_model_online: str = "paraformer-zh-streaming"
    vad_model: str = "fsmn-vad"
    punc_model: str = "ct-punc-c"
    spk_model: str = "cam++"

    # 设备配置
    device: Literal["cuda", "cpu"] = "cuda"
    ngpu: int = 1
    ncpu: int = 4

    # 热词配置
    hotwords_file: str = "hotwords.txt"
    hotwords_threshold: float = 0.85

    # LLM 优化配置 (可选)
    llm_enable: bool = False
    llm_model: str = "qwen2.5:7b"
    llm_base_url: str = "http://localhost:11434"

    # WebSocket 配置
    ws_chunk_size: int = 9600  # 600ms @ 16kHz
    ws_chunk_interval: int = 10

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

# 确保目录存在
for dir_path in [settings.data_dir, settings.models_dir, settings.hotwords_dir,
                 settings.uploads_dir, settings.outputs_dir]:
    dir_path.mkdir(parents=True, exist_ok=True)
