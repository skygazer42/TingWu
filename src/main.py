"""TingWu Speech Service 主入口"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.api import api_router
from src.api.schemas import HealthResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info(f"Starting {settings.app_name} v{settings.version}")

    # 加载所有热词相关文件
    from src.core.engine import transcription_engine
    transcription_engine.load_all()

    # 启动热词文件监视器
    from src.core.hotword.watcher import setup_hotword_watcher, stop_hotword_watcher
    setup_hotword_watcher(
        watch_dir=str(settings.hotwords_dir),
        on_hotwords_change=lambda path: transcription_engine.load_hotwords(path),
        on_rules_change=lambda path: transcription_engine.load_rules(path),
        on_rectify_change=lambda path: transcription_engine.load_rectify_history(path),
    )

    # 启动异步任务管理器
    from src.core.task_manager import task_manager
    task_manager.start()

    logger.info("Service ready!")

    yield

    # 停止任务管理器
    task_manager.stop()
    stop_hotword_watcher()
    logger.info("Shutting down...")


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="基于 FunASR + CapsWriter 的中文语音转写服务",
    lifespan=lifespan,
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(api_router)


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """健康检查"""
    return HealthResponse(status="healthy", version=settings.version)


@app.get("/", tags=["system"])
async def root():
    """服务信息"""
    return {
        "name": settings.app_name,
        "version": settings.version,
        "docs": "/docs",
    }
