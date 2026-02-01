"""WebSocket 实时转写路由"""
import json
import uuid
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.config import settings
from src.api.ws_manager import ws_manager, ConnectionState
from src.core.engine import transcription_engine
from src.models.model_manager import model_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


def _check_streaming_support() -> bool:
    """检查当前后端是否支持流式转写"""
    backend = model_manager.backend
    return backend.supports_streaming


@router.websocket("/ws/realtime")
async def websocket_realtime(websocket: WebSocket):
    """
    实时流式转写 WebSocket 接口

    协议:
    1. 客户端发送配置 (JSON): {"is_speaking": true, "mode": "2pass"}
    2. 客户端发送音频 (binary): PCM 16bit, 16kHz, mono
    3. 服务端返回结果 (JSON): {"mode": "2pass-online", "text": "...", "is_final": false}

    注意: 流式转写需要 PyTorch 后端支持。
    如果配置了其他后端，将自动回退到 PyTorch。
    """
    await websocket.accept()

    connection_id = str(uuid.uuid4())
    ws_manager.connect(websocket, connection_id)
    state = ws_manager.get_state(connection_id)

    # 检查流式支持，发送警告信息
    if not _check_streaming_support():
        backend_info = model_manager.backend.get_info()
        logger.warning(
            f"Backend {backend_info['name']} does not support streaming, "
            "falling back to PyTorch backend for WebSocket"
        )
        await websocket.send_json({
            "warning": f"当前后端 {backend_info['name']} 不支持流式，已自动切换到 PyTorch 后端",
            "backend": backend_info['name'],
        })

    frames = []
    frames_online = []

    try:
        while True:
            message = await websocket.receive()

            # 处理文本消息 (配置)
            if "text" in message:
                try:
                    config = json.loads(message["text"])
                    _handle_config(state, config)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON config: {message['text']}")
                continue

            # 处理二进制消息 (音频)
            if "bytes" in message:
                audio_chunk = message["bytes"]
                frames.append(audio_chunk)
                frames_online.append(audio_chunk)

                # 在线识别 (每 chunk_interval 帧)
                if len(frames_online) % state.chunk_interval == 0:
                    if state.mode in ("2pass", "online"):
                        audio_in = b"".join(frames_online)
                        result = await _asr_online(audio_in, state)
                        if result and result.get("text"):
                            text = result["text"]
                            # 流式去重
                            if settings.stream_dedup_enable:
                                text = state.text_merger.merge(text)
                            if text:  # 只发送非空增量
                                await websocket.send_json({
                                    "mode": "2pass-online" if state.mode == "2pass" else "online",
                                    "text": text,
                                    "is_final": False,
                                })
                        frames_online = []

                # 说话结束时执行离线识别
                if not state.is_speaking:
                    if state.mode in ("2pass", "offline") and frames:
                        audio_in = b"".join(frames)
                        result = await _asr_offline(audio_in, state)
                        if result and result.get("text"):
                            text = result["text"]
                            # 热词纠错
                            if transcription_engine._hotwords_loaded:
                                correction = transcription_engine.corrector.correct(text)
                                text = correction.text

                            # 流式去重 (最终文本)
                            if settings.stream_dedup_enable:
                                text = state.text_merger.merge_final(text)

                            await websocket.send_json({
                                "mode": "2pass-offline" if state.mode == "2pass" else "offline",
                                "text": text,
                                "is_final": True,
                            })

                    # 重置
                    frames = []
                    frames_online = []
                    state.reset()

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        ws_manager.disconnect(connection_id)


def _handle_config(state: ConnectionState, config: dict):
    """处理配置消息"""
    if "is_speaking" in config:
        state.is_speaking = config["is_speaking"]
    if "mode" in config:
        state.mode = config["mode"]
    if "hotwords" in config:
        state.hotwords = config["hotwords"]
    if "chunk_interval" in config:
        state.chunk_interval = config["chunk_interval"]


async def _asr_online(audio_in: bytes, state: ConnectionState) -> dict:
    """在线流式识别

    注意: 始终使用 PyTorch 后端的流式功能。
    """
    try:
        # 使用 PyTorch 后端的流式模型
        online_model = model_manager.loader.asr_model_online
        result = online_model.generate(
            input=audio_in,
            cache=state.asr_cache,
            is_final=not state.is_speaking,
            hotword=state.hotwords,
        )
        if result:
            state.asr_cache = result[0].get("cache", {})
            return {"text": result[0].get("text", "")}
    except Exception as e:
        logger.error(f"Online ASR error: {e}")
    return {}


async def _asr_offline(audio_in: bytes, state: ConnectionState) -> dict:
    """离线识别

    使用配置的后端进行离线识别。
    """
    try:
        backend = model_manager.backend

        # 如果后端支持，使用后端转写
        if backend.supports_streaming or backend.get_info()["type"] == "pytorch":
            # PyTorch 后端使用 loader
            offline_model = model_manager.loader.asr_model
            result = offline_model.generate(
                input=audio_in,
                hotword=state.hotwords,
            )
            if result:
                return {"text": result[0].get("text", "")}
        else:
            # 其他后端使用 backend.transcribe
            result = backend.transcribe(
                audio_in,
                hotwords=state.hotwords,
            )
            if result:
                return {"text": result.get("text", "")}
    except Exception as e:
        logger.error(f"Offline ASR error: {e}")
    return {}
