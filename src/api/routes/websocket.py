"""WebSocket 实时转写路由"""
import json
import uuid
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.ws_manager import ws_manager, ConnectionState
from src.core.engine import transcription_engine
from src.models.model_manager import model_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/realtime")
async def websocket_realtime(websocket: WebSocket):
    """
    实时流式转写 WebSocket 接口

    协议:
    1. 客户端发送配置 (JSON): {"is_speaking": true, "mode": "2pass"}
    2. 客户端发送音频 (binary): PCM 16bit, 16kHz, mono
    3. 服务端返回结果 (JSON): {"mode": "2pass-online", "text": "...", "is_final": false}
    """
    await websocket.accept()

    connection_id = str(uuid.uuid4())
    ws_manager.connect(websocket, connection_id)
    state = ws_manager.get_state(connection_id)

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
                            await websocket.send_json({
                                "mode": "2pass-online" if state.mode == "2pass" else "online",
                                "text": result["text"],
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
    """在线流式识别"""
    try:
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
    """离线识别"""
    try:
        offline_model = model_manager.loader.asr_model
        result = offline_model.generate(
            input=audio_in,
            hotword=state.hotwords,
        )
        if result:
            return {"text": result[0].get("text", "")}
    except Exception as e:
        logger.error(f"Offline ASR error: {e}")
    return {}
