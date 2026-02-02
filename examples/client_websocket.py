"""WebSocket 实时转写客户端示例"""
import asyncio
import json
import wave
import sys

try:
    import websockets
except ImportError:
    print("请安装 websockets: pip install websockets")
    sys.exit(1)

WS_URL = "ws://localhost:8000/ws/realtime"
CHUNK_SIZE = 9600  # 600ms @ 16kHz


async def realtime_transcribe(audio_path: str):
    """实时转写音频文件"""
    with wave.open(audio_path, "rb") as wf:
        assert wf.getnchannels() == 1, "需要单声道音频"
        assert wf.getsampwidth() == 2, "需要 16bit 音频"
        assert wf.getframerate() == 16000, "需要 16kHz 采样率"
        audio_data = wf.readframes(wf.getnframes())

    print(f"音频长度: {len(audio_data) / 32000:.1f} 秒")
    print("开始实时转写...")
    print("-" * 40)

    async with websockets.connect(WS_URL, subprotocols=["binary"]) as ws:
        config = {
            "is_speaking": True,
            "mode": "2pass",
            "chunk_interval": 10,
        }
        await ws.send(json.dumps(config))

        for i in range(0, len(audio_data), CHUNK_SIZE):
            chunk = audio_data[i:i + CHUNK_SIZE]
            await ws.send(chunk)

            try:
                response = await asyncio.wait_for(ws.recv(), timeout=0.1)
                result = json.loads(response)
                text = result.get("text", "")
                is_final = result.get("is_final", False)

                if text:
                    prefix = "[最终]" if is_final else "[实时]"
                    print(f"{prefix} {text}")
            except asyncio.TimeoutError:
                pass

            await asyncio.sleep(0.3)

        await ws.send(json.dumps({"is_speaking": False}))

        try:
            response = await asyncio.wait_for(ws.recv(), timeout=5.0)
            result = json.loads(response)
            if result.get("text"):
                print(f"[最终] {result['text']}")
        except asyncio.TimeoutError:
            pass

    print("-" * 40)
    print("转写完成")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python client_websocket.py <WAV文件路径>")
        print("注意: 音频需要是 16kHz, 单声道, 16bit PCM WAV 格式")
        sys.exit(1)

    asyncio.run(realtime_transcribe(sys.argv[1]))
