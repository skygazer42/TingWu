# TingWu 听悟 - 语音转写服务

<p align="center">
  <img src="frontend/public/favicon.svg" width="80" height="80" alt="TingWu Logo">
</p>

<p align="center">
  基于 FunASR + CapsWriter-Offline 的高精度中文语音转写服务，包含完整的 Web 前端界面。
</p>

## 特性

### 核心功能
- **高精度转写**: 基于阿里 FunASR Paraformer-large 模型 (60000小时训练数据)
- **热词纠错**: 音素级模糊匹配 + FAISS 向量检索 + 字形相似度重排序
- **说话人识别**: 自动识别多说话人并标注（甲/乙/丙/丁）
- **实时转写**: WebSocket 流式接口，支持 2pass/online/offline 模式
- **LLM 润色**: 支持 Ollama/OpenAI/vLLM 后端进行文本优化

### 20 项性能优化
- ONNX INT8 量化 + 模型预热
- 批量音频 API + GPU 内存优化
- WebSocket 压缩 + 心跳重连 + 自适应分块
- H-PRM 声学预检索 + FAISS 向量索引
- 音形义联合纠错 (ShapeCorrector)
- vLLM 高吞吐后端 + LRU 缓存 + 批量处理
- DeepFilterNet v3 降噪 + 自适应预处理
- 长音频智能分块 + 并行文本处理
- 置信度驱动选择性纠错
- Prometheus 可观测性指标

## 快速开始

### 方式一：Docker 部署 (推荐)

```bash
# 构建镜像 (包含前端)
docker-compose build

# 启动服务
docker-compose up -d
```

访问 **http://localhost:8000** 即可使用。

> **首次启动说明**：首次运行时会自动从 ModelScope 下载 ASR 模型（约 1-2GB），请耐心等待。模型会缓存到 Docker Volume 中，后续启动无需重新下载。

#### 容器选择

| 场景 | 命令 | 模型大小 |
|------|------|----------|
| **GPU 版本** (推荐) | `docker-compose up -d` | ~1.5GB |
| **CPU 版本** | `docker-compose -f docker-compose.cpu.yml up -d` | ~1.5GB |
| **SenseVoice 模型** | `docker-compose -f docker-compose.sensevoice.yml up -d` | ~500MB |
| **ONNX 后端** | `docker-compose -f docker-compose.onnx.yml up -d` | ~400MB |

#### 多模型按需启动（每个模型=一个容器，一个统一 API）

如果你希望**每个模型/后端单独一个容器**，并且都提供同一套 TingWu API（`/api/v1/transcribe`），可以使用：`docker-compose.models.yml`。

特点：
- 端口不同，但 API 路径一致，方便做 A/B 对比和基准测试
- 默认不启动任何模型，按需用 profile 启动

示例（按需启动）：

```bash
# 1) PyTorch Paraformer (GPU) -> http://localhost:8101
docker compose -f docker-compose.models.yml --profile pytorch up -d

# 2) ONNX (CPU) -> http://localhost:8102
docker compose -f docker-compose.models.yml --profile onnx up -d

# 3) SenseVoice (GPU) -> http://localhost:8103
docker compose -f docker-compose.models.yml --profile sensevoice up -d

# 4) Whisper (GPU) -> http://localhost:8105
docker compose -f docker-compose.models.yml --profile whisper up -d

# 5) Qwen3-ASR (远程模型容器 + TingWu 包装) -> http://localhost:8201
docker compose -f docker-compose.models.yml --profile qwen3 up -d

# 6) VibeVoice-ASR (远程模型容器 + TingWu 包装) -> http://localhost:8202
# 需要提供本地 VibeVoice 仓库路径（包含 vllm_plugin）
VIBEVOICE_REPO_PATH=/path/to/VibeVoice \
  docker compose -f docker-compose.models.yml --profile vibevoice up -d

# 7) Router (Qwen3 + VibeVoice 自动路由) -> http://localhost:8200
VIBEVOICE_REPO_PATH=/path/to/VibeVoice \
  docker compose -f docker-compose.models.yml --profile router up -d
```

停止：

```bash
docker compose -f docker-compose.models.yml down
```

提示：
- 打开任意一个 TingWu 容器的前端页面后，可在「转写选项 → 后端」里切换 `Base URL`（例如 `http://localhost:8101` / `8102` / `8201`），前端会把请求发到你选择的端口。
- Whisper 容器默认使用 `WHISPER_MODEL=large`（显存占用更高）；可通过环境变量覆盖（例如 `WHISPER_MODEL=small`）。模型权重会下载到 `WHISPER_DOWNLOAD_ROOT`（默认映射到宿主机 `./data/models/whisper`），不会把镜像撑大。
- 如果你希望**任意后端（包括 Qwen3-ASR / Whisper）都输出 `speaker_turns`（说话人1/2/3...）**，推荐启用 external diarizer（`tingwu-diarizer`，pyannote）：
  - 需要在 HuggingFace 上准备 `HF_TOKEN`（部分 pyannote 模型需要申请访问权限）
  - 启动示例（以 Qwen3 为例）：

    ```bash
    HF_TOKEN=... \
    SPEAKER_EXTERNAL_DIARIZER_ENABLE=true \
      docker compose -f docker-compose.models.yml --profile diarizer --profile qwen3 up -d
    ```

  - 模型权重会缓存到 `huggingface-cache` volume（不会把镜像撑大）
  - diarizer 失败会自动降级：后端原生支持 speaker → 回退原生；否则忽略 speaker（不硬报错）
- Qwen3-ASR **原生不支持说话人识别**。如果你不想引入 external diarizer，也可以启用“fallback diarization”（辅助 TingWu 服务生成分段）：
  - 同时启动 `--profile pytorch` 和 `--profile qwen3`
  - 并给 `tingwu-qwen3` 设置：
    - `SPEAKER_FALLBACK_DIARIZATION_ENABLE=true`
    - `SPEAKER_FALLBACK_DIARIZATION_BASE_URL=http://tingwu-pytorch:8000`
  - 如果辅助服务不可用，会自动回退为普通转写（不会报错）。

#### 请求级调参（`asr_options`，用于准确率 A/B）

在 `POST /api/v1/transcribe`（以及 batch）里可以额外传一个表单字段 `asr_options`（JSON 字符串），用于**单次请求**调参：
- `preprocess`：音频预处理（normalize/denoise/trim/DC offset）
- `chunking`：长音频分块参数（chunk 时长/重叠/合并 overlap_chars/线程数）
- `backend`：模型后端参数（不同后端支持的参数不同）
- `postprocess`：文本后处理（ITN/标点/spacing 等）
- `speaker`：会议/回忆转录输出格式（说话人标签风格、turn 合并阈值等；需 `with_speaker=true`）

返回中会额外包含一个可选字段 `text_accu`：基于时间窗口对齐的“精确拼接文本”（长音频 chunk overlap 去重更严格），更适合回忆/会议转录的最终稿输出。
当 `with_speaker=true` 时，返回会额外包含 `speaker_turns`（按说话人合并后的段落/turn 列表），并且 `transcript` 默认会优先使用 `speaker_turns` 来格式化（更适合会议阅读/回放）。

示例：强制更小的 chunk + 更激进的合并窗口（更偏准确率，慢一点）

```bash
curl -X POST "http://localhost:8102/api/v1/transcribe" \
  -F "file=@/path/to/audio.wav" \
  -F 'asr_options={"chunking":{"max_chunk_duration_s":30,"overlap_chars":40,"max_workers":1}}'
```

示例：边界重解码/对齐（更偏准确率，明显更慢；适合长音频 chunk 边界容易漏字/断词的场景）

> 该模式会对每个 chunk 边界额外转写一个小窗口（`boundary_reconcile_window_s` 表示边界左右各多少秒），
> 再把这段“桥接文本”插入合并流程来减少边界缺字/重复。

```bash
curl -X POST "http://localhost:8101/api/v1/transcribe" \
  -F "file=@/path/to/audio.wav" \
  -F 'asr_options={"chunking":{"boundary_reconcile_enable":true,"boundary_reconcile_window_s":1.0}}'
```

示例：单次请求关闭音量归一化 + DC offset（用于排查“预处理是否伤准确率”）

```bash
curl -X POST "http://localhost:8101/api/v1/transcribe" \
  -F "file=@/path/to/audio.wav" \
  -F 'asr_options={"preprocess":{"normalize_enable":false,"remove_dc_offset":false}}'
```

示例：长音频（大量静音/停顿）更稳的音量归一化（更偏准确率，避免把静音/底噪整体放大）

> `normalize_robust_rms_percentile=95` 大致表示：用最响的约 5% 帧估算“有效音量”，更接近“语音段 RMS”，而不是整段音频的平均 RMS。

```bash
curl -X POST "http://localhost:8101/api/v1/transcribe" \
  -F "file=@/path/to/audio.wav" \
  -F 'asr_options={"preprocess":{"normalize_enable":true,"normalize_robust_rms_enable":true,"normalize_robust_rms_percentile":95}}'
```

示例：自适应降噪 gating（高 SNR 自动跳过降噪，低 SNR 才启用；更偏准确率）

```bash
curl -X POST "http://localhost:8101/api/v1/transcribe" \
  -F "file=@/path/to/audio.wav" \
  -F 'asr_options={"preprocess":{"adaptive_enable":true,"snr_threshold":20,"denoise_enable":true}}'
```

示例：会议/回忆转录（说话人 + turn 合并 + 数字标签）

```bash
curl -X POST "http://localhost:8200/api/v1/transcribe" \
  -F "file=@/path/to/meeting.wav" \
  -F "with_speaker=true" \
  -F 'asr_options={"speaker":{"label_style":"numeric","turn_merge_enable":true,"turn_merge_gap_ms":800}}'
```

说明：
- `speaker_turns`：合并后的 turn 列表（更适合人类阅读/导出）
- `sentences`：句级时间戳（更适合时间轴/字幕等）
- 如果后端不支持说话人识别，行为由 `SPEAKER_UNSUPPORTED_BEHAVIOR` 控制：`error | fallback | ignore`
  - 多端口/多模型场景推荐 `ignore`（`docker-compose.models.yml` 默认已设置为 ignore）
  - 单容器部署如果希望“说话人自动回退到 PyTorch”，可用 `fallback`
- 一键脚本：`scripts/transcribe_meeting.sh /path/to/meeting.wav http://localhost:8200`

#### 模型缓存

模型存储在 Docker Volume 中，可查看下载状态：

```bash
# 查看模型下载进度
docker-compose logs -f | grep -i "download\|loading"

# 查看缓存大小
docker volume ls
docker system df -v
```

#### 常用操作

```bash
# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down

# 重新构建
docker-compose build --no-cache
```

### 方式二：本地开发

需要分别启动后端和前端：

```bash
# 终端 1 - 启动后端
pip install -r requirements.txt
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# 终端 2 - 启动前端
cd frontend
npm install
npm run dev
```

- 后端 API: http://localhost:8000
- 前端界面: http://localhost:5173

### 方式三：脚本部署

```bash
# GPU 版本
./scripts/start.sh gpu

# CPU 版本
./scripts/start.sh cpu

# 查看日志
./scripts/start.sh logs

# 停止服务
./scripts/start.sh stop
```

### 方式四：HuggingFace 模型自部署（vLLM 远程后端）

如果你希望把模型推理放到独立服务（本地/内网），TingWu 支持通过 vLLM 的 OpenAI 兼容接口接入 **Qwen3-ASR** 和 **VibeVoice-ASR**。

1) 启动 Qwen3-ASR（自动从 HuggingFace 下载权重）

```bash
pip install -U "qwen-asr[vllm]"
qwen-asr-serve Qwen/Qwen3-ASR-1.7B --host 0.0.0.0 --port 9001 --gpu-memory-utilization 0.8
```

2) 启动 VibeVoice-ASR（官方 vLLM Docker，自动从 HuggingFace 下载）

```bash
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice

docker run -d --gpus all --name vibevoice-vllm \
  --ipc=host \
  -p 9002:8000 \
  -e VIBEVOICE_FFMPEG_MAX_CONCURRENCY=64 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v $(pwd):/app \
  -w /app \
  --entrypoint bash \
  vllm/vllm-openai:latest \
  -c "python3 /app/vllm_plugin/scripts/start_server.py"
```

3) 配置 TingWu（`.env`）

```bash
ASR_BACKEND=router
QWEN3_ASR_BASE_URL=http://localhost:9001
QWEN3_ASR_MODEL=Qwen/Qwen3-ASR-1.7B
VIBEVOICE_ASR_BASE_URL=http://localhost:9002
VIBEVOICE_ASR_MODEL=vibevoice
VIBEVOICE_ASR_USE_CHAT_COMPLETIONS_FALLBACK=true
```

说明：
- `ASR_BACKEND=qwen3`：直接走 Qwen3-ASR
- `ASR_BACKEND=vibevoice`：直接走 VibeVoice-ASR（支持 `with_speaker=true`）
- `ASR_BACKEND=router`：短音频默认 Qwen3，长音频/`with_speaker=true` 默认 VibeVoice

## 前端界面

基于 React + TypeScript + Tailwind CSS + shadcn/ui 构建的现代化前端。

### 功能页面

| 页面 | 功能 |
|------|------|
| **转写** | 文件上传/URL转写、转写选项、结果展示、时间轴、历史记录、导出 |
| **实时转写** | 麦克风录制、波形显示、流式识别、模式选择、连接状态 |
| **热词管理** | 热词编辑、分组管理、搜索、导入/导出、更新、追加 |
| **配置管理** | 纠错/热词/LLM/后处理/音频配置、配置预设 |
| **系统监控** | 健康状态、请求统计、性能指标、Prometheus 集成 |

### 快捷键

按 `Ctrl + /` 查看所有快捷键。支持导航、转写控制、录制控制等。

### 前端技术栈

- **框架**: React 19 + TypeScript + Vite 7
- **样式**: Tailwind CSS v4 + shadcn/ui + CVA
- **状态**: Zustand + React Query v5
- **路由**: React Router v7
- **图表**: Recharts
- **音频**: Web Audio API + MediaRecorder
- **通知**: Sonner
- **无障碍**: WCAG 2.2 AA

### 前端构建

```bash
cd frontend

# 开发模式
npm run dev

# 生产构建
npm run build

# 预览构建
npm run preview
```

## API 使用

### 文件转写

```bash
# 基本转写
curl -X POST http://localhost:8000/api/v1/transcribe \
  -F "file=@audio.wav"

# 完整选项
curl -X POST http://localhost:8000/api/v1/transcribe \
  -F "file=@audio.wav" \
  -F "with_speaker=true" \
  -F "apply_hotword=true" \
  -F "apply_llm=false" \
  -F "llm_role=default"

# 批量转写
curl -X POST http://localhost:8000/api/v1/transcribe/batch \
  -F "files=@audio1.wav" \
  -F "files=@audio2.wav"
```

### 实时转写

参见 `examples/client_websocket.py`

WebSocket 协议:
1. 连接 `ws://localhost:8000/ws/realtime`
2. 发送配置: `{"is_speaking": true, "mode": "2pass"}`
3. 发送 PCM 音频 (16bit, 16kHz, mono)
4. 接收结果: `{"mode": "2pass-online", "text": "...", "is_final": false}`

### 热词管理

```bash
# 查看热词
curl http://localhost:8000/api/v1/hotwords

# 更新热词
curl -X POST http://localhost:8000/api/v1/hotwords \
  -H "Content-Type: application/json" \
  -d '{"hotwords": ["Claude", "FunASR", "麦当劳"]}'

# 追加热词
curl -X POST http://localhost:8000/api/v1/hotwords/append \
  -H "Content-Type: application/json" \
  -d '{"hotwords": ["新热词1", "新热词2"]}'
```

说明：
- `POST /api/v1/hotwords` 管理的是 **强制纠错热词**（`data/hotwords/hotwords.txt`），会参与音素纠错/规则替换。
- 回忆/会议转录更推荐维护 **上下文热词**（`data/hotwords/hotwords-context.txt`）：只用于“注入提示模型”，不会强制替换，风险更低（支持热加载）。

### 配置管理

```bash
# 获取配置
curl http://localhost:8000/config

# 更新配置
curl -X POST http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -d '{"updates": {"llm_enable": true, "hotwords_threshold": 0.8}}'

# 重载引擎
curl -X POST http://localhost:8000/config/reload
```

### API 文档

启动服务后访问: http://localhost:8000/docs

## 配置

通过环境变量或 `.env` 文件配置:

### 基础配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| DEVICE | 设备类型 (cuda/cpu) | cuda |
| NGPU | GPU 数量 | 1 |
| NCPU | CPU 线程数 | 4 |
| PORT | 服务端口 | 8000 |

### ASR 后端配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| ASR_BACKEND | ASR 后端 (pytorch/onnx/sensevoice/gguf/qwen3/vibevoice/router/whisper) | pytorch |
| ASR_MODEL | 离线 ASR 模型 | paraformer-zh |
| VAD_MODEL | VAD 模型 | fsmn-vad |
| PUNC_MODEL | 标点模型 | ct-punc-c |
| SPK_MODEL | 说话人模型 | cam++ |

> 模型会自动从 [ModelScope](https://modelscope.cn) 下载并缓存。
>
> 远程后端（qwen3/vibevoice/router）相关环境变量见 `.env.example`，或参考上面的「方式四：HuggingFace 模型自部署」。

### 热词配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| HOTWORDS_FILE | 强制纠错热词文件 | hotwords.txt |
| HOTWORDS_CONTEXT_FILE | 上下文提示热词文件（仅注入） | hotwords-context.txt |
| HOTWORDS_THRESHOLD | 热词匹配阈值 | 0.85 |
| HOTWORD_INJECTION_ENABLE | 启用热词前向注入 | true |
| HOTWORD_INJECTION_MAX | 最大注入热词数 | 50 |
| HOTWORD_USE_FAISS | 启用 FAISS 索引 | false |
| HOTWORD_FAISS_INDEX_TYPE | FAISS 索引类型 | IVFFlat |

### LLM 配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| LLM_ENABLE | 启用 LLM 润色 | false |
| LLM_BACKEND | LLM 后端 (auto/ollama/openai/vllm) | auto |
| LLM_CACHE_ENABLE | 启用响应缓存 | true |
| LLM_CACHE_SIZE | 缓存大小 | 1000 |

### WebSocket 配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| WS_COMPRESSION | 启用压缩 | true |
| WS_HEARTBEAT_INTERVAL | 心跳间隔 (秒) | 30 |
| WS_HEARTBEAT_TIMEOUT | 心跳超时 (秒) | 60 |

## 项目结构

```
tingwu/
├── src/                    # 后端源码
│   ├── api/                # FastAPI 接口层
│   │   ├── routes/         # 路由定义
│   │   ├── schemas.py      # 数据模型
│   │   └── ws_manager.py   # WebSocket 管理
│   ├── core/               # 核心业务逻辑
│   │   ├── hotword/        # 热词纠错 (PhonemeCorrector, ShapeCorrector)
│   │   ├── speaker/        # 说话人识别
│   │   ├── llm/            # LLM 客户端
│   │   ├── audio/          # 音频处理 (AudioChunker, Preprocessor)
│   │   ├── text_processor/ # 文本后处理
│   │   └── engine.py       # 转写引擎
│   ├── models/             # 模型加载 (支持多后端)
│   ├── utils/              # 工具类 (ServiceMetrics)
│   ├── config.py           # 配置管理
│   └── main.py             # 应用入口
├── frontend/               # 前端源码
│   ├── src/
│   │   ├── components/     # React 组件
│   │   ├── pages/          # 页面组件
│   │   ├── hooks/          # 自定义 Hooks
│   │   ├── stores/         # Zustand 状态
│   │   ├── lib/api/        # API 客户端
│   │   └── providers/      # Context Providers
│   ├── package.json
│   └── vite.config.ts
├── data/hotwords/          # 热词配置
├── examples/               # 使用示例
├── tests/                  # 测试 (165 用例, 90.9% 通过率)
├── Dockerfile
├── docker-compose.yml      # GPU 部署
└── docker-compose.cpu.yml  # CPU 部署
```

## 技术栈

### 后端
- **ASR**: FunASR (Paraformer-large + FSMN-VAD + CT-Transformer)
- **说话人**: CAMPPlus + ClusterBackend
- **热词**: CapsWriter-Offline 音素 RAG + FAISS + 字形相似度
- **LLM**: Ollama / OpenAI / vLLM
- **服务**: FastAPI + WebSocket
- **部署**: Docker + Docker Compose

### 前端
- **框架**: React 19 + TypeScript
- **构建**: Vite 7
- **样式**: Tailwind CSS v4 + shadcn/ui
- **状态**: Zustand + React Query v5
- **图表**: Recharts

## 测试

```bash
# 运行全部测试
python -m pytest tests/ -v

# 运行特定模块测试
python -m pytest tests/test_hotword.py -v

# 生成覆盖率报告
python -m pytest tests/ --cov=src --cov-report=html
```

## License

MIT
