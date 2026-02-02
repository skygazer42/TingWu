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
| ASR_BACKEND | ASR 后端 (pytorch/onnx/sensevoice) | pytorch |
| ASR_MODEL | 离线 ASR 模型 | paraformer-zh |
| VAD_MODEL | VAD 模型 | fsmn-vad |
| PUNC_MODEL | 标点模型 | ct-punc-c |
| SPK_MODEL | 说话人模型 | cam++ |

> 模型会自动从 [ModelScope](https://modelscope.cn) 下载并缓存。

### 热词配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| HOTWORDS_THRESHOLD | 热词匹配阈值 | 0.85 |
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
