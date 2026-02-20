# 多模型 / 多后端部署指南（profiles / 端口 / 说话人策略）

本项目支持“**一个模型/后端 = 一个 TingWu 服务实例**”的部署方式：每个实例都提供 **同一套 TingWu HTTP API**（例如 `/api/v1/transcribe`），只是端口不同。

这样你可以：

- 同一份前端 UI，通过切换 `Base URL` 做 A/B 对比
- 会议 / 视频 / 电话会议按需选择：快 / 准 / 低资源 / 支持 speaker

> 从 0 开始的部署请先看 `docs/DEPLOYMENT.md`。  
> 常见问题排障请看 `docs/TROUBLESHOOTING.md`。  
> API 细节请看 `docs/API.md`（或访问任意后端的 `/docs`）。

---

## 0) TL;DR（常用命令）

```bash
# 启动常用“全家桶”（不含 vibevoice/router）
docker compose -f docker-compose.models.yml --profile all up -d

# 只启动 PyTorch（建议会议）
docker compose -f docker-compose.models.yml --profile pytorch up -d

# 停止
docker compose -f docker-compose.models.yml down
```

---

## 1) 两种部署形态：单容器 vs 多模型端口

### 1.1 单容器（入门最简单）

使用根目录的 `docker-compose.yml`（或 `docker-compose.cpu.yml`）启动一个 TingWu：

- 优点：最省心，UI/API 都在 `:8000`
- 缺点：想切换模型要改 env 或换 compose 文件

### 1.2 多模型端口（推荐：对比/压测/多场景共存）

使用 `docker-compose.models.yml`：

- 每个后端一个容器（每个容器一个端口）
- API 路径一致，适合前端做“后端选择器”

---

## 2) Profile 用法（按需启动）

`docker-compose.models.yml` 通过 Compose profiles 做“按需启动”。

### 2.1 启动单个 profile

```bash
docker compose -f docker-compose.models.yml --profile pytorch up -d
docker compose -f docker-compose.models.yml --profile whisper up -d
docker compose -f docker-compose.models.yml --profile qwen3 up -d
```

### 2.2 一键启动常用“全家桶”

```bash
# 包含：diarizer + pytorch + onnx + sensevoice + gguf + whisper + qwen3-asr + tingwu-qwen3
# 不包含：vibevoice/router（需要额外挂载 VIBEVOICE_REPO_PATH）
docker compose -f docker-compose.models.yml --profile all up -d
```

### 2.3 启动 VibeVoice / Router（需要挂载 VibeVoice repo）

`vibevoice-asr` 容器需要挂载本地 VibeVoice 仓库目录（包含 `vllm_plugin` 等）。准备好后：

```bash
VIBEVOICE_REPO_PATH=/path/to/VibeVoice \
  docker compose -f docker-compose.models.yml --profile vibevoice up -d

VIBEVOICE_REPO_PATH=/path/to/VibeVoice \
  docker compose -f docker-compose.models.yml --profile router up -d
```

停止：

```bash
docker compose -f docker-compose.models.yml down
```

---

## 3) 端口对照表（默认值）

默认端口可在 `.env` 中覆盖；下表为 `.env.example` 默认值：

| 服务 | Profile | 默认端口 | 说明 |
|---|---|---:|---|
| `tingwu-pytorch` | `pytorch` / `all` | `8101` | FunASR PyTorch（建议会议） |
| `tingwu-onnx` | `onnx` / `all` | `8102` | ONNX（默认 CPU） |
| `tingwu-sensevoice` | `sensevoice` / `all` | `8103` | SenseVoice（GPU） |
| `tingwu-gguf` | `gguf` / `all` | `8104` | GGUF（默认 CPU；需要你准备模型文件） |
| `tingwu-whisper` | `whisper` / `all` | `8105` | Whisper（GPU，默认 `large`） |
| `tingwu-diarizer` | `diarizer` / `all` | `8300` | external diarizer（pyannote） |
| `qwen3-asr` | `qwen3` / `router` / `all` | `9001` | 远程 ASR server（OpenAI transcription 风格） |
| `vibevoice-asr` | `vibevoice` / `router` | `9002` | 远程 ASR server（vLLM OpenAI-compatible） |
| `tingwu-qwen3` | `qwen3` / `all` | `8201` | Qwen3 wrapper（提供 TingWu API） |
| `tingwu-vibevoice` | `vibevoice` | `8202` | VibeVoice wrapper（提供 TingWu API） |
| `tingwu-router` | `router` | `8200` | 路由后端（提供 TingWu API；不是必须） |

每个 TingWu 实例的常用入口：

- Web UI：`http://localhost:<port>`
- API Docs：`http://localhost:<port>/docs`
- Health：`http://localhost:<port>/health`

---

## 4) 前端怎么切后端（Base URL）

每个 TingWu 容器都内置相同的前端 UI（只要镜像包含 `frontend/dist`）。

建议：

1) 打开任意一个端口的 UI（例如 `http://localhost:8101`）  
2) 在「转写选项 → 后端」里切换 `Base URL`（例如 `http://localhost:8105` / `8201`）  
3) 前端会把请求发到你选择的端口

---

## 5) 权重缓存策略（不会把镜像撑大）

原则：

- **权重下载到 volume 或宿主机挂载目录**，避免写进镜像层
- 重启容器不会重复下载

`docker-compose.models.yml` 主要使用这些缓存：

- `model-cache`：ModelScope 缓存（FunASR）
- `huggingface-cache`：HuggingFace 缓存（pyannote / Qwen3 / VibeVoice 等）
- `onnx-cache`：ONNX 缓存（ONNX 后端）
- 宿主机 `./data:/app/data`：
  - Whisper 默认下载到 `./data/models/whisper`

查看 volumes：

```bash
docker volume ls | rg "model-cache|huggingface-cache|onnx-cache"
```

清理缓存（谨慎：会导致下次启动重新下载）：

```bash
docker compose -f docker-compose.models.yml down
docker volume rm tingwu_model-cache tingwu_huggingface-cache  # 名称以 docker volume ls 为准
```

---

## 6) 说话人（speaker）从哪里来？（会议场景关键）

会议转写想要输出：

```
说话人1：...
说话人2：...
```

TingWu 支持三种策略（推荐顺序）：

### 6.1 External diarizer（推荐：任意后端都能输出 `speaker_turns`）

启用 `tingwu-diarizer`（pyannote）后，TingWu 会：

1) diarizer 先把音频切成 speaker turns  
2) TingWu 再逐段调用你选择的 ASR 后端  
3) 输出 `speaker_turns`，并可格式化为 `说话人1/2/3...`

启动示例（Qwen3 wrapper + external diarizer）：

```bash
HF_TOKEN=... \
SPEAKER_EXTERNAL_DIARIZER_ENABLE=true \
  docker compose -f docker-compose.models.yml --profile diarizer --profile qwen3 up -d
```

> 注意：部分 pyannote 模型需要在 HuggingFace 申请访问权限，并提供 `HF_TOKEN`。

### 6.2 后端原生 speaker（最简单）

当后端本身支持 speaker 时，请求里带：

- `with_speaker=true`
- （可选）`asr_options.speaker.label_style="numeric"`

### 6.3 Fallback diarization（兼容：Qwen3/Whisper 这类不带 speaker 的后端）

Qwen3-ASR、Whisper 等常见转写服务 **不原生输出 speaker**。

你可以让 TingWu 额外调用一个“辅助 TingWu 服务”（通常是 `tingwu-pytorch`）生成分段信息：

- 同时启动 `--profile pytorch` 和 `--profile qwen3`
- 并设置：
  - `SPEAKER_FALLBACK_DIARIZATION_ENABLE=true`
  - `SPEAKER_FALLBACK_DIARIZATION_BASE_URL=http://tingwu-pytorch:8000`

失败会自动降级为普通转写（避免破坏“端口=模型”的预期）。

---

## 7) 远程 ASR：Qwen3 / VibeVoice 的两层结构

### 7.1 Qwen3-ASR（`qwen3-asr` + `tingwu-qwen3`）

`--profile qwen3` 会启动：

1) `qwen3-asr`：OpenAI transcription server（吃 GPU）  
2) `tingwu-qwen3`：TingWu wrapper（提供统一 TingWu API；一般 CPU 即可）

常用配置（`.env`）：

- `QWEN3_MODEL_ID`（默认 `Qwen/Qwen3-ASR-1.7B`）
- `QWEN3_GPU_MEMORY_UTILIZATION`

### 7.2 VibeVoice-ASR（`vibevoice-asr` + `tingwu-vibevoice`）

同理：

1) `vibevoice-asr`：vLLM OpenAI-compatible server（吃 GPU）  
2) `tingwu-vibevoice`：TingWu wrapper（提供统一 TingWu API）

> VibeVoice 需要 `VIBEVOICE_REPO_PATH` 挂载本地 repo（详见 `docker-compose.models.yml` 注释）。

---

## 8) 常用组合推荐（会议）

1) **稳/省心**：`pytorch` + `diarizer`（external diarizer）  
2) **多模型对比**：`--profile all` + 前端切 `Base URL`  
3) **远程 ASR + speaker**：`qwen3` + `diarizer`（或 `qwen3` + fallback diarization）

