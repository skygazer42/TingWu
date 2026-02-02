#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${QWEN3_MODEL_ID:-Qwen/Qwen3-ASR-0.6B}"
PORT="${PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEMORY_UTILIZATION:-0.85}"

echo "[qwen3-asr] MODEL_ID=${MODEL_ID}"
echo "[qwen3-asr] PORT=${PORT}"
echo "[qwen3-asr] GPU_MEMORY_UTILIZATION=${GPU_MEM_UTIL}"

exec qwen-asr-serve "${MODEL_ID}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --trust-remote-code

