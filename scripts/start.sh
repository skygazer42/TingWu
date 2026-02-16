#!/bin/bash
set -e

MODE=${1:-gpu}
PORT=${PORT:-8000}

echo "======================================"
echo "TingWu Speech Service Launcher"
echo "======================================"

case $MODE in
    gpu)
        echo "Starting with GPU support..."
        docker compose up -d
        ;;
    cpu)
        echo "Starting with CPU only..."
        docker compose -f docker-compose.cpu.yml up -d
        ;;
    build)
        echo "Building Docker image..."
        docker compose build
        ;;
    stop)
        echo "Stopping service..."
        docker compose down
        docker compose -f docker-compose.cpu.yml down 2>/dev/null || true
        docker compose -f docker-compose.models.yml down 2>/dev/null || true
        ;;
    logs)
        docker compose logs -f
        ;;
    models)
        PROFILE=${2:-}
        if [ -z "$PROFILE" ]; then
            echo "Usage: $0 models <pytorch|onnx|sensevoice|gguf|qwen3|vibevoice|router>"
            echo "Tip: vibevoice/router need VIBEVOICE_REPO_PATH=/path/to/VibeVoice"
            exit 1
        fi
        echo "Starting model profile: ${PROFILE}"
        docker compose -f docker-compose.models.yml --profile "${PROFILE}" up -d

        # Best-effort: print the expected endpoint for the selected profile.
        case "$PROFILE" in
            pytorch) MODEL_PORT=${PORT_PYTORCH:-8101} ;;
            onnx) MODEL_PORT=${PORT_ONNX:-8102} ;;
            sensevoice) MODEL_PORT=${PORT_SENSEVOICE:-8103} ;;
            gguf) MODEL_PORT=${PORT_GGUF:-8104} ;;
            qwen3) MODEL_PORT=${PORT_TINGWU_QWEN3:-8201} ;;
            vibevoice) MODEL_PORT=${PORT_TINGWU_VIBEVOICE:-8202} ;;
            router) MODEL_PORT=${PORT_TINGWU_ROUTER:-8200} ;;
            *) MODEL_PORT="" ;;
        esac
        if [ -n "${MODEL_PORT}" ]; then
            echo ""
            echo "Service URL: http://localhost:${MODEL_PORT}"
            echo "API Docs: http://localhost:${MODEL_PORT}/docs"
        fi
        ;;
    *)
        echo "Usage: $0 {gpu|cpu|models|build|stop|logs}"
        exit 1
        ;;
esac

if [ "$MODE" = "gpu" ] || [ "$MODE" = "cpu" ]; then
    echo ""
    echo "Service URL: http://localhost:${PORT}"
    echo "API Docs: http://localhost:${PORT}/docs"
    echo "WebSocket: ws://localhost:${PORT}/ws/realtime"
fi
