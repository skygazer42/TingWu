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
        ;;
    logs)
        docker compose logs -f
        ;;
    *)
        echo "Usage: $0 {gpu|cpu|build|stop|logs}"
        exit 1
        ;;
esac

echo ""
echo "Service URL: http://localhost:${PORT}"
echo "API Docs: http://localhost:${PORT}/docs"
echo "WebSocket: ws://localhost:${PORT}/ws/realtime"
