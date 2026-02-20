from __future__ import annotations

import argparse
import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


RUN_DIR_REL = Path(".run") / "local_stack"


@dataclass(frozen=True)
class ServiceSpec:
    name: str
    host: str
    port: int
    python: str
    module: str
    extra_env: Mapping[str, str]


def ensure_run_dir(root: Path) -> Path:
    run_dir = root / RUN_DIR_REL
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def is_port_open(host: str, port: int, timeout_s: float = 0.25) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_s)
        try:
            return sock.connect_ex((host, int(port))) == 0
        except OSError:
            return False


def _env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    if v is None or str(v).strip() == "":
        return int(default)
    return int(str(v).strip())


def _env_str(key: str, default: str) -> str:
    v = os.getenv(key)
    if v is None or str(v).strip() == "":
        return default
    return str(v)


def build_meeting_specs(root: Path, host: str) -> list[ServiceSpec]:
    pytorch_port = _env_int("PORT_PYTORCH", 8101)
    diarizer_port = _env_int("DIARIZER_PORT", 8300)

    tingwu_python = _env_str("TINGWU_PYTHON", sys.executable)
    diarizer_python = _env_str("DIARIZER_PYTHON", sys.executable)

    diarizer = ServiceSpec(
        name="diarizer",
        host=host,
        port=diarizer_port,
        python=diarizer_python,
        module="src.diarizer_service.app",
        extra_env={
            "DIARIZER_PORT": str(diarizer_port),
            "DIARIZER_WARMUP_ON_STARTUP": os.getenv("DIARIZER_WARMUP_ON_STARTUP", "true"),
            **({"HF_TOKEN": os.getenv("HF_TOKEN", "")} if os.getenv("HF_TOKEN") else {}),
        },
    )

    pytorch = ServiceSpec(
        name="pytorch",
        host=host,
        port=pytorch_port,
        python=tingwu_python,
        module="src.main",
        extra_env={
            "ASR_BACKEND": "pytorch",
            "PORT": str(pytorch_port),
            "SPEAKER_EXTERNAL_DIARIZER_ENABLE": "true",
            "SPEAKER_EXTERNAL_DIARIZER_BASE_URL": f"http://{host}:{diarizer_port}",
        },
    )

    return [diarizer, pytorch]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local multi-process launcher for TingWu")
    sub = parser.add_subparsers(dest="cmd", required=True)

    start = sub.add_parser("start", help="Start local services")
    start.add_argument("--mode", default="meeting", choices=["meeting"])

    sub.add_parser("stop", help="Stop local services (not yet implemented)")
    sub.add_parser("status", help="Show status (not yet implemented)")
    sub.add_parser("logs", help="Show logs (not yet implemented)")

    args = parser.parse_args(argv)

    # v1: only validates config. Lifecycle management is added in later tasks.
    root = Path.cwd()
    _ = ensure_run_dir(root)
    if args.cmd == "start":
        _ = build_meeting_specs(root, host="127.0.0.1")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

