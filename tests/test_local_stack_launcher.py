import socket
from pathlib import Path


def test_is_port_open_false_for_unused_port():
    from scripts import local_stack

    # Pick a port that is extremely unlikely to be used. If it is used on a
    # machine, the second test will still validate the True case via a bound socket.
    assert local_stack.is_port_open("127.0.0.1", 65534) is False


def test_is_port_open_true_for_bound_socket():
    from scripts import local_stack

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)

    try:
        host, port = server.getsockname()
        assert local_stack.is_port_open(host, port) is True
    finally:
        server.close()


def test_ensure_run_dir_creates_local_stack_dir(tmp_path: Path):
    from scripts import local_stack

    run_dir = local_stack.ensure_run_dir(tmp_path)
    assert run_dir.exists()
    assert run_dir.is_dir()
    assert run_dir.name == "local_stack"


def test_build_meeting_specs_has_two_services(tmp_path: Path, monkeypatch):
    from scripts import local_stack

    monkeypatch.delenv("TINGWU_PYTHON", raising=False)
    monkeypatch.delenv("DIARIZER_PYTHON", raising=False)
    monkeypatch.setenv("PORT_PYTORCH", "18101")
    monkeypatch.setenv("DIARIZER_PORT", "18300")

    specs = local_stack.build_meeting_specs(tmp_path, host="127.0.0.1")
    assert [s.name for s in specs] == ["diarizer", "pytorch"]

