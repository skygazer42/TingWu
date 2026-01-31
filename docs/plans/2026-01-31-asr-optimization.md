# ASR Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve TingWu’s ASR accuracy (A) while also reducing WebSocket realtime latency (B) and improving throughput/cost (C) on RTX 4090 + 16-core CPU.

**Architecture:** Unify the correction pipeline across HTTP/Async/WebSocket, remove async event-loop blocking by offloading sync inference + ffmpeg to threads, add observability (timings + debug), then add model-side tuning knobs (TF32/threads/preload/warmup) guarded by config flags.

**Tech Stack:** Python 3.11+, FastAPI, WebSocket, FunASR (torch), ffmpeg-python, pytest

---

## Preconditions / Conventions

- Worktree: `~/.config/superpowers/worktrees/tingwu/asr-opt-2026-01-31`
- Venv:
  - Create: `/opt/homebrew/bin/python3.11 -m venv .venv`
  - Install: `source .venv/bin/activate && pip install -r requirements.txt pytest`
- Test command: `source .venv/bin/activate && pytest -q`
- Commit frequently; keep each task small and reversible.

---

## Track A (Accuracy): pipeline consistency + debugability

### Task 1: Add request-level debug + timing container schema (non-breaking)

**Files:**
- Modify: `src/api/schemas.py`
- Test: `tests/test_api_http.py`

**Step 1: Write the failing test**

Add a new assertion that responses can include a `meta` field when requested.

In `tests/test_api_http.py`, extend `test_transcribe_endpoint`:

```python
data = response.json()
assert "meta" not in data  # default: not included
```

and add a new test:

```python
def test_transcribe_endpoint_debug_meta(client):
    with patch('src.core.engine.transcription_engine') as mock_engine, \
         patch('src.api.routes.transcribe.process_audio_file') as mock_process:
        mock_engine.transcribe.return_value = {
            "text": "你好世界",
            "sentences": [{"text": "你好世界", "start": 0, "end": 1000}],
            "raw_text": "你好世界",
            "meta": {"timings_ms": {"asr": 1.0}},
        }
        async def fake_process(file):
            yield b"\\x00" * 32000
        mock_process.side_effect = fake_process
        files = {"file": ("test.wav", io.BytesIO(b"fake"), "audio/wav")}
        response = client.post("/api/v1/transcribe", files=files, data={"debug": "true"})
        assert response.status_code == 200
        assert "meta" in response.json()
```

Expected: FAIL because request schema/route doesn’t accept `debug`, and response_model rejects extra fields.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_api_http.py::test_transcribe_endpoint_debug_meta -q`
Expected: FAIL (422 or response model validation).

**Step 3: Write minimal implementation**

In `src/api/schemas.py`, add optional fields:

```python
from typing import Any, Dict, Optional

class TranscribeMeta(BaseModel):
    timings_ms: Optional[Dict[str, float]] = None
    hotword_matches: Optional[list] = None

class TranscribeResponse(BaseModel):
    ...
    meta: Optional[Dict[str, Any]] = None
```

Keep it permissive to avoid breaking clients.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_api_http.py::test_transcribe_endpoint_debug_meta -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/api/schemas.py tests/test_api_http.py
git commit -m "feat(api): allow optional meta field for debug/timings"
```

---

### Task 2: Add debug flag to `/api/v1/transcribe` and pass-through meta

**Files:**
- Modify: `src/api/routes/transcribe.py`
- Test: `tests/test_api_http.py`

**Step 1: Write the failing test**

The test from Task 1 should still fail (422) until the endpoint accepts `debug`.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_api_http.py::test_transcribe_endpoint_debug_meta -q`
Expected: FAIL (422 validation error).

**Step 3: Write minimal implementation**

In `src/api/routes/transcribe.py` add:

```python
debug: bool = Form(default=False, description="是否返回调试信息/耗时分解"),
```

and when building `TranscribeResponse`, pass `meta=result.get("meta") if debug else None`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_api_http.py::test_transcribe_endpoint_debug_meta -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/api/routes/transcribe.py
git commit -m "feat(api): add debug flag to include meta in /transcribe"
```

---

### Task 3: Instrument `TranscriptionEngine` with timings + optional debug payload

**Files:**
- Modify: `src/core/engine.py`
- Test: `tests/test_engine.py`

**Step 1: Write the failing test**

In `tests/test_engine.py`, add:

```python
def test_transcribe_returns_meta_when_debug(mock_model_manager):
    from src.core.engine import TranscriptionEngine
    engine = TranscriptionEngine()
    engine.update_hotwords(["麦当劳"])
    result = engine.transcribe(b"fake_audio", debug=True)
    assert "meta" in result
    assert "timings_ms" in result["meta"]
```

Expected: FAIL because `debug` param doesn’t exist and no meta.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_engine.py::test_transcribe_returns_meta_when_debug -q`
Expected: FAIL

**Step 3: Write minimal implementation**

In `src/core/engine.py`:
- Add `debug: bool = False` parameter to `transcribe()` and `transcribe_async()`
- Measure timings with `time.perf_counter()` for stages:
  - `asr`
  - `hotword_rules_post`
  - `speaker`
  - `llm` (if enabled)
- When `debug` is true, include:

```python
result["meta"] = {"timings_ms": timings_ms}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_engine.py::test_transcribe_returns_meta_when_debug -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/core/engine.py tests/test_engine.py
git commit -m "feat(engine): add debug timings meta to transcribe results"
```

---

## Track B (Realtime latency): remove event-loop blocking + streaming correctness

### Task 4: Add a small helper to run sync functions in thread (reusable)

**Files:**
- Create: `src/utils/async_utils.py`
- Test: `tests/test_async_utils.py`

**Step 1: Write the failing test**

Create `tests/test_async_utils.py`:

```python
import asyncio

def test_run_sync_returns_value():
    from src.utils.async_utils import run_sync
    async def main():
        def f(x): return x + 1
        return await run_sync(f, 1)
    assert asyncio.run(main()) == 2
```

Expected: FAIL (module missing).

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_async_utils.py -q`
Expected: FAIL

**Step 3: Write minimal implementation**

Create `src/utils/async_utils.py`:

```python
from __future__ import annotations

import asyncio
from typing import Any, Callable, TypeVar

T = TypeVar("T")

async def run_sync(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    return await asyncio.to_thread(fn, *args, **kwargs)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_async_utils.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/utils/async_utils.py tests/test_async_utils.py
git commit -m "feat(utils): add run_sync helper for async offloading"
```

---

### Task 5: Offload blocking ASR inference in `transcribe_async`

**Files:**
- Modify: `src/core/engine.py`
- Test: `tests/test_engine_async_offload.py`

**Step 1: Write the failing test**

Create `tests/test_engine_async_offload.py`:

```python
import asyncio
from unittest.mock import patch, MagicMock

def test_transcribe_async_uses_run_sync():
    with patch('src.core.engine.model_manager') as mock_mm, \
         patch('src.core.engine.run_sync') as mock_run_sync:
        mock_loader = MagicMock()
        mock_mm.loader = mock_loader
        mock_run_sync.return_value = {"text": "x", "sentence_info": []}

        from src.core.engine import TranscriptionEngine
        engine = TranscriptionEngine()

        async def main():
            return await engine.transcribe_async(b"fake")
        asyncio.run(main())
        assert mock_run_sync.called
```

Expected: FAIL because `run_sync` isn’t used by `transcribe_async`.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_engine_async_offload.py -q`
Expected: FAIL

**Step 3: Write minimal implementation**

In `src/core/engine.py`:
- `from src.utils.async_utils import run_sync`
- Replace direct call to `model_manager.loader.transcribe(...)` with:

```python
raw_result = await run_sync(model_manager.loader.transcribe, audio_input, hotwords=hotwords, with_speaker=with_speaker, **kwargs)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_engine_async_offload.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/core/engine.py tests/test_engine_async_offload.py
git commit -m "perf(engine): offload sync ASR inference in transcribe_async"
```

---

### Task 6: Offload ffmpeg conversion in `process_audio_file`

**Files:**
- Modify: `src/api/dependencies.py`
- Test: `tests/test_dependencies_offload.py`

**Step 1: Write the failing test**

Create `tests/test_dependencies_offload.py`:

```python
import asyncio
from unittest.mock import patch, MagicMock

def test_process_audio_file_uses_run_sync(tmp_path):
    class DummyFile:
        filename = "a.wav"
        async def read(self): return b"fake"
    with patch("src.api.dependencies.run_sync") as mock_run_sync, \
         patch("src.api.dependencies.aiofiles.open") as mock_open:
        mock_open.return_value.__aenter__.return_value.write = MagicMock()
        mock_run_sync.return_value = (b"\\x00"*10, b"")

        from src.api.dependencies import process_audio_file
        async def main():
            agen = process_audio_file(DummyFile())
            async for _ in agen:
                break
        asyncio.run(main())
        assert mock_run_sync.called
```

Expected: FAIL (no run_sync usage).

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_dependencies_offload.py -q`
Expected: FAIL

**Step 3: Write minimal implementation**

In `src/api/dependencies.py`:
- `from src.utils.async_utils import run_sync`
- Wrap ffmpeg `.run(...)` in a sync helper and call via `await run_sync(...)`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_dependencies_offload.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/api/dependencies.py tests/test_dependencies_offload.py
git commit -m "perf(api): offload ffmpeg conversion to thread"
```

---

### Task 7: Extend `ConnectionState` to carry runtime options (apply_hotword/llm/with_speaker)

**Files:**
- Modify: `src/api/ws_manager.py`
- Test: `tests/test_api_websocket.py`

**Step 1: Write the failing test**

In `tests/test_api_websocket.py::test_connection_state_defaults`, add:

```python
assert state.apply_hotword is True
assert state.apply_llm is False
assert state.with_speaker is False
```

Expected: FAIL (fields missing).

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_api_websocket.py::test_connection_state_defaults -q`
Expected: FAIL

**Step 3: Write minimal implementation**

In `src/api/ws_manager.py`, add fields to `ConnectionState`:

```python
apply_hotword: bool = True
apply_llm: bool = False
with_speaker: bool = False
llm_role: str = "default"
debug: bool = False
```

and reset relevant fields in `reset()` (keep config fields unchanged unless explicitly desired).

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_api_websocket.py::test_connection_state_defaults -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/api/ws_manager.py tests/test_api_websocket.py
git commit -m "feat(ws): extend ConnectionState with pipeline options"
```

---

### Task 8: Update WebSocket config handler to accept the new options

**Files:**
- Modify: `src/api/routes/websocket.py`
- Test: `tests/test_ws_config.py`

**Step 1: Write the failing test**

Create `tests/test_ws_config.py`:

```python
def test_handle_config_sets_options():
    from src.api.ws_manager import ConnectionState
    from src.api.routes.websocket import _handle_config
    state = ConnectionState()
    _handle_config(state, {"apply_hotword": False, "apply_llm": True, "with_speaker": True, "llm_role": "translator"})
    assert state.apply_hotword is False
    assert state.apply_llm is True
    assert state.with_speaker is True
    assert state.llm_role == "translator"
```

Expected: FAIL (handler ignores these).

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ws_config.py -q`
Expected: FAIL

**Step 3: Write minimal implementation**

In `src/api/routes/websocket.py::_handle_config`, add:

```python
if "apply_hotword" in config: state.apply_hotword = bool(config["apply_hotword"])
if "apply_llm" in config: state.apply_llm = bool(config["apply_llm"])
if "with_speaker" in config: state.with_speaker = bool(config["with_speaker"])
if "llm_role" in config: state.llm_role = str(config["llm_role"])
if "debug" in config: state.debug = bool(config["debug"])
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ws_config.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/api/routes/websocket.py tests/test_ws_config.py
git commit -m "feat(ws): support apply_hotword/apply_llm/with_speaker in config"
```

---

### Task 9: Route WebSocket online inference through engine + offload (no blocking)

**Files:**
- Modify: `src/core/engine.py`
- Modify: `src/api/routes/websocket.py`
- Test: `tests/test_engine_streaming_async.py`

**Step 1: Write the failing test**

Create `tests/test_engine_streaming_async.py`:

```python
import asyncio
from unittest.mock import patch, MagicMock

def test_transcribe_streaming_async_offloads():
    from src.core.engine import TranscriptionEngine
    engine = TranscriptionEngine()
    with patch('src.core.engine.model_manager') as mock_mm, \
         patch('src.core.engine.run_sync') as mock_run_sync:
        online = MagicMock()
        online.generate.return_value = [{"text": "hi", "cache": {"x": 1}}]
        mock_mm.loader.asr_model_online = online
        mock_run_sync.return_value = [{"text": "hi", "cache": {"x": 1}}]
        cache = {}
        async def main():
            return await engine.transcribe_streaming_async(b"\\x00"*10, cache, is_final=False)
        out = asyncio.run(main())
        assert out["text"] != ""  # got text
        assert "asr_cache" in cache
        assert mock_run_sync.called
```

Expected: FAIL (no `transcribe_streaming_async`).

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_engine_streaming_async.py -q`
Expected: FAIL

**Step 3: Write minimal implementation**

In `src/core/engine.py`, add:

```python
async def transcribe_streaming_async(self, audio_chunk: bytes, cache: Dict[str, Any], is_final: bool = False, apply_hotword: bool = False, **kwargs) -> Dict[str, Any]:
    online_model = model_manager.loader.asr_model_online
    result = await run_sync(online_model.generate, input=audio_chunk, cache=cache.get("asr_cache", {}), is_final=is_final, **kwargs)
    ...
```

Only apply corrections when `apply_hotword` is true (default False for online to reduce jitter).

Update `src/api/routes/websocket.py` to call this async method for online messages.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_engine_streaming_async.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/core/engine.py src/api/routes/websocket.py tests/test_engine_streaming_async.py
git commit -m "perf(ws): offload online streaming inference via engine"
```

---

### Task 10: Route WebSocket offline (final) inference through engine + full correction pipeline

**Files:**
- Modify: `src/api/routes/websocket.py`
- Test: `tests/test_ws_offline_pipeline.py`

**Step 1: Write the failing test**

Create `tests/test_ws_offline_pipeline.py`:

```python
import asyncio
from unittest.mock import AsyncMock, patch

def test_ws_offline_uses_engine_transcribe_async():
    with patch('src.api.routes.websocket.transcription_engine') as engine:
        engine.transcribe_async = AsyncMock(return_value={"text": "x", "sentences": [], "raw_text": "x"})
        from src.api.routes.websocket import _asr_offline_engine
        async def main():
            await _asr_offline_engine(b"\\x00"*10, with_speaker=False, apply_hotword=True, apply_llm=False, llm_role="default", hotwords=None, debug=False)
        asyncio.run(main())
        assert engine.transcribe_async.called
```

Expected: FAIL (helper missing).

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ws_offline_pipeline.py -q`
Expected: FAIL

**Step 3: Write minimal implementation**

In `src/api/routes/websocket.py`, introduce a small helper:

```python
async def _asr_offline_engine(audio_in: bytes, *, with_speaker: bool, apply_hotword: bool, apply_llm: bool, llm_role: str, hotwords: str | None, debug: bool) -> dict:
    return await transcription_engine.transcribe_async(
        audio_in,
        with_speaker=with_speaker,
        apply_hotword=apply_hotword,
        apply_llm=apply_llm,
        llm_role=llm_role,
        hotwords=hotwords,
        debug=debug,
    )
```

Then in the WebSocket loop, replace direct `_asr_offline(...)` + manual hotword correction with this unified call.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ws_offline_pipeline.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/api/routes/websocket.py tests/test_ws_offline_pipeline.py
git commit -m "feat(ws): unify offline final inference via engine pipeline"
```

---

## Track C (Throughput/Cost + Model-side knobs): tuning + concurrency controls

### Task 11: Add torch/perf settings to `Settings`

**Files:**
- Modify: `src/config.py`
- Test: `tests/test_config_perf_flags.py`

**Step 1: Write the failing test**

Create `tests/test_config_perf_flags.py`:

```python
def test_settings_has_perf_flags():
    from src.config import Settings
    s = Settings()
    assert hasattr(s, "torch_tf32_enable")
    assert hasattr(s, "torch_matmul_precision")
    assert hasattr(s, "asr_max_concurrency")
```

Expected: FAIL (fields missing).

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config_perf_flags.py -q`
Expected: FAIL

**Step 3: Write minimal implementation**

In `src/config.py`, add:

```python
torch_tf32_enable: bool = False
torch_matmul_precision: str = "high"  # high/medium
torch_num_threads: int = 0  # 0 => leave default; else set torch.set_num_threads
torch_num_interop_threads: int = 0
asr_max_concurrency: int = 1  # GPU safety default; raise for CPU-only
asr_preload_models: bool = False
asr_warmup: bool = False
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_config_perf_flags.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/config.py tests/test_config_perf_flags.py
git commit -m "feat(config): add perf/concurrency flags for ASR tuning"
```

---

### Task 12: Centralize torch configuration (TF32/threads/matmul precision)

**Files:**
- Create: `src/models/torch_config.py`
- Modify: `src/models/asr_loader.py`
- Test: `tests/test_torch_config.py`

**Step 1: Write the failing test**

Create `tests/test_torch_config.py`:

```python
from unittest.mock import patch

def test_configure_torch_sets_threads_when_requested():
    from src.models.torch_config import configure_torch
    with patch("src.models.torch_config.torch") as torch:
        configure_torch(torch_num_threads=4, torch_num_interop_threads=2, torch_tf32_enable=True, torch_matmul_precision="high")
        assert torch.set_num_threads.called
        assert torch.set_num_interop_threads.called
```

Expected: FAIL (module missing).

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_torch_config.py -q`
Expected: FAIL

**Step 3: Write minimal implementation**

Create `src/models/torch_config.py`:

```python
def configure_torch(*, torch_num_threads: int, torch_num_interop_threads: int, torch_tf32_enable: bool, torch_matmul_precision: str) -> None:
    import torch
    if torch_num_threads and torch_num_threads > 0:
        torch.set_num_threads(torch_num_threads)
    if torch_num_interop_threads and torch_num_interop_threads > 0:
        torch.set_num_interop_threads(torch_num_interop_threads)
    try:
        torch.set_float32_matmul_precision(torch_matmul_precision)
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = bool(torch_tf32_enable)
            torch.backends.cudnn.allow_tf32 = bool(torch_tf32_enable)
        except Exception:
            pass
```

In `src/models/asr_loader.py`, call `configure_torch(...)` in `__init__` using `settings` values (plumb from `ModelManager`).

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_torch_config.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/models/torch_config.py src/models/asr_loader.py tests/test_torch_config.py
git commit -m "perf(torch): add configurable torch tuning (tf32/threads/precision)"
```

---

### Task 13: Add global inference concurrency limiter (async semaphore)

**Files:**
- Create: `src/models/inference_limiter.py`
- Modify: `src/core/engine.py`
- Test: `tests/test_inference_limiter.py`

**Step 1: Write the failing test**

Create `tests/test_inference_limiter.py`:

```python
import asyncio

def test_inference_limiter_limits_concurrency():
    from src.models.inference_limiter import InferenceLimiter
    limiter = InferenceLimiter(max_concurrency=1)
    active = 0
    peak = 0
    async def job():
        nonlocal active, peak
        async with limiter:
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
    async def main():
        await asyncio.gather(job(), job(), job())
    asyncio.run(main())
    assert peak == 1
```

Expected: FAIL (module missing).

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_inference_limiter.py -q`
Expected: FAIL

**Step 3: Write minimal implementation**

Create `src/models/inference_limiter.py`:

```python
import asyncio

class InferenceLimiter:
    def __init__(self, max_concurrency: int):
        self._sem = asyncio.Semaphore(max(1, int(max_concurrency)))
    async def __aenter__(self):
        await self._sem.acquire()
        return self
    async def __aexit__(self, exc_type, exc, tb):
        self._sem.release()
        return False
```

In `src/core/engine.py`, wrap calls to `run_sync(...transcribe...)` and `run_sync(...generate...)` with `async with limiter:`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_inference_limiter.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/models/inference_limiter.py src/core/engine.py tests/test_inference_limiter.py
git commit -m "perf(engine): add async inference limiter for concurrency control"
```

---

### Task 14: Add optional model preload/warmup on startup (4090/CPU first-hit latency)

**Files:**
- Modify: `src/main.py`
- Modify: `src/models/model_manager.py`
- Test: `tests/test_startup_preload.py`

**Step 1: Write the failing test**

Create `tests/test_startup_preload.py`:

```python
from unittest.mock import patch

def test_startup_calls_preload_when_enabled():
    with patch("src.models.startup.model_manager") as mm:
        from src.models.startup import maybe_preload_and_warmup
        maybe_preload_and_warmup(preload=True, warmup=False, with_speaker=True)
        assert mm.preload_models.called
```

Expected: FAIL (module missing / helper missing).

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_startup_preload.py -q`
Expected: FAIL

**Step 3: Write minimal implementation**

- Create `src/models/startup.py` with:

```python
from src.models.model_manager import model_manager

def maybe_preload_and_warmup(*, preload: bool, warmup: bool, with_speaker: bool) -> None:
    if preload:
        model_manager.preload_models(with_speaker=with_speaker)
    if warmup:
        model_manager.warmup()
```

- In `src/main.py` lifespan startup, call it based on `settings.asr_preload_models/settings.asr_warmup`.

Add `model_manager.warmup()` that runs a tiny dummy inference once per loaded model.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_startup_preload.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/main.py src/models/model_manager.py src/models/startup.py tests/test_startup_preload.py
git commit -m "perf(startup): add optional model preload/warmup"
```

---

## Benchmarks / Tooling

### Task 15: Add CER evaluation script (accuracy regression guard)

**Files:**
- Create: `scripts/eval_cer.py`
- Create: `scripts/data/README.md`

**Step 1: Run script to verify it fails**

Run: `python scripts/eval_cer.py --help`
Expected: FAIL (file missing)

**Step 2: Write minimal implementation**

Create `scripts/eval_cer.py` that:
- reads a jsonl/csv with `ref` and `hyp`
- computes CER via edit distance (use `editdistance` if available, else fallback)
- prints summary (mean CER, top errors)

**Step 3: Run script to verify it passes**

Run: `python scripts/eval_cer.py --help`
Expected: prints usage.

**Step 4: Commit**

```bash
git add scripts/eval_cer.py scripts/data/README.md
git commit -m "chore(eval): add CER evaluation script"
```

---

### Task 16: Add HTTP benchmark script (latency/throughput)

**Files:**
- Create: `scripts/bench_http.py`

**Step 1: Run to verify it fails**

Run: `python scripts/bench_http.py --help`
Expected: FAIL (file missing)

**Step 2: Write minimal implementation**

Implement benchmark with httpx:
- configurable concurrency, requests, file path
- prints p50/p95, RPS

**Step 3: Run to verify it passes**

Run: `python scripts/bench_http.py --help`
Expected: prints usage

**Step 4: Commit**

```bash
git add scripts/bench_http.py
git commit -m "chore(bench): add HTTP benchmark script"
```

---

### Task 17: Add WebSocket benchmark script (online/offline latency)

**Files:**
- Create: `scripts/bench_ws.py`

**Step 1: Run to verify it fails**

Run: `python scripts/bench_ws.py --help`
Expected: FAIL (file missing)

**Step 2: Write minimal implementation**

Implement benchmark with `websockets`:
- sends config + PCM chunks from a wav file
- measures time to first online partial and offline final

**Step 3: Run to verify it passes**

Run: `python scripts/bench_ws.py --help`
Expected: prints usage

**Step 4: Commit**

```bash
git add scripts/bench_ws.py
git commit -m "chore(bench): add WebSocket benchmark script"
```

---

### Task 18: Document new knobs + how to benchmark

**Files:**
- Modify: `README.md`

**Step 1: Update docs**

Add a section:
- New env vars: TF32/threads/max_concurrency/preload/warmup
- How to run `scripts/eval_cer.py`, `scripts/bench_http.py`, `scripts/bench_ws.py`

**Step 2: Quick verification**

Run: `rg -n "bench_" README.md`
Expected: shows new docs.

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add benchmarking and perf tuning instructions"
```

---

## Additional Realtime + Async API Hardening

### Task 19: Include `meta.timings_ms` in WebSocket responses when `debug=true`

**Files:**
- Modify: `src/api/routes/websocket.py`
- Test: `tests/test_ws_debug_meta.py`

**Step 1: Write the failing test**

Create `tests/test_ws_debug_meta.py`:

```python
import asyncio
from unittest.mock import AsyncMock, patch

def test_ws_offline_debug_includes_meta():
    with patch('src.api.routes.websocket.transcription_engine') as engine:
        engine.transcribe_async = AsyncMock(return_value={
            "text": "x",
            "sentences": [],
            "raw_text": "x",
            "meta": {"timings_ms": {"asr": 1.0}},
        })
        from src.api.routes.websocket import _asr_offline_engine
        async def main():
            out = await _asr_offline_engine(
                b"\\x00"*10,
                with_speaker=False,
                apply_hotword=True,
                apply_llm=False,
                llm_role="default",
                hotwords=None,
                debug=True,
            )
            assert "meta" in out
        asyncio.run(main())
```

Expected: FAIL until `_asr_offline_engine` forwards `debug` and the WebSocket send_json includes meta when debug enabled.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ws_debug_meta.py -q`
Expected: FAIL

**Step 3: Write minimal implementation**

In `src/api/routes/websocket.py`:
- Ensure `_asr_offline_engine(..., debug=...)` passes `debug` to `transcription_engine.transcribe_async`.
- When sending WS JSON, include `"meta": result.get("meta")` only if `state.debug`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ws_debug_meta.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/api/routes/websocket.py tests/test_ws_debug_meta.py
git commit -m "feat(ws): include meta timings in responses when debug enabled"
```

---

### Task 20: Feed streaming model per chunk (avoid join/copy) and keep cache correctness

**Files:**
- Modify: `src/api/routes/websocket.py`
- Test: `tests/test_ws_streaming_chunking.py`

**Step 1: Write the failing test**

Create `tests/test_ws_streaming_chunking.py`:

```python
import asyncio
from unittest.mock import AsyncMock, patch

def test_ws_online_calls_engine_with_single_chunk():
    with patch('src.api.routes.websocket.transcription_engine') as engine:
        engine.transcribe_streaming_async = AsyncMock(return_value={"text": "hi", "is_final": False})
        from src.api.routes.websocket import _asr_online_engine
        async def main():
            await _asr_online_engine(b"chunk", cache={}, is_final=False, hotwords=None, apply_hotword=False)
        asyncio.run(main())
        args, kwargs = engine.transcribe_streaming_async.call_args
        assert args[0] == b"chunk"
```

Expected: FAIL (helper missing / WS still concatenates audio).

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ws_streaming_chunking.py -q`
Expected: FAIL

**Step 3: Write minimal implementation**

In `src/api/routes/websocket.py`, create helper:

```python
async def _asr_online_engine(audio_chunk: bytes, *, cache: dict, is_final: bool, hotwords: str | None, apply_hotword: bool) -> dict:
    return await transcription_engine.transcribe_streaming_async(
        audio_chunk,
        cache,
        is_final=is_final,
        hotword=hotwords,
        apply_hotword=apply_hotword,
    )
```

Then, in the WS loop, call this per received chunk (or per `chunk_interval`), but pass a single `audio_chunk` (not `b''.join(...)`).

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ws_streaming_chunking.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/api/routes/websocket.py tests/test_ws_streaming_chunking.py
git commit -m "perf(ws): feed streaming model per chunk (no join) via engine"
```

---

### Task 21: Avoid blocking in `/api/v1/asr` (whisper-compatible) by using async engine + offloading ffmpeg

**Files:**
- Modify: `src/api/routes/async_transcribe.py`
- Test: `tests/test_whisper_compatible_offload.py`

**Step 1: Write the failing test**

Create `tests/test_whisper_compatible_offload.py`:

```python
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

def test_asr_whisper_compatible_uses_transcribe_async():
    with patch('src.api.routes.async_transcribe.transcription_engine') as engine:
        engine.transcribe_async = AsyncMock(return_value={"text": "x", "sentences": [], "raw_text": "x"})
        from src.main import app
        client = TestClient(app)
        # Minimal request: we patch conversion path to skip ffmpeg work.
        with patch('src.api.routes.async_transcribe.convert_audio_to_pcm') as conv:
            conv.return_value = True
            response = client.post(
                "/api/v1/asr",
                files={"file": ("a.wav", b"fake", "audio/wav")},
                data={"file_type": "audio", "with_speaker": "false", "apply_hotword": "false"},
            )
        assert response.status_code in (200, 500)  # depends on patched internals
        assert engine.transcribe_async.called
```

Expected: FAIL (endpoint uses sync `transcribe`).

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_whisper_compatible_offload.py -q`
Expected: FAIL

**Step 3: Write minimal implementation**

In `src/api/routes/async_transcribe.py`:
- Replace `transcription_engine.transcribe(...)` with `await transcription_engine.transcribe_async(...)`.
- Offload `convert_audio_to_pcm(...)` / `extract_audio_from_video(...)` via `run_sync`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_whisper_compatible_offload.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/api/routes/async_transcribe.py tests/test_whisper_compatible_offload.py
git commit -m "perf(api): avoid event-loop blocking in whisper-compatible /asr"
```

---

### Task 22: Offload ffmpeg in `/api/v1/trans/video` and keep inference non-blocking

**Files:**
- Modify: `src/api/routes/async_transcribe.py`
- Test: `tests/test_video_transcribe_offload.py`

**Step 1: Write the failing test**

Create `tests/test_video_transcribe_offload.py`:

```python
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

def test_transcribe_video_uses_run_sync_for_extract():
    with patch('src.api.routes.async_transcribe.run_sync') as run_sync:
        run_sync.return_value = True
        with patch('src.api.routes.async_transcribe.transcription_engine') as engine:
            engine.transcribe_async = AsyncMock(return_value={"text": "x", "sentences": [], "raw_text": "x"})
            from src.main import app
            client = TestClient(app)
            response = client.post(
                "/api/v1/trans/video",
                files={"file": ("a.mp4", b"fake", "video/mp4")},
                data={"with_speaker": "false", "apply_hotword": "false", "apply_llm": "false"},
            )
            assert response.status_code in (200, 400, 500)
            assert run_sync.called
```

Expected: FAIL until video extraction uses `run_sync`.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_video_transcribe_offload.py -q`
Expected: FAIL

**Step 3: Write minimal implementation**

In `src/api/routes/async_transcribe.py`, wrap `extract_audio_from_video(...)` via `await run_sync(...)`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_video_transcribe_offload.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/api/routes/async_transcribe.py tests/test_video_transcribe_offload.py
git commit -m "perf(api): offload ffmpeg in /trans/video"
```

---

## Completion

After all tasks:
- Run full test suite: `pytest -q`
- Run basic smoke: start uvicorn and hit `/health`, `/api/v1/transcribe` with a short wav.
- Prepare summary: what improved, which knobs to tune for 4090 vs CPU.

