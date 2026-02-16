# TingWu ASR Accuracy Optimization — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve *final* file/video transcription accuracy (especially long audio) by stabilizing chunking + merge, then iterating on preprocessing and boundary handling.

**Architecture:** Uploads are decoded to 16kHz mono PCM16LE → (optional) preprocessing → ASR → if long: chunk + overlap → robust merge → correction pipeline → optional LLM polish → final output.

**Tech Stack:** Python, FastAPI, numpy, FFmpeg, FunASR backends (pytorch/onnx/sensevoice/gguf + remote), CapsWriter-Offline-inspired text merge/hotword techniques.

---

## Status snapshot (already implemented)

These items are already done in the current workspace:

- Robust merge-by-text utility (`src/core/text_processor/text_merge.py`) + tests (`tests/test_text_merge.py`)
- `AudioChunker.merge_results` now uses the robust merge (`src/core/audio/chunker.py`) + tests (`tests/test_audio_chunker_merge.py`)
- PCM/WAV decode helpers (`src/core/audio/pcm.py`) + tests (`tests/test_pcm_utils.py`)
- HTTP `/api/v1/transcribe` and batch now call `transcribe_auto_async` to auto-chunk long audio (`src/api/routes/transcribe.py`)
- URL async handler now uses long-audio chunking (`src/api/routes/async_transcribe.py`)
- Chunk splitting now prefers the latest silence near `target_end` (`src/core/audio/chunker.py`) + tests (`tests/test_audio_chunker_split.py`)
- Chunk sentence merge now dedupes overlap sentences (time + tail-text heuristic) (`src/core/audio/chunker.py`) + tests (`tests/test_audio_chunker_sentences.py`)
- Audio diagnostics now include `dc_offset` + `clipping_ratio` (`src/core/audio/preprocessor.py`) + tests (`tests/test_audio_diagnostics.py`)
- Preprocessing now removes DC offset by default (`src/core/audio/preprocessor.py`) + tests (`tests/test_audio_preprocess_dc_offset.py`)
- ONNX + GGUF backends now accept TingWu-standard raw PCM16LE bytes (fixes HTTP/chunking compatibility) (`src/models/backends/onnx.py`, `src/models/backends/gguf/backend.py`)

If you want a clean branch/worktree flow + commits per task, run this plan in a worktree.

---

## 20-task roadmap (bite-sized, test-first)

### Task 01: Lock down merge-by-text behavior

**Files:**
- Modify: `src/core/text_processor/text_merge.py`
- Test: `tests/test_text_merge.py`

**Steps:**
1. Add edge-case tests (mixed punctuation, whitespace-only chunks, empty chunks).
2. Confirm: `pytest -q tests/test_text_merge.py`
3. Only then tweak defaults (`overlap_chars`, `error_tolerance`, `max_skip_new`) if tests show improvement.

### Task 02: Improve sentence de-duplication around overlaps

**Files:**
- Modify: `src/core/audio/chunker.py`
- Test: `tests/test_audio_chunker_merge.py` (add more cases)

**Steps:**
1. (Done) Write failing test showing duplicated `sentences` around overlap.
2. (Done) Implement minimal heuristic: drop overlap sentences when (a) they fall fully inside the overlap time window and (b) their text is already present in the merged tail text.
3. Add more tests for partial overlap / punctuation mismatch.
4. Run: `pytest -q tests/test_audio_chunker_sentences.py`

### Task 03: Add audio diagnostics (debug + tuning)

**Files:**
- Modify: `src/core/audio/preprocessor.py`
- Modify: `src/api/dependencies.py`
- Optional: `src/utils/metrics.py` / `src/utils/service_metrics.py`
- Test: add `tests/test_pcm_utils.py` or a new `tests/test_audio_diagnostics.py`

**Diagnostics to compute:**
- duration_s, rms_db, peak_db (already exists partially)
- clipping ratio (samples near +/-1.0)
- DC offset (mean)
- estimated SNR (already exists)

**Steps:**
1. Write test for clipping + DC offset metrics.
2. Implement `get_audio_diagnostics(...)` helper.
3. Log diagnostics at INFO when `settings.debug=True` (or behind a config flag).

### Task 04: Make long-audio chunking threshold explicit for HTTP

**Files:**
- Modify: `src/config.py` (new setting like `http_long_audio_threshold_s`)
- Modify: `src/core/engine.py` (`transcribe_auto_async`)
- Test: `tests/test_api_http.py` (update expectation / patching)

**Steps:**
1. Add a setting with clear semantics (seconds).
2. Route based on that setting, not VAD segment time.

### Task 05: Add `accuracy_mode` request flag (accuracy > speed)

**Files:**
- Modify: `src/api/routes/transcribe.py` (add form field)
- Modify: `src/core/engine.py` (route options to preprocessing)
- Modify: `src/api/dependencies.py` (preprocessor config selection)
- Test: `tests/test_api_http.py`

**Steps:**
1. Test: request with `accuracy_mode=accurate` hits heavier pipeline.
2. Implement as a small enum: `fast|balanced|accurate`.

### Task 06: DC offset removal + optional high-pass filter

**Files:**
- Modify: `src/core/audio/preprocessor.py`
- Test: add new unit tests under `tests/`

**Steps:**
1. Write test: DC offset removed (mean close to 0).
2. Implement `remove_dc_offset(audio)`.
3. Add simple high-pass (single-pole IIR) as optional step (accurate mode only).

### Task 07: Clipping mitigation (soft limiting)

**Files:**
- Modify: `src/core/audio/preprocessor.py`
- Test: new unit tests

**Steps:**
1. Test: severely clipped input returns reduced clipping ratio.
2. Implement soft limiter (tanh or smooth knee) behind `accuracy_mode`.

### Task 08: Loudness normalization improvements

**Files:**
- Modify: `src/core/audio/preprocessor.py`

**Steps:**
1. Add a “robust RMS” (exclude top/bottom percentile) for very noisy audio.
2. Avoid over-boosting noise (cap gain lower in accurate mode).

### Task 09: Denoise gating improvements

**Files:**
- Modify: `src/core/audio/preprocessor.py`

**Steps:**
1. Write test: very high SNR audio does not get denoised.
2. Implement: only denoise if SNR < threshold and peak/rms imply audible noise floor.

### Task 10: Upgrade chunk split point selection

**Files:**
- Modify: `src/core/audio/chunker.py`
- Test: add deterministic split tests (synthetic waveform with silence spans)

**Steps:**
1. (Partially done) Prefer the latest silence point within the search range (closest to `target_end`).
2. Upgrade to a real scoring function to choose the “best” split:
   - prefer longer continuous silence
   - prefer split closer to target_end
2. Ensure minimum chunk length constraints still hold.

### Task 11: Boundary reconciliation (reduce missing/duplicate syllables)

**Files:**
- Modify: `src/core/engine.py` (chunk transcribe strategy)
- Optional: new helper `src/core/audio/boundary.py`

**Steps:**
1. Add a mode that re-decodes a small boundary window (e.g. last/first 1–2s) and uses merge-by-text to reconcile.
2. Only enable in `accuracy_mode=accurate`.

### Task 12: Make post-processing “merge-safe”

**Files:**
- Modify: `src/core/engine.py` (ensure post-process runs after merge for long audio)
- Modify: `src/core/text_processor/post_processor.py`

**Steps:**
1. Ensure ITN/punc/spacing is not applied per chunk for long audio.
2. Add regression test: chunk merge still works when ITN enabled globally.

### Task 13: Output per-chunk diagnostics (optional)

**Files:**
- Modify: `src/core/engine.py`
- Modify: `src/api/schemas.py` (optional new debug field)

**Steps:**
1. Add optional debug output with chunk count, per-chunk text length, merge decisions.
2. Hide behind `debug=true` or `include_debug=true`.

### Task 14: Separate “context hotwords” vs “forced hotwords”

**Files:**
- Modify: `src/core/engine.py`
- Modify: `src/core/hotword/watcher.py` + configs
- Add: `data/hotwords/hotwords-context.txt` (or similar)

**Steps:**
1. Keep forced replacement list small and strict.
2. Use context list for ASR injection only.

### Task 15: Default injection uses context list

**Files:**
- Modify: `src/core/engine.py` (`_get_injection_hotwords`)

**Steps:**
1. Prefer context hotwords for injection; fallback to forced list if context missing.

### Task 16: Spoken punctuation commands ruleset

**Files:**
- Modify: `src/core/hotword/rule_corrector.py` or `src/core/text_processor/post_processor.py`
- Test: add unit tests for rules

**Examples:**
- “逗号/句号/问号/感叹号/回车” at start/end → punctuation/newline

### Task 17: Improve English acronym merging

**Files:**
- Modify: `src/core/text_processor/spacing.py` or add a small formatter module
- Test: unit tests for “A I” → “AI”, “V S Code” → “VS Code”

### Task 18: Ensure consistent post-processing across outputs

**Files:**
- Modify: `src/core/engine.py`
- Test: integration-ish unit tests (mock backend)

**Steps:**
1. Ensure `text`, every `sentences[].text`, and `transcript` are consistent with flags.

### Task 19: Add an accuracy benchmark harness for your 5-minute sample

**Files:**
- Add: `scripts/eval_accuracy.py`
- Add: `data/benchmark/custom/` (store sample + reference)

**Steps:**
1. Define reference TXT format (normalize whitespace/punc for CER).
2. Compute CER/WER + boundary-duplication metrics.
3. Run before/after comparisons for each backend.

### Task 20: Precision mode (upper bound)

**Files:**
- Modify: `src/models/backends/router.py`
- Modify: `src/core/engine.py`

**Idea:**
- For long audio: run primary backend → detect low-confidence / drift / duplication → re-check with an alternate backend (SenseVoice / remote) → strict LLM corrector fallback.

---

## Suggested next execution slice (highest ROI)

If we focus purely on “accuracy of final text” for long audio, the best next slice is:

1) Task 02 (sentence overlap de-dup)  
2) Task 03 (audio diagnostics)  
3) Task 10 (better split point selection)  
4) Task 11 (boundary reconciliation in accurate mode)
