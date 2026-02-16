"""Per-request ASR tuning options (`asr_options`) parsing + validation.

`asr_options` is provided as a JSON string in multipart form data.
We validate it early in the API layer to:
- fail fast with a clear 400 on bad inputs
- avoid silently ignoring typos / unknown keys
- keep the engine/backend kwargs surface predictable
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

__all__ = ["parse_asr_options"]


_TOP_LEVEL_KEYS = {"preprocess", "chunking", "backend", "postprocess", "debug"}

_PREPROCESS_KEYS = {
    "normalize_enable",
    "target_db",
    "trim_silence_enable",
    "silence_threshold_db",
    "min_silence_ms",
    "denoise_enable",
    "denoise_prop",
    "denoise_backend",
    "vocal_separate_enable",
    "vocal_separate_model",
    "adaptive_enable",
    "snr_threshold",
    "remove_dc_offset",
}

_CHUNKING_KEYS = {
    "max_chunk_duration_s",
    "min_chunk_duration_s",
    "overlap_duration_s",
    "silence_threshold_db",
    "min_silence_duration_s",
    "max_workers",
    "overlap_chars",
}

_POSTPROCESS_KEYS = {
    "filler_remove_enable",
    "filler_aggressive",
    "qj2bj_enable",
    "itn_enable",
    "itn_erhua_remove",
    "spacing_cjk_ascii_enable",
    "zh_convert_enable",
    "zh_convert_locale",
    "punc_convert_enable",
    "punc_add_space",
    "punc_restore_enable",
    "punc_restore_model",
    "punc_merge_enable",
    "trash_punc_enable",
    "trash_punc_chars",
}

_PREPROCESS_TYPES: Dict[str, str] = {
    "normalize_enable": "bool",
    "target_db": "number",
    "trim_silence_enable": "bool",
    "silence_threshold_db": "number",
    "min_silence_ms": "int",
    "denoise_enable": "bool",
    "denoise_prop": "number",
    "denoise_backend": "str",
    "vocal_separate_enable": "bool",
    "vocal_separate_model": "str",
    "adaptive_enable": "bool",
    "snr_threshold": "number",
    "remove_dc_offset": "bool",
}

_CHUNKING_TYPES: Dict[str, str] = {
    "max_chunk_duration_s": "number",
    "min_chunk_duration_s": "number",
    "overlap_duration_s": "number",
    "silence_threshold_db": "number",
    "min_silence_duration_s": "number",
    "max_workers": "int",
    "overlap_chars": "int",
}

_POSTPROCESS_TYPES: Dict[str, str] = {
    "filler_remove_enable": "bool",
    "filler_aggressive": "bool",
    "qj2bj_enable": "bool",
    "itn_enable": "bool",
    "itn_erhua_remove": "bool",
    "spacing_cjk_ascii_enable": "bool",
    "zh_convert_enable": "bool",
    "zh_convert_locale": "str",
    "punc_convert_enable": "bool",
    "punc_add_space": "bool",
    "punc_restore_enable": "bool",
    "punc_restore_model": "str",
    "punc_merge_enable": "bool",
    "trash_punc_enable": "bool",
    "trash_punc_chars": "str",
}


def parse_asr_options(asr_options_str: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parse and validate `asr_options`.

    Args:
        asr_options_str: JSON string or None.

    Returns:
        Parsed dict or None.

    Raises:
        ValueError: on invalid JSON or schema violations.
    """
    if asr_options_str is None:
        return None

    s = str(asr_options_str).strip()
    if not s:
        return None

    # Basic payload size guard (avoid accidental huge form fields).
    if len(s) > 200_000:
        raise ValueError("asr_options is too large (max 200k chars)")

    try:
        obj = json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"asr_options is not valid JSON: {e.msg} (pos={e.pos})") from e

    if not isinstance(obj, dict):
        raise ValueError("asr_options must be a JSON object")

    unknown_top = [k for k in obj.keys() if k not in _TOP_LEVEL_KEYS]
    if unknown_top:
        raise ValueError(f"Unknown asr_options top-level keys: {unknown_top}")

    _validate_section(obj, "preprocess", allowed_keys=_PREPROCESS_KEYS)
    _validate_section(obj, "chunking", allowed_keys=_CHUNKING_KEYS)
    _validate_section(obj, "postprocess", allowed_keys=_POSTPROCESS_KEYS)
    _validate_section_types(obj, "preprocess", type_map=_PREPROCESS_TYPES)
    _validate_section_types(obj, "chunking", type_map=_CHUNKING_TYPES)
    _validate_section_types(obj, "postprocess", type_map=_POSTPROCESS_TYPES)

    backend = obj.get("backend")
    if backend is not None:
        if not isinstance(backend, dict):
            raise ValueError("asr_options.backend must be an object")
        for k, v in backend.items():
            if not isinstance(k, str):
                raise ValueError("asr_options.backend keys must be strings")
            if not _is_json_primitive_or_list(v):
                raise ValueError(
                    f"asr_options.backend[{k!r}] must be a JSON primitive or list of primitives"
                )

    debug = obj.get("debug")
    if debug is not None:
        if not isinstance(debug, dict):
            raise ValueError("asr_options.debug must be an object")
        # Keep debug section flexible for now, but avoid deeply nested blobs.
        for k, v in debug.items():
            if not isinstance(k, str):
                raise ValueError("asr_options.debug keys must be strings")
            if not _is_json_primitive_or_list(v):
                raise ValueError(
                    f"asr_options.debug[{k!r}] must be a JSON primitive or list of primitives"
                )

    # Numeric sanity checks (best-effort; deeper validation happens in engine).
    _validate_ranges(obj)

    return obj


def _validate_section(obj: Dict[str, Any], key: str, *, allowed_keys: set[str]) -> None:
    section = obj.get(key)
    if section is None:
        return
    if not isinstance(section, dict):
        raise ValueError(f"asr_options.{key} must be an object")

    unknown = [k for k in section.keys() if k not in allowed_keys]
    if unknown:
        raise ValueError(f"Unknown asr_options.{key} keys: {unknown}")


def _validate_section_types(obj: Dict[str, Any], key: str, *, type_map: Dict[str, str]) -> None:
    section = obj.get(key)
    if section is None:
        return
    if not isinstance(section, dict):
        return

    for k, v in section.items():
        expected = type_map.get(k)
        if expected is None:
            continue
        if expected == "bool":
            if not isinstance(v, bool):
                raise ValueError(f"asr_options.{key}.{k} must be a boolean")
        elif expected == "str":
            if not isinstance(v, str):
                raise ValueError(f"asr_options.{key}.{k} must be a string")
        elif expected == "int":
            if not _is_int(v):
                raise ValueError(f"asr_options.{key}.{k} must be an integer")
        elif expected == "number":
            if not _is_number(v):
                raise ValueError(f"asr_options.{key}.{k} must be a number")


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_json_primitive_or_list(v: Any) -> bool:
    if v is None or isinstance(v, (str, int, float, bool)):
        return True
    if isinstance(v, list):
        return all(x is None or isinstance(x, (str, int, float, bool)) for x in v)
    return False


def _validate_ranges(obj: Dict[str, Any]) -> None:
    preprocess = obj.get("preprocess") or {}
    if isinstance(preprocess, dict):
        if "denoise_prop" in preprocess:
            prop = preprocess.get("denoise_prop")
            if isinstance(prop, (int, float)) and not (0.0 <= float(prop) <= 1.0):
                raise ValueError("asr_options.preprocess.denoise_prop must be within [0, 1]")

    chunking = obj.get("chunking") or {}
    if isinstance(chunking, dict):
        for k in ("max_chunk_duration_s", "min_chunk_duration_s", "overlap_duration_s", "min_silence_duration_s"):
            if k in chunking:
                v = chunking.get(k)
                if isinstance(v, (int, float)) and float(v) < 0.0:
                    raise ValueError(f"asr_options.chunking.{k} must be >= 0")
        if "max_workers" in chunking:
            mw = chunking.get("max_workers")
            if isinstance(mw, int) and mw < 1:
                raise ValueError("asr_options.chunking.max_workers must be >= 1")
        if "overlap_chars" in chunking:
            oc = chunking.get("overlap_chars")
            if isinstance(oc, int) and oc < 0:
                raise ValueError("asr_options.chunking.overlap_chars must be >= 0")
