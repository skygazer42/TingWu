"""
Robust text merge for overlapped ASR chunks.

This module ports the *behavior* of CapsWriter-Offline's merge-by-text algorithm:
- Search within a tail window of the previous text (not only the absolute suffix)
- Allow skipping a small noisy prefix in the new text
- Optional fuzzy match fallback (tolerate a few wrong characters)

It is used to reduce duplicated text when long audio is chunked with overlap.
"""

from __future__ import annotations

import logging

__all__ = ["merge_by_text"]

logger = logging.getLogger(__name__)


# Treat these characters as "punctuation/noise" for boundary matching.
# Includes ASCII + common CJK punctuation and whitespace.
_PUNCTUATION_ALL = (
    " \t\r\n"
    ",.?!:;()[]{}<>"
    "\"'`"
    "，。？！：；、（）【】《》〈〉「」『』"
    "“”‘’…—"
)


def _fuzzy_match(a: str, b: str, max_errors: int) -> bool:
    """Return True if strings are same length and differ by <= max_errors chars."""
    if len(a) != len(b):
        return False
    errors = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            errors += 1
            if errors > max_errors:
                return False
    return True


def merge_by_text(
    prev_text: str,
    new_text: str,
    *,
    overlap_chars: int = 20,
    error_tolerance: int = 3,
    max_skip_new: int = 10,
) -> str:
    """Merge `new_text` into `prev_text` by removing overlapped duplication.

    Args:
        prev_text: Accumulated previous text.
        new_text: New chunk text to append.
        overlap_chars: Tail window size (characters) from `prev_text` to search within.
        error_tolerance: Max character mismatches allowed in fuzzy matching.
        max_skip_new: Max leading characters to skip in `new_text` when matching.

    Returns:
        Merged text.
    """
    if not prev_text:
        return new_text
    if not new_text:
        return prev_text

    # 1) Prepare matching views: strip trailing punctuation from prev, and leading punctuation from new.
    prev_clean = prev_text.rstrip(_PUNCTUATION_ALL)

    new_match_start = 0
    while new_match_start < len(new_text) and new_text[new_match_start] in _PUNCTUATION_ALL:
        new_match_start += 1
    new_clean = new_text[new_match_start:]

    if not prev_clean or not new_clean:
        return prev_text + new_text

    # 2) Define search window on prev tail.
    if overlap_chars <= 0:
        search_window = prev_clean
    else:
        search_window = prev_clean[-overlap_chars:]
    window_offset = len(prev_clean) - len(search_window)

    max_to_check = min(len(search_window), len(new_clean))
    min_exact_len = 2
    min_fuzzy_len = error_tolerance + 2  # ensure correct chars > error chars

    best_match_skip_new = -1
    best_match_pos_in_window = -1
    best_match_len = 0

    # 3.1 Exact match: longer is better, less skipping is better, closer to the end is better (via rfind).
    for match_len in range(max_to_check, min_exact_len - 1, -1):
        max_skip = min(max_skip_new, len(new_clean) - match_len)
        for skip_new in range(max_skip + 1):
            target_prefix = new_clean[skip_new : skip_new + match_len]
            idx = search_window.rfind(target_prefix)
            if idx != -1:
                best_match_skip_new = skip_new
                best_match_pos_in_window = idx
                best_match_len = match_len
                break
        if best_match_len > 0:
            break

    # 3.2 Fuzzy match: fallback when exact match fails.
    if best_match_len == 0 and error_tolerance > 0 and max_to_check >= min_fuzzy_len:
        for match_len in range(max_to_check, min_fuzzy_len - 1, -1):
            max_skip = min(max_skip_new, len(new_clean) - match_len)
            for skip_new in range(max_skip + 1):
                target_prefix = new_clean[skip_new : skip_new + match_len]

                found_idx = -1
                for i in range(len(search_window) - match_len, -1, -1):
                    if _fuzzy_match(search_window[i : i + match_len], target_prefix, error_tolerance):
                        found_idx = i
                        break

                if found_idx != -1:
                    best_match_skip_new = skip_new
                    best_match_pos_in_window = found_idx
                    best_match_len = match_len
                    break
            if best_match_len > 0:
                break

    # 4) Stitch.
    if best_match_len > 0:
        keep_prev_len = window_offset + best_match_pos_in_window
        res_prev = prev_clean[:keep_prev_len]
        res_new = new_text[new_match_start + best_match_skip_new :]

        logger.debug(
            "merge_by_text: match_len=%s discard_prev=%s skip_new=%s",
            best_match_len,
            max(0, len(prev_clean) - keep_prev_len - best_match_len),
            best_match_skip_new,
        )
        return res_prev + res_new

    logger.debug("merge_by_text: no overlap found, concatenating")
    return prev_text + new_text

