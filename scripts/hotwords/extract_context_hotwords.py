from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, List, Sequence


_SPLIT_RE = re.compile(r"[,\uFF0C;\uFF1B\u3001|\t]+")
_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-*•·]|(\d+)[\.\)]|[（(]\d+[)）])\s+")
_TRIM_RE = re.compile(r"^[\s\"'“”‘’]+|[\s\"'“”‘’]+$")


def _repo_root() -> Path:
    # scripts/hotwords/extract_context_hotwords.py -> repo root is 2 parents up
    return Path(__file__).resolve().parents[2]


def _clean_phrase(s: str) -> str:
    s = s.strip()
    if not s or s.startswith("#"):
        return ""
    s = _BULLET_PREFIX_RE.sub("", s)
    s = _TRIM_RE.sub("", s)
    return s.strip()


def extract_phrases(lines: Iterable[str], *, split_delims: bool = True) -> List[str]:
    phrases: List[str] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = _BULLET_PREFIX_RE.sub("", line).strip()
        parts: Sequence[str]
        if split_delims:
            parts = [p for p in _SPLIT_RE.split(line) if p.strip()]
        else:
            parts = [line]

        for p in parts:
            cleaned = _clean_phrase(p)
            if cleaned:
                phrases.append(cleaned)
    return phrases


def _load_existing_header(path: Path) -> List[str]:
    if not path.exists():
        return []
    header: List[str] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines(True):
            if line.lstrip().startswith("#") or not line.strip():
                header.append(line)
                continue
            break
    except Exception:
        return []
    return header


def _write_output(path: Path, header_lines: List[str], phrases: List[str]) -> bool:
    content_lines: List[str] = []
    if header_lines:
        content_lines.extend(header_lines)
        if content_lines and content_lines[-1].strip():
            content_lines.append("\n")

    content_lines.extend([f"{p}\n" for p in phrases])
    new_content = "".join(content_lines)

    old_content = ""
    if path.exists():
        try:
            old_content = path.read_text(encoding="utf-8")
        except Exception:
            old_content = ""

    if old_content == new_content:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_content, encoding="utf-8")
    return True


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Extract unique context hotwords from a text file for meeting/recall transcription.",
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input text file (meeting notes / domain list).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=str(_repo_root() / "data" / "hotwords" / "hotwords-context.txt"),
        help="Output file path. Use '-' to write to stdout.",
    )
    parser.add_argument(
        "--no-split-delims",
        action="store_true",
        help="Do not split a line by delimiters like comma/semicolon/、/| (treat each line as one phrase).",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=1,
        help="Minimum non-whitespace char length to keep a phrase (default: 1).",
    )

    args = parser.parse_args(list(argv))

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file does not exist: {input_path}", file=sys.stderr)
        return 2

    split_delims = not bool(args.no_split_delims)
    phrases = extract_phrases(input_path.read_text(encoding="utf-8", errors="ignore").splitlines(), split_delims=split_delims)
    min_len = max(int(args.min_len), 0)
    phrases = [p for p in phrases if len(p.strip()) >= min_len]

    # Stable + friendly sorting
    unique_sorted = sorted(set(phrases), key=lambda s: (s.lower(), s))

    if args.output == "-":
        for p in unique_sorted:
            sys.stdout.write(p + "\n")
        return 0

    output_path = Path(args.output)
    header = _load_existing_header(output_path)
    changed = _write_output(output_path, header, unique_sorted)
    if changed:
        print(f"Wrote {len(unique_sorted)} phrases -> {output_path}")
    else:
        print(f"No changes ({len(unique_sorted)} phrases) -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

