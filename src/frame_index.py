"""Frame number <-> filename compatibility helpers.

Source frames, combined alpha, and combined masks use 6-digit names
(``000000.jpg`` / ``000000.png``). MatAnyone2 person alpha may use 6 / 4 / raw
digit names (``000123.png`` / ``0123.png`` / ``123.png``). These helpers
normalise that so the rest of the pipeline can address frames by integer index.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")
_NUM_RE = re.compile(r"(\d+)")


def find_frame_file(dir_path: Path, frame_index: int) -> Optional[Path]:
    """Return the file for ``frame_index`` in ``dir_path`` or ``None``.

    Tries zero-padded 6 / 4 digit names then the raw integer, across the
    common image extensions.
    """
    if dir_path is None:
        return None
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        return None

    stems = [
        f"{frame_index:06d}",
        f"{frame_index:04d}",
        f"{frame_index}",
    ]
    for stem in stems:
        for ext in _IMG_EXTS:
            p = dir_path / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def parse_frame_index(name: str) -> Optional[int]:
    """Extract the leading integer frame index from a filename."""
    stem = Path(name).stem
    m = _NUM_RE.search(stem)
    if not m:
        return None
    return int(m.group(1))


def list_frame_files(dir_path: Path) -> Dict[int, Path]:
    """Map ``frame_index -> path`` for every image file in ``dir_path``.

    Robust to mixed padding. If two files map to the same index, the
    lexicographically last one wins (deterministic).
    """
    out: Dict[int, Path] = {}
    if dir_path is None:
        return out
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        return out
    for p in sorted(dir_path.iterdir()):
        if p.suffix.lower() not in _IMG_EXTS:
            continue
        idx = parse_frame_index(p.name)
        if idx is None:
            continue
        out[idx] = p
    return out


def frame_indices(dir_path: Path) -> List[int]:
    """Sorted list of frame indices present in ``dir_path``."""
    return sorted(list_frame_files(dir_path).keys())


def count_frames(dir_path: Path) -> int:
    return len(list_frame_files(dir_path))


def first_last_names(dir_path: Path) -> tuple[Optional[str], Optional[str]]:
    files = list_frame_files(dir_path)
    if not files:
        return None, None
    keys = sorted(files.keys())
    return files[keys[0]].name, files[keys[-1]].name
