"""Best-effort loader for ``prompts/auto_candidates.json``.

The upstream schema is not finalised, so we extract non-person candidate boxes
defensively and tolerate several key conventions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

BBox = Tuple[float, float, float, float]

_PERSON_WORDS = {"person", "host", "assistant", "human", "people", "anchor"}


def _is_person(label: Optional[str]) -> bool:
    if not label:
        return False
    low = str(label).lower()
    return any(w in low for w in _PERSON_WORDS)


def _coerce_box(val) -> Optional[BBox]:
    if isinstance(val, (list, tuple)) and len(val) == 4:
        try:
            return tuple(float(v) for v in val)  # type: ignore
        except (TypeError, ValueError):
            return None
    return None


def load_non_person_boxes(path: Optional[str]) -> List[BBox]:
    if not path or not Path(path).is_file():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    items = data
    if isinstance(data, dict):
        for key in ("candidates", "objects", "items", "boxes", "detections"):
            if isinstance(data.get(key), list):
                items = data[key]
                break

    boxes: List[BBox] = []
    if not isinstance(items, list):
        return boxes
    for it in items:
        if not isinstance(it, dict):
            bb = _coerce_box(it)
            if bb:
                boxes.append(bb)
            continue
        label = it.get("label") or it.get("role") or it.get("class") or it.get("name")
        if _is_person(label):
            continue
        bb = None
        for key in ("box", "bbox", "xyxy", "rect"):
            bb = _coerce_box(it.get(key))
            if bb:
                break
        if bb is None and all(k in it for k in ("x0", "y0", "x1", "y1")):
            bb = (it["x0"], it["y0"], it["x1"], it["y1"])
        if bb:
            boxes.append(bb)
    return boxes
