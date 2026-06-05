"""SAM2 QC.

No per-frame bbox/area log exists upstream, so we compute our own by counting
non-zero pixels in ``masks/sam2/<object_name>/*.png`` over sampled frames.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import frame_index, mask_stats
from .findings import HIGH, MEDIUM, WARNING, Finding


def list_objects(sam2_dir: Optional[str]) -> List[Tuple[str, Path]]:
    """Return (object_name, dir) for each sub-directory holding mask PNGs."""
    out: List[Tuple[str, Path]] = []
    if not sam2_dir:
        return out
    base = Path(sam2_dir)
    if not base.is_dir():
        return out
    for child in sorted(base.iterdir()):
        if child.is_dir() and frame_index.list_frame_files(child):
            out.append((child.name, child))
    return out


def load_object_roles(prompts_path: Optional[str]) -> Dict[str, str]:
    roles: Dict[str, str] = {}
    if not prompts_path or not Path(prompts_path).is_file():
        return roles
    try:
        with open(prompts_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return roles
    objs = data.get("objects") if isinstance(data, dict) else data
    if isinstance(objs, list):
        for o in objs:
            if isinstance(o, dict):
                name = o.get("name") or o.get("object_name") or o.get("id")
                role = o.get("role") or o.get("type") or o.get("label")
                if name:
                    roles[str(name)] = str(role) if role else ""
    return roles


def _area_bbox(path: Path) -> Tuple[int, Optional[Tuple[int, int, int, int]]]:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0, None
    b = (img >= 128).astype(np.uint8)
    a = int(b.sum())
    bb = mask_stats.bbox(b, 0.5)
    return a, bb


def analyze(
    resolved,
    frames: List[int],
    fps: float,
    frame_wh: Tuple[int, int],
    config,
) -> Tuple[List[Finding], List[dict]]:
    findings: List[Finding] = []
    object_stats: List[dict] = []

    objects = list_objects(resolved.sam2_dir)
    if not objects:
        return findings, object_stats

    roles = load_object_roles(resolved.sam2_prompts)
    drop_ratio = float(config.rule("sam2_object_drop_ratio", 0.40))
    grow_ratio = float(config.rule("sam2_background_leak_ratio", 0.80))
    drift_ratio = float(config.rule("sam2_drift_ratio", 0.25))
    w, h = frame_wh
    drift_dist = max(w, h) * drift_ratio

    for name, obj_dir in objects:
        files = frame_index.list_frame_files(obj_dir)
        present = sorted(set(frames) & set(files.keys())) or sorted(files.keys())
        area_by_frame: Dict[int, int] = {}
        bbox_by_frame: Dict[int, Tuple[int, int, int, int]] = {}
        for f in present:
            a, bb = _area_bbox(files[f])
            area_by_frame[f] = a
            if bb is not None:
                bbox_by_frame[f] = bb

        object_stats.append(
            {
                "object_name": name,
                "role": roles.get(name, ""),
                "area_by_frame": {str(k): v for k, v in area_by_frame.items()},
                "bbox_by_frame": {str(k): list(v) for k, v in bbox_by_frame.items()},
            }
        )

        seq = sorted(area_by_frame.keys())
        for f in seq:
            cur = area_by_frame.get(f, 0)
            ref = mask_stats.window_median(seq, area_by_frame, f, fps, window_sec=1.0)
            if ref is None or ref < 64:
                continue
            if cur < ref * (1.0 - drop_ratio):
                findings.append(
                    Finding(
                        f,
                        "sam2_object_drop",
                        HIGH,
                        "sam2_qc",
                        {"object": name, "area": cur, "reference_median": int(ref)},
                    )
                )
            elif cur > ref * (1.0 + grow_ratio):
                findings.append(
                    Finding(
                        f,
                        "sam2_background_leak",
                        MEDIUM,
                        "sam2_qc",
                        {"object": name, "area": cur, "reference_median": int(ref)},
                    )
                )

        # drift: bbox center jump between consecutive sampled frames
        prev_center = None
        prev_f = None
        for f in sorted(bbox_by_frame.keys()):
            c = mask_stats.bbox_center(bbox_by_frame[f])
            if prev_center is not None and c is not None:
                dist = ((c[0] - prev_center[0]) ** 2 + (c[1] - prev_center[1]) ** 2) ** 0.5
                if dist > drift_dist:
                    findings.append(
                        Finding(
                            f,
                            "sam2_drift",
                            MEDIUM,
                            "sam2_qc",
                            {
                                "object": name,
                                "center_jump_px": round(dist, 1),
                                "threshold_px": round(drift_dist, 1),
                                "prev_frame": prev_f,
                            },
                        )
                    )
            prev_center = c
            prev_f = f

    return findings, object_stats
