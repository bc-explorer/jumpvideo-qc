"""Product QC.

First version only answers: "did a suspected product region survive into the
final alpha?". It leans on upstream product masks (host/assistant hand products,
table products, final_keep minus subtitles) rather than generic YOLO classes.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from . import mask_stats
from .findings import HIGH, MEDIUM, WARNING, Finding
from .yolo_auditor import FrameAudit

BBox = Tuple[float, float, float, float]


def _wrist_points(audit: FrameAudit, kp_conf: float = 0.2) -> List[Tuple[float, float]]:
    from .yolo_auditor import WRIST_KPS

    pts = []
    for pose in audit.poses:
        for i in WRIST_KPS:
            if i < len(pose.keypoints):
                x, y, c = pose.keypoints[i]
                if c >= kp_conf:
                    pts.append((x, y))
    return pts


def _box_center(box: BBox) -> Tuple[float, float]:
    x0, y0, x1, y1 = box
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def _near(center: Tuple[float, float], points: List[Tuple[float, float]], radius: float) -> bool:
    for px, py in points:
        if (center[0] - px) ** 2 + (center[1] - py) ** 2 <= radius * radius:
            return True
    return False


def check_frame_near_hand(
    frame_idx: int,
    audit: FrameAudit,
    combined_alpha: Optional[np.ndarray],
    auto_candidate_boxes: List[BBox],
    frame_wh: Tuple[int, int],
    config,
) -> List[Finding]:
    """Objects near a hand whose combined-alpha coverage is too low."""
    findings: List[Finding] = []
    if combined_alpha is None:
        return findings
    alpha_thr = float(config.rule("alpha_threshold", 0.5))
    fail = float(config.rule("product_near_hand_coverage_fail", 0.50))

    w, h = frame_wh
    radius = max(w, h) * 0.15
    wrists = _wrist_points(audit)
    if not wrists:
        return findings

    candidate_boxes: List[Tuple[str, BBox]] = []
    for det in audit.objects:
        candidate_boxes.append((f"yolo:{det.name}", det.box))
    for bb in auto_candidate_boxes:
        candidate_boxes.append(("auto_candidate", bb))

    seen = set()
    for tag, box in candidate_boxes:
        center = _box_center(box)
        if not _near(center, wrists, radius):
            continue
        key = (round(center[0]), round(center[1]))
        if key in seen:
            continue
        seen.add(key)
        cov = mask_stats.coverage_in_box(combined_alpha, box, alpha_thr)
        if cov < fail:
            findings.append(
                Finding(
                    frame_idx,
                    "product_missing_near_hand",
                    HIGH,
                    "product_qc",
                    {
                        "source": tag,
                        "box": [round(v, 1) for v in box],
                        "combined_coverage": round(cov, 3),
                    },
                )
            )
    return findings


def check_drops(
    frames: List[int],
    areas: Dict[int, int],
    fps: float,
    drop_ratio: float,
    type_code: str,
    severity: str,
    stage: str,
    min_ref_area: int = 64,
) -> List[Finding]:
    """Flag frames where mask area dropped > ``drop_ratio`` vs the +/-1s median."""
    findings: List[Finding] = []
    for f in frames:
        cur = areas.get(f, 0)
        ref = mask_stats.window_median(frames, areas, f, fps, window_sec=1.0)
        if ref is None or ref < min_ref_area:
            continue
        if cur < ref * (1.0 - drop_ratio):
            findings.append(
                Finding(
                    f,
                    type_code,
                    severity,
                    stage,
                    {
                        "area": int(cur),
                        "reference_median": int(ref),
                        "drop_ratio": round(1.0 - (cur / ref if ref else 1.0), 3),
                    },
                )
            )
    return findings


def check_final_keep_object_drop(
    frames: List[int],
    blob_counts: Dict[int, int],
    fps: float,
    stage: str = "product_qc",
) -> List[Finding]:
    """final_keep (minus subtitles) blob count drops to ~0 after being present."""
    findings: List[Finding] = []
    for f in frames:
        cur = blob_counts.get(f, 0)
        ref = mask_stats.window_median(frames, blob_counts, f, fps, window_sec=1.0)
        if ref is None or ref < 1:
            continue
        if cur == 0 and ref >= 1:
            findings.append(
                Finding(
                    f,
                    "final_keep_object_drop",
                    MEDIUM,
                    stage,
                    {"blob_count": cur, "reference_median": round(ref, 2)},
                )
            )
    return findings
