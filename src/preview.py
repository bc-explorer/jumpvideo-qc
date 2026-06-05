"""Annotated preview images for failed segments.

Draws, on the peak frame: YOLO person boxes, face ROI, hand points, the product
candidate / evidence box, combined-alpha and human-alpha boundaries, the
subtitle ignore region, and the failure label.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import alpha_loader, yolo_auditor
from .findings import HIGH, MEDIUM
from .segment_merge import Segment

# BGR colors
C_PERSON = (0, 200, 0)
C_FACE = (0, 215, 255)
C_HAND = (255, 100, 0)
C_EVIDENCE = (0, 0, 255)
C_COMBINED = (255, 255, 0)
C_HUMAN = (180, 105, 255)
C_IGNORE = (60, 60, 60)
SEV_COLOR = {HIGH: (0, 0, 255), MEDIUM: (0, 165, 255), "warning": (0, 215, 255)}


def _draw_contours(img, mask: Optional[np.ndarray], color, thickness=2):
    if mask is None:
        return
    b = (mask >= 0.5).astype(np.uint8) if mask.dtype != np.uint8 else mask
    if b.shape[:2] != img.shape[:2]:
        b = cv2.resize(b, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, thickness)


def _box(img, box, color, thickness=2):
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)


def render_segment(
    seg: Segment,
    resolved,
    config,
    subtitle_ignore,
    auditor: Optional[yolo_auditor.YoloAuditor],
    out_dir: Path,
    audit_cache: Optional[Dict[int, yolo_auditor.FrameAudit]] = None,
) -> Optional[str]:
    frame_idx = seg.peak_frame
    frame = alpha_loader.load_frame_bgr(resolved.frames_dir, frame_idx)
    if frame is None:
        return None
    h, w = frame.shape[:2]
    alpha_thr = float(config.rule("alpha_threshold", 0.5))

    combined = alpha_loader.load_binary(resolved.combined_alpha_dir, frame_idx, alpha_thr)
    human = alpha_loader.load_binary(resolved.human_alpha_dir, frame_idx, alpha_thr)

    overlay = frame.copy()

    # subtitle / ignore region (filled, faint)
    ig = subtitle_ignore.ignore_mask(frame_idx, (h, w))
    if ig is not None and ig.any():
        tint = overlay.copy()
        tint[ig > 0] = C_IGNORE
        cv2.addWeighted(tint, 0.35, overlay, 0.65, 0, overlay)

    _draw_contours(overlay, combined, C_COMBINED, 2)
    _draw_contours(overlay, human, C_HUMAN, 1)

    # YOLO overlays
    if auditor is not None:
        try:
            audit = (audit_cache or {}).get(frame_idx)
            if audit is None:
                audit = auditor.audit_frame(frame)
            for det in audit.persons:
                _box(overlay, det.box, C_PERSON, 2)
                if det.mask is not None:
                    _draw_contours(overlay, det.mask, C_PERSON, 1)
            for pose in audit.poses:
                froi = yolo_auditor.face_roi_from_pose(pose)
                if froi:
                    _box(overlay, froi, C_FACE, 1)
                from .yolo_auditor import WRIST_KPS

                for i in WRIST_KPS:
                    if i < len(pose.keypoints):
                        x, y, c = pose.keypoints[i]
                        if c >= 0.2:
                            cv2.circle(overlay, (int(x), int(y)), 6, C_HAND, -1)
        except Exception:
            pass

    # evidence box / roi
    ev = seg.evidence or {}
    for key in ("box", "roi"):
        if key in ev and isinstance(ev[key], (list, tuple)) and len(ev[key]) == 4:
            _box(overlay, ev[key], C_EVIDENCE, 2)

    # label banner
    color = SEV_COLOR.get(seg.severity, (0, 0, 255))
    label = f"{seg.type} [{seg.severity.upper()}]"
    cv2.rectangle(overlay, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.putText(
        overlay, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"seg_{seg.id:03d}_{seg.type}.jpg"
    out_path = out_dir / fname
    cv2.imwrite(str(out_path), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return str(out_path)


def render_all(
    segments: List[Segment],
    resolved,
    config,
    subtitle_ignore,
    auditor: Optional[yolo_auditor.YoloAuditor],
    report_dir: Path,
    max_previews: int = 200,
    audit_cache: Optional[Dict[int, yolo_auditor.FrameAudit]] = None,
) -> None:
    out_dir = Path(report_dir) / "previews"
    preview_by_frame: Dict[int, str] = {}
    rendered = 0
    for seg in segments:
        existing = preview_by_frame.get(seg.peak_frame)
        if existing:
            seg.preview = existing
            continue
        if rendered >= max_previews:
            break
        try:
            seg.preview = render_segment(
                seg, resolved, config, subtitle_ignore, auditor, out_dir, audit_cache
            )
            if seg.preview:
                rendered += 1
                preview_by_frame[seg.peak_frame] = seg.preview
        except Exception:
            seg.preview = None
