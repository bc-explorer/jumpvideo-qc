"""Face / hand QC against combined_alpha, using YOLO pose keypoints.

Coverage denominator is the person pixels inside the ROI (ROI box intersected
with the person instance-segmentation union), not the whole rectangular ROI.
This removes the false positives produced when a generous face/hand ROI box
includes lots of background. Falls back to plain box coverage only when no
person instance mask is available.

Hand risks are prioritised: in live commerce a product is usually near the
hand, so a hand that gets eaten by the matte often takes the product with it.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from . import mask_stats, yolo_auditor
from .findings import HIGH, MEDIUM, WARNING, Finding
from .yolo_auditor import FrameAudit

# Minimum number of person pixels inside an ROI to trust a mask-based ratio.
_MIN_ROI_PERSON_PX = 40


def _roi_coverage(
    combined_alpha: np.ndarray,
    roi: tuple,
    person_union: Optional[np.ndarray],
    alpha_thr: float,
) -> Optional[tuple]:
    """Return (coverage, basis) or None if the ROI has too few person pixels."""
    if person_union is not None:
        region = mask_stats.restrict_to_box(person_union, roi)
        if region is None or int(region.sum()) < _MIN_ROI_PERSON_PX:
            return None
        return mask_stats.coverage_in_mask(combined_alpha, region, alpha_thr), "mask"
    return mask_stats.coverage_in_box(combined_alpha, roi, alpha_thr), "box"


def check_frame(
    frame_idx: int,
    audit: FrameAudit,
    combined_alpha: Optional[np.ndarray],
    config,
) -> List[Finding]:
    findings: List[Finding] = []
    if combined_alpha is None:
        return findings

    alpha_thr = float(config.rule("alpha_threshold", 0.5))
    face_warn = float(config.rule("face_coverage_warn", 0.90))
    face_fail = float(config.rule("face_coverage_fail", 0.75))
    hand_warn = float(config.rule("hand_coverage_warn", 0.75))
    hand_fail = float(config.rule("hand_coverage_fail", 0.60))

    person_union = yolo_auditor.person_union_mask(audit, combined_alpha.shape[:2])

    for rank, pose in enumerate(audit.poses):
        face_roi = yolo_auditor.face_roi_from_pose(pose)
        if face_roi is not None:
            res = _roi_coverage(combined_alpha, face_roi, person_union, alpha_thr)
            if res is not None:
                cov, basis = res
                ev = {
                    "person_rank": rank,
                    "roi": [round(v, 1) for v in face_roi],
                    "coverage_basis": basis,
                    "face_coverage": round(cov, 3),
                }
                if cov < face_fail:
                    findings.append(
                        Finding(frame_idx, "face_alpha_missing", HIGH, "face_hand_qc", ev)
                    )
                elif cov < face_warn:
                    findings.append(
                        Finding(frame_idx, "face_alpha_missing", WARNING, "face_hand_qc", ev)
                    )

        for hand_roi in yolo_auditor.hand_rois_from_pose(pose):
            res = _roi_coverage(combined_alpha, hand_roi, person_union, alpha_thr)
            if res is None:
                continue
            cov, basis = res
            ev = {
                "person_rank": rank,
                "roi": [round(v, 1) for v in hand_roi],
                "coverage_basis": basis,
                "hand_coverage": round(cov, 3),
            }
            if cov < hand_fail:
                findings.append(
                    Finding(frame_idx, "hand_alpha_missing", HIGH, "face_hand_qc", ev)
                )
            elif cov < hand_warn:
                findings.append(
                    Finding(frame_idx, "hand_alpha_missing", WARNING, "face_hand_qc", ev)
                )

    return findings
