"""Person QC: main host / assistant / extra-person coverage checks.

Coverage uses the YOLO instance-segmentation mask of each person as the
denominator (i.e. alpha recall over actual person pixels), falling back to the
bounding box only when no instance mask is available. This avoids the false
positives caused by tall subjects never filling their rectangular bbox.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from . import mask_stats
from .findings import HIGH, MEDIUM, Finding
from .yolo_auditor import FrameAudit

# Minimum person mask area (px) to trust a mask-based coverage measurement.
_MIN_MASK_PX = 200


def _coverage(
    alpha: Optional[np.ndarray],
    det,
    alpha_thr: float,
) -> Tuple[float, str]:
    """Return (coverage, basis) where basis is 'mask' or 'box'."""
    if alpha is None:
        return 0.0, "none"
    if det.mask is not None and int(det.mask.sum()) >= _MIN_MASK_PX:
        return mask_stats.coverage_in_mask(alpha, det.mask, alpha_thr), "mask"
    return mask_stats.coverage_in_box(alpha, det.box, alpha_thr), "box"


def _person_area(det) -> int:
    if det.mask is not None:
        return int(det.mask.sum())
    x0, y0, x1, y1 = det.box
    return int(max(0.0, x1 - x0) * max(0.0, y1 - y0))


def _frame_area(*masks) -> int:
    for m in masks:
        if m is not None:
            h, w = m.shape[:2]
            return int(h * w)
    return 0


def _is_real_secondary(det, frame_area: int, min_conf: float, min_area_ratio: float) -> bool:
    """Filter out low-confidence / tiny boxes so they are not a 'second person'."""
    if det.conf < min_conf:
        return False
    if frame_area > 0 and (_person_area(det) / frame_area) < min_area_ratio:
        return False
    return True


def check_frame(
    frame_idx: int,
    audit: FrameAudit,
    combined_alpha: Optional[np.ndarray],
    human_alpha: Optional[np.ndarray],
    assistant_mask: Optional[np.ndarray],
    has_human_alpha: bool,
    config,
) -> List[Finding]:
    findings: List[Finding] = []
    alpha_thr = float(config.rule("alpha_threshold", 0.5))
    combined_warn = float(config.rule("person_combined_coverage_warn", 0.75))
    human_warn = float(config.rule("human_alpha_coverage_warn", 0.65))
    sec_min_conf = float(config.rule("second_person_min_conf", 0.35))
    sec_min_area = float(config.rule("second_person_min_area_ratio", 0.02))

    persons = audit.persons
    if not persons:
        return findings

    frame_area = _frame_area(combined_alpha, human_alpha, assistant_mask)

    # count of "real" persons: host (rank 0) + qualified secondaries
    qualified_count = 1
    for det in persons[1:]:
        if _is_real_secondary(det, frame_area, sec_min_conf, sec_min_area):
            qualified_count += 1

    # primary host = largest person box
    for rank, det in enumerate(persons):
        # skip noise boxes for secondary-person logic entirely
        if rank >= 1 and not _is_real_secondary(
            det, frame_area, sec_min_conf, sec_min_area
        ):
            continue
        comb_cov, basis = _coverage(combined_alpha, det, alpha_thr)
        human_cov = None
        if has_human_alpha:
            human_cov, _ = _coverage(human_alpha, det, alpha_thr)
        ev = {
            "person_rank": rank,
            "box": [round(v, 1) for v in det.box],
            "yolo_conf": round(det.conf, 3),
            "coverage_basis": basis,
            "person_combined_coverage": round(comb_cov, 3),
        }
        if human_cov is not None:
            ev["person_human_coverage"] = round(human_cov, 3)

        if rank == 0:
            if comb_cov < combined_warn:
                findings.append(
                    Finding(frame_idx, "person_missing", HIGH, "person_qc", dict(ev))
                )
            if has_human_alpha and human_cov is not None and human_cov < human_warn:
                findings.append(
                    Finding(
                        frame_idx, "human_alpha_missing", MEDIUM, "person_qc", dict(ev)
                    )
                )
        else:
            # second / additional person
            if comb_cov < combined_warn:
                findings.append(
                    Finding(
                        frame_idx,
                        "second_person_missing",
                        HIGH,
                        "person_qc",
                        dict(ev),
                    )
                )
            # if an assistant mask is expected but weak/absent under the 2nd person
            if assistant_mask is not None:
                if det.mask is not None and int(det.mask.sum()) >= _MIN_MASK_PX:
                    a_cov = mask_stats.coverage_in_mask(assistant_mask, det.mask, 0.5)
                else:
                    a_cov = mask_stats.coverage_in_box(assistant_mask, det.box, 0.5)
                if a_cov < 0.4:
                    ev2 = dict(ev)
                    ev2["assistant_mask_coverage"] = round(a_cov, 3)
                    findings.append(
                        Finding(
                            frame_idx,
                            "assistant_mask_missing",
                            MEDIUM,
                            "person_qc",
                            ev2,
                        )
                    )

    if qualified_count > 2:
        findings.append(
            Finding(
                frame_idx,
                "staff_or_extra_person_risk",
                MEDIUM,
                "person_qc",
                {
                    "person_count": qualified_count,
                    "raw_detections": len(persons),
                },
            )
        )

    return findings
