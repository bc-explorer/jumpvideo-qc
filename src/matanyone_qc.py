"""MatAnyone2 QC: locate person-alpha problems via area / coverage / flicker.

Operates on the human alpha series + consecutive-frame IoU collected by the
pipeline. Product problems are intentionally handled elsewhere (final_keep and
product masks), not here.
"""
from __future__ import annotations

from typing import Dict, List

from . import mask_stats
from .findings import MEDIUM, WARNING, Finding


def check_series(
    frames: List[int],
    human_area: Dict[int, int],
    human_iou_prev: Dict[int, float],
    person_present: Dict[int, bool],
    fps: float,
    config,
) -> List[Finding]:
    findings: List[Finding] = []
    drop_ratio = float(config.rule("matanyone_alpha_drop_ratio", 0.35))
    flicker_iou = float(config.rule("flicker_iou_threshold", 0.50))

    for f in frames:
        cur = human_area.get(f, 0)
        ref = mask_stats.window_median(frames, human_area, f, fps, window_sec=1.0)
        if ref is not None and ref >= 64 and cur < ref * (1.0 - drop_ratio):
            findings.append(
                Finding(
                    f,
                    "matanyone_alpha_drop",
                    MEDIUM,
                    "matanyone_qc",
                    {"area": int(cur), "reference_median": int(ref)},
                )
            )

        iou = human_iou_prev.get(f)
        if (
            iou is not None
            and iou < flicker_iou
            and person_present.get(f, False)
            and cur > 0
        ):
            findings.append(
                Finding(
                    f,
                    "matanyone_flicker",
                    MEDIUM,
                    "matanyone_qc",
                    {"iou_prev": round(iou, 3), "threshold": flicker_iou},
                )
            )

    return findings
