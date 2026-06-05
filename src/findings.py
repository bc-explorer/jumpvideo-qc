"""Shared finding / severity model emitted by every QC stage."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

# Severity ordering
WARNING = "warning"
MEDIUM = "medium"
HIGH = "high"

SEVERITY_RANK = {WARNING: 1, MEDIUM: 2, HIGH: 3}

# Canonical finding type codes -> "problem source" (for the report).
# Keeps the human-facing string aligned with which upstream step to inspect.
SOURCE_BY_TYPE = {
    "person_missing": "person_missing_from_combined_alpha",
    "human_alpha_missing": "human_alpha_missing_from_matanyone",
    "second_person_missing": "person_missing_from_combined_alpha",
    "staff_or_extra_person_risk": "extra_person_detected",
    "assistant_mask_missing": "assistant_person_mask_missing",
    "face_alpha_missing": "person_missing_from_combined_alpha",
    "hand_alpha_missing": "person_missing_from_combined_alpha",
    "product_missing_near_hand": "product_missing_from_final_keep",
    "host_product_drop": "product_missing_from_final_keep",
    "assistant_product_drop": "product_missing_from_final_keep",
    "table_product_drop": "product_missing_from_final_keep",
    "final_keep_object_drop": "product_missing_from_final_keep",
    "sam2_object_drop": "sam2_object_drop",
    "sam2_background_leak": "sam2_background_leak",
    "sam2_drift": "sam2_drift",
    "possible_object_person_merge": "sam2_object_person_merge",
    "matanyone_alpha_drop": "human_alpha_missing_from_matanyone",
    "matanyone_person_missing": "human_alpha_missing_from_matanyone",
    "matanyone_flicker": "human_alpha_missing_from_matanyone",
    "subtitle_mask_missing": "subtitle_mask_missing",
    "composited_video_missing": "composited_video_missing",
}


@dataclass
class Finding:
    frame_index: int
    type: str
    severity: str
    stage: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def source(self) -> str:
        return SOURCE_BY_TYPE.get(self.type, self.type)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["source"] = self.source
        return d


def max_severity(findings: List[Finding]) -> str | None:
    if not findings:
        return None
    return max(findings, key=lambda f: SEVERITY_RANK.get(f.severity, 0)).severity


def severity_at_least(a: str, b: str) -> bool:
    return SEVERITY_RANK.get(a, 0) >= SEVERITY_RANK.get(b, 0)
