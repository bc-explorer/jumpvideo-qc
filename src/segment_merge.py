"""Merge consecutive risk frames into failed segments.

Findings of the same type that are within ``merge_gap_seconds`` of each other
are merged; each segment is padded by ``pad_seconds`` on both sides. A per-type
minimum duration filter drops transient blips (e.g. a second person must
persist >= 0.7s to alarm).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .findings import SEVERITY_RANK, Finding, SOURCE_BY_TYPE

# Minimum on-screen duration (seconds) required for certain alarm types.
MIN_DURATION_SEC = {
    "second_person_missing": 0.7,
}


@dataclass
class Segment:
    id: int
    type: str
    severity: str
    source: str
    start_frame: int
    end_frame: int
    peak_frame: int
    start_time: float
    end_time: float
    evidence: dict = field(default_factory=dict)
    finding_count: int = 0
    preview: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "severity": self.severity,
            "source": self.source,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "peak_frame": self.peak_frame,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "start_tc": _tc(self.start_time),
            "end_tc": _tc(self.end_time),
            "evidence": self.evidence,
            "finding_count": self.finding_count,
            "preview": self.preview,
        }


def _tc(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h:
        return f"{h:02d}:{m:02d}:{s:04.1f}"
    return f"{m:02d}:{s:04.1f}"


def merge_findings(
    findings: List[Finding],
    fps: float,
    config,
    frame_count: int = 0,
) -> List[Segment]:
    merge_gap = float(config.segment.get("merge_gap_seconds", 1.0))
    pad = float(config.segment.get("pad_seconds", 1.0))
    gap_frames = max(1, int(round(merge_gap * fps)))
    pad_frames = int(round(pad * fps))

    by_type: Dict[str, List[Finding]] = {}
    for f in findings:
        by_type.setdefault(f.type, []).append(f)

    segments: List[Segment] = []
    seg_id = 0
    for ftype, items in by_type.items():
        items.sort(key=lambda x: x.frame_index)
        cluster: List[Finding] = []
        last_frame = None
        clusters: List[List[Finding]] = []
        for it in items:
            if last_frame is None or it.frame_index - last_frame <= gap_frames:
                cluster.append(it)
            else:
                clusters.append(cluster)
                cluster = [it]
            last_frame = it.frame_index
        if cluster:
            clusters.append(cluster)

        min_dur = MIN_DURATION_SEC.get(ftype, 0.0)
        for cl in clusters:
            start_f = cl[0].frame_index
            end_f = cl[-1].frame_index
            duration = (end_f - start_f) / fps if fps else 0.0
            if min_dur and duration < min_dur and len(cl) <= 1:
                # too short to be real
                continue

            peak = max(cl, key=lambda x: SEVERITY_RANK.get(x.severity, 0))
            severity = peak.severity

            ps = max(0, start_f - pad_frames)
            pe = end_f + pad_frames
            if frame_count:
                pe = min(pe, frame_count - 1)

            seg = Segment(
                id=seg_id,
                type=ftype,
                severity=severity,
                source=SOURCE_BY_TYPE.get(ftype, ftype),
                start_frame=ps,
                end_frame=pe,
                peak_frame=peak.frame_index,
                start_time=ps / fps if fps else 0.0,
                end_time=pe / fps if fps else 0.0,
                evidence=peak.evidence,
                finding_count=len(cl),
            )
            segments.append(seg)
            seg_id += 1

    segments.sort(key=lambda s: (s.start_frame, -SEVERITY_RANK.get(s.severity, 0)))
    # reassign ids in temporal order
    for i, s in enumerate(segments):
        s.id = i
    return segments
