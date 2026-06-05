"""Input integrity scan -> ``input_status``.

Reports presence/absence of each artifact rather than failing hard. Upstream
stages may not have run fully; QC must be able to tell a human what is missing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from . import frame_index
from .resolver import ResolvedPaths

# Which artifacts are critical vs optional. Missing critical artifacts disable
# specific QC stages but never crash the run.
CRITICAL = {"source_video", "frames_dir", "combined_alpha_dir"}


@dataclass
class DirStat:
    status: str  # "ok" | "missing"
    path: Optional[str] = None
    frame_count: int = 0
    first: Optional[str] = None
    last: Optional[str] = None

    def to_dict(self) -> dict:
        d = {"status": self.status}
        if self.path:
            d["path"] = self.path
        if self.frame_count:
            d["frame_count"] = self.frame_count
        if self.first:
            d["first"] = self.first
        if self.last:
            d["last"] = self.last
        return d


@dataclass
class InputScan:
    input_status: Dict[str, dict] = field(default_factory=dict)
    warnings: List[dict] = field(default_factory=list)

    def status_of(self, key: str) -> str:
        entry = self.input_status.get(key)
        if not entry:
            return "missing"
        return entry.get("status", "missing")

    def is_ok(self, key: str) -> bool:
        return self.status_of(key) == "ok"

    def to_dict(self) -> dict:
        return {"input_status": self.input_status, "warnings": self.warnings}


def _file_stat(path: Optional[str]) -> DirStat:
    if path and Path(path).is_file():
        return DirStat(status="ok", path=path)
    return DirStat(status="missing")


def _dir_stat(path: Optional[str]) -> DirStat:
    if not path or not Path(path).is_dir():
        return DirStat(status="missing")
    d = Path(path)
    count = frame_index.count_frames(d)
    first, last = frame_index.first_last_names(d)
    return DirStat(status="ok", path=path, frame_count=count, first=first, last=last)


def scan_inputs(rp: ResolvedPaths) -> InputScan:
    scan = InputScan()
    s = scan.input_status

    s["source_video"] = _file_stat(rp.source_video).to_dict()
    s["frames_dir"] = _dir_stat(rp.frames_dir).to_dict()
    s["combined_alpha_dir"] = _dir_stat(rp.combined_alpha_dir).to_dict()
    s["human_alpha_dir"] = _dir_stat(rp.human_alpha_dir).to_dict()
    s["final_keep_dir"] = _dir_stat(rp.final_keep_dir).to_dict()
    s["table_products_dir"] = _dir_stat(rp.table_products_dir).to_dict()
    s["host_hand_products_dir"] = _dir_stat(rp.host_hand_products_dir).to_dict()
    s["assistant_person_dir"] = _dir_stat(rp.assistant_person_dir).to_dict()
    s["assistant_hand_products_dir"] = _dir_stat(rp.assistant_hand_products_dir).to_dict()
    s["subtitles_dir"] = _dir_stat(rp.subtitles_dir).to_dict()
    s["sam2_dir"] = _dir_stat(rp.sam2_dir).to_dict()
    s["sam2_prompts"] = _file_stat(rp.sam2_prompts).to_dict()
    s["auto_candidates"] = _file_stat(rp.auto_candidates).to_dict()
    s["outputs_manifest"] = _file_stat(rp.outputs_manifest).to_dict()
    s["composited_video"] = _file_stat(rp.composited_video).to_dict()

    # warnings for missing optional/notable artifacts
    if scan.status_of("composited_video") == "missing":
        scan.warnings.append({"warning": "composited_video_missing"})
    if scan.status_of("final_keep_dir") == "missing":
        scan.warnings.append({"warning": "final_keep_missing"})
    if scan.status_of("subtitles_dir") == "missing":
        scan.warnings.append({"warning": "subtitle_mask_missing"})
    if scan.status_of("human_alpha_dir") == "missing":
        scan.warnings.append({"warning": "human_alpha_dir_missing"})
    if scan.status_of("sam2_dir") == "missing":
        scan.warnings.append({"warning": "sam2_masks_missing"})

    return scan


def blocking_messages(scan: InputScan) -> List[str]:
    """Human-readable notes about which QC stages will be skipped/limited."""
    msgs: List[str] = []
    if not scan.is_ok("combined_alpha_dir"):
        msgs.append("combined_alpha missing: cannot run foreground coverage QC")
    if not scan.is_ok("human_alpha_dir"):
        msgs.append("human_alpha missing: MatAnyone QC will be skipped")
    if not scan.is_ok("frames_dir") and not scan.is_ok("source_video"):
        msgs.append("frames/source and source.mp4 both missing: YOLO QC will be skipped")
    if not scan.is_ok("final_keep_dir"):
        msgs.append("final_keep missing: product QC limited to hand/table masks")
    return msgs
