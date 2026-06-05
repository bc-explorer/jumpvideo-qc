"""Performance accounting for a QC run.

Measures how much input data was actually consumed (bytes of the sampled frame
files across every source the pipeline reads), the video duration, and derives
throughput (MB/s), processing fps, and a real-time factor.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from . import frame_index
from .resolver import ResolvedPaths

# directories addressed per sampled frame by the pipeline
_PER_FRAME_DIRS = (
    "frames_dir",
    "combined_alpha_dir",
    "human_alpha_dir",
    "final_keep_dir",
    "table_products_dir",
    "host_hand_products_dir",
    "assistant_hand_products_dir",
    "assistant_person_dir",
    "subtitles_dir",
)


def _safe_size(p: Optional[Path]) -> int:
    if p is None:
        return 0
    try:
        return p.stat().st_size
    except OSError:
        return 0


def measure_processed_bytes(resolved: ResolvedPaths, frames: List[int]) -> int:
    """Sum byte sizes of the files actually read for the sampled frames."""
    total = 0
    for attr in _PER_FRAME_DIRS:
        d = getattr(resolved, attr, None)
        if not d:
            continue
        dpath = Path(d)
        for f in frames:
            total += _safe_size(frame_index.find_frame_file(dpath, f))

    # SAM2 object masks (sampled frames per object)
    if resolved.sam2_dir:
        base = Path(resolved.sam2_dir)
        if base.is_dir():
            for child in base.iterdir():
                if not child.is_dir():
                    continue
                for f in frames:
                    total += _safe_size(frame_index.find_frame_file(child, f))
    return total


def build_report(
    resolved: ResolvedPaths,
    frames: List[int],
    elapsed_seconds: float,
    fps: float,
    total_frames: int,
) -> Dict:
    processed_bytes = measure_processed_bytes(resolved, frames)
    sampled = len(frames)
    elapsed = max(1e-6, float(elapsed_seconds))

    video_duration = (total_frames / fps) if fps else 0.0
    mb = processed_bytes / (1024.0 * 1024.0)

    return {
        "elapsed_seconds": round(elapsed_seconds, 2),
        "video_duration_seconds": round(video_duration, 2),
        "total_frames": int(total_frames),
        "sampled_frames": sampled,
        "fps": round(fps, 3),
        "processed_bytes": int(processed_bytes),
        "processed_mb": round(mb, 1),
        "throughput_mb_s": round(mb / elapsed, 2),
        "frames_per_second": round(sampled / elapsed, 2),
        "realtime_factor": round(video_duration / elapsed, 2) if video_duration else 0.0,
        "seconds_per_sampled_frame": round(elapsed / sampled, 4) if sampled else 0.0,
    }


def human_summary(perf: Dict) -> str:
    return (
        f"处理 {perf['sampled_frames']}/{perf['total_frames']} 帧 "
        f"(视频时长 {perf['video_duration_seconds']}s)，"
        f"读取 {perf['processed_mb']} MB，耗时 {perf['elapsed_seconds']}s，"
        f"吞吐 {perf['throughput_mb_s']} MB/s，"
        f"{perf['frames_per_second']} 帧/s，实时倍率 {perf['realtime_factor']}x"
    )
