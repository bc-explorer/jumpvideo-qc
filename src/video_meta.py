"""Video metadata: fps / width / height / frame_count.

Falls back to frame-directory inspection when no source video is present, so
alpha-only QC can still run with a sensible fps assumption.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

from . import frame_index


DEFAULT_FPS = 25.0


@dataclass
class VideoMeta:
    fps: float = DEFAULT_FPS
    width: int = 0
    height: int = 0
    frame_count: int = 0
    source: str = "unknown"  # "video" | "frames" | "default"

    def to_dict(self) -> dict:
        return {
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
            "source": self.source,
        }


def from_video(video_path: str | Path) -> Optional[VideoMeta]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0:
        fps = DEFAULT_FPS
    return VideoMeta(fps=fps, width=width, height=height, frame_count=count, source="video")


def from_frames(frames_dir: str | Path, fps: float = DEFAULT_FPS) -> Optional[VideoMeta]:
    frames_dir = Path(frames_dir)
    files = frame_index.list_frame_files(frames_dir)
    if not files:
        return None
    first = files[min(files.keys())]
    img = cv2.imread(str(first), cv2.IMREAD_COLOR)
    if img is None:
        return VideoMeta(fps=fps, frame_count=len(files), source="frames")
    h, w = img.shape[:2]
    return VideoMeta(fps=fps, width=w, height=h, frame_count=len(files), source="frames")


def resolve_meta(
    source_video: Optional[str],
    frames_dir: Optional[str],
    fallback_fps: float = DEFAULT_FPS,
) -> VideoMeta:
    """Prefer the source video; fall back to frame directory; then defaults."""
    if source_video:
        meta = from_video(source_video)
        if meta is not None:
            # frame_count from container can be 0/unreliable -> patch from frames
            if meta.frame_count <= 0 and frames_dir:
                fcount = frame_index.count_frames(Path(frames_dir))
                if fcount:
                    meta.frame_count = fcount
            return meta
    if frames_dir:
        meta = from_frames(frames_dir, fps=fallback_fps)
        if meta is not None:
            return meta
    return VideoMeta(fps=fallback_fps, source="default")
