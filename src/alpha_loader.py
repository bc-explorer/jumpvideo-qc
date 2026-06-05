"""Reading combined / human / final_keep / subtitle masks as numpy arrays.

All loaders return ``float32`` arrays in ``[0, 1]`` (alpha) or ``uint8`` binary
masks. Missing frames return ``None`` (treated as empty downstream).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from . import frame_index


def _read_gray(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        # use alpha channel if present, else convert to gray
        if img.shape[2] == 4:
            img = img[:, :, 3]
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def load_alpha(dir_path: Optional[str], frame_idx: int) -> Optional[np.ndarray]:
    """Load an alpha map as float32 in [0, 1]. None if missing."""
    if not dir_path:
        return None
    p = frame_index.find_frame_file(Path(dir_path), frame_idx)
    if p is None:
        return None
    img = _read_gray(p)
    if img is None:
        return None
    return (img.astype(np.float32) / 255.0)


def load_binary(
    dir_path: Optional[str], frame_idx: int, threshold: float = 0.5
) -> Optional[np.ndarray]:
    """Load a mask as a uint8 {0,1} array. None if missing."""
    if not dir_path:
        return None
    p = frame_index.find_frame_file(Path(dir_path), frame_idx)
    if p is None:
        return None
    img = _read_gray(p)
    if img is None:
        return None
    # Match load_alpha(... ) >= threshold without allocating a float32 image.
    cutoff = int(np.ceil(float(threshold) * 255.0))
    cutoff = max(0, min(cutoff, 255))
    return (img >= cutoff).astype(np.uint8)


def load_frame_bgr(frames_dir: Optional[str], frame_idx: int) -> Optional[np.ndarray]:
    if not frames_dir:
        return None
    p = frame_index.find_frame_file(Path(frames_dir), frame_idx)
    if p is None:
        return None
    return cv2.imread(str(p), cv2.IMREAD_COLOR)


def frame_size(frames_dir: Optional[str]) -> Optional[Tuple[int, int]]:
    """Return (height, width) of the first frame, if any."""
    if not frames_dir:
        return None
    files = frame_index.list_frame_files(Path(frames_dir))
    if not files:
        return None
    img = cv2.imread(str(files[min(files.keys())]), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img.shape[:2]


def ensure_size(mask: Optional[np.ndarray], hw: Tuple[int, int]) -> Optional[np.ndarray]:
    """Resize a mask to (h, w) if it differs (nearest for binary, linear else)."""
    if mask is None:
        return None
    h, w = hw
    if mask.shape[:2] == (h, w):
        return mask
    interp = cv2.INTER_NEAREST if mask.dtype == np.uint8 else cv2.INTER_LINEAR
    return cv2.resize(mask, (w, h), interpolation=interp)
