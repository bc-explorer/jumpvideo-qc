"""Mask statistics: area, bbox, connected components, IoU, coverage."""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]  # x0, y0, x1, y1


def to_binary(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    if mask is None:
        return None
    if mask.dtype == np.uint8 and mask.max() <= 1:
        return mask
    return (mask >= threshold).astype(np.uint8)


def area(mask: Optional[np.ndarray], threshold: float = 0.5) -> int:
    if mask is None:
        return 0
    b = to_binary(mask, threshold)
    return int(b.sum())


def bbox(mask: Optional[np.ndarray], threshold: float = 0.5) -> Optional[BBox]:
    if mask is None:
        return None
    b = to_binary(mask, threshold)
    ys, xs = np.where(b > 0)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def bbox_center(bb: Optional[BBox]) -> Optional[Tuple[float, float]]:
    if bb is None:
        return None
    x0, y0, x1, y1 = bb
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def iou(a: Optional[np.ndarray], b: Optional[np.ndarray], threshold: float = 0.5) -> float:
    if a is None or b is None:
        return 0.0
    ba = to_binary(a, threshold)
    bb = to_binary(b, threshold)
    if ba.shape != bb.shape:
        bb = cv2.resize(bb, (ba.shape[1], ba.shape[0]), interpolation=cv2.INTER_NEAREST)
    inter = int(np.logical_and(ba, bb).sum())
    union = int(np.logical_or(ba, bb).sum())
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return inter / union


def connected_components(
    mask: Optional[np.ndarray],
    threshold: float = 0.5,
    min_area: int = 64,
) -> List[dict]:
    """Return list of blobs: {area, bbox, centroid}. Background excluded."""
    if mask is None:
        return []
    b = to_binary(mask, threshold)
    if b.sum() == 0:
        return []
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(b, connectivity=8)
    out = []
    for i in range(1, num):  # 0 is background
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < min_area:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = float(centroids[i][0]), float(centroids[i][1])
        out.append(
            {
                "area": a,
                "bbox": (x, y, x + w, y + h),
                "centroid": (cx, cy),
            }
        )
    out.sort(key=lambda d: d["area"], reverse=True)
    return out


def coverage_in_box(
    mask: Optional[np.ndarray],
    box: BBox,
    threshold: float = 0.5,
) -> float:
    """Fraction of pixels inside ``box`` where ``mask >= threshold``."""
    if mask is None:
        return 0.0
    b = to_binary(mask, threshold)
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    h, w = b.shape[:2]
    x0 = max(0, min(x0, w))
    x1 = max(0, min(x1, w))
    y0 = max(0, min(y0, h))
    y1 = max(0, min(y1, h))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    sub = b[y0:y1, x0:x1]
    if sub.size == 0:
        return 0.0
    return float(sub.sum()) / float(sub.size)


def restrict_to_box(mask: Optional[np.ndarray], box: BBox) -> Optional[np.ndarray]:
    """Return a binary copy of ``mask`` with everything outside ``box`` zeroed."""
    if mask is None:
        return None
    b = to_binary(mask, 0.5).copy()
    h, w = b.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0 = max(0, min(x0, w))
    x1 = max(0, min(x1, w))
    y0 = max(0, min(y0, h))
    y1 = max(0, min(y1, h))
    out = np.zeros_like(b)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = b[y0:y1, x0:x1]
    return out


def coverage_in_mask(
    alpha: Optional[np.ndarray],
    region: Optional[np.ndarray],
    threshold: float = 0.5,
) -> float:
    """Fraction of ``region`` pixels covered by ``alpha >= threshold``."""
    if alpha is None or region is None:
        return 0.0
    a = to_binary(alpha, threshold)
    r = to_binary(region, 0.5)
    if a.shape != r.shape:
        a = cv2.resize(a, (r.shape[1], r.shape[0]), interpolation=cv2.INTER_NEAREST)
    denom = int(r.sum())
    if denom == 0:
        return 0.0
    return float(np.logical_and(a, r).sum()) / float(denom)


def median(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.median(np.asarray(values, dtype=np.float64)))


def window_median(
    frames: List[int],
    values: dict,
    center_frame: int,
    fps: float,
    window_sec: float = 1.0,
    exclude_center: bool = True,
) -> Optional[float]:
    """Median of ``values`` over frames within +/- ``window_sec`` of center.

    ``frames`` is the sorted list of sampled frame indices; ``values`` maps
    frame index -> scalar. Returns ``None`` if no neighbours.
    """
    span = max(1.0, fps * window_sec)
    lo, hi = center_frame - span, center_frame + span
    vals = [
        values[f]
        for f in frames
        if lo <= f <= hi and (not exclude_center or f != center_frame) and f in values
    ]
    if not vals:
        return None
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def subtract(mask_a: Optional[np.ndarray], mask_b: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Binary A minus B (A AND NOT B)."""
    if mask_a is None:
        return None
    a = to_binary(mask_a, 0.5)
    if mask_b is None:
        return a
    b = to_binary(mask_b, 0.5)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.logical_and(a, np.logical_not(b)).astype(np.uint8)
