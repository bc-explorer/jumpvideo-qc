"""Subtitle / price-tag ignore handling.

Builds a per-frame "ignore" mask used to suppress false product/object
detections coming from subtitles and price banners. Prefers the upstream
``masks/combined/subtitles`` mask when available; otherwise falls back to fixed
normalized regions (lower third + top banner).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import alpha_loader, mask_stats

Region = Tuple[float, float, float, float]  # x0,y0,x1,y1 normalized


class SubtitleIgnore:
    def __init__(self, config, resolved):
        sub = config.subtitle
        self.use_upstream = bool(sub.get("use_upstream_subtitle_mask", True))
        self.ignore_lower_third = bool(sub.get("ignore_lower_third", True))
        self.lower_third_region: Region = tuple(
            sub.get("lower_third_region", [0.0, 0.72, 1.0, 1.0])
        )
        self.ignore_top_banner = bool(sub.get("ignore_top_banner", True))
        self.top_banner_region: Region = tuple(
            sub.get("top_banner_region", [0.0, 0.0, 1.0, 0.12])
        )
        self.subtitles_dir: Optional[str] = resolved.subtitles_dir
        self.alpha_threshold = float(config.rule("alpha_threshold", 0.5))

    @property
    def has_upstream(self) -> bool:
        return bool(self.use_upstream and self.subtitles_dir)

    def _region_mask(self, hw: Tuple[int, int]) -> np.ndarray:
        h, w = hw
        m = np.zeros((h, w), dtype=np.uint8)
        regions: List[Region] = []
        if self.ignore_lower_third:
            regions.append(self.lower_third_region)
        if self.ignore_top_banner:
            regions.append(self.top_banner_region)
        for (x0, y0, x1, y1) in regions:
            xa, ya = int(round(x0 * w)), int(round(y0 * h))
            xb, yb = int(round(x1 * w)), int(round(y1 * h))
            m[max(0, ya): min(h, yb), max(0, xa): min(w, xb)] = 1
        return m

    def ignore_mask(self, frame_idx: int, hw: Tuple[int, int]) -> np.ndarray:
        """Return a uint8 {0,1} ignore mask sized to ``hw`` for ``frame_idx``."""
        h, w = hw
        ignore = np.zeros((h, w), dtype=np.uint8)

        if self.has_upstream:
            sub = alpha_loader.load_binary(
                self.subtitles_dir, frame_idx, self.alpha_threshold
            )
            if sub is not None:
                sub = alpha_loader.ensure_size(sub, hw)
                ignore = np.logical_or(ignore, sub).astype(np.uint8)
            else:
                # upstream missing for this frame -> fall back to regions
                ignore = np.logical_or(ignore, self._region_mask(hw)).astype(np.uint8)
        else:
            ignore = np.logical_or(ignore, self._region_mask(hw)).astype(np.uint8)

        return ignore

    def apply(self, mask: Optional[np.ndarray], frame_idx: int) -> Optional[np.ndarray]:
        """Subtract the ignore region from ``mask``."""
        if mask is None:
            return None
        hw = mask.shape[:2]
        ig = self.ignore_mask(frame_idx, hw)
        return mask_stats.subtract(mask, ig)
