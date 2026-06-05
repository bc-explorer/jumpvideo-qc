"""YOLO11 person/object detection + pose audit.

Ultralytics + torch are imported lazily so the rest of the pipeline (resolver,
scanner, mask stats) works even on a machine without the model stack. Device is
auto-detected: cuda -> mps -> cpu.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# COCO keypoint indices (yolo pose)
KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
FACE_KPS = [KP_NOSE, KP_LEFT_EYE, KP_RIGHT_EYE, KP_LEFT_EAR, KP_RIGHT_EAR]
WRIST_KPS = [KP_LEFT_WRIST, KP_RIGHT_WRIST]

BBox = Tuple[float, float, float, float]


@dataclass
class Detection:
    box: BBox  # x0,y0,x1,y1
    conf: float
    cls: int
    name: str
    mask: Optional[np.ndarray] = None  # binary instance mask at frame resolution


@dataclass
class PoseResult:
    box: BBox
    conf: float
    keypoints: List[Tuple[float, float, float]] = field(default_factory=list)  # x,y,conf


@dataclass
class FrameAudit:
    persons: List[Detection] = field(default_factory=list)
    objects: List[Detection] = field(default_factory=list)  # non-person classes
    poses: List[PoseResult] = field(default_factory=list)


def auto_device(requested: str = "auto") -> str:
    if requested and requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class YoloAuditor:
    def __init__(self, config):
        rt = config.runtime
        # Primary model is an instance-segmentation model so that person/object
        # coverage can use the actual instance mask as denominator (not the
        # rectangular bbox, which is mostly background for tall subjects).
        self.seg_model_name = rt.get("yolo_seg_model", "yolo11s-seg.pt")
        self.det_model_name = rt.get("yolo_det_model", "yolo11s.pt")
        self.pose_model_name = rt.get("yolo_pose_model", "yolo11n-pose.pt")
        self.imgsz = int(rt.get("image_size", 640))
        self.device = auto_device(rt.get("device", "auto"))
        self.conf = float(config.rule("yolo_conf", 0.12))
        self._seg = None
        self._pose = None
        self._loaded = False
        self._load_error: Optional[str] = None

    # ----------------------------------------------------------------------
    def available(self) -> bool:
        try:
            import ultralytics  # noqa: F401
            import torch  # noqa: F401

            return True
        except Exception as exc:  # pragma: no cover - env dependent
            self._load_error = str(exc)
            return False

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        from ultralytics import YOLO

        self._seg = YOLO(self.seg_model_name)
        self._pose = YOLO(self.pose_model_name)
        self._loaded = True

    @staticmethod
    def _poly_to_mask(poly, hw: Tuple[int, int]) -> Optional[np.ndarray]:
        import cv2

        if poly is None or len(poly) < 3:
            return None
        h, w = hw
        m = np.zeros((h, w), dtype=np.uint8)
        pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(m, [pts], 1)
        return m

    # ----------------------------------------------------------------------
    def audit_frame(self, image_bgr: np.ndarray) -> FrameAudit:
        self.ensure_loaded()
        audit = FrameAudit()
        h, w = image_bgr.shape[:2]

        seg_res = self._seg.predict(
            image_bgr, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False
        )[0]
        names = seg_res.names
        # polygons in original-image coordinates, one per instance (or None)
        polys = None
        if seg_res.masks is not None:
            try:
                polys = seg_res.masks.xy
            except Exception:
                polys = None
        if seg_res.boxes is not None:
            for i, b in enumerate(seg_res.boxes):
                xyxy = b.xyxy[0].tolist()
                cls = int(b.cls[0].item())
                conf = float(b.conf[0].item())
                name = names.get(cls, str(cls)) if isinstance(names, dict) else str(cls)
                mask = None
                if polys is not None and i < len(polys):
                    mask = self._poly_to_mask(polys[i], (h, w))
                det = Detection(box=tuple(xyxy), conf=conf, cls=cls, name=name, mask=mask)
                if cls == 0:  # person
                    audit.persons.append(det)
                else:
                    audit.objects.append(det)

        pose_res = self._pose.predict(
            image_bgr, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False
        )[0]
        if pose_res.boxes is not None and pose_res.keypoints is not None:
            kpts = pose_res.keypoints.data.cpu().numpy()  # (n, 17, 3)
            for i, b in enumerate(pose_res.boxes):
                xyxy = b.xyxy[0].tolist()
                conf = float(b.conf[0].item())
                kp_list = []
                if i < len(kpts):
                    for (x, y, c) in kpts[i]:
                        kp_list.append((float(x), float(y), float(c)))
                audit.poses.append(
                    PoseResult(box=tuple(xyxy), conf=conf, keypoints=kp_list)
                )

        audit.persons.sort(key=lambda d: _box_area(d.box), reverse=True)
        return audit


def _box_area(box: BBox) -> float:
    x0, y0, x1, y1 = box
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def person_union_mask(audit: "FrameAudit", hw: Tuple[int, int]) -> Optional[np.ndarray]:
    """Union of all person instance masks, or ``None`` if no masks available."""
    h, w = hw
    union: Optional[np.ndarray] = None
    for det in audit.persons:
        if det.mask is None:
            continue
        m = det.mask
        if m.shape[:2] != (h, w):
            import cv2

            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        union = m.copy() if union is None else np.logical_or(union, m).astype(np.uint8)
    return union


def face_roi_from_pose(pose: PoseResult, kp_conf: float = 0.2) -> Optional[BBox]:
    """Estimate a face bounding box from head keypoints."""
    pts = [
        (x, y)
        for i, (x, y, c) in enumerate(pose.keypoints)
        if i in FACE_KPS and c >= kp_conf
    ]
    if len(pts) < 2:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    # head box is roughly 1.6x the eye/ear span; pad generously
    pad_x = max(w * 0.6, 12.0)
    pad_y = max(h * 1.0, 16.0)
    return (min(xs) - pad_x, min(ys) - pad_y, max(xs) + pad_x, max(ys) + pad_y)


def hand_rois_from_pose(
    pose: PoseResult, kp_conf: float = 0.2, radius: float = 40.0
) -> List[BBox]:
    """Estimate small ROIs around each detected wrist."""
    rois: List[BBox] = []
    for i in WRIST_KPS:
        if i >= len(pose.keypoints):
            continue
        x, y, c = pose.keypoints[i]
        if c < kp_conf:
            continue
        rois.append((x - radius, y - radius, x + radius, y + radius))
    return rois
