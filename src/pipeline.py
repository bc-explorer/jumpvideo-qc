"""QC pipeline orchestrator: resolve -> scan -> detect -> report."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from . import (
    alpha_loader,
    auto_candidates,
    face_hand_qc,
    frame_index,
    matanyone_qc,
    mask_stats,
    perf,
    person_qc,
    product_qc,
    sam2_qc,
)
from .config import QCConfig, load_config
from .findings import Finding
from .preview import render_all
from .report import decide, summarize, write_csv, write_html, write_json
from .resolver import ResolvedPaths, TaskResolver
from .scanner import blocking_messages, scan_inputs
from .segment_merge import merge_findings
from .subtitle_ignore import SubtitleIgnore
from .video_meta import resolve_meta
from .yolo_auditor import YoloAuditor


def _sample_frames(
    frames_dir: Optional[str],
    stride: int,
    fallback_dirs: Optional[List[Optional[str]]] = None,
) -> List[int]:
    """Sample frame indices from ``frames_dir``, or the first non-empty fallback."""
    dirs = [frames_dir] + list(fallback_dirs or [])
    indices: List[int] = []
    for d in dirs:
        if not d:
            continue
        indices = frame_index.frame_indices(Path(d))
        if indices:
            break
    if not indices:
        return []
    return indices[:: max(1, stride)]


def _mask_areas_over_frames(
    mask_dir: Optional[str],
    frames: List[int],
    threshold: float,
) -> Dict[int, int]:
    out: Dict[int, int] = {}
    if not mask_dir:
        return out
    for f in frames:
        m = alpha_loader.load_binary(mask_dir, f, threshold)
        out[f] = mask_stats.area(m, threshold)
    return out


def _final_keep_blob_counts(
    final_keep_dir: Optional[str],
    subtitles_dir: Optional[str],
    subtitle_ignore: SubtitleIgnore,
    frames: List[int],
    hw: Tuple[int, int],
    threshold: float,
) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    if not final_keep_dir:
        return counts
    for f in frames:
        fk = alpha_loader.load_binary(final_keep_dir, f, threshold)
        if fk is None:
            counts[f] = 0
            continue
        fk = subtitle_ignore.apply(fk, f)
        blobs = mask_stats.connected_components(fk, threshold, min_area=128)
        counts[f] = len(blobs)
    return counts


def _add_window(
    selected: set[int],
    frames: List[int],
    index_by_frame: Dict[int, int],
    frame_idx: int,
    radius: int,
) -> None:
    pos = index_by_frame.get(frame_idx)
    if pos is None:
        return
    for i in range(max(0, pos - radius), min(len(frames), pos + radius + 1)):
        selected.add(frames[i])


def _mask_change_candidates(
    frames: List[int],
    values: Dict[int, int],
    fps: float,
    change_ratio: float,
    min_ref: int = 64,
) -> set[int]:
    out: set[int] = set()
    for f in frames:
        cur = values.get(f, 0)
        ref = mask_stats.window_median(frames, values, f, fps, window_sec=1.0)
        if ref is None or ref < min_ref:
            continue
        ratio = abs(cur - ref) / max(ref, 1.0)
        if ratio >= change_ratio:
            out.add(f)
    return out


def _select_yolo_frames(
    frames: List[int],
    fps: float,
    cfg: QCConfig,
    area_series: List[Dict[int, int]],
    blob_counts: Dict[int, int],
) -> set[int]:
    """Choose frames for YOLO. ``all`` preserves full recall; ``smart`` samples candidates."""
    policy = str(cfg.runtime.get("yolo_frame_policy", "all")).lower()
    if policy in {"all", "full", "every"}:
        return set(frames)
    if not frames:
        return set()

    index_by_frame = {frame: i for i, frame in enumerate(frames)}
    selected: set[int] = {frames[0], frames[-1]}

    baseline_seconds = float(cfg.runtime.get("yolo_baseline_seconds", 2.0))
    baseline_step = max(1, int(round((fps or 25.0) * baseline_seconds / max(1, int(cfg.runtime.get("frame_stride", 1))))))
    for i in range(0, len(frames), baseline_step):
        selected.add(frames[i])

    change_ratio = float(cfg.runtime.get("yolo_mask_change_ratio", 0.35))
    window = max(0, int(cfg.runtime.get("yolo_candidate_window", 1)))
    candidates: set[int] = set()
    for series in area_series:
        candidates.update(_mask_change_candidates(frames, series, fps, change_ratio))
    if blob_counts:
        candidates.update(_mask_change_candidates(frames, blob_counts, fps, change_ratio, min_ref=1))

    for frame_idx in candidates:
        _add_window(selected, frames, index_by_frame, frame_idx, window)

    return selected


def run_qc(
    task_root: str | Path,
    config: Optional[QCConfig] = None,
    mode: Optional[str] = None,
    write_config: bool = True,
) -> Dict:
    """Run full QC on ``task_root`` and write outputs under ``<task_root>/qc/``."""
    wall_start = time.time()
    perf_start = time.perf_counter()
    stage_seconds = {
        "setup": 0.0,
        "mask_io": 0.0,
        "yolo": 0.0,
        "sam2": 0.0,
        "preview": 0.0,
        "report": 0.0,
    }
    task_root = Path(task_root).expanduser().resolve()
    cfg = config or load_config(task_root, mode=mode)

    if write_config:
        cfg.dump_yaml(task_root / "qc_config.yaml")

    resolver = TaskResolver(task_root)
    resolved = resolver.resolve_and_dump()
    scan = scan_inputs(resolved)
    meta = resolve_meta(resolved.source_video, resolved.frames_dir)
    stride = int(cfg.runtime.get("frame_stride", 5))
    alpha_thr = float(cfg.rule("alpha_threshold", 0.5))

    frames = _sample_frames(
        resolved.frames_dir,
        stride,
        fallback_dirs=[
            resolved.combined_alpha_dir,
            resolved.human_alpha_dir,
            resolved.final_keep_dir,
        ],
    )
    if meta.frame_count <= 0 and frames:
        meta.frame_count = max(frames) + 1

    subtitle_ignore = SubtitleIgnore(cfg, resolved)
    hw = alpha_loader.frame_size(resolved.frames_dir)
    if hw is None and resolved.source_video:
        import cv2

        cap = cv2.VideoCapture(resolved.source_video)
        if cap.isOpened():
            hw = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        cap.release()
    frame_wh = (hw[1], hw[0]) if hw else (1920, 1080)
    fps = meta.fps or 25.0

    findings: List[Finding] = []
    warnings: List[dict] = list(scan.warnings)

    # auto candidate boxes (static, reused per frame)
    auto_boxes = auto_candidates.load_non_person_boxes(resolved.auto_candidates)

    # temporal series
    human_area: Dict[int, int] = {}
    human_iou_prev: Dict[int, float] = {}
    person_present: Dict[int, bool] = {}
    audit_cache = {}
    prev_human = None

    t_stage = time.perf_counter()
    host_areas = _mask_areas_over_frames(resolved.host_hand_products_dir, frames, alpha_thr)
    assistant_areas = _mask_areas_over_frames(
        resolved.assistant_hand_products_dir, frames, alpha_thr
    )
    table_areas = _mask_areas_over_frames(resolved.table_products_dir, frames, alpha_thr)
    final_keep_blobs = _final_keep_blob_counts(
        resolved.final_keep_dir,
        resolved.subtitles_dir,
        subtitle_ignore,
        frames,
        hw or (frame_wh[1], frame_wh[0]),
        alpha_thr,
    )
    stage_seconds["mask_io"] += time.perf_counter() - t_stage
    yolo_frames = _select_yolo_frames(
        frames,
        fps,
        cfg,
        [host_areas, assistant_areas, table_areas],
        final_keep_blobs,
    )

    auditor: Optional[YoloAuditor] = None
    yolo_ok = scan.is_ok("frames_dir") or scan.is_ok("source_video")
    can_yolo = yolo_ok and scan.is_ok("combined_alpha_dir")
    if can_yolo:
        auditor = YoloAuditor(cfg)
        if not auditor.available():
            warnings.append({"warning": "yolo_unavailable", "detail": auditor._load_error})
            auditor = None

    has_human = scan.is_ok("human_alpha_dir")
    has_combined = scan.is_ok("combined_alpha_dir")
    stage_seconds["setup"] = time.perf_counter() - perf_start - stage_seconds["mask_io"]

    t0 = time.time()
    batch_size = max(1, int(cfg.runtime.get("batch_size", 1)))
    pbar = tqdm(total=len(frames), desc="QC frames", unit="frame") if frames else None
    for start in range(0, len(frames), batch_size):
        chunk = frames[start : start + batch_size]
        records = []
        for frame_idx in chunk:
            t_stage = time.perf_counter()
            combined = (
                alpha_loader.load_binary(resolved.combined_alpha_dir, frame_idx, alpha_thr)
                if has_combined
                else None
            )
            human = (
                alpha_loader.load_binary(resolved.human_alpha_dir, frame_idx, alpha_thr)
                if has_human
                else None
            )
            if hw and combined is not None:
                combined = alpha_loader.ensure_size(combined, hw)
            if hw and human is not None:
                human = alpha_loader.ensure_size(human, hw)

            if has_human:
                human_area[frame_idx] = mask_stats.area(human, alpha_thr)
                if prev_human is not None and human is not None:
                    human_iou_prev[frame_idx] = mask_stats.iou(human, prev_human, alpha_thr)
                prev_human = human

            assistant_mask = None
            if resolved.assistant_person_dir:
                assistant_mask = alpha_loader.load_binary(
                    resolved.assistant_person_dir, frame_idx, alpha_thr
                )
                if hw and assistant_mask is not None:
                    assistant_mask = alpha_loader.ensure_size(assistant_mask, hw)
            stage_seconds["mask_io"] += time.perf_counter() - t_stage

            frame_bgr = None
            if auditor is not None and frame_idx in yolo_frames:
                t_stage = time.perf_counter()
                frame_bgr = alpha_loader.load_frame_bgr(resolved.frames_dir, frame_idx)
                stage_seconds["yolo"] += time.perf_counter() - t_stage
            if auditor is not None and frame_idx in yolo_frames and frame_bgr is None and resolved.source_video:
                import cv2

                t_stage = time.perf_counter()
                cap = cv2.VideoCapture(resolved.source_video)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame_bgr = cap.read()
                cap.release()
                if not ok:
                    frame_bgr = None
                stage_seconds["yolo"] += time.perf_counter() - t_stage

            records.append(
                {
                    "frame_idx": frame_idx,
                    "combined": combined,
                    "human": human,
                    "assistant_mask": assistant_mask,
                    "frame_bgr": frame_bgr,
                }
            )

        audit_records = [record for record in records if auditor is not None and record["frame_bgr"] is not None]
        t_stage = time.perf_counter()
        audits = auditor.audit_frames([record["frame_bgr"] for record in audit_records]) if audit_records and auditor is not None else []
        stage_seconds["yolo"] += time.perf_counter() - t_stage
        for record, audit in zip(audit_records, audits):
            record["audit"] = audit

        for record in records:
            frame_idx = record["frame_idx"]
            combined = record["combined"]
            human = record["human"]
            assistant_mask = record["assistant_mask"]
            audit = record.get("audit")
            if audit is not None:
                audit_cache[frame_idx] = audit
                person_present[frame_idx] = len(audit.persons) > 0

                findings.extend(
                    person_qc.check_frame(
                        frame_idx,
                        audit,
                        combined,
                        human,
                        assistant_mask,
                        has_human,
                        cfg,
                    )
                )
                findings.extend(
                    face_hand_qc.check_frame(frame_idx, audit, combined, cfg)
                )
                findings.extend(
                    product_qc.check_frame_near_hand(
                        frame_idx,
                        audit,
                        combined,
                        auto_boxes,
                        frame_wh,
                        cfg,
                    )
                )
            elif human is not None and combined is not None:
                # matanyone person missing without yolo: low human area vs combined
                ha = mask_stats.area(human, alpha_thr)
                ca = mask_stats.area(combined, alpha_thr)
                if ca > 500 and ha < ca * 0.3:
                    from .findings import MEDIUM, Finding

                    findings.append(
                        Finding(
                            frame_idx,
                            "matanyone_person_missing",
                            MEDIUM,
                            "matanyone_qc",
                            {"human_area": ha, "combined_area": ca},
                        )
                    )
        if pbar is not None:
            pbar.update(len(chunk))
    if pbar is not None:
        pbar.close()

    # temporal product / matanyone
    if host_areas:
        findings.extend(
            product_qc.check_drops(
                frames,
                host_areas,
                fps,
                float(cfg.rule("host_product_drop_ratio", 0.40)),
                "host_product_drop",
                "high",
                "product_qc",
            )
        )
    if assistant_areas:
        findings.extend(
            product_qc.check_drops(
                frames,
                assistant_areas,
                fps,
                float(cfg.rule("assistant_product_drop_ratio", 0.40)),
                "assistant_product_drop",
                "high",
                "product_qc",
            )
        )
    if table_areas:
        findings.extend(
            product_qc.check_drops(
                frames,
                table_areas,
                fps,
                float(cfg.rule("table_product_drop_ratio", 0.50)),
                "table_product_drop",
                "medium",
                "product_qc",
            )
        )
    if final_keep_blobs:
        findings.extend(
            product_qc.check_final_keep_object_drop(frames, final_keep_blobs, fps)
        )

    if has_human:
        findings.extend(
            matanyone_qc.check_series(
                frames, human_area, human_iou_prev, person_present, fps, cfg
            )
        )

    sam2_stats: List[dict] = []
    if scan.is_ok("sam2_dir"):
        t_stage = time.perf_counter()
        sam2_findings, sam2_stats = sam2_qc.analyze(
            resolved, frames, fps, frame_wh, cfg
        )
        stage_seconds["sam2"] += time.perf_counter() - t_stage
        findings.extend(sam2_findings)

    segments = merge_findings(findings, fps, cfg, frame_count=meta.frame_count or 0)

    report_dir = task_root / cfg.outputs.get("report_dir", "qc")
    report_dir.mkdir(parents=True, exist_ok=True)

    if cfg.outputs.get("write_previews", True):
        t_stage = time.perf_counter()
        render_all(
            segments,
            resolved,
            cfg,
            subtitle_ignore,
            auditor,
            report_dir,
            max_previews=int(cfg.outputs.get("max_previews", 200)),
            audit_cache=audit_cache,
        )
        stage_seconds["preview"] += time.perf_counter() - t_stage

    summary = summarize(segments)
    decision_info = decide(summary)

    elapsed_total = time.time() - wall_start
    performance = perf.build_report(
        resolved,
        frames,
        elapsed_seconds=elapsed_total,
        fps=fps,
        total_frames=meta.frame_count or (max(frames) + 1 if frames else 0),
    )
    performance["yolo_policy"] = str(cfg.runtime.get("yolo_frame_policy", "all"))
    performance["yolo_frames"] = len(yolo_frames) if auditor is not None else 0
    performance["yolo_frame_ratio"] = round(
        (len(yolo_frames) / len(frames)) if frames and auditor is not None else 0.0,
        3,
    )
    performance["stage_seconds"] = {key: round(value, 3) for key, value in stage_seconds.items()}

    result = {
        "task_root": str(task_root),
        "profile": resolved.profile,
        "mode": cfg.mode,
        "resolved_paths": resolved.to_dict(),
        "input_status": scan.input_status,
        "blocking_messages": blocking_messages(scan),
        "video_meta": meta.to_dict(),
        "sampled_frames": len(frames),
        "frame_stride": stride,
        "findings_count": len(findings),
        "warnings": warnings,
        "sam2_object_stats": sam2_stats,
        "summary": summary,
        "decision": decision_info["decision"],
        "risk_level": decision_info["risk_level"],
        "failed_segments": [s.to_dict() for s in segments],
        "performance": performance,
        "detect_seconds": round(time.time() - t0, 2),
        "elapsed_seconds": round(elapsed_total, 2),
    }

    t_stage = time.perf_counter()
    if cfg.outputs.get("write_csv", True):
        write_csv(segments, report_dir / "failed_segments.csv")
    if cfg.outputs.get("write_html", True):
        write_html(result, segments, report_dir / "report.html")
    stage_seconds["report"] += time.perf_counter() - t_stage
    performance["stage_seconds"] = {key: round(value, 3) for key, value in stage_seconds.items()}

    if cfg.outputs.get("write_json", True):
        write_json(result, report_dir / "report.json")

    return result
