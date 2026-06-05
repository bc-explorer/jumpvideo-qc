#!/usr/bin/env python3
"""Run QC on a task directory."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.pipeline import run_qc


def main() -> int:
    parser = argparse.ArgumentParser(description="Run live-commerce matting QC")
    parser.add_argument("task_root", type=str, help="Path to ui_runs task directory")
    parser.add_argument(
        "--mode",
        choices=["conservative", "balanced", "sensitive"],
        default=None,
        help="Detection mode preset (default: from config or sensitive)",
    )
    parser.add_argument(
        "--no-write-config",
        action="store_true",
        help="Do not write qc_config.yaml into the task dir",
    )
    parser.add_argument(
        "--yolo-frame-policy",
        choices=["all", "smart"],
        default=None,
        help="YOLO sampling policy: all preserves recall; smart runs YOLO on mask-change candidates and baseline frames.",
    )
    parser.add_argument("--json", action="store_true", help="Print summary JSON to stdout")
    args = parser.parse_args()

    task_root = Path(args.task_root).expanduser().resolve()
    if not task_root.is_dir():
        print(f"Not a directory: {task_root}", file=sys.stderr)
        return 1

    cfg = load_config(task_root, mode=args.mode)
    if args.yolo_frame_policy:
        cfg.runtime["yolo_frame_policy"] = args.yolo_frame_policy
    result = run_qc(
        task_root,
        config=cfg,
        mode=args.mode,
        write_config=not args.no_write_config,
    )

    print(
        f"Decision: {result['decision']}  Risk: {result['risk_level']}  "
        f"High={result['summary']['high']} Medium={result['summary']['medium']} "
        f"Warn={result['summary']['warning']}  ({result['elapsed_seconds']}s)"
    )
    perf = result.get("performance", {})
    if perf:
        print(
            f"Perf: {perf['sampled_frames']}/{perf['total_frames']} frames, "
            f"video {perf['video_duration_seconds']}s, "
            f"{perf['processed_mb']} MB read, "
            f"{perf['throughput_mb_s']} MB/s, "
            f"{perf['frames_per_second']} frame/s, "
            f"realtime {perf['realtime_factor']}x"
        )
        stages = perf.get("stage_seconds") or {}
        if stages:
            print(
                "Stages: "
                + ", ".join(f"{name}={seconds}s" for name, seconds in stages.items())
            )
        if "yolo_frames" in perf:
            print(
                f"YOLO: policy={perf.get('yolo_policy')} "
                f"frames={perf.get('yolo_frames')}/{perf.get('sampled_frames')} "
                f"ratio={perf.get('yolo_frame_ratio')}"
            )
    print(f"Report: {task_root / cfg.outputs.get('report_dir', 'qc') / 'report.html'}")

    if args.json:
        slim = {
            k: result[k]
            for k in (
                "decision",
                "risk_level",
                "summary",
                "input_status",
                "blocking_messages",
                "warnings",
                "performance",
                "elapsed_seconds",
            )
        }
        print(json.dumps(slim, indent=2, ensure_ascii=False))

    return 0 if result["decision"] == "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
