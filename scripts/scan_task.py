#!/usr/bin/env python3
"""Scan a task directory and print input_status (no model inference)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.resolver import TaskResolver
from src.scanner import blocking_messages, scan_inputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan task artifacts")
    parser.add_argument("task_root", type=str, help="Path to ui_runs task directory")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON (input_status + resolved_paths)",
    )
    args = parser.parse_args()

    task_root = Path(args.task_root).expanduser().resolve()
    if not task_root.is_dir():
        print(f"Not a directory: {task_root}", file=sys.stderr)
        return 1

    resolver = TaskResolver(task_root)
    resolved = resolver.resolve_and_dump()
    scan = scan_inputs(resolved)

    if args.json:
        out = {
            "task_root": str(task_root),
            "profile": resolved.profile,
            "input_status": scan.input_status,
            "warnings": scan.warnings,
            "blocking_messages": blocking_messages(scan),
            "resolved_paths": resolved.to_dict(),
        }
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0

    print(f"Task: {task_root}")
    print(f"Profile: {resolved.profile}")
    print()
    for key, val in scan.input_status.items():
        st = val.get("status", "?")
        extra = ""
        if val.get("frame_count"):
            extra = f", {val['frame_count']} frames"
        print(f"  {key:28} {st}{extra}")

    msgs = blocking_messages(scan)
    if msgs:
        print("\nNotes:")
        for m in msgs:
            print(f"  - {m}")
    if scan.warnings:
        print("\nWarnings:")
        for w in scan.warnings:
            print(f"  - {w.get('warning')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
