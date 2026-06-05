"""TaskResolver: auto-discover task artifacts from a single ``task_root``.

Compatible with both ``ui_runs/live_commerce/<task_id>`` and
``ui_runs/person/<task_id>`` layouts. Nothing here fails hard on missing
directories; absence is recorded and surfaced later in ``input_status``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


def _first_existing(candidates: List[Path]) -> Optional[Path]:
    for c in candidates:
        if c.exists():
            return c
    return None


@dataclass
class ResolvedPaths:
    task_root: str
    profile: str = "auto"  # detected: live_commerce | person | unknown

    source_video: Optional[str] = None
    frames_dir: Optional[str] = None

    combined_alpha_dir: Optional[str] = None
    combined_alpha_source: Optional[str] = None  # which candidate won

    human_alpha_dir: Optional[str] = None

    final_keep_dir: Optional[str] = None
    table_products_dir: Optional[str] = None
    host_hand_products_dir: Optional[str] = None
    assistant_person_dir: Optional[str] = None
    assistant_hand_products_dir: Optional[str] = None
    subtitles_dir: Optional[str] = None

    sam2_dir: Optional[str] = None
    sam2_source: Optional[str] = None

    sam2_prompts: Optional[str] = None
    sam2_object_prompts: Optional[str] = None
    auto_candidates: Optional[str] = None
    sam2_manifest: Optional[str] = None

    outputs_manifest: Optional[str] = None
    task_json: Optional[str] = None

    composited_video: Optional[str] = None  # resolved from outputs/manifest.json

    # bookkeeping
    missing: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


class TaskResolver:
    """Resolve every artifact path from a single ``task_root``."""

    def __init__(self, task_root: str | Path):
        self.task_root = Path(task_root).expanduser().resolve()

    # public API ------------------------------------------------------------
    def resolve(self) -> ResolvedPaths:
        root = self.task_root
        rp = ResolvedPaths(task_root=str(root))

        rp.source_video = self._opt(root / "input" / "source.mp4")
        rp.frames_dir = self._opt(root / "frames" / "source", is_dir=True)

        # combined alpha priority: matanyone2 -> person
        combined, combined_src = self._priority_dir(
            [
                ("matanyone2", root / "foreground" / "matanyone2" / "combined_alpha"),
                ("person", root / "foreground" / "person" / "combined_alpha"),
            ]
        )
        rp.combined_alpha_dir = combined
        rp.combined_alpha_source = combined_src

        rp.human_alpha_dir = self._opt(
            root / "matting" / "person" / "source_person" / "pha", is_dir=True
        )

        masks_combined = root / "masks" / "combined"
        rp.final_keep_dir = self._opt(masks_combined / "final_keep", is_dir=True)
        rp.table_products_dir = self._opt(masks_combined / "table_products", is_dir=True)
        rp.host_hand_products_dir = self._opt(
            masks_combined / "host_hand_products", is_dir=True
        )
        rp.assistant_person_dir = self._opt(
            masks_combined / "assistant_person", is_dir=True
        )
        rp.assistant_hand_products_dir = self._opt(
            masks_combined / "assistant_hand_products", is_dir=True
        )
        rp.subtitles_dir = self._opt(masks_combined / "subtitles", is_dir=True)

        # sam2 priority: masks/sam2 -> masks/sam2_person_seed
        sam2, sam2_src = self._priority_dir(
            [
                ("sam2", root / "masks" / "sam2"),
                ("sam2_person_seed", root / "masks" / "sam2_person_seed"),
            ]
        )
        rp.sam2_dir = sam2
        rp.sam2_source = sam2_src

        prompts = root / "prompts"
        rp.sam2_prompts = self._opt(prompts / "sam2_prompts.json")
        rp.sam2_object_prompts = self._opt(prompts / "sam2_object_prompts.json")
        rp.auto_candidates = self._opt(prompts / "auto_candidates.json")
        rp.sam2_manifest = self._opt(root / "masks" / "sam2" / "manifest.json")

        rp.outputs_manifest = self._opt(root / "outputs" / "manifest.json")
        rp.task_json = self._opt(root / "task.json")

        rp.profile = self._detect_profile(rp)
        rp.composited_video = self._resolve_composited(rp)

        return rp

    def resolve_and_dump(self, out_path: Optional[Path] = None) -> ResolvedPaths:
        rp = self.resolve()
        out_path = out_path or (self.task_root / "resolved_paths.json")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(rp.to_dict(), f, indent=2, ensure_ascii=False)
        return rp

    # internals -------------------------------------------------------------
    def _opt(self, path: Path, is_dir: bool = False) -> Optional[str]:
        ok = path.is_dir() if is_dir else path.exists()
        if ok:
            return str(path)
        return None

    def _priority_dir(self, candidates):
        for name, path in candidates:
            if path.is_dir():
                return str(path), name
        return None, None

    def _detect_profile(self, rp: ResolvedPaths) -> str:
        # explicit hint from path
        parts = [p.lower() for p in self.task_root.parts]
        if "live_commerce" in parts:
            return "live_commerce"
        if "person" in parts:
            return "person"
        # heuristic: presence of live-commerce-only mask dirs
        live_signals = [
            rp.table_products_dir,
            rp.host_hand_products_dir,
            rp.assistant_person_dir,
            rp.assistant_hand_products_dir,
        ]
        if any(live_signals):
            return "live_commerce"
        return "person"

    def _resolve_composited(self, rp: ResolvedPaths) -> Optional[str]:
        candidates: List[str] = []
        if rp.outputs_manifest:
            try:
                with open(rp.outputs_manifest, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                candidates = self._collect_video_candidates(manifest)
            except (json.JSONDecodeError, OSError):
                candidates = []

        # Prefer manifest candidates that actually resolve to an existing file.
        # Skip template strings like "{task_id}_{index:03d}_{bg}.mp4".
        for cand in candidates:
            if "{" in cand or "}" in cand:
                continue
            p = Path(cand)
            if not p.is_absolute():
                p = self.task_root / cand
            if p.exists():
                return str(p)

        # Fallback: any real video sitting in the outputs/ directory.
        out_dir = self.task_root / "outputs"
        if out_dir.is_dir():
            vids: List[Path] = []
            for ext in ("*.mp4", "*.mov", "*.mkv"):
                vids.extend(sorted(out_dir.glob(ext)))
            if vids:
                return str(vids[0])

        return None

    _VIDEO_KEYS = (
        "output",
        "composited_video",
        "output_video",
        "final_video",
        "result_video",
        "video",
        "file",
        "path",
    )
    _TEMPLATE_KEYS = ("output_template", "template", "pattern", "name_template")

    @staticmethod
    def _collect_video_candidates(manifest) -> List[str]:
        """Ordered, de-duplicated list of video paths referenced by a manifest.

        Explicit output keys are walked first; obvious template fields are
        skipped. Field names are not standardised upstream, so this stays
        defensive and tolerant of nested dict/list structures.
        """
        out: List[str] = []
        seen = set()

        def add(s: str) -> None:
            if s not in seen:
                seen.add(s)
                out.append(s)

        def walk(node) -> None:
            if isinstance(node, str):
                if node.lower().endswith((".mp4", ".mov", ".mkv")):
                    add(node)
            elif isinstance(node, dict):
                for k in TaskResolver._VIDEO_KEYS:
                    if k in node:
                        walk(node[k])
                for k, v in node.items():
                    if k in TaskResolver._VIDEO_KEYS or k in TaskResolver._TEMPLATE_KEYS:
                        continue
                    walk(v)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(manifest)
        return out
