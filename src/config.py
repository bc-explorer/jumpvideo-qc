"""Configuration loading + detection-mode presets.

A single ``QCConfig`` object is threaded through the whole pipeline. It is
loaded from ``config/qc_config.default.yaml`` and may be overridden by a
task-local ``qc_config.yaml`` (written by the UI) and by a chosen detection
mode preset (conservative / balanced / sensitive).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml

_PKG_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = _PKG_ROOT / "config" / "qc_config.default.yaml"


# Mode presets only override a small, curated set of knobs. Everything else
# comes from the base config so users never have to tune dozens of thresholds.
MODE_PRESETS: Dict[str, Dict[str, Any]] = {
    "conservative": {
        "runtime": {"frame_stride": 10},
        "rules": {
            "yolo_conf": 0.30,
            "person_combined_coverage_warn": 0.65,
            "face_coverage_fail": 0.65,
            "hand_coverage_fail": 0.50,
            "product_near_hand_coverage_fail": 0.40,
            "mask_area_drop_ratio": 0.50,
        },
    },
    "balanced": {
        "runtime": {"frame_stride": 5},
        "rules": {
            "yolo_conf": 0.20,
            "person_combined_coverage_warn": 0.75,
            "face_coverage_fail": 0.75,
            "hand_coverage_fail": 0.60,
            "product_near_hand_coverage_fail": 0.50,
            "mask_area_drop_ratio": 0.60,
        },
    },
    "sensitive": {
        "runtime": {"frame_stride": 5},
        "rules": {
            "yolo_conf": 0.12,
            "person_combined_coverage_warn": 0.80,
            "face_coverage_fail": 0.80,
            "hand_coverage_fail": 0.65,
            "product_near_hand_coverage_fail": 0.55,
            "mask_area_drop_ratio": 0.65,
        },
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


@dataclass
class QCConfig:
    """In-memory configuration. Access nested values via the sub-dicts."""

    data: Dict[str, Any] = field(default_factory=dict)

    # convenience accessors -------------------------------------------------
    @property
    def task(self) -> Dict[str, Any]:
        return self.data.setdefault("task", {})

    @property
    def paths(self) -> Dict[str, Any]:
        return self.data.setdefault("paths", {})

    @property
    def runtime(self) -> Dict[str, Any]:
        return self.data.setdefault("runtime", {})

    @property
    def subtitle(self) -> Dict[str, Any]:
        return self.data.setdefault("subtitle", {})

    @property
    def rules(self) -> Dict[str, Any]:
        return self.data.setdefault("rules", {})

    @property
    def segment(self) -> Dict[str, Any]:
        return self.data.setdefault("segment", {})

    @property
    def outputs(self) -> Dict[str, Any]:
        return self.data.setdefault("outputs", {})

    @property
    def mode(self) -> str:
        return self.data.get("mode", "sensitive")

    # rule helper -----------------------------------------------------------
    def rule(self, name: str, default: Any = None) -> Any:
        return self.rules.get(name, default)

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.data)

    def dump_yaml(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.data, f, sort_keys=False, allow_unicode=True)


def load_default() -> Dict[str, Any]:
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_mode(data: Dict[str, Any], mode: str) -> Dict[str, Any]:
    preset = MODE_PRESETS.get(mode)
    if preset is None:
        return data
    return _deep_merge(data, preset)


def load_config(
    task_root: str | Path | None = None,
    overrides: Dict[str, Any] | None = None,
    mode: str | None = None,
) -> QCConfig:
    """Build a :class:`QCConfig`.

    Resolution order (lowest to highest precedence):
    1. packaged default yaml
    2. task-local ``qc_config.yaml`` (if present under ``task_root``)
    3. detection-mode preset
    4. explicit ``overrides`` dict
    """
    data = load_default()

    if task_root is not None:
        local = Path(task_root) / "qc_config.yaml"
        if local.is_file():
            with local.open("r", encoding="utf-8") as f:
                local_data = yaml.safe_load(f) or {}
            data = _deep_merge(data, local_data)

    chosen_mode = mode or data.get("mode", "sensitive")
    data["mode"] = chosen_mode
    data = apply_mode(data, chosen_mode)

    if overrides:
        data = _deep_merge(data, overrides)

    if task_root is not None:
        data.setdefault("task", {})["root"] = str(task_root)

    return QCConfig(data=data)
