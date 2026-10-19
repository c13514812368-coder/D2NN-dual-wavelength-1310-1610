from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Dict
from typing import Mapping

import yaml

from .networkangler import NetworkShape


def load_project_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in config file {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise TypeError(
            f"Top-level config in {path} must be a mapping, got {type(raw).__name__}"
        )

    cfg = dict(raw)
    for key in (
        "meta",
        "shape",
        "fdtd",
        "mapping",
        "source",
        "train",
        "validation",
        "hybrid",
        "export",
    ):
        value = cfg.get(key)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise TypeError(
                f"Config section {key!r} in {path} must be a mapping, got {type(value).__name__}"
            )
    return cfg


def get_meta(config: Mapping[str, Any]) -> Dict[str, Any]:
    raw = config.get("meta") or {}
    if not isinstance(raw, dict):
        raise TypeError("meta section must be a mapping")
    return dict(raw)


def get_seed(config: Mapping[str, Any]) -> int:
    return int(get_meta(config).get("seed", 0))


def build_network_shape(config: Mapping[str, Any]) -> NetworkShape:
    shape_cfg = config.get("shape") or {}
    if not isinstance(shape_cfg, dict):
        raise TypeError(f"shape section must be a mapping, got {type(shape_cfg).__name__}")

    required = (
        "num_layers",
        "neurons_per_layer",
        "total_length_um",
        "total_width_um",
        "inter_layer_spacings_um",
    )
    missing = [key for key in required if key not in shape_cfg]
    if missing:
        raise KeyError(
            "shape section missing required keys: "
            f"{missing}. present_keys={sorted(shape_cfg.keys())}"
        )
    return NetworkShape(**shape_cfg)


__all__ = [
    "build_network_shape",
    "get_meta",
    "get_seed",
    "load_project_config",
]
