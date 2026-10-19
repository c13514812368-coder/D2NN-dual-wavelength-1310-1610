from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Dict
from typing import Mapping

from .fdtd_pipeline import (
    BidirLayoutConfig,
    MappingFactorConfig,
    MonitorExtractConfig,
    PipelineConfig,
)
from .project_paths import resolve_repo_path


BASE_FDTD_LSF_NAMES = (
    "network_simulate",
    "network_bidir",
    "calc_neff_mode_xprop",
)


def resolve_fdtd_base_lsf_templates(
    config_path: str | Path,
    *,
    project_root: Path | None = None,
) -> Dict[str, Path]:
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = resolve_repo_path(cfg_path, project_root=project_root)
    cfg_path = cfg_path.resolve()
    out: Dict[str, Path] = {}
    missing: list[Path] = []
    for name in BASE_FDTD_LSF_NAMES:
        candidate = (cfg_path.parent / f"{cfg_path.stem}.{name}.lsf").resolve()
        if candidate.exists():
            out[name] = candidate
        else:
            missing.append(candidate)
    if missing:
        expected = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Missing required base LSF sidecar file(s) for config "
            f"{cfg_path}: {expected}. "
            f"Run `python scripts/sync_base_lsf.py --config {cfg_path}` to create/update only this config's paired sidecars."
        )
    return out


def get_cache_index_path(
    config: Mapping[str, Any],
    *,
    project_root: Path | None = None,
) -> Path:
    fdtd_cfg = config.get("fdtd") or {}
    cache_value = fdtd_cfg.get("cache_index", "runs/cache/simulation_index.json")
    return resolve_repo_path(cache_value, project_root=project_root)


def build_fdtd_pipeline_config(
    config: Mapping[str, Any],
    *,
    runs_dir: Path,
    config_path: str | Path,
    project_root: Path | None = None,
) -> PipelineConfig:
    fdtd_cfg = config.get("fdtd") or {}
    if not isinstance(fdtd_cfg, dict):
        raise TypeError(f"fdtd section must be a mapping, got {type(fdtd_cfg).__name__}")

    bidir_cfg = fdtd_cfg.get("bidir")
    monitor_cfg = fdtd_cfg.get("monitor")
    mapping_cfg = fdtd_cfg.get("mapping")
    for section_name, section_value in (
        ("fdtd.bidir", bidir_cfg),
        ("fdtd.monitor", monitor_cfg),
        ("fdtd.mapping", mapping_cfg),
    ):
        if section_value is None:
            continue
        if not isinstance(section_value, dict):
            raise TypeError(
                f"{section_name} in {Path(config_path).resolve()} must be a mapping, "
                f"got {type(section_value).__name__}"
            )

    return PipelineConfig(
        data_raw_dir=resolve_repo_path(
            fdtd_cfg.get("data_raw_dir", "data/raw"),
            project_root=project_root,
        ),
        runs_dir=Path(runs_dir),
        sim_subdir_name=str(fdtd_cfg.get("sim_subdir_name", "simulation_data")),
        save_name=str(fdtd_cfg.get("save_name", "name")),
        bidir=BidirLayoutConfig(**dict(bidir_cfg or {})),
        monitors=MonitorExtractConfig(**dict(monitor_cfg or {})),
        mapping=MappingFactorConfig(**dict(mapping_cfg or {})),
        debug=bool(fdtd_cfg.get("debug", True)),
        base_lsf_templates=resolve_fdtd_base_lsf_templates(
            config_path,
            project_root=project_root,
        ),
    )


def resolve_hybrid_config(
    config: Mapping[str, Any],
    *,
    project_root: Path | None = None,
) -> Dict[str, Any]:
    hybrid_cfg = config.get("hybrid") or {}
    if not isinstance(hybrid_cfg, dict):
        raise TypeError("hybrid section must be a mapping")

    out = dict(hybrid_cfg)
    for key in ("working_dir", "mode_path", "gds_path"):
        if key in out and out[key] is not None:
            out[key] = resolve_repo_path(out[key], project_root=project_root)

    for key in ("input_fields", "field_paths"):
        values = out.get(key)
        if values is None:
            continue
        if not isinstance(values, list):
            raise TypeError(f"hybrid.{key} must be a list")
        out[key] = [
            resolve_repo_path(value, project_root=project_root) for value in values
        ]

    jobs = []
    for job in out.get("gds_imports", []):
        if not isinstance(job, dict):
            raise TypeError("hybrid.gds_imports entries must be mappings")
        item = dict(job)
        item["gds_path"] = resolve_repo_path(
            item["gds_path"], project_root=project_root
        )
        jobs.append(item)
    if jobs:
        out["gds_imports"] = jobs
    return out


def resolve_export_config(
    config: Mapping[str, Any],
    *,
    project_root: Path | None = None,
) -> Dict[str, Any]:
    export_cfg = config.get("export") or {}
    if not isinstance(export_cfg, dict):
        raise TypeError("export section must be a mapping")

    out = dict(export_cfg)
    targets = []
    for target in export_cfg.get("targets", []):
        if not isinstance(target, dict):
            raise TypeError("export.targets entries must be mappings")
        if "source" not in target or "destination" not in target:
            raise KeyError("export.targets entries require source and destination")
        item = dict(target)
        item["source"] = resolve_repo_path(item["source"], project_root=project_root)
        item["destination"] = resolve_repo_path(
            item["destination"],
            project_root=project_root,
        )
        targets.append(item)
    out["targets"] = targets
    return out


def resolve_mapping_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    mapping_cfg = config.get("mapping") or {}
    if not isinstance(mapping_cfg, dict):
        raise TypeError("mapping section must be a mapping")
    if "focus_validation" in mapping_cfg:
        raise ValueError(
            "mapping.focus_validation has moved out of the map stage. "
            "Create a dedicated validation config inside the train bundle and run "
            "`python scripts/run_validation.py --config <validation-config>` instead."
        )
    if "port_focus_validation" in mapping_cfg:
        raise ValueError(
            "mapping.port_focus_validation has been removed. "
            "The previous implementation compared a trained mapped field against a static "
            "reference monitor, not against a trained physical-loaded diffraction field. "
            "If you need this check, put it in a dedicated train/hybrid validation script."
        )
    return dict(mapping_cfg)


def resolve_train_config(
    config: Mapping[str, Any],
    *,
    project_root: Path | None = None,
) -> Dict[str, Any]:
    train_cfg = config.get("train") or {}
    if not isinstance(train_cfg, dict):
        raise TypeError("train section must be a mapping")

    out = dict(train_cfg)
    dataset_cfg = out.get("dataset")
    if dataset_cfg is not None:
        if not isinstance(dataset_cfg, dict):
            raise TypeError("train.dataset must be a mapping")
        dataset_out = dict(dataset_cfg)
        for key in ("inputs_path", "targets_path", "eval_inputs_path", "eval_targets_path"):
            if dataset_out.get(key) is not None:
                dataset_out[key] = resolve_repo_path(
                    dataset_out[key],
                    project_root=project_root,
                )
        out["dataset"] = dataset_out

    for key in (
        "implementation",
        "task",
        "objective",
        "optimizer",
        "loop",
        "checkpoint",
        "model",
        "constraints",
        "shape",
        "mapping",
    ):
        value = out.get(key)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise TypeError(f"train.{key} must be a mapping")
        out[key] = dict(value)
    return out


def resolve_validation_config(
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    validation_cfg = config.get("validation") or {}
    if not isinstance(validation_cfg, dict):
        raise TypeError("validation section must be a mapping")

    out = dict(validation_cfg)
    backend = str(out.get("backend", "hybrid")).strip().lower()
    if backend not in {"hybrid", "fdtd"}:
        raise ValueError(
            f"validation.backend must be 'hybrid' or 'fdtd', got {backend!r}"
        )
    out["backend"] = backend
    return out


__all__ = [
    "BASE_FDTD_LSF_NAMES",
    "build_fdtd_pipeline_config",
    "get_cache_index_path",
    "resolve_export_config",
    "resolve_fdtd_base_lsf_templates",
    "resolve_hybrid_config",
    "resolve_mapping_config",
    "resolve_train_config",
    "resolve_validation_config",
]
