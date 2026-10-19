from __future__ import annotations

import copy
import os
import stat
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from .experiment_artifacts import (
    build_calibration_package,
    build_mapped_network_package,
    build_shape_spec,
    fingerprint_payload,
    load_monitors_bundle,
    save_json,
    save_package,
)
from .experiment_visualization import save_map_visualizations
from .fdtd_pipeline import compute_input_output_ys, run_fdtd_workflow
from .run_tracking import (
    append_log,
    create_experiment_context,
    read_experiment_manifest,
    write_experiment_manifest,
)
from .config_loading import (
    build_network_shape,
    load_project_config,
)
from .config_sections import (
    build_fdtd_pipeline_config,
    get_cache_index_path,
    resolve_mapping_config,
)
from .project_paths import (
    get_project_root,
    resolve_repo_path,
)
from .simulation_store import load_named_payload_bundle
from .workspace_layout import (
    ExperimentWorkspaceLayout,
    resolve_experiment_workspace,
    resolve_workspace_experiment_config,
)


STAGE_DIRS = {
    "shape": "shape",
    "fdtd": "fdtd",
    "map": "mapped_network",
}


def _build_fixture_fdtd_result(
    config: Mapping[str, Any],
    *,
    shape: Any,
) -> dict[str, Any]:
    fdtd_cfg = dict(config.get("fdtd") or {})
    fixture = dict(fdtd_cfg.get("fixture") or {})
    if "monitors_path" not in fixture or "emode_path" not in fixture:
        raise KeyError("fdtd.fixture requires monitors_path and emode_path")

    monitors_path = resolve_repo_path(fixture["monitors_path"])
    emode_path = resolve_repo_path(fixture["emode_path"])
    selected_wavelength_um = float(
        fixture.get(
            "selected_wavelength_um",
            getattr(shape, "wavelength_center", getattr(shape, "wavelength_um", 1.55)),
        )
    )
    neff_by = dict(
        fixture.get("neff_by_wavelength_um") or {selected_wavelength_um: fixture["neff"]}
    )
    mapping_by = dict(
        fixture.get("mapping_factor_by_wavelength_um")
        or {selected_wavelength_um: fixture["mapping_factor"]}
    )
    wavelengths = np.asarray([float(v) for v in neff_by.keys()], dtype=float)

    global_monitor_bundle_path = fixture.get("global_monitor_bundle_path")
    global_monitors = None
    if global_monitor_bundle_path is not None:
        global_monitor_bundle_path = resolve_repo_path(global_monitor_bundle_path)
        global_monitors = load_named_payload_bundle(global_monitor_bundle_path)

    return {
        "monitors": load_monitors_bundle(monitors_path),
        "global_monitors": global_monitors,
        "sim_dir": str(monitors_path.parent),
        "run_dir": None,
        "emode_path": str(emode_path),
        "monitors_bundle_path": str(monitors_path),
        "global_monitor_bundle_path": (
            None if global_monitor_bundle_path is None else str(global_monitor_bundle_path)
        ),
        "simulation_record_path": None,
        "parameter_txt_dir": None,
        "sim_signature": fingerprint_payload(
            {
                "backend": "fixture",
                "shape": getattr(shape, "to_cache_payload", lambda: {})(),
                "monitors_path": str(monitors_path),
                "emode_path": str(emode_path),
                "selected_wavelength_um": selected_wavelength_um,
                "neff_by_wavelength_um": neff_by,
                "mapping_factor_by_wavelength_um": mapping_by,
            }
        ),
        "wavelengths_um": wavelengths,
        "selected_wavelength_um": selected_wavelength_um,
        "neff_by_wavelength_um": neff_by,
        "mapping_factor_by_wavelength_um": mapping_by,
        "neff": float(fixture.get("neff", list(neff_by.values())[0])),
        "mapping_factor": float(fixture.get("mapping_factor", list(mapping_by.values())[0])),
        "source": "fixture",
        "use_cached_results": True,
        "did_run": False,
        "warnings": [],
        "diffs": [],
    }


def _copy_optional_file(source: str | Path | None, destination: Path) -> str | None:
    if source is None:
        return None
    src_path = Path(source)
    if not src_path.exists():
        raise FileNotFoundError(
            f"Expected optional artifact file does not exist: source={src_path}, destination={destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, destination)
    return str(destination)


def _resolve_stage_sequence(stage: str) -> list[str]:
    normalized = stage.lower()
    if normalized == "shape":
        return ["shape"]
    if normalized == "fdtd":
        return ["shape", "fdtd"]
    if normalized == "map":
        return ["shape", "fdtd", "map"]
    raise ValueError(f"Unsupported stage {stage!r}")


def _write_stage_event(stage_log_path: Path, message: str) -> None:
    with stage_log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{datetime.now().isoformat()}] {message}\n")


def _build_fdtd_summary_payload(
    calibration_package: Mapping[str, Any],
    *,
    include_global_monitor_bundle_path: bool,
) -> dict[str, Any]:
    summary = {
        "selected_wavelength_um": calibration_package["selected_wavelength_um"],
        "wavelengths_um": np.asarray(
            calibration_package.get("wavelengths_um", []),
            dtype=float,
        ).reshape(-1).tolist(),
        "neff": calibration_package["neff"],
        "neff_by_wavelength_um": dict(
            calibration_package.get("neff_by_wavelength_um", {})
        ),
        "mapping_factor": calibration_package["mapping_factor"],
        "mapping_factor_by_wavelength_um": dict(
            calibration_package.get("mapping_factor_by_wavelength_um", {})
        ),
        "monitor_payload_kind": calibration_package.get("monitor_payload_kind"),
        "source": calibration_package["cache_info"]["source"],
        "use_cached_results": calibration_package["cache_info"]["use_cached_results"],
        "did_run": calibration_package["cache_info"]["did_run"],
        "warnings": list(calibration_package["cache_info"]["warnings"]),
        "diffs": list(calibration_package["cache_info"]["diffs"]),
        "sim_signature": calibration_package.get("sim_signature"),
        "sim_dir": calibration_package.get("sim_dir"),
        "run_dir": calibration_package.get("run_dir"),
        "emode_path": calibration_package.get("emode_path"),
        "monitors_bundle_path": calibration_package.get("monitors_bundle_path"),
        "simulation_record_path": calibration_package.get("simulation_record_path"),
        "parameter_txt_dir": calibration_package.get("parameter_txt_dir"),
    }
    if include_global_monitor_bundle_path:
        summary["global_monitor_bundle_path"] = calibration_package.get(
            "global_monitor_bundle_path"
        )
    return summary


def _run_shape_stage(
    config: Mapping[str, Any],
    *,
    stage_dir: Path,
) -> tuple[dict[str, str], str, dict[str, Any]]:
    shape = build_network_shape(config)
    shape_spec = build_shape_spec(config.get("shape") or {}, shape=shape)
    shape_yaml = stage_dir / "shape.yaml"
    shape_yaml.write_text(
        yaml.safe_dump(
            dict(shape_spec["shape_config"]),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    shape_spec_path = save_json(stage_dir / "shape_spec.json", shape_spec)
    outputs = {
        "shape_yaml": str(shape_yaml),
        "shape_spec": str(shape_spec_path),
    }
    return outputs, fingerprint_payload(shape_spec), {"shape": shape, "shape_spec": shape_spec}


def _run_fdtd_stage(
    config: Mapping[str, Any],
    *,
    stage_dir: Path,
    project_root: Path,
    config_path: Path,
    shape_result: Mapping[str, Any],
    force_set: set[str],
) -> tuple[dict[str, str], str, dict[str, Any]]:
    shape = shape_result["shape"]
    fdtd_cfg = dict(config.get("fdtd") or {})
    backend = str(fdtd_cfg.get("backend", "live"))
    if backend == "fixture":
        calibration_result = _build_fixture_fdtd_result(config, shape=shape)
    elif backend == "live":
        use_cache = bool(fdtd_cfg.get("use_cache", True))
        cache_file = None
        if use_cache:
            cache_file = get_cache_index_path(config, project_root=project_root)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
        calibration_result = run_fdtd_workflow(
            shape,
            cfg=build_fdtd_pipeline_config(
                config,
                runs_dir=stage_dir / "runtime",
                config_path=config_path,
                project_root=project_root,
            ),
            cache_file=cache_file,
            use_cache=use_cache,
            force_redo="fdtd" in force_set or bool(fdtd_cfg.get("force_redo", False)),
        )
    else:
        raise ValueError(f"Unsupported fdtd.backend {backend!r}")

    calibration_package = build_calibration_package(
        shape_spec=shape_result["shape_spec"],
        calibration_result=calibration_result,
    )
    summary_path = save_json(
        stage_dir / "summary.json",
        _build_fdtd_summary_payload(
            calibration_package,
            include_global_monitor_bundle_path=True,
        ),
    )
    calibration_package_path = save_package(
        stage_dir / "calibration_package.pt",
        calibration_package,
    )
    outputs = {
        "summary": str(summary_path),
        "calibration_package": str(calibration_package_path),
    }
    return outputs, calibration_package["calibration_signature"], {
        "calibration_package": calibration_package,
    }


def _publish_workspace_map(
    *,
    workspace: ExperimentWorkspaceLayout,
    ctx: Any,
    shape_spec: Mapping[str, Any],
    calibration_package: Mapping[str, Any],
    mapped_network_package: Mapping[str, Any],
) -> dict[str, str]:
    workspace_map_dir = workspace.map_dir
    if workspace_map_dir.exists():
        attrs = getattr(workspace_map_dir.lstat(), "st_file_attributes", 0)
        is_reparse_point = bool(attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
        if os.path.islink(workspace_map_dir) or is_reparse_point:
            os.rmdir(workspace_map_dir)
        else:
            shutil.rmtree(workspace_map_dir)
    workspace_map_dir.mkdir(parents=True, exist_ok=False)

    shape_spec_path = save_json(workspace_map_dir / "shape_spec.json", shape_spec)
    calibration_package_path = save_package(
        workspace_map_dir / "calibration_package.pt",
        calibration_package,
    )
    mapped_network_package_path = save_package(
        workspace_map_dir / "mapped_network_package.pt",
        mapped_network_package,
    )
    fdtd_summary_path = save_json(
        workspace_map_dir / "fdtd_summary.json",
        _build_fdtd_summary_payload(
            calibration_package,
            include_global_monitor_bundle_path=False,
        ),
    )
    local_monitors_path = _copy_optional_file(
        calibration_package.get("monitors_bundle_path"),
        workspace_map_dir / "monitors.npz",
    )
    local_global_monitors_path = _copy_optional_file(
        calibration_package.get("global_monitor_bundle_path"),
        workspace_map_dir / "global_monitors.npz",
    )
    workspace_visualizations = save_map_visualizations(
        mapped_network_package,
        out_dir=workspace_map_dir / "figures",
    )
    workspace_visualization_manifest_path = save_json(
        workspace_map_dir / "visualization_manifest.json",
        workspace_visualizations,
    )
    workspace_package_meta_path = save_json(
        workspace_map_dir / "package_meta.json",
        {
            "mapped_network_signature": mapped_network_package["mapped_network_signature"],
            "shape_signature": mapped_network_package["shape_spec"]["shape_signature"],
            "calibration_signature": mapped_network_package["calibration_package"][
                "calibration_signature"
            ],
            "optical_donn_config": mapped_network_package["optical_donn_config"],
            "monitors_bundle_path": local_monitors_path,
            "global_monitor_bundle_path": local_global_monitors_path,
            "visualizations": workspace_visualizations,
        },
    )
    workspace_manifest_path = save_json(
        workspace_map_dir / "manifest.json",
        {
            "config_path": str(ctx.config_path),
            "source_run_dir": str(ctx.run_dir),
            "shape_spec": str(shape_spec_path),
            "fdtd_summary": str(fdtd_summary_path),
            "calibration_package": str(calibration_package_path),
            "mapped_network_package": str(mapped_network_package_path),
            "monitors_bundle_path": local_monitors_path,
            "global_monitor_bundle_path": local_global_monitors_path,
            "package_meta": str(workspace_package_meta_path),
            "visualization_manifest": str(workspace_visualization_manifest_path),
            "figures_dir": str(workspace_map_dir / "figures"),
        },
    )
    return {
        "workspace_map_dir": str(workspace_map_dir),
        "workspace_map_manifest": str(workspace_manifest_path),
        "workspace_mapped_network_package": str(mapped_network_package_path),
        "workspace_calibration_package": str(calibration_package_path),
        "workspace_monitors_bundle": (
            "" if local_monitors_path is None else str(local_monitors_path)
        ),
        "workspace_global_monitors_bundle": (
            "" if local_global_monitors_path is None else str(local_global_monitors_path)
        ),
    }


def _run_map_stage(
    config: Mapping[str, Any],
    *,
    stage_dir: Path,
    workspace: ExperimentWorkspaceLayout,
    ctx: Any,
    shape_result: Mapping[str, Any],
    fdtd_result: Mapping[str, Any],
) -> tuple[dict[str, str], str, dict[str, Any]]:
    shape_spec = shape_result["shape_spec"]
    calibration_package = fdtd_result["calibration_package"]
    mapped_network_package = build_mapped_network_package(
        shape_spec=shape_spec,
        calibration_package=calibration_package,
        shape=build_network_shape({"shape": shape_spec["shape_config"]}),
        mapping_config=resolve_mapping_config(config),
    )

    package_path = save_package(
        stage_dir / "mapped_network_package.pt",
        mapped_network_package,
    )
    visualizations = save_map_visualizations(
        mapped_network_package,
        out_dir=stage_dir / "figures",
    )
    visualization_manifest_path = save_json(
        stage_dir / "visualization_manifest.json",
        visualizations,
    )
    package_meta_path = save_json(
        stage_dir / "package_meta.json",
        {
            "mapped_network_signature": mapped_network_package["mapped_network_signature"],
            "shape_signature": mapped_network_package["shape_spec"]["shape_signature"],
            "calibration_signature": mapped_network_package["calibration_package"][
                "calibration_signature"
            ],
            "optical_donn_config": mapped_network_package["optical_donn_config"],
            "visualizations": visualizations,
        },
    )

    outputs: dict[str, str] = {
        "mapped_network_package": str(package_path),
        "package_meta": str(package_meta_path),
        "visualization_manifest": str(visualization_manifest_path),
    }
    for key, value in visualizations.items():
        if key != "errors" and isinstance(value, str):
            outputs[key] = value
    outputs.update(
        _publish_workspace_map(
            workspace=workspace,
            ctx=ctx,
            shape_spec=shape_spec,
            calibration_package=calibration_package,
            mapped_network_package=mapped_network_package,
        )
    )
    return outputs, mapped_network_package["mapped_network_signature"], {
        "mapped_network_package": mapped_network_package,
        "workspace_map_dir": str(workspace.map_dir),
    }


def _build_stage_input_payload(
    stage_name: str,
    *,
    config: Mapping[str, Any],
    stage_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if stage_name == "shape":
        return dict(config.get("shape") or {})
    if stage_name == "fdtd":
        return {
            "shape_signature": stage_results["shape"]["shape_spec"]["shape_signature"],
            "fdtd": dict(config.get("fdtd") or {}),
        }
    return {
        "shape_signature": stage_results["shape"]["shape_spec"]["shape_signature"],
        "calibration_signature": stage_results["fdtd"]["calibration_package"][
            "calibration_signature"
        ],
        "mapping": dict(config.get("mapping") or {}),
    }


def run_experiment(
    *,
    config_path: str | Path,
    stage: str = "map",
    runs_root: str | Path = "runs",
    force_stages: set[str] | None = None,
) -> dict[str, Any]:
    project_root = get_project_root()
    config = load_project_config(config_path)
    workspace = resolve_experiment_workspace(config_path)

    ctx = create_experiment_context(
        config_path=config_path,
        config=config,
        runs_root=runs_root,
        project_root=project_root,
    )

    resolved_shape = build_network_shape(config)
    compute_input_output_ys(resolved_shape)
    ctx.config_snapshot_path.write_text(
        yaml.safe_dump(
            {
                "meta": copy.deepcopy(dict(config.get("meta") or {})),
                "shape": (
                    resolved_shape.to_config_payload()
                    if hasattr(resolved_shape, "to_config_payload")
                    else copy.deepcopy(dict(config.get("shape") or {}))
                ),
                "fdtd": copy.deepcopy(dict(config.get("fdtd") or {})),
                "mapping": copy.deepcopy(dict(config.get("mapping") or {})),
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    append_log(ctx, f"loaded experiment config {config_path}")

    stage_sequence = _resolve_stage_sequence(stage)
    manifest = read_experiment_manifest(ctx)
    manifest.setdefault("config_path", str(config_path))
    manifest.setdefault("experiment_id", ctx.run_dir.name)
    manifest.setdefault("stages", {})
    force_set = {item.lower() for item in (force_stages or set())}
    stage_results: dict[str, dict[str, Any]] = {}

    for stage_name in stage_sequence:
        stage_input = _build_stage_input_payload(
            stage_name,
            config=config,
            stage_results=stage_results,
        )
        input_fingerprint = fingerprint_payload(stage_input)
        manifest["stages"][stage_name] = {
            "status": "running",
            "input_fingerprint": input_fingerprint,
            "outputs": {},
            "output_fingerprint": None,
            "error": None,
            "updated_at": datetime.now().isoformat(),
        }
        manifest["status"] = "running"
        write_experiment_manifest(ctx, manifest)
        append_log(ctx, f"stage {stage_name} started")
        stage_log_path = ctx.logs_dir / f"{stage_name}.log"
        _write_stage_event(stage_log_path, "started")

        stage_dir = ctx.run_dir / STAGE_DIRS[stage_name]
        stage_dir.mkdir(parents=True, exist_ok=True)

        try:
            if stage_name == "shape":
                outputs, output_fingerprint, stage_result = _run_shape_stage(
                    config,
                    stage_dir=stage_dir,
                )
            elif stage_name == "fdtd":
                outputs, output_fingerprint, stage_result = _run_fdtd_stage(
                    config,
                    stage_dir=stage_dir,
                    project_root=project_root,
                    config_path=workspace.config_path,
                    shape_result=stage_results["shape"],
                    force_set=force_set,
                )
            else:
                outputs, output_fingerprint, stage_result = _run_map_stage(
                    config,
                    stage_dir=stage_dir,
                    workspace=workspace,
                    ctx=ctx,
                    shape_result=stage_results["shape"],
                    fdtd_result=stage_results["fdtd"],
                )
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            append_log(ctx, f"stage {stage_name} failed: {error_text}")
            _write_stage_event(stage_log_path, f"failed: {error_text}")
            manifest["stages"][stage_name] = {
                "status": "failed",
                "input_fingerprint": input_fingerprint,
                "outputs": {},
                "output_fingerprint": None,
                "error": error_text,
                "updated_at": datetime.now().isoformat(),
            }
            manifest["status"] = "failed"
            write_experiment_manifest(ctx, manifest)
            raise

        stage_results[stage_name] = stage_result
        manifest["stages"][stage_name] = {
            "status": "done",
            "input_fingerprint": input_fingerprint,
            "outputs": {key: str(value) for key, value in outputs.items()},
            "output_fingerprint": output_fingerprint,
            "error": None,
            "updated_at": datetime.now().isoformat(),
        }
        manifest["status"] = "done"
        write_experiment_manifest(ctx, manifest)
        append_log(ctx, f"stage {stage_name} completed")
        _write_stage_event(stage_log_path, "completed")

    manifest = read_experiment_manifest(ctx)
    manifest["status"] = "completed"
    write_experiment_manifest(ctx, manifest)
    return {
        "run_dir": str(ctx.run_dir),
        "manifest_path": str(ctx.manifest_path),
        "stages": stage_sequence,
        "workspace_map_dir": (
            None if "map" not in stage_results else str(stage_results["map"]["workspace_map_dir"])
        ),
    }


def run_experiment_workspace(
    workspace_dir: str | Path,
    *,
    stage: str = "map",
    runs_root: str | Path = "runs",
    force_stages: set[str] | None = None,
) -> dict[str, Any]:
    return run_experiment(
        config_path=resolve_workspace_experiment_config(workspace_dir),
        stage=stage,
        runs_root=runs_root,
        force_stages=force_stages,
    )


__all__ = [
    "resolve_workspace_experiment_config",
    "run_experiment",
    "run_experiment_workspace",
]
