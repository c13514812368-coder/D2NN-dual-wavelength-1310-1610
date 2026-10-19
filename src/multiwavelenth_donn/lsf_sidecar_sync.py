from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .fdtd_pipeline import BidirLayoutConfig, compute_input_output_ys
from .networkangler import (
    make_monitor_positions,
    normalize_lumerical_material_name,
    uses_object_defined_dielectric,
)
from .runtime_config import build_network_shape, load_project_config


SYNC_TEMPLATE_DIR = Path(__file__).resolve().parent / "sidecar_sync_templates"
_TOKEN_RE = re.compile(r"{{\s*([A-Za-z0-9_]+)\s*}}")
_AUTO_SYNC_BEGIN = "# >>> AUTO-SYNC VARS BEGIN"
_AUTO_SYNC_END = "# <<< AUTO-SYNC VARS END"
BASE_FDTD_LSF_NAMES = (
    "network_simulate",
    "network_bidir",
    "calc_neff_mode_xprop",
)


@dataclass(frozen=True)
class SyncResult:
    path: Path
    changed: bool


def _ensure_1d_array(values: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={arr.shape}")
    return arr.reshape(-1)


def _encode_lsf_value(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    if isinstance(value, np.ndarray):
        if value.ndim > 1:
            raise ValueError(f"Only 1D arrays are supported, got shape={value.shape}")
        return "[" + ", ".join(_encode_lsf_value(item) for item in value.tolist()) + "]"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_encode_lsf_value(item) for item in value) + "]"
    raise TypeError(f"Unsupported LSF value type: {type(value)!r}")


def _render_template(template_name: str, context: Mapping[str, Any]) -> str:
    path = (SYNC_TEMPLATE_DIR / f"{template_name}.lsf.in").resolve()
    template_text = path.read_text(encoding="utf-8")
    missing: set[str] = set()

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in context:
            missing.add(key)
            return match.group(0)
        return _encode_lsf_value(context[key])

    rendered = _TOKEN_RE.sub(_replace, template_text)
    if missing:
        raise KeyError(f"Missing template keys for {template_name}: {sorted(missing)}")
    unresolved = sorted(set(_TOKEN_RE.findall(rendered)))
    if unresolved:
        raise KeyError(f"Unresolved template keys for {template_name}: {unresolved}")
    return rendered.rstrip() + "\n"


def _find_auto_sync_block_lines(
    content: str,
    *,
    source: str,
) -> tuple[list[str], int, int]:
    lines = content.splitlines(keepends=True)
    start_idx: int | None = None
    end_idx: int | None = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == _AUTO_SYNC_BEGIN:
            start_idx = idx
        if stripped == _AUTO_SYNC_END:
            end_idx = idx
            break
    if start_idx is None or end_idx is None or end_idx < start_idx:
        raise RuntimeError(
            f"Missing managed auto-sync block in {source}. "
            f"Expected lines {_AUTO_SYNC_BEGIN!r} and {_AUTO_SYNC_END!r}."
        )
    return lines, start_idx, end_idx


def _extract_auto_sync_block(content: str, *, source: str) -> str:
    lines, start_idx, end_idx = _find_auto_sync_block_lines(content, source=source)
    block = "".join(lines[start_idx : end_idx + 1])
    if not block.endswith("\n"):
        block += "\n"
    return block


def _merge_managed_block(existing: str, managed_block: str, *, path: Path) -> str:
    lines, start_idx, end_idx = _find_auto_sync_block_lines(existing, source=str(path))
    merged = "".join(lines[:start_idx]) + managed_block + "".join(lines[end_idx + 1 :])
    return merged.rstrip() + "\n"


def _build_sidecar_content(path: Path, rendered: str) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required sidecar LSF file: {path}. "
            "This sync helper only updates existing config-paired sidecars."
        )
    managed_block = _extract_auto_sync_block(rendered, source="sync template output")
    existing = path.read_text(encoding="utf-8")
    return _merge_managed_block(existing, managed_block, path=path)


def _write_if_changed(path: Path, content: str, *, check: bool) -> SyncResult:
    existing = path.read_text(encoding="utf-8") if path.exists() else None
    changed = existing != content
    if check:
        if changed:
            raise RuntimeError(f"Sidecar LSF out of sync: {path}")
    else:
        path.write_text(content, encoding="utf-8")
    return SyncResult(path=path.resolve(), changed=changed)


def _resolve_wavelength_spec_um(shape: Any) -> dict[str, float | int]:
    if hasattr(shape, "resolve_wavelength_spec"):
        spec = shape.resolve_wavelength_spec()
        start_um = float(spec.get("start_um", getattr(shape, "wavelength_um", 1.55)))
        stop_um = float(spec.get("stop_um", start_um))
        center_um = float(spec.get("center_um", 0.5 * (start_um + stop_um)))
        points = max(1, int(spec.get("points", getattr(shape, "wavelength_points", 1))))
    else:
        start_um = float(getattr(shape, "wavelength_start", getattr(shape, "wavelength_um", 1.55)))
        stop_um = float(getattr(shape, "wavelength_stop", start_um))
        center_um = float(getattr(shape, "wavelength_center", 0.5 * (start_um + stop_um)))
        points = max(1, int(getattr(shape, "wavelength_points", 1)))
    if abs(stop_um - start_um) <= 0.0:
        points = 1
    return {
        "start_um": start_um,
        "stop_um": stop_um,
        "center_um": center_um,
        "points": points,
    }


def _material_context(prefix: str, material: object, index_val: object) -> dict[str, Any]:
    material_name = normalize_lumerical_material_name(material)
    return {
        f"{prefix}_material_name": material_name,
        f"{prefix}_material_index": float(index_val),
        f"{prefix}_set_index": uses_object_defined_dielectric(material_name),
    }


def _resolve_monitor_positions_um(shape: Any) -> np.ndarray:
    for attr in ("monitor_positions", "monitor_positions_x_um"):
        values = getattr(shape, attr, None)
        if values is None:
            continue
        arr = _ensure_1d_array(values, name=attr)
        if arr.size > 0:
            return arr
    positions = make_monitor_positions(
        int(getattr(shape, "num_layers", 0)),
        np.asarray(getattr(shape, "inter_layer_spacings_um", []), dtype=float),
    )
    arr = _ensure_1d_array(positions, name="monitor_positions")
    target_count = int(getattr(shape, "num_layers", 0))
    if target_count > 0 and arr.size > target_count:
        arr = arr[:target_count]
    return arr


def build_network_sync_context(shape: Any) -> dict[str, Any]:
    wl_spec = _resolve_wavelength_spec_um(shape)
    input_ys, output_ys = compute_input_output_ys(shape)
    monitor_positions_um = _resolve_monitor_positions_um(shape)

    use_taper = bool(getattr(shape, "use_taper", False))
    local_map_enabled = bool(getattr(shape, "local_map_enabled", False))
    total_length_um = float(shape.total_length_um)
    total_width_um = float(shape.total_width_um)
    dist_from_sub_um = float(getattr(shape, "dist_from_sub_to_fdtd_um", 0.0))
    propagation_thickness_um = float(shape.propagation_thickness_um)
    sub_thickness_um = float(shape.sub_thickness_um)
    waveguide_length_um = float(shape.waveguide_length_um)
    waveguide_width_um = float(shape.waveguide_width_um)
    in_taper_length_um = float(getattr(shape, "in_taper_length_um", 0.0))
    out_taper_length_um = float(getattr(shape, "out_taper_length_um", 0.0))
    taper_lead_length_um = float(getattr(shape, "taper_lead_length_um", 10.0))
    inter = np.asarray(getattr(shape, "inter_layer_spacings_um", [0.0]), dtype=float)
    inter_sum_um = float(np.sum(inter)) if inter.size else 0.0

    prop_center_x_m = 0.0
    sub_center_x_m = 0.0
    x_in_center_m = -0.5 * total_length_um * 1e-6
    x_out_center_m = 0.5 * total_length_um * 1e-6
    local_map_left_pad_um = float(getattr(shape, "local_map_left_pad_um", 10.0))
    local_map_right_pad_um = float(getattr(shape, "local_map_right_pad_um", 10.0))

    if local_map_enabled:
        if not use_taper:
            raise ValueError("shape.local_map_enabled currently requires shape.use_taper=true")
        if monitor_positions_um.size == 0:
            raise ValueError(
                "shape.local_map_enabled requires at least one monitor position; "
                "set shape.monitor_positions to taper-local x offsets in um"
            )
        if np.any(np.asarray(monitor_positions_um, dtype=float) < 0.0):
            raise ValueError(
                "shape.local_map_enabled interprets shape.monitor_positions as taper-local x "
                f"offsets >= 0 um, got {np.asarray(monitor_positions_um, dtype=float).tolist()}"
            )
        x_in_center_m = 0.0
        x_out_center_m = float(np.max(np.asarray(monitor_positions_um, dtype=float))) * 1e-6
        fdtd_left_edge_m = (
            x_in_center_m
            - in_taper_length_um * 1e-6
            - taper_lead_length_um * 1e-6
            - local_map_left_pad_um * 1e-6
        )
        fdtd_right_edge_m = x_out_center_m + local_map_right_pad_um * 1e-6
        x_span_m = max(fdtd_right_edge_m - fdtd_left_edge_m, 1e-9)
        fdtd_center_x_m = 0.5 * (fdtd_left_edge_m + fdtd_right_edge_m)
        prop_x_span_m = max(x_out_center_m + local_map_right_pad_um * 1e-6, 1e-9)
        prop_center_x_m = 0.5 * prop_x_span_m
        sub_x_span_m = max(3.0 * x_span_m, 1e-9)
        sub_center_x_m = fdtd_center_x_m
        monitor_positions_m = np.asarray(monitor_positions_um, dtype=float) * 1e-6
    elif use_taper:
        x_span_m = (
            total_length_um
            + in_taper_length_um
            + out_taper_length_um
        ) * 1e-6
        fdtd_center_x_m = 0.0
        prop_x_span_m = total_length_um * 1e-6
        sub_x_span_m = 3.0 * total_length_um * 1e-6
        monitor_positions_m = (monitor_positions_um - 0.5 * total_length_um) * 1e-6
    else:
        x_span_m = (total_length_um + 2.0 * waveguide_length_um) * 1e-6
        fdtd_center_x_m = 0.0
        prop_x_span_m = total_length_um * 1e-6
        sub_x_span_m = 3.0 * total_length_um * 1e-6
        monitor_positions_m = (monitor_positions_um - 0.5 * total_length_um) * 1e-6

    context = {
        "use_var_fdtd": bool(getattr(shape, "use_var_fdtd", False)),
        "use_taper": use_taper,
        "local_map_enabled": local_map_enabled,
        "multi_wavelength": int(wl_spec["points"]) > 1,
        "lam_start_m": float(wl_spec["start_um"]) * 1e-6,
        "lam_stop_m": float(wl_spec["stop_um"]) * 1e-6,
        "wl_points": int(wl_spec["points"]),
        "mesh_accuracy": int(getattr(shape, "mesh_accuracy", 3)),
        "fdtd_x_span_m": x_span_m,
        "fdtd_y_span_m": total_width_um * 1e-6,
        "fdtd_z_span_m": 5.0 * propagation_thickness_um * 1e-6,
        "fdtd_center_x_m": fdtd_center_x_m,
        "simulation_time_s": max(1e-15, 1e-15 * 50.0 * max(1.0, inter_sum_um)),
        "prop_x_span_m": prop_x_span_m,
        "prop_center_x_m": prop_center_x_m,
        "prop_y_span_m": (total_width_um + 2.0 * dist_from_sub_um) * 1e-6 + 1e-7,
        "prop_z_span_m": propagation_thickness_um * 1e-6,
        "sub_x_span_m": sub_x_span_m,
        "sub_center_x_m": sub_center_x_m,
        "sub_y_span_m": 3.0 * ((total_width_um + 2.0 * dist_from_sub_um) * 1e-6 + 1e-7),
        "sub_z_span_m": sub_thickness_um * 1e-6,
        "sub_center_z_m": (-0.5 * propagation_thickness_um - 0.5 * sub_thickness_um) * 1e-6,
        "monitor_positions_m": monitor_positions_m,
        "monitor_y_span_m": total_width_um * 1e-6,
        "input_ys_m": np.asarray(input_ys, dtype=float) * 1e-6,
        "output_ys_m": np.asarray(output_ys, dtype=float) * 1e-6,
        "x_in_center_m": x_in_center_m,
        "x_out_center_m": x_out_center_m,
        "local_map_left_pad_m": local_map_left_pad_um * 1e-6,
        "local_map_right_pad_m": local_map_right_pad_um * 1e-6,
        "wg_len_m": waveguide_length_um * 1e-6,
        "wg_width_m": waveguide_width_um * 1e-6,
        "source_y_span_m": 2.0 * waveguide_width_um * 1e-6,
        "monitor_port_y_span_m": 1.5 * waveguide_width_um * 1e-6,
        "input_taper_len_m": in_taper_length_um * 1e-6,
        "output_taper_len_m": out_taper_length_um * 1e-6,
        "input_taper_wide_m": float(getattr(shape, "in_taper_width_um", 12.0)) * 1e-6,
        "output_taper_wide_m": float(getattr(shape, "out_taper_width_um", 12.0)) * 1e-6,
        "taper_narrow_m": waveguide_width_um * 1e-6,
        "taper_lead_len_m": taper_lead_length_um * 1e-6,
    }
    context.update(
        _material_context(
            "background",
            getattr(shape, "background_material"),
            getattr(shape, "background_material_index", 1.0),
        )
    )
    context.update(
        _material_context(
            "propagation",
            getattr(shape, "propagation_material"),
            getattr(shape, "propagation_material_index", 1.0),
        )
    )
    context.update(
        _material_context(
            "sub",
            getattr(shape, "sub_material"),
            getattr(shape, "sub_material_index", 1.0),
        )
    )
    return context


def build_bidir_sync_context(shape: Any, bidir_cfg: BidirLayoutConfig) -> dict[str, Any]:
    wl_spec = _resolve_wavelength_spec_um(shape)
    propagation_thickness_um = float(shape.propagation_thickness_um)
    sub_thickness_um = float(shape.sub_thickness_um)
    total_len_m = float(bidir_cfg.total_len_m)
    total_wid_m = float(bidir_cfg.total_wid_m)
    wid_m = float(bidir_cfg.wid_m)
    wg_len_m = float(bidir_cfg.wg_len_m)
    wg_width_m = float(shape.waveguide_width_um) * 1e-6

    context = {
        "use_var_fdtd": bool(getattr(shape, "use_var_fdtd", False)),
        "multi_wavelength": int(wl_spec["points"]) > 1,
        "lam_start_m": float(wl_spec["start_um"]) * 1e-6,
        "lam_stop_m": float(wl_spec["stop_um"]) * 1e-6,
        "wl_points": int(wl_spec["points"]),
        "mesh_accuracy": int(getattr(shape, "mesh_accuracy", 3)),
        "fdtd_x_span_m": total_len_m + 2.0 * float(bidir_cfg.dist_from_sub_tofdtd),
        "fdtd_y_span_m": total_wid_m + float(bidir_cfg.dist_from_sub_tofdtd),
        "fdtd_z_span_m": 5.0 * propagation_thickness_um * 1e-6,
        "simulation_time_s": max(1e-15, (1e-9) * 1000.0 * float(total_len_m + 2.0 * wg_len_m) / 7.0),
        "prop_x_span_m": total_len_m + 1e-7,
        "prop_y_span_m": wid_m,
        "prop_z_span_m": propagation_thickness_um * 1e-6,
        "sub_x_span_m": 10.0 * total_len_m + 1e-7,
        "sub_y_span_m": 10.0 * wid_m,
        "sub_z_span_m": sub_thickness_um * 1e-6 + 1e-8,
        "sub_center_z_m": (-0.5 * sub_thickness_um - 0.5 * propagation_thickness_um) * 1e-6,
        "total_len_m": total_len_m,
        "total_wid_m": total_wid_m,
        "wg_len_m": wg_len_m,
        "wg_width_m": wg_width_m,
        "wg_bias_m": float(bidir_cfg.wg_bias) * 1e-6,
        "x_left_m": -0.5 * total_len_m - 0.5 * wg_len_m,
        "x_right_m": 0.5 * total_len_m + 0.5 * wg_len_m,
        "y0_m": 0.0,
    }
    context.update(
        _material_context(
            "background",
            getattr(shape, "background_material"),
            getattr(shape, "background_material_index", 1.0),
        )
    )
    context.update(
        _material_context(
            "propagation",
            getattr(shape, "propagation_material"),
            getattr(shape, "propagation_material_index", 1.0),
        )
    )
    context.update(
        _material_context(
            "sub",
            getattr(shape, "sub_material"),
            getattr(shape, "sub_material_index", 1.0),
        )
    )
    return context


def build_neff_sync_context(shape: Any) -> dict[str, Any]:
    wl_um = float(getattr(shape, "wavelength_um", getattr(shape, "wavelength_center", 1.55)))
    core_h_um = float(getattr(shape, "propagation_thickness_um"))
    lateral_span_um = max(float(getattr(shape, "waveguide_width_um", 0.0)) * 2.0, core_h_um * 6.0)
    context = {
        "wavelength_m": wl_um * 1e-6,
        "core_h_m": core_h_um * 1e-6,
        "lateral_span_m": lateral_span_um * 1e-6,
        "top_cladding_center_z_m": 0.5 * (core_h_um + lateral_span_um) * 1e-6,
        "bottom_cladding_center_z_m": -0.5 * (core_h_um + lateral_span_um) * 1e-6,
    }
    context.update(
        _material_context(
            "top",
            getattr(shape, "background_material"),
            getattr(shape, "background_material_index", 1.0),
        )
    )
    context.update(
        _material_context(
            "bottom",
            getattr(shape, "sub_material"),
            getattr(shape, "sub_material_index", 1.0),
        )
    )
    context.update(
        _material_context(
            "core",
            getattr(shape, "propagation_material"),
            getattr(shape, "propagation_material_index", 1.0),
        )
    )
    return context


def _sidecar_path(config_path: Path, template_name: str) -> Path:
    return (config_path.parent / f"{config_path.stem}.{template_name}.lsf").resolve()


def sync_fdtd_sidecars(
    config_path: str | Path,
    *,
    check: bool = False,
) -> list[SyncResult]:
    cfg_path = Path(config_path).resolve()
    config = load_project_config(cfg_path)
    shape = build_network_shape(config)
    fdtd_cfg = dict(config.get("fdtd") or {})
    bidir_cfg = BidirLayoutConfig(**dict(fdtd_cfg.get("bidir") or {}))

    contexts = {
        "network_simulate": build_network_sync_context(shape),
        "network_bidir": build_bidir_sync_context(shape, bidir_cfg),
        "calc_neff_mode_xprop": build_neff_sync_context(shape),
    }

    results: list[SyncResult] = []
    for template_name in BASE_FDTD_LSF_NAMES:
        rendered = _render_template(template_name, contexts[template_name])
        final_content = _build_sidecar_content(_sidecar_path(cfg_path, template_name), rendered)
        results.append(
            _write_if_changed(
                _sidecar_path(cfg_path, template_name),
                final_content,
                check=check,
            )
        )
    return results
