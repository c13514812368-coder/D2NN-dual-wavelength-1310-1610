from __future__ import annotations

import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch

from .experiment_artifacts import extract_monitor_lines_from_calibration_package
from .experiment_artifacts import instantiate_optical_donn_from_mapped_package
from .lumerical_scripts import (
    extract_ey_1d_at_wavelength,
    get_fields,
    monitor_payload_from_fields,
    select_field_component_at_wavelength,
)
from .networkangler import NetworkShape
from .wavelength_tools import canonical_wavelength_key


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _ensure_float_array(values: Any) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=float).reshape(-1)


def _ensure_complex_line(values: Any) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    arr = np.asarray(values, dtype=np.complex128).reshape(-1)
    if arr.size == 0:
        raise ValueError("empty field line")
    return arr


def _resolve_centered_layer_x_um(shape_cfg: Mapping[str, Any], count: int) -> np.ndarray:
    explicit = shape_cfg.get("layer_positions_um")
    if explicit is not None:
        layer_x = _ensure_float_array(explicit)
        if layer_x.size != count:
            raise ValueError("shape.layer_positions_um size does not match monitor count")
        return layer_x

    spacings = _ensure_float_array(shape_cfg.get("inter_layer_spacings_um", []))
    if count <= 1:
        return np.asarray([0.0], dtype=float)
    if spacings.size != count - 1:
        total_length = float(shape_cfg.get("total_length_um", 0.0))
        return np.linspace(-0.5 * total_length, 0.5 * total_length, count, dtype=float)

    positions = [-0.5 * float(np.sum(spacings))]
    for spacing in spacings:
        positions.append(positions[-1] + float(spacing))
    return np.asarray(positions, dtype=float)


def _resolve_field_y_axis_um(shape_cfg: Mapping[str, Any], sample_count: int) -> np.ndarray:
    total_width_um = float(shape_cfg["total_width_um"])
    return np.linspace(-0.5 * total_width_um, 0.5 * total_width_um, sample_count, dtype=float)


def _resolve_model_y_axis_um(model: Any) -> np.ndarray:
    positions = getattr(model, "positions", None)
    if not positions:
        raise ValueError("model.positions is empty")
    return _ensure_float_array(positions[0])


def _resolve_io_y_um(shape_cfg: Mapping[str, Any], optical_cfg: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    total_width_um = float(shape_cfg["total_width_um"])
    input_positions = _ensure_float_array(optical_cfg["input_positions_um"]) - 0.5 * total_width_um
    output_positions = _ensure_float_array(optical_cfg["output_positions_um"]) - 0.5 * total_width_um
    return input_positions, output_positions


def _require_payload_axis_um(payload: Mapping[str, Any], key: str) -> np.ndarray:
    if key not in payload:
        raise KeyError(f"missing {key} in payload")
    axis = _ensure_float_array(payload[key])
    if axis.size == 0:
        raise ValueError(f"{key} is empty")
    return axis


def _select_wavelength_slice(
    values: Any,
    wavelengths_um: Any,
    *,
    wavelength_um: float,
) -> tuple[np.ndarray, float]:
    arr = np.asarray(values)
    wl = _ensure_float_array(wavelengths_um)
    if wl.size <= 1:
        selected = np.squeeze(arr)
        chosen = float(wl[0]) if wl.size == 1 else float(wavelength_um)
        return np.asarray(selected), chosen

    candidates = [idx for idx, size in enumerate(arr.shape) if size == wl.size]
    if not candidates:
        raise ValueError(
            f"cannot locate wavelength axis of size {wl.size} in shape {arr.shape}"
        )
    wl_axis = candidates[-1]
    wl_idx = int(np.argmin(np.abs(wl - float(wavelength_um))))
    selected = np.take(arr, indices=wl_idx, axis=wl_axis)
    return np.asarray(np.squeeze(selected)), float(wl[wl_idx])


def _to_plane(
    values: Any,
    *,
    axis0_um: np.ndarray,
    axis1_um: np.ndarray,
) -> np.ndarray:
    arr = np.asarray(values)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D plane, got shape={arr.shape}")
    if arr.shape == (axis0_um.size, axis1_um.size):
        return np.asarray(arr)
    if arr.shape == (axis1_um.size, axis0_um.size):
        return np.asarray(arr).T
    raise ValueError(
        "plane shape does not match axes: "
        f"shape={arr.shape}, axis0={axis0_um.size}, axis1={axis1_um.size}"
    )


def _extract_global_index_plane(
    calibration_package: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    payload = calibration_package.get("global_index_monitor")
    if not isinstance(payload, Mapping):
        raise KeyError("calibration_package.global_index_monitor is missing")
    x_axis_um = _require_payload_axis_um(payload, "x_um")
    y_axis_um = _require_payload_axis_um(payload, "y_um")
    selected_wl_um = float(calibration_package["selected_wavelength_um"])
    plane, chosen_wl_um = _select_wavelength_slice(
        payload["index"],
        payload.get("lambda_um", []),
        wavelength_um=selected_wl_um,
    )
    return _to_plane(plane, axis0_um=x_axis_um, axis1_um=y_axis_um), x_axis_um, y_axis_um, chosen_wl_um


def _extract_global_index_plane_zx(
    calibration_package: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    payload = calibration_package.get("global_index_monitor_zx")
    if not isinstance(payload, Mapping):
        raise KeyError("calibration_package.global_index_monitor_zx is missing")
    x_axis_um = _require_payload_axis_um(payload, "x_um")
    z_axis_um = _require_payload_axis_um(payload, "z_um")
    selected_wl_um = float(calibration_package["selected_wavelength_um"])
    plane, chosen_wl_um = _select_wavelength_slice(
        payload["index"],
        payload.get("lambda_um", []),
        wavelength_um=selected_wl_um,
    )
    return _to_plane(plane, axis0_um=x_axis_um, axis1_um=z_axis_um), x_axis_um, z_axis_um, chosen_wl_um


def _extract_global_field_maps(
    calibration_package: Mapping[str, Any],
) -> dict[str, Any]:
    payload = calibration_package.get("global_field_monitor")
    if not isinstance(payload, Mapping):
        raise KeyError("calibration_package.global_field_monitor is missing")

    x_axis_um = _require_payload_axis_um(payload, "x_um")
    y_axis_um = _require_payload_axis_um(payload, "y_um")
    selected_wl_um = float(calibration_package["selected_wavelength_um"])

    component_planes: dict[str, np.ndarray] = {}
    chosen_wl_um = selected_wl_um
    for component_name in ("Ex", "Ey", "Ez"):
        try:
            selected = select_field_component_at_wavelength(
                payload,
                component=component_name,
                wavelength_um=selected_wl_um,
                selection="nearest",
            )
        except Exception:
            continue
        component_planes[component_name] = _to_plane(
            selected["field"],
            axis0_um=x_axis_um,
            axis1_um=y_axis_um,
        )
        chosen_wl_um = float(selected["selected_wavelength_um"])

    if not component_planes:
        raise ValueError("global_field_monitor does not contain any selectable E component")

    intensity = np.zeros((x_axis_um.size, y_axis_um.size), dtype=float)
    for component_plane in component_planes.values():
        intensity += np.abs(component_plane) ** 2
    amplitude = np.sqrt(intensity)
    phase_ref = component_planes.get("Ey")
    if phase_ref is None:
        phase_ref = next(iter(component_planes.values()))

    return {
        "x_axis_um": x_axis_um,
        "y_axis_um": y_axis_um,
        "selected_wavelength_um": chosen_wl_um,
        "amplitude": amplitude,
        "intensity": intensity,
        "phase": np.angle(phase_ref),
    }


def _resolve_material_index_guess(shape_cfg: Mapping[str, Any], prefix: str) -> float:
    material_name = str(shape_cfg.get(f"{prefix}_material", "")).strip()
    material_index = float(shape_cfg.get(f"{prefix}_material_index", 1.0) or 1.0)
    if "Object defined dielectric" in material_name:
        return material_index
    if "SiO2" in material_name or "Glass" in material_name:
        return 1.44
    if "Silicon" in material_name or material_name == "Si":
        return 3.47
    return material_index


def _build_derived_index_zx(
    calibration_package: Mapping[str, Any],
    shape_cfg: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index_xy, x_axis_um, y_axis_um, _ = _extract_global_index_plane(calibration_package)
    center_y_idx = int(np.argmin(np.abs(y_axis_um)))
    slab_line = np.asarray(index_xy[:, center_y_idx], dtype=float)

    propagation_thickness_um = float(shape_cfg.get("propagation_thickness_um", 0.22))
    propagation_center_z_um = float(shape_cfg.get("propagation_center_z_um", 0.0))
    sub_thickness_um = float(shape_cfg.get("sub_thickness_um", 2.0))
    prop_z0 = propagation_center_z_um - 0.5 * propagation_thickness_um
    prop_z1 = propagation_center_z_um + 0.5 * propagation_thickness_um
    sub_z1 = prop_z0
    sub_z0 = sub_z1 - sub_thickness_um

    z_margin_um = max(0.2, 0.5 * propagation_thickness_um)
    z_axis_um = np.linspace(sub_z0 - z_margin_um, prop_z1 + z_margin_um, 320, dtype=float)

    background_n = _resolve_material_index_guess(shape_cfg, "background")
    sub_n = _resolve_material_index_guess(shape_cfg, "sub")
    plane = np.full((x_axis_um.size, z_axis_um.size), background_n, dtype=float)

    sub_mask = (z_axis_um >= sub_z0) & (z_axis_um <= sub_z1)
    plane[:, sub_mask] = sub_n

    prop_mask = (z_axis_um >= prop_z0) & (z_axis_um <= prop_z1)
    plane[:, prop_mask] = slab_line[:, None]
    return plane, x_axis_um, z_axis_um


def _geometry_x_limits_um(
    shape_cfg: Mapping[str, Any],
    layer_x_um: np.ndarray,
) -> tuple[float, float]:
    wg_len_um = float(shape_cfg.get("waveguide_length_um", 3.0))
    taper_pad = 0.0
    if bool(shape_cfg.get("use_taper", False)):
        taper_pad = max(
            float(shape_cfg.get("in_taper_length_um", 0.0)),
            float(shape_cfg.get("out_taper_length_um", 0.0)),
        )
    pad_um = max(wg_len_um, taper_pad, 2.0)
    return float(layer_x_um[0] - pad_um), float(layer_x_um[-1] + pad_um)


def _save_layout_overview(
    mapped_network_package: Mapping[str, Any],
    *,
    out_dir: Path,
) -> Path:
    plt = _pyplot()
    optical_cfg = dict(mapped_network_package["optical_donn_config"])
    shape_cfg = dict(mapped_network_package["shape_spec"]["shape_config"])
    calibration = dict(mapped_network_package["calibration_package"])

    total_width_um = float(shape_cfg["total_width_um"])
    layer_x_um = _resolve_centered_layer_x_um(shape_cfg, int(optical_cfg["num_layers"]))
    positions_span_um = _ensure_float_array(optical_cfg["positions_span_um"])
    neurons_per_layer = [int(v) for v in optical_cfg["num_neurons_each_layer"]]
    input_y_um, output_y_um = _resolve_io_y_um(shape_cfg, optical_cfg)
    x_min_um, x_max_um = _geometry_x_limits_um(shape_cfg, layer_x_um)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(
        "Mapped DONN Layout\n"
        f"wavelength={float(optical_cfg['wavelength_um']):.4g} um, "
        f"neff={float(calibration['neff']):.4g}, "
        f"mapping_factor={float(calibration['mapping_factor']):.4g}"
    )

    for layer_idx, x_um in enumerate(layer_x_um):
        ax.axvline(x_um, color="#cbd5e1", linewidth=1.0, zorder=0)
        span_um = float(positions_span_um[min(layer_idx, positions_span_um.size - 1)])
        count = neurons_per_layer[layer_idx]
        if count == 1:
            centers_um = np.asarray([0.0], dtype=float)
        else:
            centers_um = np.linspace(-0.5 * span_um, 0.5 * span_um, count, dtype=float)
        ax.scatter(
            np.full_like(centers_um, x_um),
            centers_um,
            s=90,
            color="#0f766e",
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.text(
            x_um,
            0.5 * total_width_um + 1.4,
            f"L{layer_idx}\nN={count}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    if input_y_um.size:
        ax.scatter(
            np.full_like(input_y_um, x_min_um + 0.3 * float(shape_cfg.get("waveguide_length_um", 3.0))),
            input_y_um,
            marker=">",
            s=120,
            color="#2563eb",
            label="inputs",
            zorder=4,
        )
    if output_y_um.size:
        ax.scatter(
            np.full_like(output_y_um, x_max_um - 0.3 * float(shape_cfg.get("waveguide_length_um", 3.0))),
            output_y_um,
            marker="s",
            s=90,
            color="#c2410c",
            label="outputs",
            zorder=4,
        )

    ax.set_xlabel("Propagation x (um)")
    ax.set_ylabel("Lateral y (um)")
    ax.set_xlim(x_min_um - 1.0, x_max_um + 1.0)
    ax.set_ylim(-0.5 * total_width_um - 1.5, 0.5 * total_width_um + 3.0)
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.legend(loc="upper right")
    fig.tight_layout()

    out_path = out_dir / "layout_overview.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _save_index_monitor_xy(
    mapped_network_package: Mapping[str, Any],
    *,
    out_dir: Path,
) -> Path:
    plt = _pyplot()
    calibration = dict(mapped_network_package["calibration_package"])
    index_plane, x_axis_um, y_axis_um, chosen_wl_um = _extract_global_index_plane(calibration)

    fig, ax = plt.subplots(figsize=(11, 6.2))
    im = ax.imshow(
        index_plane.T,
        aspect="auto",
        extent=[float(x_axis_um[0]), float(x_axis_um[-1]), float(y_axis_um[0]), float(y_axis_um[-1])],
        origin="lower",
        interpolation="nearest",
        cmap="viridis",
    )
    fig.colorbar(im, ax=ax, label="n")
    ax.set_title(f"Global Index Monitor In XY Plane (lambda={chosen_wl_um:.6g} um)")
    ax.set_xlabel("Propagation x (um)")
    ax.set_ylabel("Lateral y (um)")
    fig.tight_layout()

    out_path = out_dir / "index_monitor_xy.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _save_index_zx(
    mapped_network_package: Mapping[str, Any],
    *,
    out_dir: Path,
) -> Path:
    plt = _pyplot()
    shape_cfg = dict(mapped_network_package["shape_spec"]["shape_config"])
    calibration = dict(mapped_network_package["calibration_package"])
    index_plane, x_axis_um, z_axis_um = _build_derived_index_zx(calibration, shape_cfg)
    chosen_wl_um = float(calibration["selected_wavelength_um"])

    fig, ax = plt.subplots(figsize=(11, 4.5))
    im = ax.imshow(
        index_plane.T,
        aspect="auto",
        extent=[float(x_axis_um[0]), float(x_axis_um[-1]), float(z_axis_um[0]), float(z_axis_um[-1])],
        origin="lower",
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label="n")
    ax.set_title(
        "Derived ZX Index Cross-section\n"
        f"from global XY index monitor + z stack (lambda={chosen_wl_um:.6g} um)"
    )
    ax.set_xlabel("Propagation x (um)")
    ax.set_ylabel("Vertical z (um)")
    fig.tight_layout()

    out_path = out_dir / "index_monitor_zx.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _run_reference_forward(
    mapped_network_package: Mapping[str, Any],
    *,
    out_dir: Path,
) -> tuple[Any, np.ndarray]:
    model = instantiate_optical_donn_from_mapped_package(
        mapped_network_package,
        work_dir=out_dir / "model_workspace",
        device="cpu",
    )
    model.eval()
    input_tensor = torch.zeros((1, len(model.inputposition)), dtype=torch.float32)
    with torch.no_grad():
        _ = _forward_selected_wavelength(model, input_tensor)
    outputs = _forward_selected_wavelength(
        model,
        torch.zeros((1, len(model.inputposition)), dtype=torch.float32),
    ).detach().cpu().reshape(-1).numpy()
    return model, outputs


def _forward_selected_wavelength(model: Any, input_tensor: torch.Tensor) -> torch.Tensor:
    selected_wavelength_um = getattr(model, "selected_wavelength_um", None)
    if selected_wavelength_um is None:
        output = model(input_tensor)
    else:
        output = model(input_tensor, wavelength_um=float(selected_wavelength_um))
    output = torch.as_tensor(output, dtype=torch.float32, device=input_tensor.device)
    if output.ndim == 1:
        output = output.unsqueeze(0)
    return output


def _forward_requested_wavelength(
    model: Any,
    input_tensor: torch.Tensor,
    *,
    wavelength_um: float | None = None,
) -> torch.Tensor:
    if wavelength_um is None:
        return _forward_selected_wavelength(model, input_tensor)
    output = model(input_tensor, wavelength_um=float(wavelength_um))
    output = torch.as_tensor(output, dtype=torch.float32, device=input_tensor.device)
    if output.ndim == 1:
        output = output.unsqueeze(0)
    return output


def _resolve_focus_validation_wavelength(
    model: Any,
    *,
    wavelength_um: float | None = None,
) -> float:
    if wavelength_um is None:
        selected = getattr(model, "selected_wavelength_um", None)
        if selected is None:
            return float(getattr(model, "wavelength", 1.55))
        return float(selected)
    if hasattr(model, "_resolve_wavelength_context"):
        context = model._resolve_wavelength_context(wavelength_um=float(wavelength_um))
        return float(context["selected_wavelength_um"])
    return float(wavelength_um)


def _focus_validation_suffix(value: object) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def _stack_lines(lines: Sequence[Any]) -> np.ndarray:
    rows = [_ensure_complex_line(line) for line in lines]
    base_len = rows[0].size
    for row in rows:
        if row.size != base_len:
            raise ValueError("field lines have inconsistent lengths")
    return np.stack(rows, axis=0)


def _interp_complex_line(
    y_src_um: np.ndarray,
    field_src: np.ndarray,
    y_tgt_um: np.ndarray,
) -> np.ndarray:
    real = np.interp(y_tgt_um, y_src_um, np.real(field_src))
    imag = np.interp(y_tgt_um, y_src_um, np.imag(field_src))
    return real + 1j * imag


def _normalize_amplitude(line: Any) -> np.ndarray:
    amplitude = np.abs(_ensure_complex_line(line))
    scale = float(np.max(amplitude))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("field amplitude is empty or non-positive")
    return amplitude / scale


def _field_intensity(line: Any) -> np.ndarray:
    return np.abs(_ensure_complex_line(line)) ** 2


def _normalize_sum(values: Any) -> np.ndarray:
    arr = _ensure_float_array(values)
    total = float(np.sum(arr))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"profile sum must be positive, got {total}")
    return arr / total


def _normalize_peak(values: Any) -> np.ndarray:
    arr = _ensure_float_array(values)
    peak = float(np.max(arr))
    if not np.isfinite(peak) or peak <= 0.0:
        raise ValueError(f"profile peak must be positive, got {peak}")
    return arr / peak


def _build_gaussian_target_profile(
    y_axis_um: np.ndarray,
    *,
    center_um: float,
    sigma_um: float,
) -> np.ndarray:
    if sigma_um <= 0.0:
        raise ValueError(f"sigma_um must be positive, got {sigma_um}")
    profile = np.exp(-0.5 * ((y_axis_um - float(center_um)) / float(sigma_um)) ** 2)
    return _normalize_sum(profile)


def _windowed_intensity_at_positions(
    y_axis_um: np.ndarray,
    field_line: Any,
    sample_positions_um: np.ndarray,
    *,
    window_um: float,
) -> np.ndarray:
    if window_um <= 0.0:
        raise ValueError(f"window_um must be positive, got {window_um}")
    if sample_positions_um.size == 0:
        raise ValueError("sample_positions_um is empty")

    intensity = _field_intensity(field_line)
    energies: list[float] = []
    half_window = 0.5 * float(window_um)
    for center_um in sample_positions_um.tolist():
        mask = np.abs(y_axis_um - float(center_um)) <= half_window
        if int(np.sum(mask)) >= 2:
            energy = float(np.trapz(intensity[mask], y_axis_um[mask]))
        else:
            idx = int(np.argmin(np.abs(y_axis_um - float(center_um))))
            energy = float(intensity[idx])
        energies.append(energy)
    return np.asarray(energies, dtype=float)


def _sample_amplitude_at_positions(
    y_axis_um: np.ndarray,
    field_line: Any,
    sample_positions_um: np.ndarray,
) -> np.ndarray:
    if sample_positions_um.size == 0:
        raise ValueError("sample_positions_um is empty")
    if np.any(sample_positions_um < float(y_axis_um[0])) or np.any(sample_positions_um > float(y_axis_um[-1])):
        raise ValueError(
            "Output port positions fall outside the field axis: "
            f"field_range=[{float(y_axis_um[0]):.6g}, {float(y_axis_um[-1]):.6g}] um, "
            f"ports={sample_positions_um.tolist()}"
        )
    return np.interp(sample_positions_um, y_axis_um, np.abs(_ensure_complex_line(field_line)))


def _save_focus_validation_hybrid(
    mapped_network_package: Mapping[str, Any],
    shape_cfg: Mapping[str, Any],
    *,
    out_dir: Path,
    wavelength_um: float | None = None,
    artifact_suffix: str | None = None,
) -> tuple[Path, Dict[str, Any], list[str]]:
    plt = _pyplot()
    mapping_cfg = dict(mapped_network_package.get("mapping_config") or {})
    focus_cfg = dict(mapping_cfg.get("focus_validation") or {})
    if not bool(focus_cfg.get("enabled", False)):
        raise ValueError("mapping.focus_validation.enabled is false")

    artifact_label = None if artifact_suffix is None else str(artifact_suffix).strip()
    validation_dir = (
        out_dir / "focus_validation_workspace"
        if not artifact_label
        else out_dir / f"focus_validation_workspace_{artifact_label}"
    )
    validation_dir.mkdir(parents=True, exist_ok=True)

    seed = int(focus_cfg.get("train_seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = instantiate_optical_donn_from_mapped_package(
        mapped_network_package,
        work_dir=validation_dir / "model_workspace",
        device="cpu",
    )
    if bool(focus_cfg.get("reinitialize_parameters", True)):
        if "initcanshu" in focus_cfg:
            model.initcanshu = focus_cfg["initcanshu"]
        elif "initcanshu" in mapping_cfg:
            model.initcanshu = mapping_cfg["initcanshu"]
        model.init_parameters()

    focus_wavelength_um = _resolve_focus_validation_wavelength(
        model,
        wavelength_um=wavelength_um,
    )

    target_port_index = int(focus_cfg.get("target_port_index", 0))
    if target_port_index < 0 or target_port_index >= len(model.outputposition):
        raise ValueError(
            "mapping.focus_validation.target_port_index is out of range: "
            f"target_port_index={target_port_index}, num_outputs={len(model.outputposition)}"
        )

    epochs = int(focus_cfg.get("epochs", 20))
    if epochs <= 0:
        raise ValueError(f"mapping.focus_validation.epochs must be positive, got {epochs}")
    lr = float(focus_cfg.get("lr", 1.0e-2))
    if lr <= 0.0:
        raise ValueError(f"mapping.focus_validation.lr must be positive, got {lr}")
    weight_decay = float(focus_cfg.get("weight_decay", 0.0))
    clamp_min = float(focus_cfg.get("clamp_min", 0.0))
    slot_length_min_um = float(focus_cfg.get("slot_length_min_um", 1.0e-6))
    slot_length_max_raw = focus_cfg.get("slot_length_max_um")
    slot_length_max_um = None if slot_length_max_raw is None else float(slot_length_max_raw)
    if slot_length_max_um is not None and slot_length_max_um <= slot_length_min_um:
        raise ValueError(
            "mapping.focus_validation.slot_length_max_um must be greater than "
            f"slot_length_min_um, got {slot_length_max_um} <= {slot_length_min_um}"
        )

    input_phase = focus_cfg.get("input_phase")
    if input_phase is None:
        input_phase = [0.0] * len(model.inputposition)
    input_phase_arr = np.asarray(input_phase, dtype=np.float32).reshape(-1)
    if input_phase_arr.size != len(model.inputposition):
        raise ValueError(
            "mapping.focus_validation.input_phase size mismatch: "
            f"expected {len(model.inputposition)}, got {input_phase_arr.size}"
        )

    trainable = [param for param in model.parameters() if param.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters available for mapping.focus_validation")
    optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    input_tensor = torch.tensor(input_phase_arr[None, :], dtype=torch.float32, device=next(model.parameters()).device)
    optical_cfg = dict(mapped_network_package["optical_donn_config"])
    _, output_y_um = _resolve_io_y_um(shape_cfg, optical_cfg)
    mathematical_y_um = _resolve_model_y_axis_um(model)
    target_center_um = float(output_y_um[target_port_index])
    target_sigma_um = float(
        focus_cfg.get(
            "target_sigma_um",
            max(float(optical_cfg.get("output_mapping_span_um", 2.0)) * 0.5, 1.0),
        )
    )
    target_profile = _build_gaussian_target_profile(
        mathematical_y_um,
        center_um=target_center_um,
        sigma_um=target_sigma_um,
    )
    target_profile_tensor = torch.tensor(
        target_profile[None, :],
        dtype=torch.float32,
        device=input_tensor.device,
    )
    port_window_um = float(
        focus_cfg.get(
            "port_window_um",
            float(optical_cfg.get("output_mapping_span_um", 2.0)),
        )
    )
    if port_window_um <= 0.0:
        raise ValueError(f"mapping.focus_validation.port_window_um must be positive, got {port_window_um}")

    history: list[float] = []
    best_loss = float("inf")
    best_state_dict: dict[str, torch.Tensor] | None = None

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        raw_output = _forward_requested_wavelength(
            model,
            input_tensor,
            wavelength_um=focus_wavelength_um,
        )
        if not hasattr(model, "us") or len(model.us) == 0:
            raise AttributeError("OpticalDONN did not expose mathematical fields in model.us during focus training")
        field = torch.as_tensor(model.us[-1], dtype=torch.complex64, device=input_tensor.device)
        if field.ndim == 1:
            field = field.unsqueeze(0)
        intensity = torch.abs(field) ** 2
        intensity = intensity / intensity.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        loss = loss_fn(intensity.to(torch.float32), target_profile_tensor)
        loss.backward()
        optimizer.step()
        for param in getattr(model, "slot_length_um", []):
            if not getattr(param, "requires_grad", False):
                continue
            if slot_length_max_um is None:
                param.data.clamp_(min=slot_length_min_um)
            else:
                param.data.clamp_(min=slot_length_min_um, max=slot_length_max_um)

        loss_value = float(loss.detach().cpu().item())
        history.append(loss_value)
        if loss_value <= best_loss:
            best_loss = loss_value
            best_state_dict = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

    if best_state_dict is None:
        raise RuntimeError("mapping.focus_validation did not record a best state dict")
    model.load_state_dict(best_state_dict)
    model.eval()

    with torch.no_grad():
        math_raw_output = _forward_requested_wavelength(
            model,
            input_tensor,
            wavelength_um=focus_wavelength_um,
        )
    input_phase_support = list(getattr(model, "last_input_phase_support_summary", []) or [])
    math_output = math_raw_output.clamp_min(clamp_min)
    math_output = math_output / math_output.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    math_output_response = math_output.detach().cpu().reshape(-1).numpy()

    if not hasattr(model, "us") or len(model.us) == 0:
        raise AttributeError("OpticalDONN did not expose mathematical fields in model.us")
    mathematical_line = _ensure_complex_line(model.us[-1])
    mathematical_y_um = _resolve_model_y_axis_um(model)
    mathematical_intensity = _normalize_sum(_field_intensity(mathematical_line))

    internal_layer_count = int(model.num_layers) - 2
    if internal_layer_count <= 0:
        raise ValueError(
            "mapping.focus_validation requires at least one internal diffraction layer, "
            f"got num_layers={model.num_layers}"
        )

    mapping_factor = float(model._mapping_factor_for_wavelength(wavelength_um=focus_wavelength_um))
    if mapping_factor == 0.0:
        raise ValueError("calibration_package.mapping_factor must be non-zero for focus validation")

    slot_pitch_um = float(focus_cfg.get("slot_pitch_um", shape_cfg.get("unit_spacing_um", 0.0)))
    if slot_pitch_um <= 0.0:
        raise ValueError(f"mapping.focus_validation.slot_pitch_um must be positive, got {slot_pitch_um}")
    slot_width_um = float(
        focus_cfg.get(
            "slot_width_um",
            float(shape_cfg.get("duty_cycle", 0.0)) * float(shape_cfg.get("unit_spacing_um", 0.0)),
        )
    )
    if slot_width_um <= 0.0:
        raise ValueError(f"mapping.focus_validation.slot_width_um must be positive, got {slot_width_um}")
    slot_material = str(focus_cfg.get("slot_material", "etch"))
    if not slot_material:
        raise ValueError("mapping.focus_validation.slot_material must be a non-empty string")

    shape = NetworkShape(**dict(shape_cfg))
    setattr(shape, "before_um", float(focus_cfg.get("before_um", 0.1)))
    setattr(shape, "after_um", float(focus_cfg.get("after_um", 3.0)))
    calibration_package = dict(mapped_network_package["calibration_package"])
    sim_dir_raw = calibration_package.get("sim_dir")
    if sim_dir_raw is None:
        raise KeyError("calibration_package.sim_dir is required for mapping.focus_validation hybrid verification")
    sim_dir = Path(sim_dir_raw)
    if not sim_dir.exists():
        raise FileNotFoundError(f"calibration_package.sim_dir does not exist: {sim_dir}")

    hybrid_mode_path_raw = focus_cfg.get("hybrid_mode_path")
    if hybrid_mode_path_raw is None:
        candidate_mode_path = sim_dir / "mode1_E.mat"
        if candidate_mode_path.exists():
            hybrid_mode_path = candidate_mode_path
        else:
            hybrid_mode_path = Path(str(getattr(model, "mode_path", "")))
    else:
        hybrid_mode_path = Path(hybrid_mode_path_raw)
        if not hybrid_mode_path.is_absolute():
            hybrid_mode_path = (sim_dir / hybrid_mode_path).resolve()
    if not hybrid_mode_path.exists():
        raise FileNotFoundError(
            "mapping.focus_validation could not resolve a usable hybrid mode file: "
            f"resolved={hybrid_mode_path}, sim_dir={sim_dir}, "
            f"model.mode_path={getattr(model, 'mode_path', None)}"
        )
    model.mode_path = str(hybrid_mode_path)

    from .lsf_special_scripts import iter_hybrid_init_scripts, write_add_silicon_slots_lsf
    from .lumdata import _safe_close

    try:
        import lumapi
    except Exception as exc:
        raise ImportError("lumapi is required for mapping.focus_validation hybrid verification") from exc

    hybrid_workdir = validation_dir / "hybrid_fdtd"
    hybrid_workdir.mkdir(parents=True, exist_ok=True)
    hide_fdtd_cad = bool(focus_cfg.get("hide_fdtd_cad", True))
    hybrid_mesh_accuracy = int(focus_cfg.get("hybrid_mesh_accuracy", 5))
    if hybrid_mesh_accuracy <= 0:
        raise ValueError(
            "mapping.focus_validation.hybrid_mesh_accuracy must be positive, "
            f"got {hybrid_mesh_accuracy}"
        )

    internal_counts = [int(v) for v in shape.neurons_per_layer[1:-1]]
    if len(internal_counts) != internal_layer_count:
        raise ValueError(
            "Internal layer count mismatch while building focus validation geometry: "
            f"internal_counts={internal_counts}, internal_layer_count={internal_layer_count}"
        )

    handles: list[Any] = []
    warnings: list[str] = []
    try:
        for handle_idx, script in enumerate(iter_hybrid_init_scripts(shape, acc=hybrid_mesh_accuracy)):
            fdtd = lumapi.FDTD(hide=hide_fdtd_cad)
            fdtd.cd(str(hybrid_workdir))
            fdtd.eval(script)
            project_path = hybrid_workdir / f"hybrid_layer_{handle_idx}.fsp"
            fdtd.save(str(project_path))
            handles.append(fdtd)

        if len(handles) != internal_layer_count:
            raise RuntimeError(
                "Hybrid handle count does not match internal layer count: "
                f"handles={len(handles)}, internal_layer_count={internal_layer_count}"
            )

        phase_wrap_count = 0
        phase_wrap_by_layer: list[int] = []
        physical_lengths_by_layer: list[np.ndarray] = []
        raw_delta_by_layer: list[np.ndarray] = []
        wrapped_delta_by_layer: list[np.ndarray] = []

        for internal_idx, count in enumerate(internal_counts):
            raw_delta = (
                torch.as_tensor(
                    model.get_delta_phi_by_wavelength(wavelength_um=focus_wavelength_um)[internal_idx + 1]
                )
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )
            if raw_delta.size != count:
                raise ValueError(
                    "Internal delta_phi size does not match shape.neurons_per_layer: "
                    f"layer_index={internal_idx + 1}, expected={count}, got={raw_delta.size}"
                )

            wrapped_delta = raw_delta.copy()
            if mapping_factor < 0.0:
                positive_mask = wrapped_delta > 0.0
                phase_wrap_count += int(np.sum(positive_mask))
                phase_wrap_by_layer.append(int(np.sum(positive_mask)))
                wrapped_delta[positive_mask] = wrapped_delta[positive_mask] - 2.0 * np.ceil(
                    wrapped_delta[positive_mask] / 2.0
                )
            else:
                phase_wrap_by_layer.append(int(np.sum(wrapped_delta < 0.0)))
                phase_wrap_count += phase_wrap_by_layer[-1]
                wrapped_delta = np.mod(wrapped_delta, 2.0)

            slot_length_params = list(getattr(model, "slot_length_um", []))
            if slot_length_params and internal_idx + 1 < len(slot_length_params):
                physical_lengths_um = (
                    torch.as_tensor(slot_length_params[internal_idx + 1])
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )
                if physical_lengths_um.size != count:
                    raise ValueError(
                        "Internal slot_length_um size does not match shape.neurons_per_layer: "
                        f"layer_index={internal_idx + 1}, expected={count}, got={physical_lengths_um.size}"
                    )
            else:
                physical_lengths_um = wrapped_delta * mapping_factor
            if np.any(~np.isfinite(physical_lengths_um)):
                raise ValueError(
                    "Non-finite slot lengths encountered in focus validation: "
                    f"layer_index={internal_idx + 1}, lengths={physical_lengths_um.tolist()}"
                )
            if np.any(physical_lengths_um <= 0.0):
                bad = physical_lengths_um[physical_lengths_um <= 0.0]
                raise ValueError(
                    "Non-positive slot lengths encountered in focus validation: "
                    f"layer_index={internal_idx + 1}, "
                    f"bad_count={bad.size}, min_length_um={float(np.min(bad))}"
                )

            raw_delta_by_layer.append(raw_delta)
            wrapped_delta_by_layer.append(wrapped_delta)
            physical_lengths_by_layer.append(physical_lengths_um)

            local_shape = SimpleNamespace(
                propagation_thickness_um=float(shape.propagation_thickness_um),
                total_width_um=float(shape.total_width_um),
                unit_span_um=float(slot_width_um),
                unit_spacing_um=float(slot_pitch_um),
                repeats=int(getattr(shape, "repeats", 1)),
                neurons_per_layer=[1, count, 1],
            )
            y0_um = -0.5 * float(shape.total_width_um) + 0.5 * slot_pitch_um
            slot_script_path = write_add_silicon_slots_lsf(
                local_shape,
                hybrid_workdir / f"hybrid_add_slots_layer_{internal_idx}.lsf",
                params=physical_lengths_um.tolist(),
                positions=[(0.0, y0_um)],
                material=slot_material,
                z_span_um=float(shape.propagation_thickness_um),
                width_um=slot_width_um,
                spacing_um=slot_pitch_um,
                rect_prefix=f"layer{internal_idx}_rect",
            )
            handles[internal_idx].cd(str(hybrid_workdir))
            handles[internal_idx].eval(slot_script_path.read_text(encoding="utf-8"))
            handles[internal_idx].save(str(hybrid_workdir / f"hybrid_layer_{internal_idx}.fsp"))

        model.sim = SimpleNamespace(
            hybrid_fdtds=handles,
            hybrid_monitor_name=str(focus_cfg.get("hybrid_monitor_name", "hybrid_monitor")),
        )
        model.layer_spacing_hybride = [float(v) for v in np.asarray(shape.inter_layer_spacings_um, dtype=float)]
        model.hybrid_workdir = hybrid_workdir
        model.hybrid_zlink = float(focus_cfg.get("hybrid_zlink", 0.0))
        model.setuphs_hybride(
            mode=str(focus_cfg.get("propagation_mode", "full")),
            ky_cut_ratio=float(focus_cfg.get("ky_cut_ratio", 1.0)),
            coarse_factor=float(focus_cfg.get("coarse_factor", 1.0)),
            smooth_rolloff=float(focus_cfg.get("smooth_rolloff", 0.0)),
            pad_factor=int(focus_cfg.get("pad_factor", 16)),
        )

        with torch.no_grad():
            physical_raw_output = model.forward_hybride(
                input_tensor,
                wavelength_um=focus_wavelength_um,
            )
        physical_raw_output = torch.as_tensor(
            physical_raw_output,
            dtype=torch.float32,
            device=input_tensor.device,
        )
        if physical_raw_output.ndim == 1:
            physical_raw_output = physical_raw_output.unsqueeze(0)
        physical_output = physical_raw_output.clamp_min(clamp_min)
        physical_output = physical_output / physical_output.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        physical_output_response = physical_output.detach().cpu().reshape(-1).numpy()

        if not hasattr(model, "us_H") or len(model.us_H) == 0:
            raise AttributeError("OpticalDONN did not expose hybrid physical fields in model.us_H")
        physical_line = _ensure_complex_line(model.us_H[-1])
        physical_y_um = _resolve_model_y_axis_um(model)

        if physical_line.size != mathematical_line.size or not np.allclose(physical_y_um, mathematical_y_um):
            physical_line = _interp_complex_line(physical_y_um, physical_line, mathematical_y_um)
            physical_y_um = mathematical_y_um
        physical_intensity = _normalize_sum(_field_intensity(physical_line))
        mathematical_port_energy = _windowed_intensity_at_positions(
            mathematical_y_um,
            mathematical_line,
            output_y_um,
            window_um=port_window_um,
        )
        physical_port_energy = _windowed_intensity_at_positions(
            physical_y_um,
            physical_line,
            output_y_um,
            window_um=port_window_um,
        )
        math_window_response = _normalize_sum(mathematical_port_energy)
        physical_window_response = _normalize_sum(physical_port_energy)
        math_peak_sample_index = int(np.argmax(mathematical_intensity))
        physical_peak_sample_index = int(np.argmax(physical_intensity))
        math_peak_y_um = float(mathematical_y_um[math_peak_sample_index])
        physical_peak_y_um = float(physical_y_um[physical_peak_sample_index])
        math_nearest_port_index = int(np.argmin(np.abs(output_y_um - math_peak_y_um)))
        physical_nearest_port_index = int(np.argmin(np.abs(output_y_um - physical_peak_y_um)))
        math_peak_distance_um = float(np.abs(math_peak_y_um - float(output_y_um[math_nearest_port_index])))
        physical_peak_distance_um = float(np.abs(physical_peak_y_um - float(output_y_um[physical_nearest_port_index])))
        math_peak_port_index = math_nearest_port_index if math_peak_distance_um <= 0.5 * port_window_um else -1
        physical_peak_port_index = (
            physical_nearest_port_index if physical_peak_distance_um <= 0.5 * port_window_um else -1
        )
        math_output_peak_port_index = int(np.argmax(math_output_response))
        physical_output_peak_port_index = int(np.argmax(physical_output_response))
        math_peak_matches_target = (
            math_peak_port_index == target_port_index
        )
        physical_peak_matches_target = (
            physical_peak_port_index == target_port_index
        )

        math_intensity_plot = _normalize_peak(mathematical_intensity)
        physical_intensity_plot = _normalize_peak(physical_intensity)
        target_profile_plot = _normalize_peak(target_profile)

        fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.4))

        loss_ax = axes[0, 0]
        loss_ax.plot(np.arange(1, len(history) + 1), history, color="#0f766e", linewidth=1.5)
        loss_ax.set_xlabel("epoch")
        loss_ax.set_ylabel("mse loss")
        loss_ax.set_title("Single-target focus training loss")
        loss_ax.grid(alpha=0.2, linewidth=0.5)

        output_ax = axes[0, 1]
        port_indices = np.arange(output_y_um.size)
        bar_width = 0.36
        output_ax.bar(
            port_indices - 0.5 * bar_width,
            math_output_response,
            width=bar_width,
            color="#dc2626",
            label="math overlap output",
        )
        output_ax.bar(
            port_indices + 0.5 * bar_width,
            physical_output_response,
            width=bar_width,
            color="#2563eb",
            label="physical overlap output",
        )
        output_ax.set_xlabel("output port")
        output_ax.set_ylabel("normalized response")
        output_ax.set_title(
            "Output overlap response\n"
            f"target=P{target_port_index}, math_peak=P{math_output_peak_port_index}, "
            f"physical_peak=P{physical_output_peak_port_index}"
        )
        output_ax.set_xticks(port_indices)
        output_ax.set_ylim(0.0, 1.05)
        output_ax.legend(loc="upper right")
        output_ax.grid(alpha=0.2, linewidth=0.5, axis="y")

        field_ax = axes[1, 0]
        field_ax.plot(
            mathematical_y_um,
            math_intensity_plot,
            color="#dc2626",
            linewidth=1.4,
            label="math output-plane intensity",
        )
        field_ax.plot(
            physical_y_um,
            physical_intensity_plot,
            color="#2563eb",
            linewidth=1.2,
            linestyle="--",
            label="hybrid physical intensity",
        )
        field_ax.plot(
            mathematical_y_um,
            target_profile_plot,
            color="#111827",
            linewidth=1.1,
            linestyle=":",
            label="target intensity profile",
        )
        for port_idx, port_y_um in enumerate(output_y_um):
            color = "#f59e0b" if port_idx == target_port_index else "#64748b"
            field_ax.axvline(float(port_y_um), color=color, linewidth=0.9, alpha=0.8)
            field_ax.text(
                float(port_y_um),
                1.02,
                f"P{port_idx}",
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
            )
        field_ax.set_xlabel("lateral y (um)")
        field_ax.set_ylabel("normalized intensity")
        field_ax.set_title(
            "Output-plane intensity comparison\n"
            f"math_peak={'off-port' if math_peak_port_index < 0 else f'P{math_peak_port_index}'} @ {math_peak_y_um:.2f} um, "
            f"physical_peak={'off-port' if physical_peak_port_index < 0 else f'P{physical_peak_port_index}'} @ {physical_peak_y_um:.2f} um"
        )
        field_ax.grid(alpha=0.2, linewidth=0.5)
        field_ax.legend(loc="upper right")

        port_ax = axes[1, 1]
        port_ax.bar(
            port_indices - 0.5 * bar_width,
            math_window_response,
            width=bar_width,
            color="#dc2626",
            label="math windowed energy",
        )
        port_ax.bar(
            port_indices + 0.5 * bar_width,
            physical_window_response,
            width=bar_width,
            color="#2563eb",
            label="physical windowed energy",
        )
        port_ax.set_xlabel("output port")
        port_ax.set_ylabel("normalized windowed energy")
        port_ax.set_title(
            "Port-window intensity response\n"
            f"window={port_window_um:.3g} um, sigma={target_sigma_um:.3g} um"
        )
        port_ax.set_xticks(port_indices)
        port_ax.legend(loc="upper right")
        port_ax.grid(alpha=0.2, linewidth=0.5, axis="y")

        fig.suptitle(
            "Map Focus Validation With Hybrid Physical Forward\n"
            "Train the mapped model to concentrate output-plane intensity at one target port, then compare "
            f"the mathematical intensity against the hybrid physical diffraction result at {focus_wavelength_um:.4f} um.",
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        artifact_stem = (
            "focus_validation_hybrid"
            if not artifact_label
            else f"focus_validation_hybrid_{artifact_label}"
        )
        out_path = out_dir / f"{artifact_stem}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)

        data_path = out_dir / f"{artifact_stem}_data.npz"
        np.savez(
            data_path,
            history=np.asarray(history, dtype=float),
            output_y_um=np.asarray(output_y_um, dtype=float),
            mathematical_y_um=np.asarray(mathematical_y_um, dtype=float),
            physical_y_um=np.asarray(physical_y_um, dtype=float),
            mathematical_line=mathematical_line,
            physical_line=physical_line,
            target_profile=np.asarray(target_profile, dtype=float),
            mathematical_intensity=np.asarray(mathematical_intensity, dtype=float),
            physical_intensity=np.asarray(physical_intensity, dtype=float),
            math_output_response=np.asarray(math_output_response, dtype=float),
            physical_output_response=np.asarray(physical_output_response, dtype=float),
            mathematical_port_energy=np.asarray(mathematical_port_energy, dtype=float),
            physical_port_energy=np.asarray(physical_port_energy, dtype=float),
            math_window_response=np.asarray(math_window_response, dtype=float),
            physical_window_response=np.asarray(physical_window_response, dtype=float),
            raw_delta_by_layer=np.asarray(raw_delta_by_layer, dtype=object),
            wrapped_delta_by_layer=np.asarray(wrapped_delta_by_layer, dtype=object),
            physical_lengths_um_by_layer=np.asarray(physical_lengths_by_layer, dtype=object),
        )

        metrics: Dict[str, Any] = {
            "focus_validation_hybrid": str(out_path),
            "focus_validation_hybrid_data": str(data_path),
            "focus_validation_backend": "hybrid",
            "focus_validation_selected_wavelength_um": focus_wavelength_um,
            "focus_validation_mode_path": str(hybrid_mode_path),
            "focus_validation_target_port_index": target_port_index,
            "focus_validation_target_center_um": target_center_um,
            "focus_validation_target_sigma_um": target_sigma_um,
            "focus_validation_port_window_um": port_window_um,
            "focus_validation_input_phase": input_phase_arr.tolist(),
            "focus_validation_input_phase_support": input_phase_support,
            "focus_validation_epochs": epochs,
            "focus_validation_lr": lr,
            "focus_validation_weight_decay": weight_decay,
            "focus_validation_slot_length_min_um": slot_length_min_um,
            "focus_validation_slot_length_max_um": slot_length_max_um,
            "focus_validation_best_loss": best_loss,
            "focus_validation_math_output_response": math_output_response.tolist(),
            "focus_validation_physical_output_response": physical_output_response.tolist(),
            "focus_validation_math_field_peak_port_index": math_peak_port_index,
            "focus_validation_physical_field_peak_port_index": physical_peak_port_index,
            "focus_validation_math_field_peak_nearest_port_index": math_nearest_port_index,
            "focus_validation_physical_field_peak_nearest_port_index": physical_nearest_port_index,
            "focus_validation_math_global_peak_y_um": math_peak_y_um,
            "focus_validation_physical_global_peak_y_um": physical_peak_y_um,
            "focus_validation_math_global_peak_distance_to_port_um": math_peak_distance_um,
            "focus_validation_physical_global_peak_distance_to_port_um": physical_peak_distance_um,
            "focus_validation_math_output_peak_port_index": math_output_peak_port_index,
            "focus_validation_physical_output_peak_port_index": physical_output_peak_port_index,
            "focus_validation_math_matches_target": bool(math_peak_matches_target),
            "focus_validation_physical_matches_target": bool(physical_peak_matches_target),
            "focus_validation_math_window_energy": mathematical_port_energy.tolist(),
            "focus_validation_physical_window_energy": physical_port_energy.tolist(),
            "focus_validation_math_window_energy_response": math_window_response.tolist(),
            "focus_validation_physical_window_energy_response": physical_window_response.tolist(),
            "focus_validation_output_positions_um": output_y_um.tolist(),
            "focus_validation_slot_pitch_um": slot_pitch_um,
            "focus_validation_slot_width_um": slot_width_um,
            "focus_validation_slot_material": slot_material,
            "focus_validation_mapping_factor_um": mapping_factor,
            "focus_validation_phase_wrap_count": phase_wrap_count,
            "focus_validation_phase_wrap_count_by_layer": phase_wrap_by_layer,
            "focus_validation_raw_delta_phi_min_by_layer": [
                float(np.min(values)) for values in raw_delta_by_layer
            ],
            "focus_validation_raw_delta_phi_max_by_layer": [
                float(np.max(values)) for values in raw_delta_by_layer
            ],
            "focus_validation_physical_length_min_um_by_layer": [
                float(np.min(values)) for values in physical_lengths_by_layer
            ],
            "focus_validation_physical_length_max_um_by_layer": [
                float(np.max(values)) for values in physical_lengths_by_layer
            ],
            "focus_validation_hybrid_workdir": str(hybrid_workdir),
        }
        if phase_wrap_count > 0:
            warnings.append(
                "focus_validation used 2pi-equivalent phase wrapping to convert trained delta_phi values "
                "into positive physical slot lengths."
            )
        return out_path, metrics, warnings
    finally:
        for handle in handles:
            _safe_close(handle)


def _save_focus_validation_fdtd(
    mapped_network_package: Mapping[str, Any],
    shape_cfg: Mapping[str, Any],
    *,
    out_dir: Path,
) -> tuple[Path, Dict[str, Any], list[str]]:
    plt = _pyplot()
    mapping_cfg = dict(mapped_network_package.get("mapping_config") or {})
    focus_cfg = dict(mapping_cfg.get("focus_validation") or {})
    if not bool(focus_cfg.get("enabled", False)):
        raise ValueError("mapping.focus_validation.enabled is false")

    validation_dir = out_dir / "focus_validation_workspace"
    validation_dir.mkdir(parents=True, exist_ok=True)

    seed = int(focus_cfg.get("train_seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = instantiate_optical_donn_from_mapped_package(
        mapped_network_package,
        work_dir=validation_dir / "model_workspace",
        device="cpu",
    )
    if bool(focus_cfg.get("reinitialize_parameters", True)):
        if "initcanshu" in focus_cfg:
            model.initcanshu = focus_cfg["initcanshu"]
        elif "initcanshu" in mapping_cfg:
            model.initcanshu = mapping_cfg["initcanshu"]
        model.init_parameters()

    target_port_index = int(focus_cfg.get("target_port_index", 0))
    if target_port_index < 0 or target_port_index >= len(model.outputposition):
        raise ValueError(
            "mapping.focus_validation.target_port_index is out of range: "
            f"target_port_index={target_port_index}, num_outputs={len(model.outputposition)}"
        )

    epochs = int(focus_cfg.get("epochs", 20))
    if epochs <= 0:
        raise ValueError(f"mapping.focus_validation.epochs must be positive, got {epochs}")
    lr = float(focus_cfg.get("lr", 1.0e-2))
    if lr <= 0.0:
        raise ValueError(f"mapping.focus_validation.lr must be positive, got {lr}")
    weight_decay = float(focus_cfg.get("weight_decay", 0.0))
    clamp_min = float(focus_cfg.get("clamp_min", 0.0))
    slot_length_min_um = float(focus_cfg.get("slot_length_min_um", 1.0e-6))
    slot_length_max_raw = focus_cfg.get("slot_length_max_um")
    slot_length_max_um = None if slot_length_max_raw is None else float(slot_length_max_raw)
    if slot_length_max_um is not None and slot_length_max_um <= slot_length_min_um:
        raise ValueError(
            "mapping.focus_validation.slot_length_max_um must be greater than "
            f"slot_length_min_um, got {slot_length_max_um} <= {slot_length_min_um}"
        )

    input_phase = focus_cfg.get("input_phase")
    if input_phase is None:
        input_phase = [0.0] * len(model.inputposition)
    input_phase_arr = np.asarray(input_phase, dtype=np.float32).reshape(-1)
    if input_phase_arr.size != len(model.inputposition):
        raise ValueError(
            "mapping.focus_validation.input_phase size mismatch: "
            f"expected {len(model.inputposition)}, got {input_phase_arr.size}"
        )

    trainable = [param for param in model.parameters() if param.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters available for mapping.focus_validation")
    optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    input_tensor = torch.tensor(
        input_phase_arr[None, :],
        dtype=torch.float32,
        device=next(model.parameters()).device,
    )
    target_tensor = torch.zeros(
        (1, len(model.outputposition)),
        dtype=torch.float32,
        device=input_tensor.device,
    )
    target_tensor[0, target_port_index] = 1.0

    history: list[float] = []
    best_loss = float("inf")
    best_state_dict: dict[str, torch.Tensor] | None = None

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        raw_output = _forward_selected_wavelength(model, input_tensor)
        transformed = raw_output.clamp_min(clamp_min)
        transformed = transformed / transformed.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        loss = loss_fn(transformed, target_tensor)
        loss.backward()
        optimizer.step()
        for param in getattr(model, "slot_length_um", []):
            if not getattr(param, "requires_grad", False):
                continue
            if slot_length_max_um is None:
                param.data.clamp_(min=slot_length_min_um)
            else:
                param.data.clamp_(min=slot_length_min_um, max=slot_length_max_um)

        loss_value = float(loss.detach().cpu().item())
        history.append(loss_value)
        if loss_value <= best_loss:
            best_loss = loss_value
            best_state_dict = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

    if best_state_dict is None:
        raise RuntimeError("mapping.focus_validation did not record a best state dict")
    model.load_state_dict(best_state_dict)
    model.eval()

    with torch.no_grad():
        math_raw_output = _forward_selected_wavelength(model, input_tensor)
    input_phase_support = list(getattr(model, "last_input_phase_support_summary", []) or [])
    math_output = math_raw_output.clamp_min(clamp_min)
    math_output = math_output / math_output.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    math_output_response = math_output.detach().cpu().reshape(-1).numpy()

    if not hasattr(model, "us") or len(model.us) == 0:
        raise AttributeError("OpticalDONN did not expose mathematical fields in model.us")
    mathematical_line = _ensure_complex_line(model.us[-1])
    mathematical_y_um = _resolve_model_y_axis_um(model)

    internal_layer_count = int(model.num_layers) - 2
    if internal_layer_count <= 0:
        raise ValueError(
            "mapping.focus_validation requires at least one internal diffraction layer, "
            f"got num_layers={model.num_layers}"
        )

    mapping_factor = float(mapped_network_package["calibration_package"]["mapping_factor"])
    if mapping_factor == 0.0:
        raise ValueError("calibration_package.mapping_factor must be non-zero for focus validation")

    slot_pitch_um = float(focus_cfg.get("slot_pitch_um", shape_cfg.get("unit_spacing_um", 0.0)))
    if slot_pitch_um <= 0.0:
        raise ValueError(f"mapping.focus_validation.slot_pitch_um must be positive, got {slot_pitch_um}")
    slot_width_um = float(
        focus_cfg.get(
            "slot_width_um",
            float(shape_cfg.get("duty_cycle", 0.0)) * float(shape_cfg.get("unit_spacing_um", 0.0)),
        )
    )
    if slot_width_um <= 0.0:
        raise ValueError(f"mapping.focus_validation.slot_width_um must be positive, got {slot_width_um}")
    slot_material = str(focus_cfg.get("slot_material", "etch"))
    if not slot_material:
        raise ValueError("mapping.focus_validation.slot_material must be a non-empty string")

    shape = NetworkShape(**dict(shape_cfg))
    calibration_package = dict(mapped_network_package["calibration_package"])
    sim_dir_raw = calibration_package.get("sim_dir")
    if sim_dir_raw is None:
        raise KeyError("calibration_package.sim_dir is required for mapping.focus_validation")
    sim_dir = Path(sim_dir_raw)
    if not sim_dir.exists():
        raise FileNotFoundError(f"calibration_package.sim_dir does not exist: {sim_dir}")

    project_basename = str(focus_cfg.get("physical_project_name", "experiment")).strip()
    if not project_basename:
        raise ValueError("mapping.focus_validation.physical_project_name must be a non-empty string")
    project_suffix = ".lms" if bool(shape.use_var_fdtd) else ".fsp"
    source_project_path = sim_dir / f"{project_basename}{project_suffix}"
    if not source_project_path.exists():
        candidates = sorted(sim_dir.glob(f"*{project_suffix}"))
        if len(candidates) == 1:
            source_project_path = candidates[0]
        else:
            raise FileNotFoundError(
                "Cannot resolve the physical FDTD project for mapping.focus_validation: "
                f"requested={source_project_path}, candidates={[path.name for path in candidates]}"
            )

    try:
        import lumapi
    except Exception as exc:
        raise ImportError("lumapi is required for mapping.focus_validation full-FDTD verification") from exc

    from .lsf_special_scripts import write_add_silicon_slots_lsf

    internal_counts = [int(v) for v in shape.neurons_per_layer[1:-1]]
    if len(internal_counts) != internal_layer_count:
        raise ValueError(
            "Internal layer count mismatch while building focus validation geometry: "
            f"internal_counts={internal_counts}, internal_layer_count={internal_layer_count}"
        )

    phase_wrap_count = 0
    phase_wrap_by_layer: list[int] = []
    physical_lengths_by_layer: list[np.ndarray] = []
    raw_delta_by_layer: list[np.ndarray] = []
    wrapped_delta_by_layer: list[np.ndarray] = []
    for internal_idx, count in enumerate(internal_counts):
        raw_delta = torch.as_tensor(model.delta_phi[internal_idx + 1]).detach().cpu().numpy().reshape(-1)
        if raw_delta.size != count:
            raise ValueError(
                "Internal delta_phi size does not match shape.neurons_per_layer: "
                f"layer_index={internal_idx + 1}, expected={count}, got={raw_delta.size}"
            )

        wrapped_delta = raw_delta.copy()
        if mapping_factor < 0.0:
            positive_mask = wrapped_delta > 0.0
            phase_wrap_count += int(np.sum(positive_mask))
            phase_wrap_by_layer.append(int(np.sum(positive_mask)))
            wrapped_delta[positive_mask] = wrapped_delta[positive_mask] - 2.0 * np.ceil(
                wrapped_delta[positive_mask] / 2.0
            )
        else:
            negative_mask = wrapped_delta < 0.0
            phase_wrap_count += int(np.sum(negative_mask))
            phase_wrap_by_layer.append(int(np.sum(negative_mask)))
            wrapped_delta = np.mod(wrapped_delta, 2.0)

        slot_length_params = list(getattr(model, "slot_length_um", []))
        if slot_length_params and internal_idx + 1 < len(slot_length_params):
            physical_lengths_um = (
                torch.as_tensor(slot_length_params[internal_idx + 1])
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )
            if physical_lengths_um.size != count:
                raise ValueError(
                    "Internal slot_length_um size does not match shape.neurons_per_layer: "
                    f"layer_index={internal_idx + 1}, expected={count}, got={physical_lengths_um.size}"
                )
        else:
            physical_lengths_um = wrapped_delta * mapping_factor
        if np.any(~np.isfinite(physical_lengths_um)):
            raise ValueError(
                "Non-finite slot lengths encountered in focus validation: "
                f"layer_index={internal_idx + 1}, lengths={physical_lengths_um.tolist()}"
            )
        if np.any(physical_lengths_um <= 0.0):
            bad = physical_lengths_um[physical_lengths_um <= 0.0]
            raise ValueError(
                "Non-positive slot lengths encountered in focus validation: "
                f"layer_index={internal_idx + 1}, bad_count={bad.size}, min_length_um={float(np.min(bad))}"
            )

        raw_delta_by_layer.append(raw_delta)
        wrapped_delta_by_layer.append(wrapped_delta)
        physical_lengths_by_layer.append(physical_lengths_um)

    internal_layer_x_um = _resolve_centered_layer_x_um(shape_cfg, int(shape.num_layers))[1:-1]
    if internal_layer_x_um.size != len(internal_counts):
        raise ValueError(
            "Resolved internal layer x positions do not match internal layer count: "
            f"internal_layer_x_um={internal_layer_x_um.tolist()}, internal_counts={internal_counts}"
        )
    slot_positions = [
        (
            float(internal_layer_x_um[layer_idx]),
            -0.5 * float(shape.total_width_um) + 0.5 * slot_pitch_um,
        )
        for layer_idx, count in enumerate(internal_counts)
    ]

    physical_workdir = validation_dir / "physical_fdtd"
    physical_workdir.mkdir(parents=True, exist_ok=True)
    copied_project_path = physical_workdir / source_project_path.name
    shutil.copy2(source_project_path, copied_project_path)
    slot_script_path = write_add_silicon_slots_lsf(
        shape,
        physical_workdir / "focus_validation_add_slots.lsf",
        params=np.concatenate(physical_lengths_by_layer).tolist(),
        positions=slot_positions,
        material=slot_material,
        z_span_um=float(shape.propagation_thickness_um),
        width_um=slot_width_um,
        spacing_um=slot_pitch_um,
        rect_prefix="focus_rect",
    )

    fdtd = (lumapi.MODE if bool(shape.use_var_fdtd) else lumapi.FDTD)(
        hide=bool(focus_cfg.get("hide_fdtd_cad", True))
    )
    try:
        fdtd.cd(str(physical_workdir))
        fdtd.load(str(copied_project_path))
        fdtd.eval("switchtolayout;")

        for source_idx in range(len(model.inputposition)):
            source_name = f"mode_src{source_idx}"
            if int(fdtd.getnamednumber(source_name)) < 1:
                raise ValueError(
                    "mapping.focus_validation could not find mode source in copied project: "
                    f"project={copied_project_path}, source_name={source_name}"
                )

        if np.any(np.abs(input_phase_arr) > 1e-8):
            phase_lines = ["switchtolayout;"]
            for source_idx, phase_pi in enumerate(input_phase_arr.tolist()):
                phase_lines.append(
                    f"setnamed('mode_src{source_idx}','phase',{float(phase_pi) * 180.0});"
                )
            fdtd.eval("\n".join(phase_lines))

        fdtd.eval(slot_script_path.read_text(encoding="utf-8"))
        fdtd.save(str(copied_project_path))
        fdtd.eval("run;")

        output_monitor_name = f"monitor{len(calibration_package['monitors']) - 1}"
        if int(fdtd.getnamednumber(output_monitor_name)) < 1:
            raise ValueError(
                "mapping.focus_validation could not find output monitor in copied project: "
                f"project={copied_project_path}, output_monitor_name={output_monitor_name}"
            )

        physical_fields = get_fields(
            fdtd,
            monitor_name=output_monitor_name,
            field_result_name="focus_validation_fields",
            get_eps=False,
            get_D=False,
            get_H=False,
            nointerpolation=False,
            unfold_symmetry=False,
        )
        physical_payload = monitor_payload_from_fields(physical_fields, include_axes=True)
        physical_pick = extract_ey_1d_at_wavelength(
            physical_payload,
            wavelength_um=float(calibration_package["selected_wavelength_um"]),
            selection="nearest",
        )
        physical_line = _ensure_complex_line(physical_pick["ey"])
        physical_y_um = _ensure_float_array(physical_pick["y"])
    finally:
        try:
            fdtd.close()
        except Exception:
            pass

    if physical_line.size != mathematical_line.size or not np.allclose(physical_y_um, mathematical_y_um):
        physical_line = _interp_complex_line(physical_y_um, physical_line, mathematical_y_um)
        physical_y_um = mathematical_y_um

    _, output_y_um = _resolve_io_y_um(
        shape_cfg,
        dict(mapped_network_package["optical_donn_config"]),
    )
    mathematical_port_amplitude = _sample_amplitude_at_positions(
        mathematical_y_um,
        mathematical_line,
        output_y_um,
    )
    physical_port_amplitude = _sample_amplitude_at_positions(
        physical_y_um,
        physical_line,
        output_y_um,
    )
    math_field_port_response = mathematical_port_amplitude / np.clip(
        float(np.sum(mathematical_port_amplitude)),
        1e-12,
        None,
    )
    physical_field_port_response = physical_port_amplitude / np.clip(
        float(np.sum(physical_port_amplitude)),
        1e-12,
        None,
    )
    math_peak_port_index = int(np.argmax(math_field_port_response))
    physical_peak_port_index = int(np.argmax(physical_field_port_response))

    fig = plt.figure(figsize=(15.0, 10.0))
    axes = fig.subplots(2, 2)

    loss_ax = axes[0, 0]
    loss_ax.plot(np.arange(1, len(history) + 1), history, color="#0f172a", linewidth=1.8)
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("MSE loss")
    loss_ax.set_title("Port-Focus Training Loss")
    loss_ax.grid(alpha=0.25, linewidth=0.6)

    response_ax = axes[0, 1]
    port_indices = np.arange(len(output_y_um))
    width = 0.36
    response_ax.bar(
        port_indices - 0.5 * width,
        math_field_port_response,
        width=width,
        color="#2563eb",
        label="Math field peak",
    )
    response_ax.bar(
        port_indices + 0.5 * width,
        physical_field_port_response,
        width=width,
        color="#dc2626",
        label="Full-FDTD field peak",
    )
    response_ax.set_xticks(port_indices)
    response_ax.set_xlabel("Output port")
    response_ax.set_ylabel("Normalized port-center amplitude")
    response_ax.set_title("Output Port Peak Comparison")
    response_ax.grid(alpha=0.25, linewidth=0.6)
    response_ax.legend(loc="upper right")

    field_ax = axes[1, 0]
    field_ax.plot(
        mathematical_y_um,
        np.abs(mathematical_line),
        color="#2563eb",
        linewidth=1.8,
        label="Math final field",
    )
    field_ax.plot(
        physical_y_um,
        np.abs(physical_line),
        color="#dc2626",
        linewidth=1.6,
        linestyle="--",
        label="Full-FDTD final field",
    )
    top_level = float(max(np.max(np.abs(mathematical_line)), np.max(np.abs(physical_line))))
    for port_idx, port_y_um in enumerate(output_y_um):
        color = "#16a34a" if port_idx == target_port_index else "#94a3b8"
        field_ax.axvline(float(port_y_um), color=color, linewidth=1.1, alpha=0.8)
        field_ax.text(
            float(port_y_um),
            top_level * 1.02,
            f"P{port_idx}",
            rotation=90,
            va="bottom",
            ha="center",
            fontsize=8,
            color=color,
        )
    field_ax.set_xlabel("Lateral y (um)")
    field_ax.set_ylabel("|Ey|")
    field_ax.set_title("Output Field Line Comparison")
    field_ax.grid(alpha=0.25, linewidth=0.6)
    field_ax.legend(loc="upper right")

    slot_ax = axes[1, 1]
    for layer_idx, physical_lengths_um in enumerate(physical_lengths_by_layer):
        slot_ax.plot(
            np.arange(physical_lengths_um.size),
            physical_lengths_um,
            linewidth=1.2,
            label=f"Layer {layer_idx + 1}",
        )
    slot_ax.set_xlabel("Neuron index")
    slot_ax.set_ylabel("Mapped slot length (um)")
    slot_ax.set_title("Learned Physical Slot Lengths")
    slot_ax.grid(alpha=0.25, linewidth=0.6)
    slot_ax.legend(loc="upper right")

    fig.suptitle(
        "Focus validation: train a small port-focusing target, then compare the mathematical output field "
        "against a full-FDTD rerun of the mapped physical structure.",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = out_dir / "focus_validation_fdtd.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    data_path = out_dir / "focus_validation_fdtd_data.npz"
    np.savez(
        data_path,
        history=np.asarray(history, dtype=float),
        input_phase=np.asarray(input_phase_arr, dtype=float),
        output_y_um=np.asarray(output_y_um, dtype=float),
        mathematical_y_um=np.asarray(mathematical_y_um, dtype=float),
        physical_y_um=np.asarray(physical_y_um, dtype=float),
        mathematical_line=mathematical_line,
        physical_line=physical_line,
        math_training_output_response=np.asarray(math_output_response, dtype=float),
        math_field_port_response=np.asarray(math_field_port_response, dtype=float),
        physical_field_port_response=np.asarray(physical_field_port_response, dtype=float),
        raw_delta_by_layer=np.asarray(raw_delta_by_layer, dtype=object),
        wrapped_delta_by_layer=np.asarray(wrapped_delta_by_layer, dtype=object),
        physical_lengths_um_by_layer=np.asarray(physical_lengths_by_layer, dtype=object),
    )

    metrics: Dict[str, Any] = {
        "focus_validation_fdtd": str(out_path),
        "focus_validation_fdtd_data": str(data_path),
        "focus_validation_target_port_index": target_port_index,
        "focus_validation_input_phase": input_phase_arr.tolist(),
        "focus_validation_input_phase_support": input_phase_support,
        "focus_validation_epochs": epochs,
        "focus_validation_lr": lr,
        "focus_validation_weight_decay": weight_decay,
        "focus_validation_slot_length_min_um": slot_length_min_um,
        "focus_validation_slot_length_max_um": slot_length_max_um,
        "focus_validation_best_loss": best_loss,
        "focus_validation_math_training_output_response": math_output_response.tolist(),
        "focus_validation_math_field_port_response": math_field_port_response.tolist(),
        "focus_validation_physical_field_port_response": physical_field_port_response.tolist(),
        "focus_validation_math_field_peak_port_index": math_peak_port_index,
        "focus_validation_physical_field_peak_port_index": physical_peak_port_index,
        "focus_validation_math_matches_target": bool(math_peak_port_index == target_port_index),
        "focus_validation_physical_matches_target": bool(physical_peak_port_index == target_port_index),
        "focus_validation_output_positions_um": output_y_um.tolist(),
        "focus_validation_slot_pitch_um": slot_pitch_um,
        "focus_validation_slot_width_um": slot_width_um,
        "focus_validation_slot_material": slot_material,
        "focus_validation_mapping_factor_um": mapping_factor,
        "focus_validation_phase_wrap_count": phase_wrap_count,
        "focus_validation_phase_wrap_count_by_layer": phase_wrap_by_layer,
        "focus_validation_raw_delta_phi_min_by_layer": [
            float(np.min(values)) for values in raw_delta_by_layer
        ],
        "focus_validation_raw_delta_phi_max_by_layer": [
            float(np.max(values)) for values in raw_delta_by_layer
        ],
        "focus_validation_physical_length_min_um_by_layer": [
            float(np.min(values)) for values in physical_lengths_by_layer
        ],
        "focus_validation_physical_length_max_um_by_layer": [
            float(np.max(values)) for values in physical_lengths_by_layer
        ],
        "focus_validation_physical_project_source": str(source_project_path),
        "focus_validation_physical_project_copy": str(copied_project_path),
        "focus_validation_slot_script": str(slot_script_path),
        "focus_validation_output_monitor_name": output_monitor_name,
        "focus_validation_internal_layer_x_um": internal_layer_x_um.tolist(),
    }
    warnings: list[str] = []
    if phase_wrap_count > 0:
        warnings.append(
            "focus_validation used 2pi-equivalent phase wrapping to convert trained delta_phi values "
            "into positive physical slot lengths."
        )
    return out_path, metrics, warnings


def _save_layer_line_plots(
    field_map: np.ndarray,
    y_axis_um: np.ndarray,
    *,
    out_dir: Path,
) -> Dict[str, Any]:
    plt = _pyplot()
    outputs: Dict[str, Path] = {}
    amplitude_paths: list[str] = []
    phase_paths: list[str] = []

    for layer_idx in range(field_map.shape[0]):
        layer = np.asarray(field_map[layer_idx])

        amp_fig, amp_ax = plt.subplots(figsize=(9.2, 4.6))
        amp_ax.plot(y_axis_um, np.abs(layer), color="#2563eb", linewidth=1.2)
        amp_ax.set_xlabel("Lateral y (um)")
        amp_ax.set_ylabel("|E|")
        amp_ax.set_title(f"Mathematical Layer {layer_idx} Field Amplitude")
        amp_ax.grid(alpha=0.2, linewidth=0.5)
        amp_fig.tight_layout()
        amp_path = out_dir / f"layer_{layer_idx}_field_amplitude.png"
        amp_fig.savefig(amp_path, dpi=180)
        plt.close(amp_fig)
        amplitude_paths.append(str(amp_path))

        phase_fig, phase_ax = plt.subplots(figsize=(9.2, 4.6))
        phase_ax.plot(y_axis_um, np.angle(layer), color="#dc2626", linewidth=1.1)
        phase_ax.set_xlabel("Lateral y (um)")
        phase_ax.set_ylabel("phase (rad)")
        phase_ax.set_title(f"Mathematical Layer {layer_idx} Field Phase")
        phase_ax.grid(alpha=0.2, linewidth=0.5)
        phase_fig.tight_layout()
        phase_path = out_dir / f"layer_{layer_idx}_field_phase.png"
        phase_fig.savefig(phase_path, dpi=180)
        plt.close(phase_fig)
        phase_paths.append(str(phase_path))

    outputs["layer_field_amplitudes"] = amplitude_paths
    outputs["layer_field_phases"] = phase_paths
    return outputs


def _save_output_ports(
    outputs: np.ndarray,
    *,
    out_dir: Path,
) -> Path:
    plt = _pyplot()
    values = np.asarray(outputs, dtype=float).reshape(-1)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(np.arange(values.size), values, color="#7c3aed")
    ax.set_xlabel("Output port")
    ax.set_ylabel("Response")
    ax.set_title("Mapped Output Response (zero input phase)")
    ax.set_xticks(np.arange(values.size))
    fig.tight_layout()
    out_path = out_dir / "output_ports.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _save_physical_monitor_maps(
    calibration_package: Mapping[str, Any],
    *,
    out_dir: Path,
) -> Dict[str, Path]:
    plt = _pyplot()
    global_fields = _extract_global_field_maps(calibration_package)
    x_axis_um = global_fields["x_axis_um"]
    y_axis_um = global_fields["y_axis_um"]
    extent = [float(x_axis_um[0]), float(x_axis_um[-1]), float(y_axis_um[0]), float(y_axis_um[-1])]
    outputs: Dict[str, Path] = {}

    amp_fig, amp_ax = plt.subplots(figsize=(9.5, 5.2))
    amp_im = amp_ax.imshow(
        np.asarray(global_fields["amplitude"]).T,
        aspect="auto",
        extent=extent,
        origin="lower",
        interpolation="nearest",
    )
    amp_fig.colorbar(amp_im, ax=amp_ax, label="|E|")
    amp_ax.set_xlabel("Propagation x (um)")
    amp_ax.set_ylabel("Lateral y (um)")
    amp_ax.set_title(
        "Physical Global Field Amplitude In XY Plane\n"
        f"lambda={float(global_fields['selected_wavelength_um']):.6g} um"
    )
    amp_fig.tight_layout()
    amp_path = out_dir / "physical_monitor_amplitude.png"
    amp_fig.savefig(amp_path, dpi=180)
    plt.close(amp_fig)
    outputs["physical_monitor_amplitude"] = amp_path

    phase_fig, phase_ax = plt.subplots(figsize=(9.5, 5.2))
    phase_im = phase_ax.imshow(
        np.asarray(global_fields["phase"]).T,
        aspect="auto",
        extent=extent,
        origin="lower",
        interpolation="nearest",
        cmap="twilight",
    )
    phase_fig.colorbar(phase_im, ax=phase_ax, label="phase (rad)")
    phase_ax.set_xlabel("Propagation x (um)")
    phase_ax.set_ylabel("Lateral y (um)")
    phase_ax.set_title("Physical Global Field Phase In XY Plane (phase of dominant E component)")
    phase_fig.tight_layout()
    phase_path = out_dir / "physical_monitor_phase.png"
    phase_fig.savefig(phase_path, dpi=180)
    plt.close(phase_fig)
    outputs["physical_monitor_phase"] = phase_path

    intensity_fig, intensity_ax = plt.subplots(figsize=(9.5, 5.2))
    intensity_im = intensity_ax.imshow(
        np.asarray(global_fields["intensity"]).T,
        aspect="auto",
        extent=extent,
        origin="lower",
        interpolation="nearest",
        cmap="inferno",
    )
    intensity_fig.colorbar(intensity_im, ax=intensity_ax, label="|E|^2")
    intensity_ax.set_xlabel("Propagation x (um)")
    intensity_ax.set_ylabel("Lateral y (um)")
    intensity_ax.set_title("Physical Global Field Intensity In XY Plane")
    intensity_fig.tight_layout()
    intensity_path = out_dir / "physical_global_intensity.png"
    intensity_fig.savefig(intensity_path, dpi=180)
    plt.close(intensity_fig)
    outputs["physical_global_intensity"] = intensity_path
    return outputs


def _compute_model_propagation_slices(model: Any, *, step_um: float = 0.25) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(model.us) < 1:
        raise RuntimeError("forward() did not populate fields (us)")

    span_um = float(model.positions_span[0])
    y_axis_um = np.linspace(-0.5 * span_um, 0.5 * span_um, int(model.us[0].shape[-1]), dtype=float)
    dy_m = (span_um * 1e-6) / len(y_axis_um)
    ky = 2 * np.pi * np.fft.fftfreq(len(y_axis_um), d=dy_m)

    slices = []
    x_positions = []
    current_x = 0.0
    layer_spacings = np.asarray(model.layer_spacing, dtype=float).reshape(-1)

    for layer_idx in range(len(model.us) - 1):
        u_layer = np.asarray(torch.as_tensor(model.us[layer_idx]).detach().cpu()).reshape(-1)
        transmission = np.asarray(model.transmission_coefficient(layer_idx).detach().cpu()).reshape(-1)
        u_layer = u_layer * transmission

        remaining = float(layer_spacings[layer_idx]) if layer_idx < layer_spacings.size else 0.0
        while remaining > 1e-9:
            d_um = min(step_um, remaining)
            H = np.asarray(
                model.build_propagator_kernel(
                    d_um,
                    ky=ky,
                    mode="full",
                    ky_cut_ratio=1.0,
                    smooth_rolloff=0.0,
                    coarse_factor=1.0,
                    pad_factor=1,
                    as_tensor=False,
                )
            )
            U = np.fft.fft(np.fft.ifftshift(u_layer))
            u_layer = np.fft.fftshift(np.fft.ifft(U * H))
            current_x += d_um
            x_positions.append(current_x)
            slices.append(np.asarray(u_layer, dtype=np.complex128))
            remaining -= d_um

    if not slices:
        raise RuntimeError("No mathematical propagation slices generated")
    return np.stack(slices, axis=0), y_axis_um, np.asarray(x_positions, dtype=float)


def _save_mathematical_propagation_maps(
    model: Any,
    *,
    out_dir: Path,
) -> Dict[str, Path]:
    plt = _pyplot()
    field_map, y_axis_um, x_axis_um = _compute_model_propagation_slices(model)
    extent = [float(y_axis_um[0]), float(y_axis_um[-1]), float(x_axis_um[0]), float(x_axis_um[-1])]
    outputs: Dict[str, Path] = {}

    amp_fig, amp_ax = plt.subplots(figsize=(9.5, 5.2))
    amp_im = amp_ax.imshow(
        np.abs(field_map),
        aspect="auto",
        extent=extent,
        origin="lower",
    )
    amp_fig.colorbar(amp_im, ax=amp_ax, label="|E|")
    amp_ax.set_xlabel("Lateral y (um)")
    amp_ax.set_ylabel("Propagation x (um)")
    amp_ax.set_title("Mathematical Optical Field Amplitude")
    amp_fig.tight_layout()
    amp_path = out_dir / "mathematical_propagation_amplitude.png"
    amp_fig.savefig(amp_path, dpi=180)
    plt.close(amp_fig)
    outputs["mathematical_propagation_amplitude"] = amp_path

    phase_fig, phase_ax = plt.subplots(figsize=(9.5, 5.2))
    phase_im = phase_ax.imshow(
        np.angle(field_map),
        aspect="auto",
        extent=extent,
        origin="lower",
        cmap="twilight",
    )
    phase_fig.colorbar(phase_im, ax=phase_ax, label="phase (rad)")
    phase_ax.set_xlabel("Lateral y (um)")
    phase_ax.set_ylabel("Propagation x (um)")
    phase_ax.set_title("Mathematical Optical Field Phase")
    phase_fig.tight_layout()
    phase_path = out_dir / "mathematical_propagation_phase.png"
    phase_fig.savefig(phase_path, dpi=180)
    plt.close(phase_fig)
    outputs["mathematical_propagation_phase"] = phase_path
    return outputs


def _save_layer_reference_compare(
    calibration_package: Mapping[str, Any],
    model: Any,
    shape_cfg: Mapping[str, Any],
    *,
    out_dir: Path,
) -> tuple[Path, Dict[str, float]]:
    plt = _pyplot()
    monitor_view = extract_monitor_lines_from_calibration_package(calibration_package)
    physical = _stack_lines(monitor_view["monitors"])
    y_axis_um = _resolve_field_y_axis_um(shape_cfg, physical.shape[1])
    mathematical_raw = _stack_lines(model.us)
    mathematical_y_um = _resolve_model_y_axis_um(model)
    layer_count = physical.shape[0]

    fig, axes = plt.subplots(layer_count, 2, figsize=(12, 3.2 * layer_count), squeeze=False)
    amp_max_diff = 0.0
    phase_max_diff = 0.0
    for layer_idx in range(layer_count):
        phys = physical[layer_idx]
        math_line = mathematical_raw[layer_idx]
        if math_line.size != y_axis_um.size or not np.allclose(mathematical_y_um, y_axis_um):
            math_line = _interp_complex_line(mathematical_y_um, math_line, y_axis_um)

        amp_max_diff = max(amp_max_diff, float(np.max(np.abs(np.abs(phys) - np.abs(math_line)))))
        phase_delta = np.angle(np.exp(1j * (np.angle(phys) - np.angle(math_line))))
        phase_max_diff = max(phase_max_diff, float(np.max(np.abs(phase_delta))))

        amp_ax = axes[layer_idx, 0]
        amp_ax.plot(y_axis_um, np.abs(phys), label="physical", color="#2563eb", linewidth=1.1)
        amp_ax.plot(y_axis_um, np.abs(math_line), label="networkangler", color="#dc2626", linewidth=1.0, linestyle="--")
        amp_ax.set_ylabel(f"L{layer_idx} |E|")
        if layer_idx == 0:
            amp_ax.set_title("Per-layer physical vs networkangler amplitude")
            amp_ax.legend(loc="upper right")

        phase_ax = axes[layer_idx, 1]
        phase_ax.plot(y_axis_um, np.angle(phys), label="physical", color="#2563eb", linewidth=1.1)
        phase_ax.plot(y_axis_um, np.angle(math_line), label="networkangler", color="#dc2626", linewidth=1.0, linestyle="--")
        phase_ax.set_ylabel(f"L{layer_idx} phase")
        if layer_idx == 0:
            phase_ax.set_title("Per-layer physical vs networkangler phase")

    for ax in axes[-1, :]:
        ax.set_xlabel("Lateral y (um)")
    fig.tight_layout()

    out_path = out_dir / "layer_reference_compare.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    metrics = {
        "physical_vs_networkangler_amplitude_max_abs_diff": amp_max_diff,
        "physical_vs_networkangler_phase_max_abs_diff_rad": phase_max_diff,
    }
    return out_path, metrics


def _diagnose_visualization_state(
    mapped_network_package: Mapping[str, Any],
    model: Any,
) -> list[str]:
    warnings: list[str] = []
    shape_cfg = dict(mapped_network_package["shape_spec"]["shape_config"])
    optical_cfg = dict(mapped_network_package["optical_donn_config"])

    physical_span_um = float(shape_cfg["total_width_um"])
    model_span_um = float(model.positions_span[0])
    if abs(model_span_um - physical_span_um) > 1e-6:
        warnings.append(
            "model field span does not match physical monitor span: "
            f"model={model_span_um:.6g} um, physical={physical_span_um:.6g} um"
        )

    input_y_um, output_y_um = _resolve_io_y_um(shape_cfg, optical_cfg)
    half_model_span = 0.5 * model_span_um
    if input_y_um.size and np.any(np.abs(input_y_um) > half_model_span + 1e-6):
        warnings.append(
            "input positions extend outside model field span; mathematical mapping may clip inputs"
        )
    if output_y_um.size and np.any(np.abs(output_y_um) > half_model_span + 1e-6):
        warnings.append(
            "output positions extend outside model field span; output overlap may be inconsistent"
        )
    return warnings


def save_map_visualizations(
    mapped_network_package: Mapping[str, Any],
    *,
    out_dir: str | Path,
) -> Dict[str, Any]:
    figures_dir = Path(out_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    saved: Dict[str, Any] = {}
    errors: list[str] = []
    warnings: list[str] = []

    shape_cfg = dict(mapped_network_package["shape_spec"]["shape_config"])
    calibration = dict(mapped_network_package["calibration_package"])

    try:
        saved["layout_overview"] = str(_save_layout_overview(mapped_network_package, out_dir=figures_dir))
    except Exception as exc:
        errors.append(f"layout_overview: {exc}")

    try:
        saved["index_monitor_xy"] = str(_save_index_monitor_xy(mapped_network_package, out_dir=figures_dir))
        saved["index_monitor_zx"] = str(_save_index_zx(mapped_network_package, out_dir=figures_dir))
        warnings.append(
            "index_monitor_zx is derived from the global XY index monitor plus the configured z stack; "
            "the current varFDTD workflow does not expose a true global Y-normal monitor."
        )
    except Exception as exc:
        errors.append(f"geometry_visualization: {exc}")

    try:
        for key, path in _save_physical_monitor_maps(
            calibration,
            out_dir=figures_dir,
        ).items():
            saved[key] = str(path)
    except Exception as exc:
        errors.append(f"physical_monitor_visualization: {exc}")

    try:
        model, outputs = _run_reference_forward(
            mapped_network_package,
            out_dir=figures_dir,
        )
        math_layers = _stack_lines(model.us)
        model_y_axis_um = _resolve_model_y_axis_um(model)
        for key, value in _save_layer_line_plots(
            math_layers,
            model_y_axis_um,
            out_dir=figures_dir,
        ).items():
            saved[key] = value
        for key, path in _save_mathematical_propagation_maps(model, out_dir=figures_dir).items():
            saved[key] = str(path)
        saved["output_ports"] = str(_save_output_ports(outputs, out_dir=figures_dir))
        compare_path, metrics = _save_layer_reference_compare(
            calibration,
            model,
            shape_cfg,
            out_dir=figures_dir,
        )
        saved["layer_reference_compare"] = str(compare_path)
        saved.update(metrics)
        warnings.extend(_diagnose_visualization_state(mapped_network_package, model))
    except Exception as exc:
        errors.append(f"mathematical_visualization: {exc}")

    focus_cfg = dict((mapped_network_package.get("mapping_config") or {}).get("focus_validation") or {})
    if bool(focus_cfg.get("enabled", False)):
        try:
            focus_backend = str(focus_cfg.get("backend", "hybrid")).strip().lower()
            if focus_backend == "hybrid":
                requested_wavelengths = focus_cfg.get("wavelengths_um")
                if requested_wavelengths is None:
                    focus_path, focus_metrics, focus_warnings = _save_focus_validation_hybrid(
                        mapped_network_package,
                        shape_cfg,
                        out_dir=figures_dir,
                    )
                    saved["focus_validation"] = str(focus_path)
                    saved.update(focus_metrics)
                    warnings.extend(focus_warnings)
                else:
                    if isinstance(requested_wavelengths, (str, bytes)) or not isinstance(
                        requested_wavelengths, Sequence
                    ):
                        raise TypeError("mapping.focus_validation.wavelengths_um must be a sequence of wavelengths")
                    requested_values = [float(value) for value in requested_wavelengths]
                    if len(requested_values) == 0:
                        raise ValueError("mapping.focus_validation.wavelengths_um must not be empty")

                    summary: Dict[str, Any] = {}
                    for idx, requested_wavelength_um in enumerate(requested_values):
                        focus_path, focus_metrics, focus_warnings = _save_focus_validation_hybrid(
                            mapped_network_package,
                            shape_cfg,
                            out_dir=figures_dir,
                            wavelength_um=requested_wavelength_um,
                            artifact_suffix=f"{idx:02d}_{_focus_validation_suffix(requested_wavelength_um)}",
                        )
                        resolved_wavelength_um = float(
                            focus_metrics["focus_validation_selected_wavelength_um"]
                        )
                        wavelength_key = canonical_wavelength_key(resolved_wavelength_um)
                        saved[f"focus_validation_hybrid_{wavelength_key}"] = str(focus_path)
                        summary[wavelength_key] = dict(focus_metrics)
                        warnings.extend(focus_warnings)

                    summary_path = figures_dir / "focus_validation_hybrid_summary.json"
                    summary_path.write_text(
                        json.dumps(summary, indent=2, sort_keys=True),
                        encoding="utf-8",
                    )
                    saved["focus_validation"] = str(summary_path)
                    saved["focus_validation_summary"] = str(summary_path)
                    saved["focus_validation_requested_wavelengths_um"] = requested_values
            elif focus_backend in ("fdtd", "full_fdtd"):
                focus_path, focus_metrics, focus_warnings = _save_focus_validation_fdtd(
                    mapped_network_package,
                    shape_cfg,
                    out_dir=figures_dir,
                )
                saved["focus_validation"] = str(focus_path)
                saved.update(focus_metrics)
                warnings.extend(focus_warnings)
            else:
                raise ValueError(
                    "mapping.focus_validation.backend must be 'hybrid' or 'fdtd', "
                    f"got {focus_backend!r}"
                )
        except Exception as exc:
            errors.append(f"focus_validation: {exc}")

    saved["figures_dir"] = str(figures_dir)
    saved["errors"] = errors
    saved["warnings"] = warnings
    return saved
