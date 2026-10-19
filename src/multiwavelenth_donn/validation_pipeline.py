from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch

from .config_loading import load_project_config
from .config_sections import resolve_validation_config
from .experiment_artifacts import instantiate_optical_donn_from_trained_package
from .experiment_artifacts import load_package
from .experiment_artifacts import save_json
from .experiment_visualization import _ensure_complex_line
from .experiment_visualization import _field_intensity
from .experiment_visualization import _forward_requested_wavelength
from .experiment_visualization import _interp_complex_line
from .experiment_visualization import _normalize_sum
from .experiment_visualization import _pyplot
from .experiment_visualization import _resolve_centered_layer_x_um
from .experiment_visualization import _resolve_focus_validation_wavelength
from .experiment_visualization import _resolve_io_y_um
from .experiment_visualization import _resolve_model_y_axis_um
from .experiment_visualization import _sample_amplitude_at_positions
from .experiment_visualization import _windowed_intensity_at_positions
from .lumerical_scripts import extract_ey_1d_at_wavelength
from .lumerical_scripts import get_fields
from .lumerical_scripts import monitor_payload_from_fields
from .networkangler import NetworkShape
from .project_paths import get_project_root
from .workspace_layout import ValidationJobLayout
from .workspace_layout import resolve_validation_config_path
from .workspace_layout import resolve_validation_job_layout


def _resolve_validation_input(
    model: Any,
    trained_network_package: Mapping[str, Any],
    validation_cfg: Mapping[str, Any],
    *,
    layout: ValidationJobLayout,
) -> np.ndarray:
    input_phase = validation_cfg.get("input_phase")
    if input_phase is not None:
        resolved = np.asarray(input_phase, dtype=np.float32)
    else:
        reference_artifacts = dict(trained_network_package.get("reference_artifacts") or {})
        reference_input_path = reference_artifacts.get("reference_input_path")
        if reference_input_path is None:
            resolved = np.zeros(len(model.inputposition), dtype=np.float32)
        else:
            path = Path(reference_input_path)
            if not path.is_absolute():
                path = (layout.train_bundle_dir / path).resolve()
            if not path.exists():
                raise FileNotFoundError(
                    "validation could not find trained reference input file: "
                    f"resolved={path}, train_bundle_dir={layout.train_bundle_dir}"
                )
            resolved = np.asarray(np.load(path), dtype=np.float32)

    resolved = np.asarray(resolved, dtype=np.float32).reshape(-1)
    if resolved.size != len(model.inputposition):
        raise ValueError(
            "validation input phase size mismatch: "
            f"expected {len(model.inputposition)}, got {resolved.size}"
        )
    return resolved


def _resolve_expected_port_index(
    validation_cfg: Mapping[str, Any],
    *,
    num_outputs: int,
) -> int | None:
    raw = validation_cfg.get("expected_port_index")
    if raw is None:
        return None
    expected_port_index = int(raw)
    if expected_port_index < 0 or expected_port_index >= num_outputs:
        raise ValueError(
            "validation.expected_port_index is out of range: "
            f"expected_port_index={expected_port_index}, num_outputs={num_outputs}"
        )
    return expected_port_index


def _prepare_job_dir(layout: ValidationJobLayout) -> Path | None:
    job_dir = layout.job_dir
    backup_dir: Path | None = None
    if job_dir.exists() and any(job_dir.iterdir()):
        archive_root = job_dir.parent / "artifacts_history"
        archive_root.mkdir(parents=True, exist_ok=True)
        backup_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = archive_root / backup_name
        suffix = 1
        while backup_dir.exists():
            backup_dir = archive_root / f"{backup_name}_{suffix:02d}"
            suffix += 1
        shutil.copytree(job_dir, backup_dir)
        shutil.rmtree(job_dir)

    job_dir.mkdir(parents=True, exist_ok=True)
    return backup_dir


def _resolve_validation_geometry(
    model: Any,
    *,
    shape_cfg: Mapping[str, Any],
    validation_cfg: Mapping[str, Any],
    wavelength_um: float,
) -> dict[str, Any]:
    internal_layer_count = int(model.num_layers) - 2
    if internal_layer_count <= 0:
        raise ValueError(
            "validation requires at least one internal diffraction layer, "
            f"got num_layers={model.num_layers}"
        )

    mapping_factor = float(model._mapping_factor_for_wavelength(wavelength_um=wavelength_um))
    if mapping_factor == 0.0:
        raise ValueError(
            "calibration_package.mapping_factor must be non-zero for validation"
        )

    slot_pitch_um = float(
        validation_cfg.get("slot_pitch_um", shape_cfg.get("unit_spacing_um", 0.0))
    )
    if slot_pitch_um <= 0.0:
        raise ValueError(
            f"validation.slot_pitch_um must be positive, got {slot_pitch_um}"
        )

    slot_width_um = float(
        validation_cfg.get(
            "slot_width_um",
            float(shape_cfg.get("duty_cycle", 0.0))
            * float(shape_cfg.get("unit_spacing_um", 0.0)),
        )
    )
    if slot_width_um <= 0.0:
        raise ValueError(
            f"validation.slot_width_um must be positive, got {slot_width_um}"
        )

    slot_material = str(validation_cfg.get("slot_material", "etch")).strip()
    if not slot_material:
        raise ValueError("validation.slot_material must be a non-empty string")

    shape = NetworkShape(**dict(shape_cfg))
    internal_counts = [int(v) for v in shape.neurons_per_layer[1:-1]]
    if len(internal_counts) != internal_layer_count:
        raise ValueError(
            "Internal layer count mismatch while building validation geometry: "
            f"internal_counts={internal_counts}, internal_layer_count={internal_layer_count}"
        )

    phase_wrap_count = 0
    phase_wrap_by_layer: list[int] = []
    raw_delta_by_layer: list[np.ndarray] = []
    wrapped_delta_by_layer: list[np.ndarray] = []
    physical_lengths_by_layer: list[np.ndarray] = []
    delta_phi_by_layer = model.get_delta_phi_by_wavelength(wavelength_um=wavelength_um)

    for internal_idx, count in enumerate(internal_counts):
        raw_delta = (
            torch.as_tensor(delta_phi_by_layer[internal_idx + 1])
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
            wraps = int(np.sum(positive_mask))
            wrapped_delta[positive_mask] = wrapped_delta[positive_mask] - 2.0 * np.ceil(
                wrapped_delta[positive_mask] / 2.0
            )
        else:
            negative_mask = wrapped_delta < 0.0
            wraps = int(np.sum(negative_mask))
            wrapped_delta = np.mod(wrapped_delta, 2.0)
        phase_wrap_count += wraps
        phase_wrap_by_layer.append(wraps)

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
                "Non-finite slot lengths encountered in validation: "
                f"layer_index={internal_idx + 1}, lengths={physical_lengths_um.tolist()}"
            )
        if np.any(physical_lengths_um <= 0.0):
            bad = physical_lengths_um[physical_lengths_um <= 0.0]
            raise ValueError(
                "Non-positive slot lengths encountered in validation: "
                f"layer_index={internal_idx + 1}, bad_count={bad.size}, "
                f"min_length_um={float(np.min(bad))}"
            )

        raw_delta_by_layer.append(raw_delta)
        wrapped_delta_by_layer.append(wrapped_delta)
        physical_lengths_by_layer.append(physical_lengths_um)

    return {
        "shape": shape,
        "internal_counts": internal_counts,
        "mapping_factor": mapping_factor,
        "slot_pitch_um": slot_pitch_um,
        "slot_width_um": slot_width_um,
        "slot_material": slot_material,
        "phase_wrap_count": phase_wrap_count,
        "phase_wrap_count_by_layer": phase_wrap_by_layer,
        "raw_delta_by_layer": raw_delta_by_layer,
        "wrapped_delta_by_layer": wrapped_delta_by_layer,
        "physical_lengths_by_layer": physical_lengths_by_layer,
        "internal_layer_x_um": _resolve_centered_layer_x_um(
            shape_cfg, int(shape.num_layers)
        )[1:-1],
    }


def _resolve_math_validation_state(
    model: Any,
    *,
    input_phase_arr: np.ndarray,
    wavelength_um: float,
    clamp_min: float,
) -> dict[str, Any]:
    input_tensor = torch.tensor(
        input_phase_arr[None, :],
        dtype=torch.float32,
        device=next(model.parameters()).device,
    )

    model.eval()
    with torch.no_grad():
        math_raw_output = _forward_requested_wavelength(
            model,
            input_tensor,
            wavelength_um=wavelength_um,
        )
    input_phase_support = list(
        getattr(model, "last_input_phase_support_summary", []) or []
    )
    math_output = math_raw_output.clamp_min(clamp_min)
    math_output = math_output / math_output.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    math_output_response = math_output.detach().cpu().reshape(-1).numpy()

    if not hasattr(model, "us") or len(model.us) == 0:
        raise AttributeError("OpticalDONN did not expose mathematical fields in model.us")

    mathematical_line = _ensure_complex_line(model.us[-1])
    mathematical_y_um = _resolve_model_y_axis_um(model)
    return {
        "input_tensor": input_tensor,
        "input_phase_support": input_phase_support,
        "math_output_response": math_output_response,
        "mathematical_line": mathematical_line,
        "mathematical_y_um": mathematical_y_um,
        "mathematical_intensity": _normalize_sum(_field_intensity(mathematical_line)),
    }


def _build_slot_length_summary(
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "phase_wrap_count": int(geometry["phase_wrap_count"]),
        "phase_wrap_count_by_layer": list(geometry["phase_wrap_count_by_layer"]),
        "raw_delta_phi_min_by_layer": [
            float(np.min(values)) for values in geometry["raw_delta_by_layer"]
        ],
        "raw_delta_phi_max_by_layer": [
            float(np.max(values)) for values in geometry["raw_delta_by_layer"]
        ],
        "physical_length_min_um_by_layer": [
            float(np.min(values)) for values in geometry["physical_lengths_by_layer"]
        ],
        "physical_length_max_um_by_layer": [
            float(np.max(values)) for values in geometry["physical_lengths_by_layer"]
        ],
    }


def _resolve_expected_label(
    expected_port_index: int | None,
    *,
    fallback_port_index: int,
) -> tuple[int | None, str]:
    if expected_port_index is None:
        return None, f"peak=P{fallback_port_index}"
    return expected_port_index, f"expected=P{expected_port_index}"


def _plot_validation_summary(
    *,
    out_path: Path,
    title: str,
    output_y_um: np.ndarray,
    mathematical_y_um: np.ndarray,
    physical_y_um: np.ndarray,
    mathematical_line: np.ndarray,
    physical_line: np.ndarray,
    math_output_response: np.ndarray,
    physical_output_response: np.ndarray,
    math_window_response: np.ndarray,
    physical_window_response: np.ndarray,
    physical_lengths_by_layer: list[np.ndarray],
    expected_port_index: int | None,
) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.4))

    peak_math = int(np.argmax(math_output_response))
    peak_physical = int(np.argmax(physical_output_response))
    expected_label_idx, expected_label = _resolve_expected_label(
        expected_port_index,
        fallback_port_index=peak_math,
    )

    output_ax = axes[0, 0]
    port_indices = np.arange(output_y_um.size)
    width = 0.36
    output_ax.bar(
        port_indices - 0.5 * width,
        math_output_response,
        width=width,
        color="#dc2626",
        label="math response",
    )
    output_ax.bar(
        port_indices + 0.5 * width,
        physical_output_response,
        width=width,
        color="#2563eb",
        label="physical response",
    )
    output_ax.set_xlabel("output port")
    output_ax.set_ylabel("normalized response")
    output_ax.set_title(
        f"Output Response Comparison\n{expected_label}, math_peak=P{peak_math}, physical_peak=P{peak_physical}"
    )
    output_ax.set_xticks(port_indices)
    output_ax.set_ylim(0.0, 1.05)
    output_ax.legend(loc="upper right")
    output_ax.grid(alpha=0.2, linewidth=0.5, axis="y")

    field_ax = axes[0, 1]
    field_ax.plot(
        mathematical_y_um,
        _normalize_sum(np.abs(mathematical_line) ** 2),
        color="#dc2626",
        linewidth=1.4,
        label="math output-plane intensity",
    )
    field_ax.plot(
        physical_y_um,
        _normalize_sum(np.abs(physical_line) ** 2),
        color="#2563eb",
        linewidth=1.2,
        linestyle="--",
        label="physical output-plane intensity",
    )
    for port_idx, port_y_um in enumerate(output_y_um):
        highlight = expected_label_idx is not None and port_idx == expected_label_idx
        color = "#f59e0b" if highlight else "#64748b"
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
    field_ax.set_title("Output-Plane Field Comparison")
    field_ax.grid(alpha=0.2, linewidth=0.5)
    field_ax.legend(loc="upper right")

    port_ax = axes[1, 0]
    port_ax.bar(
        port_indices - 0.5 * width,
        math_window_response,
        width=width,
        color="#dc2626",
        label="math windowed energy",
    )
    port_ax.bar(
        port_indices + 0.5 * width,
        physical_window_response,
        width=width,
        color="#2563eb",
        label="physical windowed energy",
    )
    port_ax.set_xlabel("output port")
    port_ax.set_ylabel("normalized windowed energy")
    port_ax.set_title("Port-Window Energy Comparison")
    port_ax.set_xticks(port_indices)
    port_ax.legend(loc="upper right")
    port_ax.grid(alpha=0.2, linewidth=0.5, axis="y")

    slot_ax = axes[1, 1]
    for layer_idx, physical_lengths_um in enumerate(physical_lengths_by_layer):
        slot_ax.plot(
            np.arange(physical_lengths_um.size),
            physical_lengths_um,
            linewidth=1.2,
            label=f"Layer {layer_idx + 1}",
        )
    slot_ax.set_xlabel("neuron index")
    slot_ax.set_ylabel("mapped slot length (um)")
    slot_ax.set_title("Trained Physical Slot Lengths")
    slot_ax.grid(alpha=0.2, linewidth=0.5)
    slot_ax.legend(loc="upper right")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _run_hybrid_validation(
    trained_network_package: Mapping[str, Any],
    validation_cfg: Mapping[str, Any],
    *,
    layout: ValidationJobLayout,
    job_dir: Path,
    shape_cfg: Mapping[str, Any],
    model: Any,
    input_phase_arr: np.ndarray,
    selected_wavelength_um: float,
    clamp_min: float,
    expected_port_index: int | None,
) -> dict[str, Any]:
    geometry = _resolve_validation_geometry(
        model,
        shape_cfg=shape_cfg,
        validation_cfg=validation_cfg,
        wavelength_um=selected_wavelength_um,
    )
    math_state = _resolve_math_validation_state(
        model,
        input_phase_arr=input_phase_arr,
        wavelength_um=selected_wavelength_um,
        clamp_min=clamp_min,
    )

    optical_cfg = dict(trained_network_package["mapped_network_package"]["optical_donn_config"])
    _, output_y_um = _resolve_io_y_um(shape_cfg, optical_cfg)
    port_window_um = float(
        validation_cfg.get(
            "port_window_um",
            float(optical_cfg.get("output_mapping_span_um", 2.0)),
        )
    )
    if port_window_um <= 0.0:
        raise ValueError(
            f"validation.port_window_um must be positive, got {port_window_um}"
        )

    calibration_package = dict(
        trained_network_package["mapped_network_package"]["calibration_package"]
    )
    sim_dir_raw = calibration_package.get("sim_dir")
    if sim_dir_raw is None:
        raise KeyError(
            "calibration_package.sim_dir is required for validation backend 'hybrid'"
        )
    sim_dir = Path(sim_dir_raw)
    if not sim_dir.exists():
        raise FileNotFoundError(f"calibration_package.sim_dir does not exist: {sim_dir}")

    hybrid_mode_path_raw = validation_cfg.get("hybrid_mode_path")
    if hybrid_mode_path_raw is None:
        candidate_mode_path = sim_dir / "mode1_E.mat"
        if candidate_mode_path.exists():
            hybrid_mode_path = candidate_mode_path
        else:
            hybrid_mode_path = Path(str(getattr(model, "mode_path", "")))
    else:
        hybrid_mode_path = Path(hybrid_mode_path_raw)
        if not hybrid_mode_path.is_absolute():
            bundle_relative = (layout.validation_bundle_dir / hybrid_mode_path).resolve()
            hybrid_mode_path = (
                bundle_relative
                if bundle_relative.exists()
                else (sim_dir / hybrid_mode_path).resolve()
            )
    if not hybrid_mode_path.exists():
        raise FileNotFoundError(
            "validation could not resolve a usable hybrid mode file: "
            f"resolved={hybrid_mode_path}, sim_dir={sim_dir}"
        )
    model.mode_path = str(hybrid_mode_path)

    try:
        import lumapi
    except Exception as exc:
        raise ImportError("lumapi is required for hybrid validation") from exc

    from .lsf_special_scripts import iter_hybrid_init_scripts
    from .lsf_special_scripts import write_add_silicon_slots_lsf
    from .lumdata import _safe_close

    hybrid_workdir = job_dir / "hybrid_fdtd"
    hybrid_workdir.mkdir(parents=True, exist_ok=True)
    hybrid_mesh_accuracy = int(validation_cfg.get("hybrid_mesh_accuracy", 5))
    if hybrid_mesh_accuracy <= 0:
        raise ValueError(
            "validation.hybrid_mesh_accuracy must be positive, "
            f"got {hybrid_mesh_accuracy}"
        )

    hide_fdtd_cad = bool(validation_cfg.get("hide_fdtd_cad", True))
    handles: list[Any] = []
    try:
        for handle_idx, script in enumerate(
            iter_hybrid_init_scripts(geometry["shape"], acc=hybrid_mesh_accuracy)
        ):
            fdtd = lumapi.FDTD(hide=hide_fdtd_cad)
            fdtd.cd(str(hybrid_workdir))
            fdtd.eval(script)
            project_path = hybrid_workdir / f"hybrid_layer_{handle_idx}.fsp"
            fdtd.save(str(project_path))
            handles.append(fdtd)

        if len(handles) != len(geometry["internal_counts"]):
            raise RuntimeError(
                "Hybrid handle count does not match internal layer count: "
                f"handles={len(handles)}, internal_layer_count={len(geometry['internal_counts'])}"
            )

        for internal_idx, count in enumerate(geometry["internal_counts"]):
            local_shape = SimpleNamespace(
                propagation_thickness_um=float(geometry["shape"].propagation_thickness_um),
                total_width_um=float(geometry["shape"].total_width_um),
                unit_span_um=float(geometry["slot_width_um"]),
                unit_spacing_um=float(geometry["slot_pitch_um"]),
                repeats=int(getattr(geometry["shape"], "repeats", 1)),
                neurons_per_layer=[1, count, 1],
            )
            y0_um = (
                -0.5 * float(geometry["shape"].total_width_um)
                + 0.5 * geometry["slot_pitch_um"]
            )
            slot_script_path = write_add_silicon_slots_lsf(
                local_shape,
                hybrid_workdir / f"validation_add_slots_layer_{internal_idx}.lsf",
                params=geometry["physical_lengths_by_layer"][internal_idx].tolist(),
                positions=[(0.0, y0_um)],
                material=geometry["slot_material"],
                z_span_um=float(geometry["shape"].propagation_thickness_um),
                width_um=geometry["slot_width_um"],
                spacing_um=geometry["slot_pitch_um"],
                rect_prefix=f"validation_layer_{internal_idx}_rect",
            )
            handles[internal_idx].cd(str(hybrid_workdir))
            handles[internal_idx].eval(slot_script_path.read_text(encoding="utf-8"))
            handles[internal_idx].save(str(hybrid_workdir / f"hybrid_layer_{internal_idx}.fsp"))

        model.sim = SimpleNamespace(
            hybrid_fdtds=handles,
            hybrid_monitor_name=str(
                validation_cfg.get("hybrid_monitor_name", "hybrid_monitor")
            ),
            hybrid_xy_monitor_name=str(
                validation_cfg.get("hybrid_xy_monitor_name", "hybrid_xy_monitor")
            ),
        )
        model.layer_spacing_hybride = [
            float(v)
            for v in np.asarray(geometry["shape"].inter_layer_spacings_um, dtype=float)
        ]
        model.hybrid_workdir = hybrid_workdir
        model.hybrid_zlink = float(validation_cfg.get("hybrid_zlink", 0.0))
        model.setuphs_hybride(
            mode=str(validation_cfg.get("propagation_mode", "full")),
            ky_cut_ratio=float(validation_cfg.get("ky_cut_ratio", 1.0)),
            coarse_factor=float(validation_cfg.get("coarse_factor", 1.0)),
            smooth_rolloff=float(validation_cfg.get("smooth_rolloff", 0.0)),
            pad_factor=int(validation_cfg.get("pad_factor", 16)),
        )

        with torch.no_grad():
            physical_raw_output = model.forward_hybride(
                math_state["input_tensor"],
                wavelength_um=selected_wavelength_um,
            )
        physical_raw_output = torch.as_tensor(
            physical_raw_output,
            dtype=torch.float32,
            device=math_state["input_tensor"].device,
        )
        if physical_raw_output.ndim == 1:
            physical_raw_output = physical_raw_output.unsqueeze(0)
        physical_output = physical_raw_output.clamp_min(clamp_min)
        physical_output = physical_output / physical_output.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        physical_output_response = physical_output.detach().cpu().reshape(-1).numpy()

        if not hasattr(model, "us_H") or len(model.us_H) == 0:
            raise AttributeError(
                "OpticalDONN did not expose hybrid physical fields in model.us_H"
            )
        physical_line = _ensure_complex_line(model.us_H[-1])
        physical_y_um = _resolve_model_y_axis_um(model)
    finally:
        for handle in handles:
            _safe_close(handle)

    if physical_line.size != math_state["mathematical_line"].size or not np.allclose(
        physical_y_um, math_state["mathematical_y_um"]
    ):
        physical_line = _interp_complex_line(
            physical_y_um,
            physical_line,
            math_state["mathematical_y_um"],
        )
        physical_y_um = math_state["mathematical_y_um"]

    mathematical_port_energy = _windowed_intensity_at_positions(
        math_state["mathematical_y_um"],
        math_state["mathematical_line"],
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

    figure_path = job_dir / "validation_hybrid.png"
    data_path = job_dir / "validation_hybrid_data.npz"
    _plot_validation_summary(
        out_path=figure_path,
        title=(
            "Hybrid Validation\n"
            "Run the trained mapped parameters through the mathematical model and the hybrid physical forward, "
            "then compare output response and output-plane fields."
        ),
        output_y_um=np.asarray(output_y_um, dtype=float),
        mathematical_y_um=np.asarray(math_state["mathematical_y_um"], dtype=float),
        physical_y_um=np.asarray(physical_y_um, dtype=float),
        mathematical_line=np.asarray(math_state["mathematical_line"]),
        physical_line=np.asarray(physical_line),
        math_output_response=np.asarray(math_state["math_output_response"], dtype=float),
        physical_output_response=np.asarray(physical_output_response, dtype=float),
        math_window_response=np.asarray(math_window_response, dtype=float),
        physical_window_response=np.asarray(physical_window_response, dtype=float),
        physical_lengths_by_layer=list(geometry["physical_lengths_by_layer"]),
        expected_port_index=expected_port_index,
    )
    np.savez(
        data_path,
        input_phase=np.asarray(input_phase_arr, dtype=float),
        output_y_um=np.asarray(output_y_um, dtype=float),
        mathematical_y_um=np.asarray(math_state["mathematical_y_um"], dtype=float),
        physical_y_um=np.asarray(physical_y_um, dtype=float),
        mathematical_line=np.asarray(math_state["mathematical_line"]),
        physical_line=np.asarray(physical_line),
        math_output_response=np.asarray(math_state["math_output_response"], dtype=float),
        physical_output_response=np.asarray(physical_output_response, dtype=float),
        math_window_response=np.asarray(math_window_response, dtype=float),
        physical_window_response=np.asarray(physical_window_response, dtype=float),
        raw_delta_by_layer=np.asarray(geometry["raw_delta_by_layer"], dtype=object),
        wrapped_delta_by_layer=np.asarray(
            geometry["wrapped_delta_by_layer"], dtype=object
        ),
        physical_lengths_um_by_layer=np.asarray(
            geometry["physical_lengths_by_layer"], dtype=object
        ),
    )

    metrics = {
        "backend": "hybrid",
        "selected_wavelength_um": float(selected_wavelength_um),
        "expected_port_index": expected_port_index,
        "input_phase": input_phase_arr.tolist(),
        "input_phase_support": math_state["input_phase_support"],
        "math_output_response": math_state["math_output_response"].tolist(),
        "physical_output_response": physical_output_response.tolist(),
        "math_window_energy_response": math_window_response.tolist(),
        "physical_window_energy_response": physical_window_response.tolist(),
        "output_positions_um": np.asarray(output_y_um, dtype=float).tolist(),
        "port_window_um": float(port_window_um),
        "hybrid_mode_path": str(hybrid_mode_path),
        "hybrid_workdir": str(hybrid_workdir),
        "figure_path": str(figure_path),
        "data_path": str(data_path),
        "slot_pitch_um": float(geometry["slot_pitch_um"]),
        "slot_width_um": float(geometry["slot_width_um"]),
        "slot_material": str(geometry["slot_material"]),
        "mapping_factor_um": float(geometry["mapping_factor"]),
        **_build_slot_length_summary(geometry),
    }
    return metrics


def _run_fdtd_validation(
    trained_network_package: Mapping[str, Any],
    validation_cfg: Mapping[str, Any],
    *,
    layout: ValidationJobLayout,
    job_dir: Path,
    shape_cfg: Mapping[str, Any],
    model: Any,
    input_phase_arr: np.ndarray,
    selected_wavelength_um: float,
    clamp_min: float,
    expected_port_index: int | None,
) -> dict[str, Any]:
    geometry = _resolve_validation_geometry(
        model,
        shape_cfg=shape_cfg,
        validation_cfg=validation_cfg,
        wavelength_um=selected_wavelength_um,
    )
    math_state = _resolve_math_validation_state(
        model,
        input_phase_arr=input_phase_arr,
        wavelength_um=selected_wavelength_um,
        clamp_min=clamp_min,
    )

    optical_cfg = dict(trained_network_package["mapped_network_package"]["optical_donn_config"])
    _, output_y_um = _resolve_io_y_um(shape_cfg, optical_cfg)
    mathematical_port_amplitude = _sample_amplitude_at_positions(
        math_state["mathematical_y_um"],
        math_state["mathematical_line"],
        output_y_um,
    )
    math_field_response = mathematical_port_amplitude / np.clip(
        float(np.sum(mathematical_port_amplitude)),
        1e-12,
        None,
    )

    calibration_package = dict(
        trained_network_package["mapped_network_package"]["calibration_package"]
    )
    sim_dir_raw = calibration_package.get("sim_dir")
    if sim_dir_raw is None:
        raise KeyError(
            "calibration_package.sim_dir is required for validation backend 'fdtd'"
        )
    sim_dir = Path(sim_dir_raw)
    if not sim_dir.exists():
        raise FileNotFoundError(f"calibration_package.sim_dir does not exist: {sim_dir}")

    project_basename = str(validation_cfg.get("physical_project_name", "experiment")).strip()
    if not project_basename:
        raise ValueError(
            "validation.physical_project_name must be a non-empty string"
        )
    project_suffix = ".lms" if bool(geometry["shape"].use_var_fdtd) else ".fsp"
    source_project_path = sim_dir / f"{project_basename}{project_suffix}"
    if not source_project_path.exists():
        candidates = sorted(sim_dir.glob(f"*{project_suffix}"))
        if len(candidates) == 1:
            source_project_path = candidates[0]
        else:
            raise FileNotFoundError(
                "Cannot resolve the physical FDTD project for validation: "
                f"requested={source_project_path}, candidates={[path.name for path in candidates]}"
            )

    try:
        import lumapi
    except Exception as exc:
        raise ImportError("lumapi is required for full-FDTD validation") from exc

    from .lsf_special_scripts import write_add_silicon_slots_lsf

    slot_positions = [
        (
            float(geometry["internal_layer_x_um"][layer_idx]),
            -0.5 * float(geometry["shape"].total_width_um) + 0.5 * geometry["slot_pitch_um"],
        )
        for layer_idx, _ in enumerate(geometry["internal_counts"])
    ]

    physical_workdir = job_dir / "physical_fdtd"
    physical_workdir.mkdir(parents=True, exist_ok=True)
    copied_project_path = physical_workdir / source_project_path.name
    shutil.copy2(source_project_path, copied_project_path)
    slot_script_path = write_add_silicon_slots_lsf(
        geometry["shape"],
        physical_workdir / "validation_add_slots.lsf",
        params=np.concatenate(geometry["physical_lengths_by_layer"]).tolist(),
        positions=slot_positions,
        material=geometry["slot_material"],
        z_span_um=float(geometry["shape"].propagation_thickness_um),
        width_um=geometry["slot_width_um"],
        spacing_um=geometry["slot_pitch_um"],
        rect_prefix="validation_rect",
    )

    fdtd = (lumapi.MODE if bool(geometry["shape"].use_var_fdtd) else lumapi.FDTD)(
        hide=bool(validation_cfg.get("hide_fdtd_cad", True))
    )
    try:
        fdtd.cd(str(physical_workdir))
        fdtd.load(str(copied_project_path))
        fdtd.eval("switchtolayout;")

        for source_idx in range(len(model.inputposition)):
            source_name = f"mode_src{source_idx}"
            if int(fdtd.getnamednumber(source_name)) < 1:
                raise ValueError(
                    "validation could not find mode source in copied project: "
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

        output_monitor_name = str(
            validation_cfg.get(
                "output_monitor_name",
                f"monitor{len(calibration_package['monitors']) - 1}",
            )
        )
        if int(fdtd.getnamednumber(output_monitor_name)) < 1:
            raise ValueError(
                "validation could not find output monitor in copied project: "
                f"project={copied_project_path}, output_monitor_name={output_monitor_name}"
            )

        physical_fields = get_fields(
            fdtd,
            monitor_name=output_monitor_name,
            field_result_name="validation_fields",
            get_eps=False,
            get_D=False,
            get_H=False,
            nointerpolation=False,
            unfold_symmetry=False,
        )
        physical_payload = monitor_payload_from_fields(physical_fields, include_axes=True)
        physical_pick = extract_ey_1d_at_wavelength(
            physical_payload,
            wavelength_um=float(selected_wavelength_um),
            selection="nearest",
        )
        physical_line = _ensure_complex_line(physical_pick["ey"])
        physical_y_um = np.asarray(physical_pick["y"], dtype=float).reshape(-1)
    finally:
        try:
            fdtd.close()
        except Exception:
            pass

    if physical_line.size != math_state["mathematical_line"].size or not np.allclose(
        physical_y_um, math_state["mathematical_y_um"]
    ):
        physical_line = _interp_complex_line(
            physical_y_um,
            physical_line,
            math_state["mathematical_y_um"],
        )
        physical_y_um = math_state["mathematical_y_um"]

    physical_port_amplitude = _sample_amplitude_at_positions(
        physical_y_um,
        physical_line,
        output_y_um,
    )
    physical_field_response = physical_port_amplitude / np.clip(
        float(np.sum(physical_port_amplitude)),
        1e-12,
        None,
    )
    port_window_um = float(
        validation_cfg.get(
            "port_window_um",
            float(optical_cfg.get("output_mapping_span_um", 2.0)),
        )
    )
    if port_window_um <= 0.0:
        raise ValueError(
            f"validation.port_window_um must be positive, got {port_window_um}"
        )
    math_window_response = _normalize_sum(
        _windowed_intensity_at_positions(
            math_state["mathematical_y_um"],
            math_state["mathematical_line"],
            output_y_um,
            window_um=port_window_um,
        )
    )
    physical_window_response = _normalize_sum(
        _windowed_intensity_at_positions(
            physical_y_um,
            physical_line,
            output_y_um,
            window_um=port_window_um,
        )
    )

    figure_path = job_dir / "validation_fdtd.png"
    data_path = job_dir / "validation_fdtd_data.npz"
    _plot_validation_summary(
        out_path=figure_path,
        title=(
            "Full-FDTD Validation\n"
            "Load the trained parameters into the physical project, rerun the output monitor, "
            "and compare the mathematical and physical output fields."
        ),
        output_y_um=np.asarray(output_y_um, dtype=float),
        mathematical_y_um=np.asarray(math_state["mathematical_y_um"], dtype=float),
        physical_y_um=np.asarray(physical_y_um, dtype=float),
        mathematical_line=np.asarray(math_state["mathematical_line"]),
        physical_line=np.asarray(physical_line),
        math_output_response=np.asarray(math_field_response, dtype=float),
        physical_output_response=np.asarray(physical_field_response, dtype=float),
        math_window_response=np.asarray(math_window_response, dtype=float),
        physical_window_response=np.asarray(physical_window_response, dtype=float),
        physical_lengths_by_layer=list(geometry["physical_lengths_by_layer"]),
        expected_port_index=expected_port_index,
    )
    np.savez(
        data_path,
        input_phase=np.asarray(input_phase_arr, dtype=float),
        output_y_um=np.asarray(output_y_um, dtype=float),
        mathematical_y_um=np.asarray(math_state["mathematical_y_um"], dtype=float),
        physical_y_um=np.asarray(physical_y_um, dtype=float),
        mathematical_line=np.asarray(math_state["mathematical_line"]),
        physical_line=np.asarray(physical_line),
        math_output_response=np.asarray(math_state["math_output_response"], dtype=float),
        math_field_port_response=np.asarray(math_field_response, dtype=float),
        physical_field_port_response=np.asarray(physical_field_response, dtype=float),
        math_window_response=np.asarray(math_window_response, dtype=float),
        physical_window_response=np.asarray(physical_window_response, dtype=float),
        raw_delta_by_layer=np.asarray(geometry["raw_delta_by_layer"], dtype=object),
        wrapped_delta_by_layer=np.asarray(
            geometry["wrapped_delta_by_layer"], dtype=object
        ),
        physical_lengths_um_by_layer=np.asarray(
            geometry["physical_lengths_by_layer"], dtype=object
        ),
    )

    metrics = {
        "backend": "fdtd",
        "selected_wavelength_um": float(selected_wavelength_um),
        "expected_port_index": expected_port_index,
        "input_phase": input_phase_arr.tolist(),
        "input_phase_support": math_state["input_phase_support"],
        "math_output_response": math_state["math_output_response"].tolist(),
        "math_field_port_response": math_field_response.tolist(),
        "physical_field_port_response": physical_field_response.tolist(),
        "math_window_energy_response": math_window_response.tolist(),
        "physical_window_energy_response": physical_window_response.tolist(),
        "output_positions_um": np.asarray(output_y_um, dtype=float).tolist(),
        "port_window_um": float(port_window_um),
        "physical_project_source": str(source_project_path),
        "physical_project_copy": str(copied_project_path),
        "slot_script_path": str(slot_script_path),
        "figure_path": str(figure_path),
        "data_path": str(data_path),
        "slot_pitch_um": float(geometry["slot_pitch_um"]),
        "slot_width_um": float(geometry["slot_width_um"]),
        "slot_material": str(geometry["slot_material"]),
        "mapping_factor_um": float(geometry["mapping_factor"]),
        "output_monitor_name": output_monitor_name,
        "internal_layer_x_um": np.asarray(
            geometry["internal_layer_x_um"], dtype=float
        ).tolist(),
        **_build_slot_length_summary(geometry),
    }
    return metrics


def run_local_validation_job(
    *,
    config_path: str | Path,
    project_root: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    _ = project_root or get_project_root()
    config_file = resolve_validation_config_path(config_path)
    config = load_project_config(config_file)
    validation_cfg = resolve_validation_config(config)
    layout = resolve_validation_job_layout(config_file)

    trained_network_package_path = layout.trained_network_package_path
    override_path = validation_cfg.get("trained_network_package")
    if override_path is not None:
        candidate = Path(override_path)
        if not candidate.is_absolute():
            candidate = (layout.validation_bundle_dir / candidate).resolve()
        trained_network_package_path = candidate
    if not trained_network_package_path.exists():
        raise FileNotFoundError(
            "Missing trained network package for validation job, expected "
            f"{trained_network_package_path}. Run the train bundle first."
        )

    backup_dir = _prepare_job_dir(layout)
    layout.job_dir.joinpath("validation.snapshot.yaml").write_text(
        layout.validation_config_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    manifest: dict[str, Any] = {
        "validation_config_path": str(layout.validation_config_path),
        "validation_bundle_dir": str(layout.validation_bundle_dir),
        "validation_script_path": (
            str(layout.validation_script_path)
            if layout.validation_script_path is not None
            else None
        ),
        "train_bundle_dir": str(layout.train_bundle_dir),
        "workspace_dir": str(layout.workspace_dir),
        "map_dir": str(layout.map_dir),
        "train_artifacts_dir": str(layout.train_artifacts_dir),
        "trained_network_package": str(trained_network_package_path),
        "backend": validation_cfg["backend"],
        "state_source": str(validation_cfg.get("state_source", "best")).strip().lower(),
        "previous_artifacts_backup": (
            str(backup_dir) if backup_dir is not None else None
        ),
        "status": "dry_run" if dry_run else "running",
    }

    if dry_run:
        manifest_path = save_json(layout.job_dir / "manifest.json", manifest)
        return {
            "job_dir": str(layout.job_dir),
            "backup_dir": str(backup_dir) if backup_dir is not None else None,
            "manifest_path": str(manifest_path),
        }

    trained_network_package = load_package(trained_network_package_path)
    state_source = manifest["state_source"]
    model = instantiate_optical_donn_from_trained_package(
        trained_network_package,
        work_dir=layout.job_dir / "model_workspace",
        device="cpu",
        state_source=state_source,
    )

    mapped_network_package = dict(trained_network_package["mapped_network_package"])
    shape_cfg = dict(mapped_network_package["shape_spec"]["shape_config"])
    selected_wavelength_um = _resolve_focus_validation_wavelength(
        model,
        wavelength_um=(
            None
            if validation_cfg.get("wavelength_um") is None
            else float(validation_cfg["wavelength_um"])
        ),
    )
    input_phase_arr = _resolve_validation_input(
        model,
        trained_network_package,
        validation_cfg,
        layout=layout,
    )
    clamp_min = float(validation_cfg.get("clamp_min", 0.0))
    expected_port_index = _resolve_expected_port_index(
        validation_cfg,
        num_outputs=len(model.outputposition),
    )

    if validation_cfg["backend"] == "hybrid":
        metrics = _run_hybrid_validation(
            trained_network_package,
            validation_cfg,
            layout=layout,
            job_dir=layout.job_dir,
            shape_cfg=shape_cfg,
            model=model,
            input_phase_arr=input_phase_arr,
            selected_wavelength_um=selected_wavelength_um,
            clamp_min=clamp_min,
            expected_port_index=expected_port_index,
        )
    else:
        metrics = _run_fdtd_validation(
            trained_network_package,
            validation_cfg,
            layout=layout,
            job_dir=layout.job_dir,
            shape_cfg=shape_cfg,
            model=model,
            input_phase_arr=input_phase_arr,
            selected_wavelength_um=selected_wavelength_um,
            clamp_min=clamp_min,
            expected_port_index=expected_port_index,
        )

    manifest.update(
        {
            "status": "completed",
            "selected_wavelength_um": float(selected_wavelength_um),
            "expected_port_index": expected_port_index,
            "input_phase": input_phase_arr.tolist(),
            "metrics": metrics,
        }
    )
    metrics_path = save_json(layout.job_dir / "metrics.json", metrics)
    manifest["metrics_path"] = str(metrics_path)
    manifest_path = save_json(layout.job_dir / "manifest.json", manifest)
    return {
        "job_dir": str(layout.job_dir),
        "backup_dir": str(backup_dir) if backup_dir is not None else None,
        "manifest_path": str(manifest_path),
        "metrics_path": str(metrics_path),
    }


def run_local_validation_bundle(
    bundle_dir: str | Path,
    *,
    project_root: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    return run_local_validation_job(
        config_path=resolve_validation_config_path(bundle_dir),
        project_root=project_root,
        dry_run=dry_run,
    )


__all__ = [
    "resolve_validation_config_path",
    "resolve_validation_job_layout",
    "run_local_validation_bundle",
    "run_local_validation_job",
]
