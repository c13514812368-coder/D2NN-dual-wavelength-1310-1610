from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch

from .fdtd_pipeline import compute_input_output_ys
from .lumerical_scripts import extract_ey_1d_at_wavelength
from .networkangler import NetworkShape, OpticalDONN
from .overlap import Overlap
from .simulation_store import load_monitors_bundle, save_monitors_bundle
from .wavelength_tools import resolve_wavelength_choice
from .wavelength_tools import select_value_by_wavelength
from .wavelength_tools import sorted_wavelengths_from_dict


SCHEMA_VERSION = "1"


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


def _normalize_for_fingerprint(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value)
        return {
            "__ndarray__": True,
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "sha256": hashlib.sha256(arr.view(np.uint8)).hexdigest(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _normalize_for_fingerprint(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_fingerprint(v) for v in value]
    return value


def fingerprint_payload(payload: Any) -> str:
    normalized = _normalize_for_fingerprint(payload)
    dump = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(dump.encode("utf-8")).hexdigest()


def save_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(_to_serializable(dict(payload)), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return out


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_package(path: str | Path, payload: Mapping[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload_dict = dict(payload)
    try:
        torch.save(payload_dict, out)
    except MemoryError:
        # Fallback for large payloads on memory-constrained machines.
        torch.save(payload_dict, out, _use_new_zipfile_serialization=False)
    return out


def load_package(path: str | Path) -> Dict[str, Any]:
    package_path = Path(path)
    try:
        data = torch.load(package_path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(package_path, map_location="cpu")
    if not isinstance(data, dict):
        raise TypeError(f"Expected package payload dict in {package_path}")
    return data


def _copy_monitor(monitor: Any) -> Any:
    if isinstance(monitor, dict):
        return {str(k): np.asarray(v).copy() for k, v in monitor.items() if v is not None}
    return np.asarray(monitor).copy()


def _copy_named_payload(payload: Any) -> Any:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise TypeError("named payload must be a mapping")
    return {
        str(key): np.asarray(value).copy()
        for key, value in payload.items()
        if value is not None
    }


def _monitor_payload_kind(monitors: Sequence[Any]) -> str:
    if not monitors:
        return "unknown"
    return "full_fields" if isinstance(monitors[0], Mapping) else "ey_1d"


def _resolve_calibration_wavelength_grid(
    calibration_package: Mapping[str, Any],
) -> np.ndarray:
    wavelengths = np.asarray(calibration_package.get("wavelengths_um", []), dtype=float).reshape(-1)
    if wavelengths.size == 0:
        wavelengths = sorted_wavelengths_from_dict(
            calibration_package.get("neff_by_wavelength_um", {})
        )
    if wavelengths.size == 0:
        wavelengths = sorted_wavelengths_from_dict(
            calibration_package.get("mapping_factor_by_wavelength_um", {})
        )
    if wavelengths.size == 0:
        selected_wl = calibration_package.get("selected_wavelength_um")
        if selected_wl is None:
            raise KeyError(
                "calibration_package requires wavelengths_um, wavelength dictionaries, "
                "or selected_wavelength_um"
            )
        wavelengths = np.asarray([float(selected_wl)], dtype=float)
    return wavelengths


def _select_calibration_operating_point(
    calibration_package: Mapping[str, Any],
) -> Dict[str, Any]:
    wavelengths_um = _resolve_calibration_wavelength_grid(calibration_package)
    requested_wl = calibration_package.get("selected_wavelength_um")
    chosen = resolve_wavelength_choice(
        wavelengths_um,
        wavelength_um=(None if requested_wl is None else float(requested_wl)),
        selection="nearest",
    )
    selected_wavelength_um = float(chosen["selected_wavelength_um"])

    neff_by_wavelength_um = dict(calibration_package.get("neff_by_wavelength_um", {}))
    if neff_by_wavelength_um:
        neff = float(
            select_value_by_wavelength(
                neff_by_wavelength_um,
                wavelength_um=selected_wavelength_um,
                selection="nearest",
            )["value"]
        )
    else:
        neff = float(calibration_package["neff"])

    mapping_by_wavelength_um = dict(
        calibration_package.get("mapping_factor_by_wavelength_um", {})
    )
    if mapping_by_wavelength_um:
        mapping_factor = float(
            select_value_by_wavelength(
                mapping_by_wavelength_um,
                wavelength_um=selected_wavelength_um,
                selection="nearest",
            )["value"]
        )
    else:
        mapping_factor = float(calibration_package["mapping_factor"])

    return {
        "wavelengths_um": np.asarray(wavelengths_um, dtype=float),
        "selected_wavelength_um": selected_wavelength_um,
        "neff": neff,
        "mapping_factor": mapping_factor,
    }


def _build_calibration_signature_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "shape_signature": payload["shape_signature"],
        "selected_wavelength_um": payload["selected_wavelength_um"],
        "wavelengths_um": payload["wavelengths_um"],
        "neff_by_wavelength_um": payload["neff_by_wavelength_um"],
        "mapping_factor_by_wavelength_um": payload["mapping_factor_by_wavelength_um"],
        "monitor_payload_kind": payload.get("monitor_payload_kind"),
        "monitors": payload["monitors"],
        "global_field_monitor": payload["global_field_monitor"],
        "global_index_monitor": payload["global_index_monitor"],
        "global_field_monitor_zx": payload["global_field_monitor_zx"],
        "global_index_monitor_zx": payload["global_index_monitor_zx"],
        "overlap_mode": payload["overlap_mode"],
    }


def _build_mapped_network_signature_payload(
    *,
    shape_spec: Mapping[str, Any],
    calibration_package: Mapping[str, Any],
    optical_donn_config: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "shape_signature": shape_spec["shape_signature"],
        "calibration_signature": calibration_package["calibration_signature"],
        "optical_donn_config": optical_donn_config,
    }


def extract_monitor_lines_from_calibration_package(
    calibration_package: Mapping[str, Any],
    *,
    wavelength_um: float | None = None,
    wavelength_index: int | None = None,
) -> Dict[str, Any]:
    selected = _select_calibration_operating_point(
        dict(calibration_package)
        if wavelength_um is None and wavelength_index is None
        else {
            **dict(calibration_package),
            "selected_wavelength_um": (
                calibration_package.get("selected_wavelength_um")
                if wavelength_um is None
                else float(wavelength_um)
            ),
        }
    )
    if wavelength_um is not None or wavelength_index is not None:
        selected = resolve_wavelength_choice(
            selected["wavelengths_um"],
            wavelength_um=wavelength_um,
            wavelength_index=wavelength_index,
            selection="nearest",
        )
    selected_wavelength_um = float(selected["selected_wavelength_um"])
    monitor_lines: List[np.ndarray] = []
    for monitor_idx, monitor in enumerate(calibration_package.get("monitors", [])):
        if isinstance(monitor, Mapping):
            picked = extract_ey_1d_at_wavelength(
                monitor,
                wavelength_um=selected_wavelength_um,
                selection="nearest",
            )
            line = np.asarray(picked["ey"])
        else:
            line = np.asarray(monitor)
        line = np.asarray(line).reshape(-1).copy()
        if line.size == 0:
            raise ValueError(
                f"calibration monitor {monitor_idx} is empty at wavelength "
                f"{selected_wavelength_um:.6g} um"
            )
        monitor_lines.append(line)

    return {
        "selected_wavelength_um": selected_wavelength_um,
        "selected_wavelength_index": int(selected.get("selected_wavelength_index", 0)),
        "wavelengths_um": np.asarray(selected["wavelengths_um"], dtype=float),
        "monitor_payload_kind": _monitor_payload_kind(calibration_package.get("monitors", [])),
        "monitors": monitor_lines,
    }


def _resolve_grad_mask(num_layers: int, raw_mask: Any) -> List[int]:
    if raw_mask is None or raw_mask == "interior_only":
        if num_layers <= 1:
            return [0] * max(num_layers, 1)
        return [0] + [1] * max(num_layers - 2, 0) + [0]
    if raw_mask == "none":
        return [0] * num_layers
    if not isinstance(raw_mask, Sequence) or isinstance(raw_mask, (str, bytes)):
        raise TypeError("mapping.grad_mask must be 'interior_only', 'none', or a list")
    mask = [int(v) for v in raw_mask]
    if len(mask) != num_layers:
        raise ValueError("mapping.grad_mask length must match shape.num_layers")
    if mask and (mask[0] != 0 or mask[-1] != 0):
        raise ValueError("mapping.grad_mask must keep the first and last layer frozen")
    return mask


def _resolve_positions_span(shape: NetworkShape) -> List[float]:
    raw = getattr(shape, "positions_span_um", None)
    if raw is None:
        return [float(shape.total_width_um)] * int(shape.num_layers)
    arr = np.asarray(raw, dtype=float).reshape(-1)
    if arr.size == 1:
        return [float(arr[0])] * int(shape.num_layers)
    if int(arr.size) != int(shape.num_layers):
        raise ValueError("shape.positions_span_um must have one value or num_layers values")
    return [float(v) for v in arr.tolist()]


def _overlap_payload_from_mode_path(mode_path: str | Path) -> Dict[str, Any]:
    overlap = Overlap(None, None, modepath=Path(mode_path))
    meta = {
        str(k): _to_serializable(v)
        for k, v in getattr(overlap, "meta", {}).items()
    }
    return {
        "mode_path": str(mode_path),
        "y_m": np.asarray(overlap.y_m),
        "Ex_m": (None if overlap.Ex_m is None else np.asarray(overlap.Ex_m)),
        "Ey_m": np.asarray(overlap.Ey_m),
        "Ez_m": (None if overlap.Ez_m is None else np.asarray(overlap.Ez_m)),
        "Hx_m": (None if overlap.Hx_m is None else np.asarray(overlap.Hx_m)),
        "Hy_m": (None if overlap.Hy_m is None else np.asarray(overlap.Hy_m)),
        "Hz_m": (None if overlap.Hz_m is None else np.asarray(overlap.Hz_m)),
        "zeta_m": (None if overlap.zeta_m is None else np.asarray(overlap.zeta_m)),
        "meta": meta,
    }


def materialize_overlap(
    overlap_payload: Mapping[str, Any],
    *,
    out_dir: str | Path,
    tag: str = "mapped_overlap",
) -> Overlap:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    meta = overlap_payload.get("meta") or {}
    wavelength_m = meta.get("wavelength_m")
    if wavelength_m is None:
        wavelength_m = meta.get("wavelength_um")
        if wavelength_m is not None:
            wavelength_m = float(wavelength_m) * 1e-6
    if wavelength_m is None:
        wavelength_m = 1.55e-6
    return Overlap(
        overlap_payload.get("y_m"),
        overlap_payload.get("Ey_m"),
        Ex_m=overlap_payload.get("Ex_m"),
        Ez_m=overlap_payload.get("Ez_m"),
        Hx_m=overlap_payload.get("Hx_m"),
        Hy_m=overlap_payload.get("Hy_m"),
        Hz_m=overlap_payload.get("Hz_m"),
        out_dir=out_root,
        tag=tag,
        wavelength_m=wavelength_m,
        origin=str(meta.get("origin", "left")),
    )


def build_shape_spec(
    shape_config: Mapping[str, Any],
    *,
    shape: NetworkShape,
) -> Dict[str, Any]:
    compute_input_output_ys(shape)
    payload = shape.to_cache_payload()
    resolved_shape_config = (
        shape.to_config_payload()
        if hasattr(shape, "to_config_payload")
        else copy.deepcopy(dict(shape_config))
    )
    return {
        "package_type": "shape_spec",
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(),
        "shape_input_config": copy.deepcopy(dict(shape_config)),
        "shape_config": resolved_shape_config,
        "shape_payload": payload,
        "shape_signature": fingerprint_payload(payload),
    }


def build_calibration_package(
    *,
    shape_spec: Mapping[str, Any],
    calibration_result: Mapping[str, Any],
) -> Dict[str, Any]:
    if "emode_path" not in calibration_result or calibration_result.get("emode_path") is None:
        raise KeyError("calibration_result requires emode_path")

    monitors = [_copy_monitor(monitor) for monitor in calibration_result.get("monitors", [])]
    wavelengths = calibration_result.get("wavelengths_um", [])
    wavelengths_arr = np.asarray(wavelengths, dtype=float).reshape(-1)
    if wavelengths_arr.size == 0:
        neff_by = dict(calibration_result.get("neff_by_wavelength_um", {}))
        mapping_by = dict(calibration_result.get("mapping_factor_by_wavelength_um", {}))
        if neff_by:
            wavelengths_arr = np.asarray([float(k) for k in neff_by.keys()], dtype=float)
        elif mapping_by:
            wavelengths_arr = np.asarray([float(k) for k in mapping_by.keys()], dtype=float)

    selected_wl_raw = calibration_result.get("selected_wavelength_um")
    if selected_wl_raw is None:
        if wavelengths_arr.size == 0:
            raise KeyError(
                "calibration_result requires selected_wavelength_um or a non-empty wavelengths_um grid"
            )
        selected_wl_um = float(wavelengths_arr[0])
    else:
        selected_wl_um = float(selected_wl_raw)
        if wavelengths_arr.size == 0:
            wavelengths_arr = np.asarray([selected_wl_um], dtype=float)

    neff_by = {
        str(k): float(v)
        for k, v in dict(calibration_result.get("neff_by_wavelength_um", {})).items()
    }
    mapping_by = {
        str(k): float(v)
        for k, v in dict(calibration_result.get("mapping_factor_by_wavelength_um", {})).items()
    }
    operating_point = _select_calibration_operating_point(
        {
            "selected_wavelength_um": selected_wl_um,
            "wavelengths_um": wavelengths_arr,
            "neff_by_wavelength_um": neff_by,
            "mapping_factor_by_wavelength_um": mapping_by,
            "neff": calibration_result.get("neff", 0.0),
            "mapping_factor": calibration_result.get("mapping_factor", 0.0),
        }
    )

    global_monitors = calibration_result.get("global_monitors") or {}

    payload = {
        "package_type": "calibration_package",
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(),
        "shape_signature": str(shape_spec["shape_signature"]),
        "selected_wavelength_um": float(operating_point["selected_wavelength_um"]),
        "wavelengths_um": np.asarray(operating_point["wavelengths_um"], dtype=float),
        "neff_by_wavelength_um": neff_by,
        "mapping_factor_by_wavelength_um": mapping_by,
        "neff": float(operating_point["neff"]),
        "mapping_factor": float(operating_point["mapping_factor"]),
        "monitor_payload_kind": _monitor_payload_kind(monitors),
        "monitors": monitors,
        "global_field_monitor": _copy_named_payload(global_monitors.get("global_field_monitor")),
        "global_index_monitor": _copy_named_payload(global_monitors.get("global_index_monitor")),
        "global_field_monitor_zx": _copy_named_payload(global_monitors.get("global_field_monitor_zx")),
        "global_index_monitor_zx": _copy_named_payload(global_monitors.get("global_index_monitor_zx")),
        "sim_signature": calibration_result.get("sim_signature"),
        "emode_path": str(calibration_result.get("emode_path")),
        "monitors_bundle_path": (
            None
            if calibration_result.get("monitors_bundle_path") is None
            else str(calibration_result.get("monitors_bundle_path"))
        ),
        "global_monitor_bundle_path": (
            None
            if calibration_result.get("global_monitor_bundle_path") is None
            else str(calibration_result.get("global_monitor_bundle_path"))
        ),
        "simulation_record_path": (
            None
            if calibration_result.get("simulation_record_path") is None
            else str(calibration_result.get("simulation_record_path"))
        ),
        "sim_dir": (None if calibration_result.get("sim_dir") is None else str(calibration_result.get("sim_dir"))),
        "run_dir": (None if calibration_result.get("run_dir") is None else str(calibration_result.get("run_dir"))),
        "parameter_txt_dir": (
            None
            if calibration_result.get("parameter_txt_dir") is None
            else str(calibration_result.get("parameter_txt_dir"))
        ),
        "cache_info": {
            "source": calibration_result.get("source"),
            "use_cached_results": bool(calibration_result.get("use_cached_results", False)),
            "did_run": bool(calibration_result.get("did_run", True)),
            "warnings": list(calibration_result.get("warnings", [])),
            "diffs": list(calibration_result.get("diffs", [])),
        },
        "overlap_mode": _overlap_payload_from_mode_path(calibration_result["emode_path"]),
    }
    payload["calibration_signature"] = fingerprint_payload(
        _build_calibration_signature_payload(payload)
    )
    return payload


def build_optical_donn_config(
    *,
    shape: NetworkShape,
    calibration_package: Mapping[str, Any],
    mapping_config: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    mapping_cfg = dict(mapping_config or {})
    selected = _select_calibration_operating_point(calibration_package)
    input_ys, output_ys = compute_input_output_ys(shape)
    grad_mask = _resolve_grad_mask(int(shape.num_layers), mapping_cfg.get("grad_mask"))

    return {
        "input_positions_um": (input_ys + float(shape.total_width_um) / 2.0).tolist(),
        "output_positions_um": (output_ys + float(shape.total_width_um) / 2.0).tolist(),
        "positions_span_um": _resolve_positions_span(shape),
        "num_layers": int(shape.num_layers),
        "num_neurons_each_layer": [int(v) for v in shape.neurons_per_layer],
        "wavelength_um": float(selected["selected_wavelength_um"]),
        "selected_wavelength_um": float(selected["selected_wavelength_um"]),
        "wavelengths_um": np.asarray(selected["wavelengths_um"], dtype=float).tolist(),
        "n_slab": float(selected["neff"]),
        "n_slab_by_wavelength_um": copy.deepcopy(
            dict(calibration_package.get("neff_by_wavelength_um", {}))
        ),
        "mapping_factor": float(selected["mapping_factor"]),
        "mapping_factor_by_wavelength_um": copy.deepcopy(
            dict(calibration_package.get("mapping_factor_by_wavelength_um", {}))
        ),
        "layer_spacing_um": np.asarray(shape.inter_layer_spacings_um, dtype=float).reshape(-1).tolist(),
        "grad_mask": grad_mask,
        "initcanshu": mapping_cfg.get("initcanshu", 0),
        "waveguide_mapping_span_um": float(mapping_cfg.get("waveguide_mapping_span_um", 5.0)),
        "output_mapping_span_um": float(mapping_cfg.get("output_mapping_span_um", 2.0)),
        "wavelength_reduction": str(mapping_cfg.get("wavelength_reduction", "mean_power")),
        "loss_type": str(mapping_cfg.get("loss_type", "direct_mse")),
        "device": str(mapping_cfg.get("device", "cpu")),
    }


def build_mapped_network_package(
    *,
    shape_spec: Mapping[str, Any],
    calibration_package: Mapping[str, Any],
    shape: NetworkShape,
    mapping_config: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    optical_cfg = build_optical_donn_config(
        shape=shape,
        calibration_package=calibration_package,
        mapping_config=mapping_config,
    )
    payload = {
        "package_type": "mapped_network_package",
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(),
        "shape_spec": copy.deepcopy(dict(shape_spec)),
        "calibration_package": copy.deepcopy(dict(calibration_package)),
        "mapping_config": copy.deepcopy(dict(mapping_config or {})),
        "optical_donn_config": optical_cfg,
    }
    payload["mapped_network_signature"] = fingerprint_payload(
        _build_mapped_network_signature_payload(
            shape_spec=shape_spec,
            calibration_package=calibration_package,
            optical_donn_config=optical_cfg,
        )
    )
    return payload


def _cpu_state_dict(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        out[str(key)] = value.detach().cpu()
    return out


def build_trained_network_package(
    *,
    mapped_network_package: Mapping[str, Any],
    train_config: Mapping[str, Any],
    metrics_summary: Mapping[str, Any],
    best_state_dict: Mapping[str, torch.Tensor],
    final_state_dict: Mapping[str, torch.Tensor],
    reference_artifacts: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    payload = {
        "package_type": "trained_network_package",
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(),
        "mapped_network_package": copy.deepcopy(dict(mapped_network_package)),
        "train_config": copy.deepcopy(dict(train_config)),
        "metrics_summary": copy.deepcopy(dict(metrics_summary)),
        "best_state_dict": _cpu_state_dict(best_state_dict),
        "final_state_dict": _cpu_state_dict(final_state_dict),
        "reference_artifacts": copy.deepcopy(dict(reference_artifacts or {})),
    }
    payload["trained_network_signature"] = fingerprint_payload(
        {
            "mapped_network_signature": mapped_network_package["mapped_network_signature"],
            "metrics_summary": payload["metrics_summary"],
            "best_state_dict": payload["best_state_dict"],
            "final_state_dict": payload["final_state_dict"],
        }
    )
    return payload


def instantiate_optical_donn_from_mapped_package(
    package: Mapping[str, Any],
    *,
    work_dir: str | Path,
    device: str | None = None,
    state_dict: Mapping[str, torch.Tensor] | None = None,
) -> OpticalDONN:
    optical_cfg = dict(package["optical_donn_config"])
    calibration = dict(package["calibration_package"])
    monitor_view = extract_monitor_lines_from_calibration_package(calibration)
    wavelengths_um = optical_cfg.get("wavelengths_um") or calibration.get("wavelengths_um")
    n_slab_by_wavelength_um = (
        optical_cfg.get("n_slab_by_wavelength_um")
        or calibration.get("neff_by_wavelength_um")
        or {}
    )
    mapping_factor_by_wavelength_um = (
        optical_cfg.get("mapping_factor_by_wavelength_um")
        or calibration.get("mapping_factor_by_wavelength_um")
        or {}
    )
    overlap = materialize_overlap(
        calibration["overlap_mode"],
        out_dir=Path(work_dir) / "overlap_mode",
        tag="mapped_overlap",
    )

    model = OpticalDONN(
        optical_cfg["input_positions_um"],
        optical_cfg["output_positions_um"],
        optical_cfg["positions_span_um"],
        int(optical_cfg["num_layers"]),
        list(optical_cfg["num_neurons_each_layer"]),
        float(optical_cfg["wavelength_um"]),
        float(optical_cfg["n_slab"]),
        float(optical_cfg["mapping_factor"]),
        list(optical_cfg["layer_spacing_um"]),
        [_copy_monitor(monitor) for monitor in monitor_view["monitors"]],
        list(optical_cfg["grad_mask"]),
        overlap,
        device=device or str(optical_cfg.get("device", "cpu")),
        initcanshu=optical_cfg.get("initcanshu", 0),
        wavelengths_um=wavelengths_um,
        n_slab_by_wavelength_um=n_slab_by_wavelength_um,
        mapping_factor_by_wavelength_um=mapping_factor_by_wavelength_um,
        monitor_payloads=[_copy_monitor(monitor) for monitor in calibration.get("monitors", [])],
        wavelength_reduction=str(optical_cfg.get("wavelength_reduction", "mean_power")),
    )
    model.waveguide_maping_span = float(optical_cfg["waveguide_mapping_span_um"])
    model.out_maping_span = float(optical_cfg["output_mapping_span_um"])
    model.loss_type = str(optical_cfg.get("loss_type", "direct_mse"))
    model.neff = float(optical_cfg["n_slab"])
    model.mode_path = str(overlap.mode_path)
    model.selected_wavelength_um = float(monitor_view["selected_wavelength_um"])
    model.available_wavelengths_um = np.asarray(
        wavelengths_um if wavelengths_um is not None else [model.selected_wavelength_um],
        dtype=float,
    ).tolist()
    model.n_slab_by_wavelength_um = copy.deepcopy(dict(n_slab_by_wavelength_um))
    model.mapping_factor_by_wavelength_um = copy.deepcopy(dict(mapping_factor_by_wavelength_um))
    model.monitor_payload_kind = str(
        calibration.get("monitor_payload_kind", monitor_view["monitor_payload_kind"])
    )
    if state_dict is not None:
        model.load_state_dict(dict(state_dict))
    return model


def instantiate_optical_donn_from_trained_package(
    package: Mapping[str, Any],
    *,
    work_dir: str | Path,
    device: str | None = None,
    state_source: str = "best",
) -> OpticalDONN:
    source = str(state_source).strip().lower()
    if source == "best":
        state_dict = package["best_state_dict"]
    elif source == "final":
        state_dict = package["final_state_dict"]
    else:
        raise ValueError(
            f"state_source must be 'best' or 'final', got {state_source!r}"
        )

    mapped_network_package = package.get("mapped_network_package")
    if not isinstance(mapped_network_package, Mapping):
        raise TypeError(
            "trained_network_package.mapped_network_package must be a mapping"
        )
    if not isinstance(state_dict, Mapping):
        raise TypeError(
            f"trained_network_package.{source}_state_dict must be a mapping"
        )

    return instantiate_optical_donn_from_mapped_package(
        mapped_network_package,
        work_dir=work_dir,
        device=device,
        state_dict=state_dict,
    )
