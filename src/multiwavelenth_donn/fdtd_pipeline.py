from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np

from .lumdata import SimuData
from .lumerical_scripts import (
    extract_ey_1d_at_wavelength,
    get_fields,
    index_payload_from_monitor,
    monitor_payload_from_fields,
)
from .overlap import Overlap
from .parameter_text_store import (
    compute_parameter_match_key,
    write_parameter_text_bundle,
)
from .simulation_cache import run_and_extract
from .simulation_store import (
    save_monitors_bundle,
    save_named_payload_bundle,
    write_simulation_record,
)
from .wavelength_tools import (
    normalize_wavelength_dict,
    resolve_wavelength_choice,
    select_value_by_wavelength,
)

MakeTsDirFn = Callable[[Path, str], Path]


@dataclass
class BidirLayoutConfig:
    total_len_m: float = 3e-6
    total_wid_m: float = 2e-6
    wid_m: float = 5e-6
    wg_len_m: float = 18e-6
    dist_from_sub_tofdtd: float = 17e-6
    wg_bias: float = 0.0


@dataclass
class MonitorExtractConfig:
    mode: str = "full_fields"  # ey_1d | full_fields
    wavelength_um: Optional[float] = None
    include_axes: bool = True
    print_spacing: bool = True


@dataclass
class MappingFactorConfig:
    explicit_value: Optional[float] = None
    redo: bool = False
    default_value: float = -0.5434706664488476
    analytic_from_neff_fallback: bool = False
    robust_mode: str = "mad_ols"  # mad_ols | theil_sen
    mad_z_thresh: float = 3.5
    use_unwrap_pi: bool = True
    unwrap_discont: float = 0.75 * np.pi
    save_plot: bool = True


@dataclass
class PipelineConfig:
    data_raw_dir: Path
    runs_dir: Path
    sim_subdir_name: str = "simulation_data"
    save_name: str = "name"
    bidir: BidirLayoutConfig = field(default_factory=BidirLayoutConfig)
    monitors: MonitorExtractConfig = field(default_factory=MonitorExtractConfig)
    mapping: MappingFactorConfig = field(default_factory=MappingFactorConfig)
    debug: bool = True
    base_lsf_templates: Dict[str, Path] = field(default_factory=dict)


def make_ts_dir_default(parent: Path, name: str) -> Path:
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    for _ in range(16):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out = parent / f"{name}_{ts}"
        try:
            out.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return out
    raise RuntimeError(f"failed to create unique timestamped directory under {parent}")


def _debug(enabled: bool, msg: str) -> None:
    if enabled:
        print(msg)


def _run_step(step: str, enabled: bool, fn: Callable[[], Any]) -> Any:
    _debug(enabled, f"[STEP] {step} START")
    try:
        out = fn()
    except Exception as exc:
        raise RuntimeError(f"[STEP] {step} FAILED: {exc}") from exc
    _debug(enabled, f"[STEP] {step} DONE")
    return out


def _json_fingerprint(payload: Mapping[str, Any]) -> str:
    dump = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(dump.encode("utf-8")).hexdigest()


def _hash_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_sim_signature(shape: Any, cfg: PipelineConfig) -> str:
    if not hasattr(shape, "to_cache_payload"):
        raise AttributeError("shape must expose to_cache_payload()")
    if hasattr(shape, "wavelength_grid_um"):
        wl_grid = np.asarray(shape.wavelength_grid_um(), dtype=float).reshape(-1).tolist()
    else:
        wl_grid = [float(getattr(shape, "wavelength_center", getattr(shape, "wavelength_um", 1.55)))]
    payload = {
        "shape": shape.to_cache_payload(),
        "wavelength_grid_um": wl_grid,
        "monitor_mode": cfg.monitors.mode,
        "monitor_include_axes": bool(cfg.monitors.include_axes),
        "mapping_redo": bool(cfg.mapping.redo),
        "mapping_robust_mode": cfg.mapping.robust_mode,
        "mapping_default_value": float(cfg.mapping.default_value),
        "bidir": {
            "total_len_m": float(cfg.bidir.total_len_m),
            "total_wid_m": float(cfg.bidir.total_wid_m),
            "wid_m": float(cfg.bidir.wid_m),
            "wg_len_m": float(cfg.bidir.wg_len_m),
            "dist_from_sub_tofdtd": float(cfg.bidir.dist_from_sub_tofdtd),
            "wg_bias": float(cfg.bidir.wg_bias),
        },
        "base_lsf_sha256": {
            str(name): _hash_file(Path(path))
            for name, path in sorted(cfg.base_lsf_templates.items())
        },
    }
    return _json_fingerprint(payload)


def compute_input_output_ys(shape: Any) -> tuple[np.ndarray, np.ndarray]:
    for attr in ("total_width_um", "num_input_waveguides", "num_output_waveguides"):
        if not hasattr(shape, attr):
            raise AttributeError(f"shape is missing required attribute {attr!r}")

    total_width_um = float(shape.total_width_um)
    num_input_waveguides = int(shape.num_input_waveguides)
    num_output_waveguides = int(shape.num_output_waveguides)
    if total_width_um <= 0.0:
        raise ValueError(f"shape.total_width_um must be positive, got {total_width_um}")
    if num_input_waveguides <= 0:
        raise ValueError(
            f"shape.num_input_waveguides must be positive, got {num_input_waveguides}"
        )
    if num_output_waveguides <= 0:
        raise ValueError(
            f"shape.num_output_waveguides must be positive, got {num_output_waveguides}"
        )

    input_waveguide_dist = total_width_um / (num_input_waveguides + 1)
    output_waveguide_dist = total_width_um / (num_output_waveguides + 1)

    input_ys = getattr(shape, "input_ys", None)
    output_ys = getattr(shape, "output_ys", None)

    if input_ys is None:
        input_ys = np.linspace(
            -total_width_um / 2 + input_waveguide_dist,
            total_width_um / 2 - input_waveguide_dist,
            num_input_waveguides,
        )
        shape.input_centers_y_um = input_ys
    else:
        input_ys = np.asarray(input_ys, dtype=float).reshape(-1)
        if input_ys.size != num_input_waveguides:
            raise ValueError(
                "shape.input_ys length mismatch: "
                f"expected {num_input_waveguides}, got {input_ys.size}"
            )
        shape.input_centers_y_um = input_ys

    if output_ys is None:
        output_ys = np.linspace(
            -total_width_um / 2 + output_waveguide_dist,
            total_width_um / 2 - output_waveguide_dist,
            num_output_waveguides,
        )
        shape.output_centers_y_um = output_ys
    else:
        output_ys = np.asarray(output_ys, dtype=float).reshape(-1)
        if output_ys.size != num_output_waveguides:
            raise ValueError(
                "shape.output_ys length mismatch: "
                f"expected {num_output_waveguides}, got {output_ys.size}"
            )
        shape.output_centers_y_um = output_ys

    return np.asarray(input_ys, dtype=float), np.asarray(output_ys, dtype=float)


def _monitor_count_from_shape(shape: Any) -> int:
    for key in ("monitor_positions_x_um", "monitor_positions"):
        vals = getattr(shape, key, None)
        if vals is None:
            continue
        arr = np.asarray(vals)
        if arr.size > 0:
            return int(arr.shape[0])
    return int(getattr(shape, "num_layers", 0))


def _fit_ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    m, b = np.polyfit(x, y, 1)
    yhat = m * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(m), float(b), float(r2), yhat


def _fit_theil_sen(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    n = int(len(x))
    slopes = []
    for i in range(n - 1):
        dx = x[i + 1 :] - x[i]
        dy = y[i + 1 :] - y[i]
        ok = dx != 0
        if np.any(ok):
            slopes.extend((dy[ok] / dx[ok]).tolist())
    m = float(np.median(slopes))
    b = float(np.median(y - m * x))
    yhat = m * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return m, b, r2, yhat


def _mad_outliers(residuals: np.ndarray, z_thresh: float) -> np.ndarray:
    med = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - med)))
    if mad == 0:
        q1, q3 = np.percentile(residuals, [25, 75])
        iqr = float(q3 - q1)
        if iqr == 0:
            return np.zeros_like(residuals, dtype=bool)
        low, high = med - 3 * iqr, med + 3 * iqr
        return (residuals < low) | (residuals > high)
    z = np.abs(residuals - med) / (1.4826 * mad)
    return z > z_thresh


def _load_phase_column(path: Path) -> np.ndarray:
    import re

    text = Path(path).read_text(encoding="utf-8")
    vals = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if len(vals) < 2:
        raise ValueError(f"Phase text file {path} has fewer than 2 numeric values")
    return np.asarray([float(v) for v in vals], dtype=float)


@contextmanager
def _temporary_single_wavelength(shape: Any, wavelength_um: float):
    attrs = (
        "wavelength_um",
        "wavelength_start",
        "wavelength_stop",
        "wavelength_center",
        "wavelength_points",
    )
    sentinel = object()
    backup: Dict[str, Any] = {}
    for attr in attrs:
        backup[attr] = getattr(shape, attr, sentinel)

    try:
        wl_um = float(wavelength_um)
        shape.wavelength_um = wl_um
        shape.wavelength_start = wl_um
        shape.wavelength_stop = wl_um
        shape.wavelength_center = wl_um
        shape.wavelength_points = 1
        yield
    finally:
        for attr in attrs:
            old = backup[attr]
            if old is sentinel:
                if hasattr(shape, attr):
                    delattr(shape, attr)
            else:
                setattr(shape, attr, old)


def _compute_mapping_factor(
    sim: SimuData,
    sim_dir: Path,
    cfg: MappingFactorConfig,
    debug: bool,
    *,
    wavelength_um: Optional[float] = None,
) -> float:
    if cfg.explicit_value is not None:
        _debug(debug, f"[MAP] explicit_value={cfg.explicit_value}")
        return float(cfg.explicit_value)

    if not cfg.redo:
        _debug(debug, f"[MAP] use default_value={cfg.default_value}")
        return float(cfg.default_value)

    txt_path = Path(sim_dir) / "out_txt.txt"
    try:
        shape = getattr(sim, "network_shape", None)
        if wavelength_um is not None and shape is not None:
            with _temporary_single_wavelength(shape, float(wavelength_um)):
                mapping_lsf = sim.get_mapping_factor()
                sim.eval_saved_lsf(
                    sim.neff_fdtd,
                    mapping_lsf,
                    label=f"prepare mapping factor simulation at {float(wavelength_um):.6g} um",
                )
        else:
            mapping_lsf = sim.get_mapping_factor()
            sim.eval_saved_lsf(
                sim.neff_fdtd,
                mapping_lsf,
                label="prepare mapping factor simulation",
            )

        if not txt_path.exists():
            raise FileNotFoundError(
                f"Expected mapping factor output file {txt_path} after running mapping LSF"
            )
        y = _load_phase_column(txt_path)

        if cfg.use_unwrap_pi:
            y = np.unwrap(y * np.pi, discont=cfg.unwrap_discont) / np.pi

        x = np.arange(len(y), dtype=float)
        outlier_idx = []
        mode_name = "MAD outlier + OLS"

        if cfg.robust_mode.lower() == "theil_sen":
            m, b, _, yhat = _fit_theil_sen(x, y)
            mode_name = "Theil-Sen"
            _ = b
        else:
            m0, b0, _, yhat0 = _fit_ols(x, y)
            _ = b0
            resid = y - yhat0
            mask_out = _mad_outliers(resid, z_thresh=cfg.mad_z_thresh)
            mask_in = ~mask_out
            outlier_idx = np.where(mask_out)[0].tolist()
            if int(np.sum(mask_in)) < 2:
                m, b, _, yhat = _fit_theil_sen(x, y)
                mode_name = "Fallback Theil-Sen"
                outlier_idx = []
                _ = b
            else:
                m, b, _, _ = _fit_ols(x[mask_in], y[mask_in])
                yhat = m * x + b

        if cfg.save_plot:
            import matplotlib.pyplot as plt

            out_dir = txt_path.parent
            plt.figure()
            plt.scatter(x, y, label="data")
            plt.plot(x, yhat, label=mode_name)
            if outlier_idx:
                plt.scatter(x[outlier_idx], y[outlier_idx], marker="x", s=100, label="outliers")
            plt.xlabel("Index")
            plt.ylabel("Phase / pi" if cfg.use_unwrap_pi else "Value")
            plt.title("Phase Fit")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "fit.png", dpi=150)
            plt.close()

        if m == 0:
            raise ZeroDivisionError("phase slope is zero")
        mapping_factor = 0.1 / m
        _debug(debug, f"[MAP] slope={m}, mapping_factor={mapping_factor}")
        return float(mapping_factor)
    except Exception as exc:
        raise RuntimeError(
            "mapping factor computation failed while redo=True. "
            f"sim_dir={Path(sim_dir)}, wavelength_um={wavelength_um}, "
            f"txt_path={txt_path}, error_type={type(exc).__name__}, reason={exc}"
        ) from exc


def _print_spacing_info(monitor_idx: int, ey: np.ndarray, fields_obj: Any, enabled: bool) -> None:
    if not enabled:
        return
    coord_candidates = {
        "x": np.asarray(getattr(fields_obj, "x", []), dtype=float).reshape(-1),
        "y": np.asarray(getattr(fields_obj, "y", []), dtype=float).reshape(-1),
        "z": np.asarray(getattr(fields_obj, "z", []), dtype=float).reshape(-1),
    }
    coord_vals = None
    coord_name = None
    for name, vals in coord_candidates.items():
        if vals.size > 1 and vals.size == ey.size:
            coord_vals = vals
            coord_name = name
            break
    if coord_vals is None:
        return

    diffs = np.diff(coord_vals)
    dy = float(np.mean(diffs)) if diffs.size else 0.0
    max_dev = float(np.max(np.abs(diffs - dy))) if diffs.size else 0.0
    if max_dev <= 1e-9:
        print(f"[MONITOR {monitor_idx}] axis={coord_name}, uniform grid, d={dy}")
    else:
        print(
            f"[MONITOR {monitor_idx}] axis={coord_name}, non-uniform grid, mean={dy}, "
            f"min={float(np.min(diffs))}, max={float(np.max(diffs))}"
        )


def _resolve_wavelength_grid_um(shape: Any) -> np.ndarray:
    if hasattr(shape, "wavelength_grid_um"):
        grid = np.asarray(shape.wavelength_grid_um(), dtype=float).reshape(-1)
    else:
        start_um = float(getattr(shape, "wavelength_start", getattr(shape, "wavelength_um", 1.55)))
        stop_um = float(getattr(shape, "wavelength_stop", start_um))
        center_um = float(getattr(shape, "wavelength_center", 0.5 * (start_um + stop_um)))
        points = int(getattr(shape, "wavelength_points", 1))
        if points <= 0:
            raise ValueError(f"shape.wavelength_points must be positive, got {points}")
        if abs(stop_um - start_um) <= 0.0 and points != 1:
            raise ValueError(
                "shape.wavelength_start equals shape.wavelength_stop, "
                f"so shape.wavelength_points must be 1, got {points}"
            )
        if abs(stop_um - start_um) <= 0.0 or points == 1:
            grid = np.asarray([center_um], dtype=float)
        else:
            grid = np.linspace(start_um, stop_um, points, dtype=float)
    if grid.size == 0:
        raise ValueError("empty wavelength grid")
    return grid


def _resolve_selected_wavelength(shape: Any, cfg: MonitorExtractConfig, wl_grid_um: np.ndarray) -> Dict[str, Any]:
    req_wl_um = cfg.wavelength_um
    if req_wl_um is None:
        req_wl_um = float(getattr(shape, "wavelength_center", getattr(shape, "wavelength_um", 1.55)))
    return resolve_wavelength_choice(wl_grid_um, wavelength_um=float(req_wl_um), selection="nearest")


def _extract_monitors(sim: SimuData, shape: Any, cfg: MonitorExtractConfig, debug: bool) -> list[Any]:
    monitor_count = _monitor_count_from_shape(shape)
    if monitor_count <= 0:
        raise ValueError("monitor_count <= 0; cannot extract monitor fields")

    req_wl_um = cfg.wavelength_um
    if req_wl_um is None:
        req_wl_um = float(
            getattr(shape, "wavelength_center", getattr(shape, "wavelength_um", 1.55))
        )

    monitors: list[Any] = []
    for i in range(monitor_count):
        fields_obj = get_fields(
            sim.fdtd,
            monitor_name=f"monitor{i}",
            field_result_name="fields",
            get_eps=False,
            get_D=False,
            get_H=False,
            nointerpolation=False,
            unfold_symmetry=False,
        )
        if cfg.mode == "full_fields":
            payload = monitor_payload_from_fields(fields_obj, include_axes=cfg.include_axes)
            monitors.append(payload)
            _debug(
                debug,
                f"[MONITOR {i}] full_fields saved: E_shape={np.asarray(payload['E']).shape}, "
                f"lambda_n={int(np.asarray(payload['lambda_um']).size)}",
            )
            continue

        picked = extract_ey_1d_at_wavelength(
            fields_obj,
            wavelength_um=float(req_wl_um),
            selection="nearest",
        )
        ey = np.asarray(picked["ey"])
        monitors.append(ey)
        _debug(
            debug,
            f"[MONITOR {i}] ey_1d saved: len={ey.size}, selected_lambda_um={picked['selected_wavelength_um']}",
        )
        if cfg.print_spacing:
            _print_spacing_info(i, ey, fields_obj, enabled=True)

    return monitors


def _extract_global_monitors(sim: SimuData, debug: bool) -> dict[str, Any]:
    global_monitors: dict[str, Any] = {}

    global_fields = get_fields(
        sim.fdtd,
        monitor_name="global_fields",
        field_result_name="global_fields_payload",
        get_eps=False,
        get_D=False,
        get_H=False,
        nointerpolation=False,
        unfold_symmetry=False,
    )
    global_monitors["global_field_monitor"] = monitor_payload_from_fields(
        global_fields,
        include_axes=True,
    )
    _debug(
        debug,
        "[GLOBAL] global_field_monitor saved: "
        f"E_shape={np.asarray(global_monitors['global_field_monitor']['E']).shape}",
    )

    global_monitors["global_index_monitor"] = index_payload_from_monitor(
        sim.fdtd,
        monitor_name="global_fields",
        unfold_symmetry=False,
    )
    _debug(
        debug,
        "[GLOBAL] global_index_monitor saved: "
        f"index_shape={np.asarray(global_monitors['global_index_monitor']['index']).shape}",
    )
    return global_monitors


def _compute_neff_by_wavelength(sim: SimuData, wl_grid_um: np.ndarray, debug: bool) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for wl_um in np.asarray(wl_grid_um, dtype=float).reshape(-1):
        raw = sim.getneff(wavelength_um=float(wl_um))
        val = float(np.real(np.asarray(raw).squeeze()))
        out[f"{float(wl_um):.9f}".rstrip("0").rstrip(".")] = val
        _debug(debug, f"[NEFF] wl={wl_um}um -> {val}")
    return out


def _compute_mapping_factor_by_wavelength(
    sim: SimuData,
    sim_dir: Path,
    wl_grid_um: np.ndarray,
    cfg: MappingFactorConfig,
    debug: bool,
    *,
    neff_by_wavelength_um: Mapping[str, float],
    reference_wavelength_um: float,
) -> Dict[str, float]:
    reference_neff = float(
        select_value_by_wavelength(
            neff_by_wavelength_um,
            wavelength_um=float(reference_wavelength_um),
            selection="nearest",
        )["value"]
    )
    reference_mapping_factor = float(
        cfg.explicit_value if cfg.explicit_value is not None else cfg.default_value
    )
    out: Dict[str, float] = {}
    for wl_um in np.asarray(wl_grid_um, dtype=float).reshape(-1):
        try:
            val = _compute_mapping_factor(
                sim,
                sim_dir,
                cfg,
                debug,
                wavelength_um=float(wl_um),
            )
        except Exception as exc:
            if not bool(cfg.analytic_from_neff_fallback):
                raise
            neff_value = float(
                select_value_by_wavelength(
                    neff_by_wavelength_um,
                    wavelength_um=float(wl_um),
                    selection="nearest",
                )["value"]
            )
            if abs(reference_neff) <= 1e-12 or abs(neff_value) <= 1e-12:
                raise RuntimeError(
                    "analytic_from_neff_fallback requires non-zero neff values: "
                    f"reference_neff={reference_neff}, neff({float(wl_um):.6g}um)={neff_value}"
                ) from exc
            val = (
                reference_mapping_factor
                * (float(wl_um) / float(reference_wavelength_um))
                * (reference_neff / neff_value)
            )
            _debug(
                debug,
                "[MAP] analytic_from_neff_fallback "
                f"wl={float(wl_um):.6g}um mapping_factor={float(val):.12g} "
                f"reason={type(exc).__name__}: {exc}",
            )
        out[f"{float(wl_um):.9f}".rstrip("0").rstrip(".")] = float(val)
        _debug(debug, f"[MAP] wl={float(wl_um):.6g}um -> {float(val)}")
    return out


def run_fdtd_and_extract_once(
    shape: Any,
    *,
    cfg: PipelineConfig,
    make_ts_dir: Optional[MakeTsDirFn] = None,
    run_dir: Optional[Path] = None,
    reuse_parameters: Optional[Mapping[str, Any]] = None,
    reuse_parameters_only: bool = False,
) -> Dict[str, Any]:
    make_dir = make_ts_dir or make_ts_dir_default

    simulation_root = Path(cfg.data_raw_dir) / cfg.sim_subdir_name
    simulation_root.mkdir(parents=True, exist_ok=True)

    sim_dir = _run_step("create_sim_dir", cfg.debug, lambda: make_dir(simulation_root, "simulation"))
    if run_dir is None:
        run_dir = _run_step(
            "create_run_dir",
            cfg.debug,
            lambda: make_dir(Path(cfg.runs_dir), "fdtd_run"),
        )
    else:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        _debug(cfg.debug, f"[RUN_DIR] using explicit run dir {run_dir}")

    sim = _run_step(
        "init_simu",
        cfg.debug,
        lambda: SimuData(
            network_shape=shape,
            output_dir=str(sim_dir),
            base_lsf_templates=cfg.base_lsf_templates,
        ),
    )

    _run_step(
        "write_paths",
        cfg.debug,
        lambda: (run_dir / "paths.txt").write_text(
            f"simulation_raw={sim_dir}\ncreated_at={datetime.now().isoformat()}\n",
            encoding="utf-8",
        ),
    )

    wl_grid_um = _resolve_wavelength_grid_um(shape)
    selected_wl = _resolve_selected_wavelength(shape, cfg.monitors, wl_grid_um)
    selected_wl_um = float(selected_wl["selected_wavelength_um"])

    input_ys, output_ys = _run_step(
        "compute_io_ys",
        cfg.debug,
        lambda: compute_input_output_ys(shape),
    )
    _debug(cfg.debug, f"[IO] input_ys={input_ys.tolist()}, output_ys={output_ys.tolist()}")

    network_lsf_path = _run_step(
        "resolve_network_lsf",
        cfg.debug,
        lambda: sim.get_base_lsf_path("network_simulate"),
    )
    bidir_lsf_path = _run_step(
        "resolve_bidir_lsf",
        cfg.debug,
        lambda: sim.get_base_lsf_path("network_bidir"),
    )
    _debug(cfg.debug, f"[LSF] network_simulate={network_lsf_path}")
    _debug(cfg.debug, f"[LSF] network_bidir={bidir_lsf_path}")
    with (run_dir / "paths.txt").open("a", encoding="utf-8") as handle:
        handle.write(f"network_lsf={network_lsf_path}\n")
        handle.write(f"bidir_lsf={bidir_lsf_path}\n")

    _run_step("save_fsp", cfg.debug, lambda: sim.save(cfg.save_name))

    def _build_overlap_mode() -> Path:
        sim.eval_saved_lsf(
            sim.cuple_fdtd,
            bidir_lsf_path,
            label="load bidirectional overlap setup",
        )
        sim.cuple_fdtd.eval("select('mode_src_fwd');set('enabled',0);")
        sim.cuple_fdtd.eval("run;")

        mod_right = get_fields(
            sim.cuple_fdtd,
            monitor_name="P_right",
            field_result_name="fields",
            get_eps=False,
            get_D=False,
            get_H=True,
            nointerpolation=False,
            unfold_symmetry=False,
        )
        picked = extract_ey_1d_at_wavelength(
            mod_right,
            wavelength_um=selected_wl_um,
            selection="nearest",
        )
        overlap = Overlap(
            y_m=np.asarray(picked["y"]).reshape(-1),
            Ey_m=np.asarray(picked["ey"]).reshape(-1),
            out_dir=sim_dir,
            tag="P_right",
            wavelength_m=selected_wl_um * 1e-6,
            origin="left",
        )
        return Path(overlap.mode_path)

    emode_path = _run_step("build_overlap_mode", cfg.debug, _build_overlap_mode)
    _run_step(
        "run_network",
        cfg.debug,
        lambda: sim.eval_saved_lsf(
            sim.fdtd,
            network_lsf_path,
            label="load network simulation setup",
        ),
    )
    _run_step("run_fdtd", cfg.debug, lambda: sim.fdtd.eval("run;"))

    if reuse_parameters is not None:
        neff_by_wl = normalize_wavelength_dict(
            reuse_parameters["neff_by_wavelength_um"],
            required_wavelengths_um=wl_grid_um,
        )
        mapping_by_wl = normalize_wavelength_dict(
            reuse_parameters["mapping_factor_by_wavelength_um"],
            required_wavelengths_um=wl_grid_um,
        )
        _debug(cfg.debug, "[PARAM] reused neff/mapping_factor from parameter text")
    else:
        neff_by_wl = _run_step(
            "compute_neff_by_wavelength",
            cfg.debug,
            lambda: _compute_neff_by_wavelength(sim, wl_grid_um, cfg.debug),
        )
        mapping_by_wl = _run_step(
            "compute_mapping_factor_by_wavelength",
            cfg.debug,
            lambda: _compute_mapping_factor_by_wavelength(
                sim,
                sim_dir,
                wl_grid_um,
                cfg.mapping,
                cfg.debug,
                neff_by_wavelength_um=neff_by_wl,
                reference_wavelength_um=selected_wl_um,
            ),
        )

    if reuse_parameters_only:
        _debug(cfg.debug, "[PARAM] reuse_parameters_only=True; skipped parameter recomputation")

    monitors = _run_step(
        "extract_monitors",
        cfg.debug,
        lambda: _extract_monitors(sim, shape, cfg.monitors, cfg.debug),
    )
    global_monitors = _run_step(
        "extract_global_monitors",
        cfg.debug,
        lambda: _extract_global_monitors(sim, cfg.debug),
    )

    neff_pick = select_value_by_wavelength(
        neff_by_wl,
        wavelength_um=selected_wl_um,
        selection="nearest",
    )
    mapping_pick = select_value_by_wavelength(
        mapping_by_wl,
        wavelength_um=selected_wl_um,
        selection="nearest",
    )
    selected_wl_um = float(neff_pick["selected_wavelength_um"])

    sim_signature = build_sim_signature(shape, cfg)
    shape_payload = shape.to_cache_payload() if hasattr(shape, "to_cache_payload") else {}
    shape_config = (
        shape.to_config_payload()
        if hasattr(shape, "to_config_payload")
        else dict(shape_payload)
    )
    created_at = datetime.now().isoformat()
    match_key = compute_parameter_match_key(
        shape_payload=shape_payload,
        sim_signature=sim_signature,
        monitor_mode=cfg.monitors.mode,
        wavelengths_um=wl_grid_um,
    )
    params_dir = write_parameter_text_bundle(
        Path(sim_dir) / "params_txt",
        shape_payload=shape_payload,
        shape_config=shape_config,
        sim_signature=sim_signature,
        monitor_mode=cfg.monitors.mode,
        selected_wavelength_um=selected_wl_um,
        neff_by_wavelength_um=neff_by_wl,
        mapping_factor_by_wavelength_um=mapping_by_wl,
        source="fdtd_pipeline",
        created_at=created_at,
    )
    monitors_bundle_path = save_monitors_bundle(
        monitors,
        bundle_path=Path(sim_dir) / "monitors.npz",
        fields_dir=Path(sim_dir) / "fields",
    )
    global_monitor_bundle_path = save_named_payload_bundle(
        global_monitors,
        bundle_path=Path(sim_dir) / "global_monitors.npz",
        fields_dir=Path(sim_dir) / "global_fields",
    )
    simulation_record_path = write_simulation_record(
        sim_dir,
        shape_payload=shape_payload,
        monitors_bundle_path=monitors_bundle_path,
        selected_wavelength_um=selected_wl_um,
        neff_by_wavelength_um=neff_by_wl,
        mapping_factor_by_wavelength_um=mapping_by_wl,
        neff=float(neff_pick["value"]),
        mapping_factor=float(mapping_pick["value"]),
        sim_signature=sim_signature,
        run_dir=run_dir,
        emode_path=emode_path,
        parameter_txt_dir=params_dir,
        global_monitor_bundle_path=global_monitor_bundle_path,
        created_at=created_at,
    )

    return {
        "monitors": monitors,
        "global_monitors": global_monitors,
        "sim_dir": Path(sim_dir),
        "run_dir": Path(run_dir),
        "emode_path": Path(emode_path),
        "parameter_txt_dir": Path(params_dir),
        "monitors_bundle_path": Path(monitors_bundle_path),
        "global_monitor_bundle_path": Path(global_monitor_bundle_path),
        "simulation_record_path": Path(simulation_record_path),
        "sim_signature": sim_signature,
        "match_key": match_key,
        "wavelengths_um": np.asarray(wl_grid_um, dtype=float),
        "selected_wavelength_um": float(selected_wl_um),
        "neff_by_wavelength_um": dict(neff_by_wl),
        "mapping_factor_by_wavelength_um": dict(mapping_by_wl),
        "neff": float(neff_pick["value"]),
        "mapping_factor": float(mapping_pick["value"]),
        "created_at": created_at,
    }


def run_fdtd_workflow(
    shape: Any,
    *,
    cfg: PipelineConfig,
    cache_file: Path | str | None = None,
    make_ts_dir: Optional[MakeTsDirFn] = None,
    run_dir: Optional[Path] = None,
    use_cache: bool = True,
    force_redo: bool = False,
) -> Dict[str, Any]:
    """Run one FDTD workflow directly, with optional cache lookup."""
    if not use_cache or cache_file is None:
        result = run_fdtd_and_extract_once(
            shape,
            cfg=cfg,
            make_ts_dir=make_ts_dir,
            run_dir=run_dir,
        )
        out = dict(result)
        out["source"] = "fresh_run"
        out["use_cached_results"] = False
        out["did_run"] = True
        return out

    sim_signature = build_sim_signature(shape, cfg)

    def _run_live(shape_obj: Any, **kwargs: Any) -> Dict[str, Any]:
        return run_fdtd_and_extract_once(
            shape_obj,
            cfg=cfg,
            make_ts_dir=make_ts_dir,
            run_dir=run_dir,
            **kwargs,
        )

    result = run_and_extract(
        shape,
        cache_file=cache_file,
        run_fn=_run_live,
        force_redo=force_redo,
        sim_signature=sim_signature,
    )
    return result
