from __future__ import annotations

import ast
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from .wavelength_tools import (
    canonical_wavelength_key,
    normalize_wavelength_dict,
    sorted_wavelengths_from_dict,
)


PARAMETER_TEXT_SCHEMA_VERSION = "2"


def _shape_fingerprint(shape_payload: Mapping[str, Any]) -> str:
    dump = json.dumps(dict(shape_payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(dump.encode("utf-8")).hexdigest()


def compute_parameter_match_key(
    *,
    shape_payload: Mapping[str, Any],
    sim_signature: str,
    monitor_mode: str,
    wavelengths_um: np.ndarray,
) -> str:
    payload = {
        "shape_fingerprint": _shape_fingerprint(shape_payload),
        "sim_signature": str(sim_signature),
        "monitor_mode": str(monitor_mode),
        "wavelengths_um": np.asarray(wavelengths_um, dtype=float).reshape(-1).tolist(),
    }
    dump = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(dump.encode("utf-8")).hexdigest()


def _write_kv_file(path: Path, values: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for key in sorted(values.keys()):
        val = values[key]
        if isinstance(val, (dict, list, tuple)):
            text = json.dumps(val, sort_keys=True, separators=(",", ":"))
        else:
            text = str(val)
        lines.append(f"{key}={text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _read_kv_file(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _parse_kv_value(text: str) -> Any:
    value = text.strip()
    if value == "":
        return ""
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(value)
        except Exception:
            continue
    return value


def _read_typed_kv_file(path: Path) -> Dict[str, Any]:
    return {key: _parse_kv_value(value) for key, value in _read_kv_file(path).items()}


def _wavelength_file_name(wavelength_um: float) -> str:
    nm = float(wavelength_um) * 1000.0
    return f"wl_{nm:.3f}nm.txt"


def write_parameter_text_bundle(
    params_dir: Path,
    *,
    shape_payload: Mapping[str, Any],
    shape_config: Mapping[str, Any] | None = None,
    sim_signature: str,
    monitor_mode: str,
    selected_wavelength_um: float,
    neff_by_wavelength_um: Mapping[Any, Any],
    mapping_factor_by_wavelength_um: Mapping[Any, Any],
    source: str = "fdtd_pipeline",
    created_at: str | None = None,
) -> Path:
    params_dir = Path(params_dir)
    params_dir.mkdir(parents=True, exist_ok=True)
    resolved_shape_config = dict(shape_payload if shape_config is None else shape_config)

    neff = normalize_wavelength_dict(neff_by_wavelength_um)
    mapping = normalize_wavelength_dict(mapping_factor_by_wavelength_um)
    wl_grid = sorted_wavelengths_from_dict(neff)
    if wl_grid.size == 0:
        raise ValueError("Cannot write parameter bundle with empty wavelength grid")
    if set(neff.keys()) != set(mapping.keys()):
        raise ValueError("neff and mapping_factor dictionaries must share the same wavelength keys")

    created = created_at or datetime.now().isoformat()
    match_key = compute_parameter_match_key(
        shape_payload=shape_payload,
        sim_signature=sim_signature,
        monitor_mode=monitor_mode,
        wavelengths_um=wl_grid,
    )

    manifest = {
        "schema_version": PARAMETER_TEXT_SCHEMA_VERSION,
        "created_at": created,
        "source": str(source),
        "sim_signature": str(sim_signature),
        "monitor_mode": str(monitor_mode),
        "match_key": match_key,
        "shape_fingerprint": _shape_fingerprint(shape_payload),
        "selected_wavelength_um": float(selected_wavelength_um),
        "shape_config_file": "shape.txt",
        "shape_payload_file": "shape_payload.txt",
        "wavelengths_um": ",".join(canonical_wavelength_key(v) for v in wl_grid.tolist()),
    }
    _write_kv_file(params_dir / "manifest.txt", manifest)
    _write_kv_file(params_dir / "shape.txt", resolved_shape_config)
    _write_kv_file(params_dir / "shape_payload.txt", shape_payload)

    for wl_um in wl_grid.tolist():
        key = canonical_wavelength_key(wl_um)
        _write_kv_file(
            params_dir / _wavelength_file_name(wl_um),
            {
                "schema_version": PARAMETER_TEXT_SCHEMA_VERSION,
                "created_at": created,
                "wavelength_um": key,
                "neff": float(neff[key]),
                "mapping_factor": float(mapping[key]),
                "sim_signature": str(sim_signature),
                "match_key": match_key,
            },
        )

    return params_dir


def read_parameter_text_bundle(params_dir: Path) -> Dict[str, Any]:
    params_dir = Path(params_dir)
    manifest_path = params_dir / "manifest.txt"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    manifest = _read_kv_file(manifest_path)
    if manifest.get("schema_version") != PARAMETER_TEXT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported parameter text schema_version={manifest.get('schema_version')!r}"
        )

    shape_config_path = params_dir / "shape.txt"
    shape_payload_path = params_dir / "shape_payload.txt"
    shape_config = (
        _read_typed_kv_file(shape_config_path)
        if shape_config_path.exists()
        else None
    )
    shape_payload = (
        _read_typed_kv_file(shape_payload_path)
        if shape_payload_path.exists()
        else None
    )

    neff_by_wl: Dict[str, float] = {}
    mapping_by_wl: Dict[str, float] = {}
    for wl_file in sorted(params_dir.glob("wl_*nm.txt")):
        payload = _read_kv_file(wl_file)
        if payload.get("schema_version") != PARAMETER_TEXT_SCHEMA_VERSION:
            continue
        wl_key = canonical_wavelength_key(float(payload["wavelength_um"]))
        neff_by_wl[wl_key] = float(payload["neff"])
        mapping_by_wl[wl_key] = float(payload["mapping_factor"])

    if not neff_by_wl:
        raise ValueError(f"No wavelength parameter files found in {params_dir}")
    if set(neff_by_wl.keys()) != set(mapping_by_wl.keys()):
        raise ValueError("Parameter text bundle has inconsistent wavelength keys")

    wl_grid = sorted_wavelengths_from_dict(neff_by_wl)
    selected_wl = float(manifest.get("selected_wavelength_um", wl_grid[len(wl_grid) // 2]))

    return {
        "schema_version": manifest["schema_version"],
        "created_at": manifest.get("created_at"),
        "source": manifest.get("source"),
        "sim_signature": manifest.get("sim_signature"),
        "monitor_mode": manifest.get("monitor_mode"),
        "match_key": manifest.get("match_key"),
        "shape_fingerprint": manifest.get("shape_fingerprint"),
        "shape_config": shape_config,
        "shape_payload": shape_payload,
        "selected_wavelength_um": selected_wl,
        "wavelengths_um": wl_grid,
        "neff_by_wavelength_um": neff_by_wl,
        "mapping_factor_by_wavelength_um": mapping_by_wl,
    }
