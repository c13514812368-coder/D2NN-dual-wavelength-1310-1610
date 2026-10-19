from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np

from .lumerical_scripts import extract_ey_1d_at_wavelength, select_field_component_at_wavelength
from .networkangler import NetworkShape
from .simulation_store import (
    SimulationRecord,
    compute_shape_fingerprint,
    load_named_payload_bundle,
    load_monitors_bundle,
    load_simulation_record,
)
from .wavelength_tools import (
    normalize_wavelength_dict,
    resolve_wavelength_choice,
    select_value_by_wavelength,
)


CACHE_INDEX_SCHEMA_VERSION = "1"
RunFn = Callable[..., Dict[str, Any]]


@dataclass
class CacheEntry:
    fingerprint: str
    record_path: Path
    created_at: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "record_path": str(self.record_path),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CacheEntry":
        return cls(
            fingerprint=str(data["fingerprint"]),
            record_path=Path(data["record_path"]).resolve(),
            created_at=data.get("created_at"),
        )


def _record_wavelengths_um(record: SimulationRecord) -> np.ndarray:
    if record.neff_by_wavelength_um:
        return np.asarray([float(key) for key in record.neff_by_wavelength_um.keys()], dtype=float)
    if record.mapping_factor_by_wavelength_um:
        return np.asarray(
            [float(key) for key in record.mapping_factor_by_wavelength_um.keys()],
            dtype=float,
        )
    return np.asarray([float(record.selected_wavelength_um)], dtype=float)


def _invoke_run_fn(run_fn: RunFn, shape: NetworkShape, **kwargs: Any) -> Dict[str, Any]:
    if not kwargs:
        return run_fn(shape)

    try:
        sig = inspect.signature(run_fn)
    except Exception:
        return run_fn(shape)

    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
    )
    if accepts_kwargs:
        return run_fn(shape, **kwargs)

    accepted: Dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in sig.parameters:
            accepted[key] = value
    if accepted:
        return run_fn(shape, **accepted)
    return run_fn(shape)


def _result_from_record(
    record: SimulationRecord,
    *,
    monitors: Optional[list[Any]],
    global_monitors: Optional[dict[str, Any]],
    use_cached_results: bool,
    did_run: bool,
    source: str,
) -> Dict[str, Any]:
    resolved_monitors = list(load_monitors_bundle(record.monitors_bundle_path) if monitors is None else monitors)
    resolved_global_monitors = (
        (
            None
            if record.global_monitor_bundle_path is None
            else load_named_payload_bundle(record.global_monitor_bundle_path)
        )
        if global_monitors is None
        else dict(global_monitors)
    )
    return {
        "use_cached_results": bool(use_cached_results),
        "did_run": bool(did_run),
        "cache_entry": record,
        "monitors": resolved_monitors,
        "global_monitors": resolved_global_monitors,
        "diffs": [],
        "warnings": [],
        "wavelengths_um": _record_wavelengths_um(record),
        "neff_by_wavelength_um": dict(record.neff_by_wavelength_um),
        "mapping_factor_by_wavelength_um": dict(record.mapping_factor_by_wavelength_um),
        "selected_wavelength_um": float(record.selected_wavelength_um),
        "neff": float(record.neff),
        "mapping_factor": float(record.mapping_factor),
        "emode_path": record.emode_path,
        "sim_dir": record.sim_dir,
        "run_dir": record.run_dir,
        "parameter_txt_dir": record.parameter_txt_dir,
        "simulation_record_path": record.sim_dir / "simulation_record.json",
        "monitors_bundle_path": record.monitors_bundle_path,
        "global_monitor_bundle_path": record.global_monitor_bundle_path,
        "sim_signature": record.sim_signature,
        "source": source,
    }


class SimulationCache:
    def __init__(self, cache_file: Path):
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._entries: Dict[str, CacheEntry] = {}
        self._load_index()

    @staticmethod
    def fingerprint_shape(shape: NetworkShape) -> str:
        return compute_shape_fingerprint(shape.to_cache_payload())

    def _load_index(self) -> None:
        if not self.cache_file.exists():
            self._entries = {}
            return

        try:
            raw = json.loads(self.cache_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid simulation cache index JSON: {self.cache_file}") from exc

        if not isinstance(raw, dict):
            raise TypeError(f"simulation cache index must be a JSON object: {self.cache_file}")
        if str(raw.get("schema_version")) != CACHE_INDEX_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported simulation cache index schema_version="
                f"{raw.get('schema_version')!r} in {self.cache_file}"
            )

        entries_raw = raw.get("entries")
        if not isinstance(entries_raw, dict):
            raise TypeError(f"simulation cache index entries must be a JSON object: {self.cache_file}")

        entries: Dict[str, CacheEntry] = {}
        for fingerprint, payload in entries_raw.items():
            if not isinstance(payload, dict):
                raise TypeError(f"simulation cache entry {fingerprint!r} must be a JSON object")
            entry = CacheEntry.from_dict(payload)
            if entry.fingerprint != fingerprint:
                raise RuntimeError(
                    f"simulation cache entry key mismatch for {fingerprint!r} in {self.cache_file}"
                )
            entries[fingerprint] = entry
        self._entries = entries

    def _save_index(self) -> None:
        payload = {
            "schema_version": CACHE_INDEX_SCHEMA_VERSION,
            "entries": {fingerprint: entry.to_dict() for fingerprint, entry in self._entries.items()},
        }
        tmp = self.cache_file.with_suffix(self.cache_file.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.cache_file)

    def lookup(self, shape: NetworkShape, *, sim_signature: Optional[str] = None) -> Optional[SimulationRecord]:
        shape_payload = shape.to_cache_payload()
        fingerprint = compute_shape_fingerprint(shape_payload)
        entry = self._entries.get(fingerprint)
        if entry is None:
            return None

        record = load_simulation_record(entry.record_path)
        if record.fingerprint != fingerprint:
            raise RuntimeError(
                f"cache entry {fingerprint} points to mismatched simulation record {entry.record_path}"
            )
        if record.shape_payload != shape_payload:
            raise RuntimeError(
                f"cache entry {fingerprint} shape payload does not match current query"
            )
        if sim_signature is not None and record.sim_signature != str(sim_signature):
            return None
        return record

    def store_result(self, shape: NetworkShape, sim_result: Mapping[str, Any]) -> SimulationRecord:
        if "simulation_record_path" not in sim_result:
            raise KeyError("run_fn result missing required key: simulation_record_path")

        shape_payload = shape.to_cache_payload()
        fingerprint = compute_shape_fingerprint(shape_payload)
        record = load_simulation_record(sim_result["simulation_record_path"])
        if record.fingerprint != fingerprint:
            raise RuntimeError("simulation record fingerprint does not match queried shape")
        if record.shape_payload != shape_payload:
            raise RuntimeError("simulation record shape_payload does not match queried shape")

        self._entries[fingerprint] = CacheEntry(
            fingerprint=fingerprint,
            record_path=Path(sim_result["simulation_record_path"]).resolve(),
            created_at=record.created_at,
        )
        self._save_index()
        return record


def load_or_run(
    cache: SimulationCache,
    shape: NetworkShape,
    *,
    run_fn: RunFn,
    force_redo: bool = False,
    sim_signature: Optional[str] = None,
) -> Dict[str, Any]:
    if force_redo:
        sim_result = _invoke_run_fn(run_fn, shape)
        record = cache.store_result(shape, sim_result)
        return _result_from_record(
            record,
            monitors=list(sim_result["monitors"]),
            global_monitors=dict(sim_result.get("global_monitors") or {}),
            use_cached_results=False,
            did_run=True,
            source="ran_force",
        )

    record = cache.lookup(shape, sim_signature=sim_signature)
    if record is not None:
        return _result_from_record(
            record,
            monitors=None,
            global_monitors=None,
            use_cached_results=True,
            did_run=False,
            source="exact_cache",
        )

    sim_result = _invoke_run_fn(run_fn, shape)
    record = cache.store_result(shape, sim_result)
    return _result_from_record(
        record,
        monitors=list(sim_result["monitors"]),
        global_monitors=dict(sim_result.get("global_monitors") or {}),
        use_cached_results=False,
        did_run=True,
        source="ran_no_cache",
    )


def run_and_extract(
    shape: NetworkShape,
    *,
    cache_file: Path | str,
    run_fn: RunFn,
    force_redo: bool = False,
    sim_signature: Optional[str] = None,
) -> Dict[str, Any]:
    cache = SimulationCache(Path(cache_file))
    return load_or_run(
        cache,
        shape,
        run_fn=run_fn,
        force_redo=force_redo,
        sim_signature=sim_signature,
    )


def select_simulation_wavelength(
    sim_data: Mapping[str, Any],
    *,
    wavelength_um: float | None = None,
    wavelength_index: int | None = None,
    selection: str = "nearest",
    monitor_output: str = "ey_1d",
    component: str = "Ey",
) -> Dict[str, Any]:
    neff_by = normalize_wavelength_dict(sim_data.get("neff_by_wavelength_um", {}))
    mapping_by = normalize_wavelength_dict(sim_data.get("mapping_factor_by_wavelength_um", {}))

    if neff_by:
        wl_grid = np.asarray([float(key) for key in neff_by.keys()], dtype=float)
    elif mapping_by:
        wl_grid = np.asarray([float(key) for key in mapping_by.keys()], dtype=float)
    else:
        wl_grid = np.asarray([], dtype=float)

    if wl_grid.size == 0:
        for monitor in sim_data.get("monitors", []):
            if isinstance(monitor, dict) and "lambda_um" in monitor:
                wl_grid = np.asarray(monitor["lambda_um"], dtype=float).reshape(-1)
                break
    if wl_grid.size == 0:
        raise ValueError("No wavelength information in simulation data")

    chosen = resolve_wavelength_choice(
        wl_grid,
        wavelength_um=wavelength_um,
        wavelength_index=wavelength_index,
        selection=selection,
    )
    selected_wl_um = float(chosen["selected_wavelength_um"])
    selected_idx = int(chosen["selected_wavelength_index"])

    out_neff = (
        select_value_by_wavelength(neff_by, wavelength_um=selected_wl_um, selection="nearest")["value"]
        if neff_by
        else None
    )
    out_mapping = (
        select_value_by_wavelength(mapping_by, wavelength_um=selected_wl_um, selection="nearest")["value"]
        if mapping_by
        else None
    )

    monitors_out = []
    for monitor in sim_data.get("monitors", []):
        if monitor_output == "full_fields":
            out = select_field_component_at_wavelength(
                monitor,
                component=component,
                wavelength_um=selected_wl_um,
                selection="nearest",
            )
            monitors_out.append(out["field"])
        else:
            out = extract_ey_1d_at_wavelength(
                monitor,
                wavelength_um=selected_wl_um,
                selection="nearest",
            )
            monitors_out.append(out["ey"])

    return {
        "selected_wavelength_um": selected_wl_um,
        "selected_wavelength_index": selected_idx,
        "neff": out_neff,
        "mapping_factor": out_mapping,
        "monitors": monitors_out,
    }
