from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

from .wavelength_tools import normalize_wavelength_dict


SIMULATION_RECORD_SCHEMA_VERSION = "1"


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


def compute_shape_fingerprint(shape_payload: Mapping[str, Any]) -> str:
    dump = json.dumps(dict(shape_payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(dump.encode("utf-8")).hexdigest()


def save_monitors_bundle(
    monitors: Iterable[Any],
    *,
    bundle_path: str | Path,
    fields_dir: str | Path | None = None,
) -> Path:
    out_path = Path(bundle_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    field_root = None if fields_dir is None else Path(fields_dir)
    if field_root is not None:
        field_root.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, np.ndarray] = {}
    for idx, monitor in enumerate(monitors):
        if isinstance(monitor, dict):
            for key, value in monitor.items():
                if value is None:
                    continue
                arr = np.asarray(value)
                payload[f"monitor_{idx}__field__{key}"] = arr
                if field_root is not None:
                    np.save(field_root / f"monitor_{idx}__field__{key}.npy", arr)
        else:
            arr = np.asarray(monitor)
            payload[f"monitor_{idx}"] = arr
            if field_root is not None:
                np.save(field_root / f"monitor_{idx}.npy", arr)

    np.savez_compressed(out_path, **payload)
    return out_path


def load_monitors_bundle(bundle_path: str | Path) -> List[Any]:
    path = Path(bundle_path)
    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(path, allow_pickle=False)
    keys = list(data.files)
    if all("__" not in key for key in keys):
        ordered = sorted(keys, key=lambda item: int(item.split("_")[-1]))
        return [np.asarray(data[key]) for key in ordered]

    monitor_indices: List[int] = []
    for key in keys:
        if not key.startswith("monitor_"):
            continue
        stem = key.split("__", 1)[0]
        try:
            idx = int(stem.split("_")[-1])
        except ValueError:
            continue
        if idx not in monitor_indices:
            monitor_indices.append(idx)
    monitor_indices.sort()

    out: List[Any] = []
    for idx in monitor_indices:
        raw_key = f"monitor_{idx}"
        if raw_key in data.files:
            out.append(np.asarray(data[raw_key]))
            continue
        prefix = f"monitor_{idx}__field__"
        fields = {
            key[len(prefix) :]: np.asarray(data[key])
            for key in keys
            if key.startswith(prefix)
        }
        out.append(fields)
    return out


def save_named_payload_bundle(
    payloads: Mapping[str, Any],
    *,
    bundle_path: str | Path,
    fields_dir: str | Path | None = None,
) -> Path:
    out_path = Path(bundle_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    field_root = None if fields_dir is None else Path(fields_dir)
    if field_root is not None:
        field_root.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, np.ndarray] = {}
    for name, item in payloads.items():
        if not isinstance(item, Mapping):
            raise TypeError(f"named payload {name!r} must be a mapping")
        for key, value in item.items():
            if value is None:
                continue
            arr = np.asarray(value)
            payload_key = f"{name}__field__{key}"
            payload[payload_key] = arr
            if field_root is not None:
                np.save(field_root / f"{payload_key}.npy", arr)

    np.savez_compressed(out_path, **payload)
    return out_path


def load_named_payload_bundle(bundle_path: str | Path) -> Dict[str, Dict[str, np.ndarray]]:
    path = Path(bundle_path)
    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(path, allow_pickle=False)
    keys = list(data.files)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for key in keys:
        if "__field__" not in key:
            raise ValueError(f"Invalid named payload key {key!r} in {path}")
        name, field_key = key.split("__field__", 1)
        item = out.setdefault(name, {})
        item[field_key] = np.asarray(data[key])
    return out


def _validate_child_path(parent: Path, child: Path, *, label: str) -> None:
    try:
        child.relative_to(parent)
    except ValueError as exc:
        raise RuntimeError(f"{label} must be inside sim_dir: {child}") from exc


@dataclass
class SimulationRecord:
    fingerprint: str
    shape_payload: Dict[str, Any]
    sim_dir: Path
    monitors_bundle_path: Path
    selected_wavelength_um: float
    neff_by_wavelength_um: Dict[str, float]
    mapping_factor_by_wavelength_um: Dict[str, float]
    neff: float
    mapping_factor: float
    sim_signature: str | None = None
    run_dir: Optional[Path] = None
    emode_path: Optional[Path] = None
    parameter_txt_dir: Optional[Path] = None
    global_monitor_bundle_path: Optional[Path] = None
    created_at: str | None = None
    schema_version: str = SIMULATION_RECORD_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SIMULATION_RECORD_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported simulation record schema_version={self.schema_version!r}"
            )
        if compute_shape_fingerprint(self.shape_payload) != self.fingerprint:
            raise RuntimeError("simulation record fingerprint does not match shape_payload")

        sim_dir = self.sim_dir.resolve()
        if not sim_dir.exists():
            raise FileNotFoundError(sim_dir)
        if not sim_dir.is_dir():
            raise NotADirectoryError(sim_dir)

        monitors_bundle_path = self.monitors_bundle_path.resolve()
        if not monitors_bundle_path.exists():
            raise FileNotFoundError(monitors_bundle_path)
        _validate_child_path(sim_dir, monitors_bundle_path, label="monitors_bundle_path")

        if self.global_monitor_bundle_path is not None:
            global_monitor_bundle_path = self.global_monitor_bundle_path.resolve()
            if not global_monitor_bundle_path.exists():
                raise FileNotFoundError(global_monitor_bundle_path)
            _validate_child_path(
                sim_dir,
                global_monitor_bundle_path,
                label="global_monitor_bundle_path",
            )

        if self.emode_path is None:
            raise RuntimeError("simulation record missing emode_path")
        emode_path = self.emode_path.resolve()
        if not emode_path.exists():
            raise FileNotFoundError(emode_path)
        _validate_child_path(sim_dir, emode_path, label="emode_path")

        if self.parameter_txt_dir is None:
            raise RuntimeError("simulation record missing parameter_txt_dir")
        parameter_txt_dir = self.parameter_txt_dir.resolve()
        if not parameter_txt_dir.exists():
            raise FileNotFoundError(parameter_txt_dir)
        if not parameter_txt_dir.is_dir():
            raise NotADirectoryError(parameter_txt_dir)
        _validate_child_path(sim_dir, parameter_txt_dir, label="parameter_txt_dir")
        manifest_path = parameter_txt_dir / "manifest.txt"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)

        if self.run_dir is not None and not self.run_dir.resolve().exists():
            raise FileNotFoundError(self.run_dir)

        neff_by = normalize_wavelength_dict(self.neff_by_wavelength_um)
        mapping_by = normalize_wavelength_dict(self.mapping_factor_by_wavelength_um)
        if set(neff_by.keys()) != set(mapping_by.keys()):
            raise ValueError(
                "simulation record neff and mapping_factor dictionaries must share keys"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fingerprint": self.fingerprint,
            "shape_payload": _to_serializable(self.shape_payload),
            "sim_dir": str(self.sim_dir),
            "monitors_bundle_path": str(self.monitors_bundle_path),
            "selected_wavelength_um": float(self.selected_wavelength_um),
            "neff_by_wavelength_um": dict(self.neff_by_wavelength_um),
            "mapping_factor_by_wavelength_um": dict(self.mapping_factor_by_wavelength_um),
            "neff": float(self.neff),
            "mapping_factor": float(self.mapping_factor),
            "sim_signature": self.sim_signature,
            "run_dir": (None if self.run_dir is None else str(self.run_dir)),
            "emode_path": (None if self.emode_path is None else str(self.emode_path)),
            "parameter_txt_dir": (
                None if self.parameter_txt_dir is None else str(self.parameter_txt_dir)
            ),
            "global_monitor_bundle_path": (
                None
                if self.global_monitor_bundle_path is None
                else str(self.global_monitor_bundle_path)
            ),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SimulationRecord":
        neff_by = normalize_wavelength_dict(data["neff_by_wavelength_um"])
        mapping_by = normalize_wavelength_dict(data["mapping_factor_by_wavelength_um"])
        return cls(
            schema_version=str(data["schema_version"]),
            fingerprint=str(data["fingerprint"]),
            shape_payload=dict(data["shape_payload"]),
            sim_dir=Path(data["sim_dir"]).resolve(),
            monitors_bundle_path=Path(data["monitors_bundle_path"]).resolve(),
            selected_wavelength_um=float(data["selected_wavelength_um"]),
            neff_by_wavelength_um=neff_by,
            mapping_factor_by_wavelength_um=mapping_by,
            neff=float(data["neff"]),
            mapping_factor=float(data["mapping_factor"]),
            sim_signature=(
                None if data.get("sim_signature") is None else str(data["sim_signature"])
            ),
            run_dir=(None if data.get("run_dir") is None else Path(data["run_dir"]).resolve()),
            emode_path=(
                None if data.get("emode_path") is None else Path(data["emode_path"]).resolve()
            ),
            parameter_txt_dir=(
                None
                if data.get("parameter_txt_dir") is None
                else Path(data["parameter_txt_dir"]).resolve()
            ),
            global_monitor_bundle_path=(
                None
                if data.get("global_monitor_bundle_path") is None
                else Path(data["global_monitor_bundle_path"]).resolve()
            ),
            created_at=data.get("created_at"),
        )


def write_simulation_record(
    sim_dir: str | Path,
    *,
    shape_payload: Mapping[str, Any],
    monitors_bundle_path: str | Path,
    selected_wavelength_um: float,
    neff_by_wavelength_um: Mapping[Any, Any],
    mapping_factor_by_wavelength_um: Mapping[Any, Any],
    neff: float,
    mapping_factor: float,
    sim_signature: str | None = None,
    run_dir: str | Path | None = None,
    emode_path: str | Path | None = None,
    parameter_txt_dir: str | Path | None = None,
    global_monitor_bundle_path: str | Path | None = None,
    created_at: str | None = None,
) -> Path:
    sim_dir_path = Path(sim_dir).resolve()
    record = SimulationRecord(
        fingerprint=compute_shape_fingerprint(shape_payload),
        shape_payload=dict(shape_payload),
        sim_dir=sim_dir_path,
        monitors_bundle_path=Path(monitors_bundle_path).resolve(),
        selected_wavelength_um=float(selected_wavelength_um),
        neff_by_wavelength_um=normalize_wavelength_dict(neff_by_wavelength_um),
        mapping_factor_by_wavelength_um=normalize_wavelength_dict(mapping_factor_by_wavelength_um),
        neff=float(neff),
        mapping_factor=float(mapping_factor),
        sim_signature=(None if sim_signature is None else str(sim_signature)),
        run_dir=(None if run_dir is None else Path(run_dir).resolve()),
        emode_path=(None if emode_path is None else Path(emode_path).resolve()),
        parameter_txt_dir=(
            None if parameter_txt_dir is None else Path(parameter_txt_dir).resolve()
        ),
        global_monitor_bundle_path=(
            None
            if global_monitor_bundle_path is None
            else Path(global_monitor_bundle_path).resolve()
        ),
        created_at=created_at,
    )
    record.validate()

    record_path = sim_dir_path / "simulation_record.json"
    record_path.write_text(
        json.dumps(record.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return record_path


def load_simulation_record(record_path: str | Path) -> SimulationRecord:
    path = Path(record_path)
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"simulation record must be a JSON object: {path}")
    record = SimulationRecord.from_dict(data)
    record.validate()
    return record
