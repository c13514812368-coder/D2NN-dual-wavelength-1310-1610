from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np


def canonical_wavelength_key(wavelength_um: float) -> str:
    """Return a stable string key for wavelength dictionaries."""
    text = f"{float(wavelength_um):.9f}".rstrip("0").rstrip(".")
    return text if text else "0"


def wavelength_key_to_float(key: Any) -> float:
    if isinstance(key, (float, int, np.floating, np.integer)):
        return float(key)
    return float(str(key).strip())


def normalize_wavelength_dict(
    values: Mapping[Any, Any],
    *,
    required_wavelengths_um: Sequence[float] | None = None,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in values.items():
        wl = wavelength_key_to_float(k)
        out[canonical_wavelength_key(wl)] = float(v)

    if required_wavelengths_um is not None:
        missing: list[float] = []
        for wl in np.asarray(required_wavelengths_um, dtype=float).reshape(-1):
            key = canonical_wavelength_key(float(wl))
            if key not in out:
                missing.append(float(wl))
        if missing:
            raise KeyError(f"Missing wavelength values for: {missing}")
    return out


def sorted_wavelengths_from_dict(values_by_wavelength_um: Mapping[Any, Any]) -> np.ndarray:
    if not values_by_wavelength_um:
        return np.asarray([], dtype=float)
    vals = [wavelength_key_to_float(k) for k in values_by_wavelength_um.keys()]
    return np.asarray(sorted(vals), dtype=float)


def resolve_wavelength_choice(
    wavelengths_um: Iterable[float],
    *,
    wavelength_um: float | None = None,
    wavelength_index: int | None = None,
    selection: str = "nearest",
) -> Dict[str, Any]:
    wl = np.asarray(list(wavelengths_um), dtype=float).reshape(-1)
    if wl.size == 0:
        raise ValueError("No wavelength grid available")

    if wavelength_index is not None:
        idx = int(wavelength_index)
    elif wavelength_um is None:
        idx = wl.size // 2
    elif selection == "exact":
        hits = np.where(np.isclose(wl, float(wavelength_um), rtol=1e-9, atol=1e-12))[0]
        if hits.size == 0:
            raise ValueError(f"Wavelength {wavelength_um} um not found")
        idx = int(hits[0])
    else:
        idx = int(np.argmin(np.abs(wl - float(wavelength_um))))

    if not (0 <= idx < wl.size):
        raise IndexError(f"wavelength_index out of range: {idx}, valid [0, {wl.size - 1}]")

    return {
        "wavelengths_um": wl,
        "selected_wavelength_index": int(idx),
        "selected_wavelength_um": float(wl[idx]),
    }


def select_value_by_wavelength(
    values_by_wavelength_um: Mapping[Any, Any],
    *,
    wavelength_um: float | None = None,
    wavelength_index: int | None = None,
    selection: str = "nearest",
) -> Dict[str, Any]:
    wl = sorted_wavelengths_from_dict(values_by_wavelength_um)
    chosen = resolve_wavelength_choice(
        wl,
        wavelength_um=wavelength_um,
        wavelength_index=wavelength_index,
        selection=selection,
    )
    key = canonical_wavelength_key(chosen["selected_wavelength_um"])
    if key not in values_by_wavelength_um:
        normalized = normalize_wavelength_dict(values_by_wavelength_um)
        value = normalized[key]
    else:
        value = float(values_by_wavelength_um[key])
    out = dict(chosen)
    out["value"] = float(value)
    return out

