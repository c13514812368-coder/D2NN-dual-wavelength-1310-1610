from __future__ import annotations

"""
Standalone, minimal hybrid 3D FDTD helper.

Goal: drop this single file alongside lumapi + common scientific packages, and
call the provided functions to (1) set up a simple hybrid FDTD project,
(2) bridge a 1D field into 3D via an import source, and (3) read back the
resampled Ey field 鈥?without importing any other project modules.
"""

from pathlib import Path
from typing import Any, Mapping, Sequence

from .lumerical_scripts import (
    coordinates_to_um,
    extract_ey_1d_at_wavelength,
    get_fields,
    select_field_component_at_wavelength,
)
import h5py
import numpy as np

try:
    import torch  # optional; only used for convenience
except Exception:  # pragma: no cover
    torch = None

try:
    import lumapi  # Lumerical/FDTD Solutions Python API
except Exception as exc:  # pragma: no cover
    lumapi = None
    _lumapi_import_error = exc
else:
    _lumapi_import_error = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _project_runs_dir(name: str) -> Path:
    return (_project_root() / "runs" / name).resolve()


# ---------------------------------------------------------------------------
# Utilities copied from the main project (trimmed to essentials)
# ---------------------------------------------------------------------------.


def load_lumerical_modeE(path: str | Path) -> dict:
    """
    Load a 1D mode field (MATLAB v7.3/HDF5) exported by Lumerical MODE/FDE.
    Returns dict with Ex, Ey, Ez, z grid and metadata.
    """
    path = Path(path)

    def _read_node(f, node):
        if isinstance(node, h5py.Dataset):
            if node.dtype == h5py.ref_dtype:
                ref = node[()]
                if ref.shape == ():
                    return _read_node(f, f[ref])
                out = np.empty(ref.shape, dtype=object)
                it = np.nditer(ref, flags=["multi_index", "refs_ok"])
                for r in it:
                    out[it.multi_index] = _read_node(f, f[r.item()])
                return out
            return node[()]
        if isinstance(node, h5py.Group):
            return {k: _read_node(f, node[k]) for k in node.keys()}
        return None

    def _unwrap(a):
        while isinstance(a, np.ndarray) and a.dtype == object and a.size == 1:
            a = a.item()
        return a

    def _to_complex(a):
        a = _unwrap(a)
        if isinstance(a, np.ndarray) and a.dtype.fields:
            keys = set(a.dtype.fields.keys())
            if {"real", "imag"}.issubset(keys):
                return a["real"] + 1j * a["imag"]
        if isinstance(a, dict) and "real" in a and "imag" in a:
            return np.asarray(a["real"]) + 1j * np.asarray(a["imag"])
        return np.asarray(a)

    with h5py.File(str(path), "r") as f:
        if "E" in f:
            top = _read_node(f, f["E"])
        else:

            def _get(name, default=None):
                return f[name][()] if name in f else default

            top = {
                "Ex": _get("Ex"),
                "Ey": _get("Ey"),
                "Ez": _get("Ez"),
                "x": _get("x", 0.0),
                "y": _get("y", 0.0),
                "z": _get("z", []),
                "lambda": _get("lambda", None),
                "f": _get("f", None),
            }

    if isinstance(top, dict):
        if "E" in top:
            E = _to_complex(top.get("E", top))
        else:
            Exd = _to_complex(top.get("Ex", None))
            Eyd = _to_complex(top.get("Ey", None))
            Ezd = _to_complex(top.get("Ez", None))
            if Exd is None or Eyd is None or Ezd is None:
                raise RuntimeError("Missing Ex/Ey/Ez datasets in mode file")
            Exd = np.squeeze(Exd)
            Eyd = np.squeeze(Eyd)
            Ezd = np.squeeze(Ezd)

            def _collapse(arr):
                if arr.ndim == 3:
                    return arr
                if arr.ndim == 2:
                    return arr[np.newaxis, ...]
                if arr.ndim == 1:
                    return arr[np.newaxis, np.newaxis, ...]
                raise RuntimeError(f"Unsupported mode field shape: {arr.shape}")

            Exd = _collapse(Exd)
            Eyd = _collapse(Eyd)
            Ezd = _collapse(Ezd)
            Nz = Exd.shape[-1]
            if not (Eyd.shape[-1] == Ezd.shape[-1] == Nz):
                raise RuntimeError("Ex/Ey/Ez length mismatch in mode file")
            E = np.stack([Exd[0], Eyd[0], Ezd[0]], axis=0)

        x = float(np.asarray(_unwrap(top.get("x", 0.0))).squeeze())
        y = float(np.asarray(_unwrap(top.get("y", 0.0))).squeeze())
        z = np.asarray(_unwrap(top.get("z", []))).squeeze()
        lam = top.get("lambda", None)
        lam = float(np.asarray(_unwrap(lam)).squeeze()) if lam is not None else None
        f0 = top.get("f", None)
        f0 = float(np.asarray(_unwrap(f0)).squeeze()) if f0 is not None else None
    else:
        E = _to_complex(top)
        x = 0.0
        y = 0.0
        z = np.array([])
        lam = None
        f0 = None

    if E.ndim == 3 and E.shape[0] == 1 and E.shape[1] == 3:
        Ex, Ey, Ez = E[0, 0, :], E[0, 1, :], E[0, 2, :]
    elif E.ndim == 2 and E.shape[0] == 3:
        Ex, Ey, Ez = E[0, :], E[1, :], E[2, :]
    elif E.ndim == 2 and E.shape[1] == 3:
        Ex, Ey, Ez = E[:, 0], E[:, 1], E[:, 2]
    else:
        raise RuntimeError(f"Unsupported E shape: {E.shape}")

    if z.size and Ex.shape[-1] != z.shape[0]:
        raise RuntimeError(
            f"E length {Ex.shape[-1]} does not match z length {z.shape[0]}"
        )

    return {
        "Ex": Ex,
        "Ey": Ey,
        "Ez": Ez,
        "z": z,
        "x0": x,
        "y0": y,
        "lambda0": lam,
        "f0": f0,
    }


def build_yz_vector_zlink0(
    pred_Ey_y,
    y,
    mode_Ex_z,
    mode_Ey_z,
    mode_Ez_z,
    z,
    *,
    zlink: float = 0.0,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Expand a 1D Ey(y) field into 2D Ey(y,z) using the modal polarization along z.
    """

    def _interp_complex(x, v, x0):
        x = np.asarray(x).ravel()
        v = np.asarray(v)
        vr = np.interp(x0, x, v.real)
        vi = np.interp(x0, x, v.imag)
        return vr + 1j * vi

    y = np.asarray(y).squeeze()
    z = np.asarray(z).squeeze()
    pred = np.asarray(pred_Ey_y).astype(np.complex128).squeeze()
    Exm = np.asarray(mode_Ex_z).astype(np.complex128).squeeze()
    Eym = np.asarray(mode_Ey_z).astype(np.complex128).squeeze()
    Ezm = np.asarray(mode_Ez_z).astype(np.complex128).squeeze()

    if y.ndim != 1 or z.ndim != 1:
        raise ValueError("y and z must be 1D arrays")
    if pred.shape != (y.size,):
        raise ValueError(f"pred_Ey_y shape {pred.shape} != (len(y),)")
    if not (Exm.shape == Eym.shape == Ezm.shape == (z.size,)):
        raise ValueError("mode Ex/Ey/Ez length must match z")

    Ey0 = _interp_complex(z, Eym, zlink)
    if np.abs(Ey0) < eps:
        raise RuntimeError("Ey_mode(zlink) is too small to normalize")

    s = Eym / Ey0
    Ey_yz = pred[:, None] * s[None, :]

    ok = np.abs(Eym) >= eps
    rx = np.zeros_like(Eym, dtype=np.complex128)
    rz = np.zeros_like(Eym, dtype=np.complex128)
    rx[ok] = Exm[ok] / Eym[ok]
    rz[ok] = Ezm[ok] / Eym[ok]

    Ex_yz = Ey_yz * rx[None, :]
    Ez_yz = Ey_yz * rz[None, :]
    return Ex_yz, Ey_yz, Ez_yz


def save_mat_v73_for_lumerical_h5py(
    path: str | Path,
    *,
    x0: float,
    y,
    z,
    Ex_yz,
    Ey_yz,
    Ez_yz,
    f0: float | None = None,
) -> None:
    """
    Save rectilinear Ex/Ey/Ez(y,z) data into a MATLAB v7.3 file usable by Lumerical import source.
    """

    def _write_mat73_header(fname: str):
        magic = bytes.fromhex("00000000000000000002494D")
        s = "MATLAB 7.3 MAT-file, Platform: CPython, Created by Codex CLI."
        b = s.encode("ascii", "ignore")[:116].ljust(116, b" ")
        header128 = b + magic
        with open(fname, "r+b") as fobj:
            fobj.seek(0)
            fobj.write(header128)

    def _matlab_attrs(dset, matlab_class=b"double", is_complex=False):
        dset.attrs["MATLAB_class"] = np.bytes_(matlab_class)
        if is_complex:
            dset.attrs["MATLAB_complex"] = np.uint8(1)

    def _as_fortran(a):
        # Keep data Fortran-ordered so h5py avoids extra copies when writing
        return np.asfortranarray(a)

    def _save_real(fh, name, a):
        a = _as_fortran(np.asarray(a, np.float64))
        d = fh.create_dataset(name, data=a)
        _matlab_attrs(d, b"double", is_complex=False)

    def _save_complex(fh, name, a):
        a = np.asarray(a, np.complex128)
        comp = np.empty(
            a.shape, dtype=[("real", np.float64), ("imag", np.float64)], order="F"
        )
        comp["real"] = _as_fortran(a.real)
        comp["imag"] = _as_fortran(a.imag)
        d = fh.create_dataset(name, data=comp)
        _matlab_attrs(d, b"double", is_complex=True)

    path = Path(path)
    y = np.asarray(y, np.float64).reshape(-1)
    z = np.asarray(z, np.float64).reshape(-1)
    Ny, Nz = y.size, z.size

    Ex = np.asarray(Ex_yz, np.complex128).reshape(1, Ny, Nz)
    Ey = np.asarray(Ey_yz, np.complex128).reshape(1, Ny, Nz)
    Ez = np.asarray(Ez_yz, np.complex128).reshape(1, Ny, Nz)

    with h5py.File(str(path), "w", userblock_size=512) as fh:
        fh.create_group("#refs#")
        _save_real(fh, "x", np.array([[float(x0)]], np.float64))
        _save_real(fh, "y", y[None, :])
        _save_real(fh, "z", z[None, :])
        if f0 is not None:
            _save_real(fh, "f", np.array([[float(f0)]], np.float64))
        _save_complex(fh, "Ex", Ex)
        _save_complex(fh, "Ey", Ey)
        _save_complex(fh, "Ez", Ez)

    _write_mat73_header(str(path))


def interp_complex(y, x_old, x_new, fill=0.0):
    """
    1D complex interpolation with optional fill outside range.
    """
    y = np.asarray(y)
    x_old = np.asarray(x_old, dtype=float)
    x_new = np.asarray(x_new, dtype=float)

    order = np.argsort(x_old)
    x = x_old[order]
    y = y[order] if y.ndim == 1 else y[order, :]

    in_range = (x_new >= x[0]) & (x_new <= x[-1])

    if y.ndim == 1:
        re = np.full_like(x_new, fill, dtype=float)
        im = np.full_like(x_new, fill, dtype=float)
        re[in_range] = np.interp(x_new[in_range], x, np.real(y))
        im[in_range] = np.interp(x_new[in_range], x, np.imag(y))
        return re + 1j * im

    out = np.full((x_new.size, y.shape[1]), fill, dtype=np.complex128)
    for c in range(y.shape[1]):
        re = np.full_like(x_new, fill, dtype=float)
        im = np.full_like(x_new, fill, dtype=float)
        re[in_range] = np.interp(x_new[in_range], x, np.real(y[:, c]))
        im[in_range] = np.interp(x_new[in_range], x, np.imag(y[:, c]))
        out[:, c] = re + 1j * im
    return out


def _extract_coord_um(fields_obj, axis_name: str) -> np.ndarray:
    if hasattr(fields_obj, axis_name):
        return coordinates_to_um(getattr(fields_obj, axis_name))
    if isinstance(fields_obj, dict):
        key_um = f"{axis_name}_um"
        if key_um in fields_obj:
            return np.asarray(fields_obj[key_um], dtype=float).reshape(-1)
        if axis_name in fields_obj:
            return coordinates_to_um(fields_obj[axis_name])
    return np.asarray([], dtype=float)


def _collapse_except_axes(arr: np.ndarray, keep_axes: Sequence[int]) -> np.ndarray:
    out = np.asarray(arr)
    keep = {int(ax) % out.ndim for ax in keep_axes}
    for ax in reversed(range(out.ndim)):
        if ax in keep:
            continue
        out = np.take(out, indices=out.shape[ax] // 2, axis=ax)
        keep = {k - 1 if k > ax else k for k in keep}
    return np.asarray(out)


def extract_xy_plane_at_wavelength(
    fields_obj,
    *,
    component: str = "Ey",
    wavelength_um: float | None = None,
    wavelength_index: int | None = None,
    selection: str = "nearest",
) -> dict[str, Any]:
    """
    Extract a 2D x-y complex field map at one wavelength and return it in (Ny, Nx) order.
    """
    selected = select_field_component_at_wavelength(
        fields_obj,
        component=component,
        wavelength_um=wavelength_um,
        wavelength_index=wavelength_index,
        selection=selection,
        squeeze=False,
    )
    field = np.asarray(selected["field"])
    x_um = _extract_coord_um(fields_obj, "x")
    y_um = _extract_coord_um(fields_obj, "y")
    if x_um.size == 0 or y_um.size == 0:
        raise ValueError(
            f"x/y coordinates are required for XY plane extraction, got x={x_um.size}, y={y_um.size}"
        )
    if field.ndim == 0:
        raise ValueError(f"Selected field is scalar; expected at least 2D, got shape={field.shape}")

    x_axes = [idx for idx, size in enumerate(field.shape) if size == x_um.size]
    y_axes = [idx for idx, size in enumerate(field.shape) if size == y_um.size]

    plane = None
    for x_axis in x_axes:
        for y_axis in y_axes:
            if x_axis == y_axis:
                continue
            candidate = _collapse_except_axes(field, (x_axis, y_axis))
            if candidate.ndim != 2:
                candidate = np.squeeze(candidate)
            if candidate.ndim != 2:
                continue
            if candidate.shape == (x_um.size, y_um.size):
                plane = candidate.T
                break
            if candidate.shape == (y_um.size, x_um.size):
                plane = candidate
                break
        if plane is not None:
            break

    if plane is None:
        squeezed = np.squeeze(field)
        if squeezed.ndim != 2:
            raise ValueError(
                f"Cannot reshape selected field to XY plane: shape={field.shape}, squeezed={squeezed.shape}, "
                f"len(x)={x_um.size}, len(y)={y_um.size}"
            )
        if squeezed.shape == (x_um.size, y_um.size):
            plane = squeezed.T
        elif squeezed.shape == (y_um.size, x_um.size):
            plane = squeezed
        else:
            raise ValueError(
                f"Selected field shape {squeezed.shape} does not match x/y sizes {(x_um.size, y_um.size)}"
            )

    x_order = np.argsort(x_um)
    y_order = np.argsort(y_um)
    plane = np.asarray(plane)[y_order, :][:, x_order]

    return {
        "plane_yx": plane,
        "x_um": np.asarray(x_um, dtype=float).reshape(-1)[x_order],
        "y_um": np.asarray(y_um, dtype=float).reshape(-1)[y_order],
        "selected_wavelength_um": float(selected["selected_wavelength_um"]),
        "selected_wavelength_index": int(selected["selected_wavelength_index"]),
    }


def compute_diffraction_xy_map(
    Ey_1d,
    *,
    span_um: float,
    propagation_distance_um: float | None = None,
    x_positions_um=None,
    wavelength_um: float,
    n_slab: float,
    pad_factor: int = 4,
    mode: str = "full",
    ky_cut_ratio: float = 1.0,
    coarse_factor: float = 1.0,
    smooth_rolloff: float = 0.0,
) -> dict[str, Any]:
    """
    Build an h-y complex field map from a 1D input field using the same angular-spectrum propagator.
    Returns the field map in (Ny, Nx) order for plotting.
    """
    ey_1d = _to_numpy_complex(Ey_1d)
    if ey_1d.ndim != 1:
        ey_1d = ey_1d.reshape(-1)

    if x_positions_um is None:
        if propagation_distance_um is None:
            raise ValueError("Either x_positions_um or propagation_distance_um must be provided")
        x_positions_um = np.linspace(0.0, float(propagation_distance_um), 201)
    x_positions_um = np.asarray(x_positions_um, dtype=float).reshape(-1)
    if x_positions_um.size == 0:
        raise ValueError("x_positions_um must contain at least one sample")

    x_rel_um = x_positions_um - float(np.min(x_positions_um))
    pad_factor = int(max(1, pad_factor))

    N0 = int(ey_1d.size)
    Np = int(max(N0, pad_factor * N0))
    pad_start = (Np - N0) // 2
    dy_um = float(span_um) / N0

    u0 = np.zeros(Np, dtype=np.complex128)
    u0[pad_start : pad_start + N0] = ey_1d
    spectrum = np.fft.fft(np.fft.ifftshift(u0))

    ky = 2 * np.pi * np.fft.fftfreq(Np, d=dy_um)
    ky_abs = np.abs(ky)
    k = 2 * np.pi * float(n_slab) / float(wavelength_um)
    ky_nyq = np.pi / dy_um
    ky_nyq_eff = ky_nyq / float(coarse_factor) if mode == "coarse" else ky_nyq
    ky_cut = min(float(ky_cut_ratio) * k, ky_nyq_eff)

    if smooth_rolloff and smooth_rolloff > 0:
        ratio = float(smooth_rolloff)
        k1 = ky_cut * (1.0 - ratio)
        k2 = ky_cut
        filt = np.ones_like(ky_abs, dtype=np.float64)
        filt[ky_abs >= k2] = 0.0
        mid = (ky_abs >= k1) & (ky_abs < k2)
        filt[mid] = 0.5 * (1.0 + np.cos(np.pi * (ky_abs[mid] - k1) / (k2 - k1)))
    else:
        filt = (ky_abs <= ky_cut).astype(np.float64)

    slices = []
    for distance_um in x_rel_um:
        if mode in ("bandlimit", "coarse"):
            kx = np.sqrt(np.maximum(k * k - ky * ky, 0.0))
            H = np.exp(1j * kx * distance_um) * filt
        elif mode == "full":
            kx = np.sqrt((k * k - ky * ky).astype(np.complex128))
            H = np.exp(1j * kx * distance_um)
        else:
            raise ValueError(f"Unknown mode={mode}")

        propagated = np.fft.fftshift(np.fft.ifft(spectrum * H))
        slices.append(propagated[pad_start : pad_start + N0])

    field_x_y = np.stack(slices, axis=0)
    y_um = np.linspace(-0.5 * float(span_um), 0.5 * float(span_um), N0)
    return {
        "plane_yx": np.asarray(field_x_y, dtype=np.complex128).T,
        "x_um": x_rel_um,
        "y_um": y_um,
    }


def save_side_by_side_field_plot(
    *,
    fdtd_plane_yx,
    fdtd_x_um,
    fdtd_y_um,
    diffraction_plane_yx,
    diffraction_h_um,
    diffraction_y_um,
    output_path: str | Path,
    quantity: str = "abs",
    title_prefix: str = "",
) -> Path:
    """
    Save a stitched two-panel plot: left is the z=0 FDTD x-y plane, right is the analytical h-y map.
    """
    try:
        from PIL import Image, ImageDraw, ImageOps
    except Exception as exc:  # pragma: no cover - depends on local env
        raise ImportError("Pillow is required to save hybrid field plots") from exc

    fdtd_plane_yx = np.asarray(fdtd_plane_yx)
    diffraction_plane_yx = np.asarray(diffraction_plane_yx)
    fdtd_x_um = np.asarray(fdtd_x_um, dtype=float).reshape(-1)
    fdtd_y_um = np.asarray(fdtd_y_um, dtype=float).reshape(-1)
    diffraction_h_um = np.asarray(diffraction_h_um, dtype=float).reshape(-1)
    diffraction_y_um = np.asarray(diffraction_y_um, dtype=float).reshape(-1)

    def _phase_to_rgb(field: np.ndarray) -> np.ndarray:
        phase = np.angle(field)
        hue = (phase + np.pi) / (2 * np.pi)
        s = np.ones_like(hue)
        v = np.ones_like(hue)

        i = np.floor(hue * 6.0).astype(int)
        f = hue * 6.0 - i
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        i = i % 6

        rgb = np.zeros(field.shape + (3,), dtype=np.float64)
        masks = [i == idx for idx in range(6)]
        rgb[masks[0]] = np.stack([v, t, p], axis=-1)[masks[0]]
        rgb[masks[1]] = np.stack([q, v, p], axis=-1)[masks[1]]
        rgb[masks[2]] = np.stack([p, v, t], axis=-1)[masks[2]]
        rgb[masks[3]] = np.stack([p, q, v], axis=-1)[masks[3]]
        rgb[masks[4]] = np.stack([t, p, v], axis=-1)[masks[4]]
        rgb[masks[5]] = np.stack([v, p, q], axis=-1)[masks[5]]
        return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)

    def _real_to_rgb(values: np.ndarray, vmax: float) -> np.ndarray:
        scaled = 0.5 + 0.5 * values / max(vmax, 1e-12)
        scaled = np.clip(scaled, 0.0, 1.0)
        rgb = np.zeros(values.shape + (3,), dtype=np.uint8)
        rgb[..., 0] = np.clip(255.0 * scaled, 0.0, 255.0).astype(np.uint8)
        rgb[..., 2] = np.clip(255.0 * (1.0 - scaled), 0.0, 255.0).astype(np.uint8)
        rgb[..., 1] = np.clip(255.0 * (1.0 - np.abs(2.0 * scaled - 1.0)), 0.0, 255.0).astype(
            np.uint8
        )
        return rgb

    def _make_panel(
        field: np.ndarray,
        *,
        value_label: str,
        x_label: str,
        y_label: str,
        quantity: str,
        vmin: float,
        vmax: float,
    ):
        panel_w = 640
        panel_h = 420
        text_h = 90
        panel = Image.new("RGB", (panel_w, panel_h + text_h), color="white")

        if quantity == "phase":
            rgb = _phase_to_rgb(field)
            image = Image.fromarray(np.flipud(rgb), mode="RGB")
        elif quantity == "real":
            rgb = _real_to_rgb(np.real(field), vmax=max(abs(vmin), abs(vmax)))
            image = Image.fromarray(np.flipud(rgb), mode="RGB")
        else:
            norm = np.asarray(np.abs(field), dtype=float)
            norm = np.clip((norm - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)
            image = Image.fromarray(np.flipud((norm * 255.0).astype(np.uint8)), mode="L")
            image = ImageOps.colorize(image, black="#0b1020", white="#ffd166")

        image = image.resize((panel_w, panel_h), resample=Image.BICUBIC)
        panel.paste(image, (0, 0))

        draw = ImageDraw.Draw(panel)
        draw.text((12, panel_h + 8), value_label, fill="black")
        draw.text((12, panel_h + 30), x_label, fill="black")
        draw.text((12, panel_h + 52), y_label, fill="black")
        return panel

    if quantity == "abs":
        vmax = float(
            max(np.max(np.abs(fdtd_plane_yx)), np.max(np.abs(diffraction_plane_yx)), 1e-12)
        )
        vmin = 0.0
        value_label = "|Ey|"
    elif quantity == "real":
        vmax = float(
            max(
                np.max(np.abs(np.real(fdtd_plane_yx))),
                np.max(np.abs(np.real(diffraction_plane_yx))),
                1e-12,
            )
        )
        vmin = -vmax
        value_label = "Re(Ey)"
    elif quantity == "phase":
        vmin = -np.pi
        vmax = np.pi
        value_label = "phase(Ey) [rad]"
    else:
        raise ValueError(f"Unsupported quantity={quantity}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    left_panel = _make_panel(
        fdtd_plane_yx,
        value_label=f"{title_prefix}3D FDTD z=0 x-y | {value_label}",
        x_label=f"x: [{fdtd_x_um[0]:.3f}, {fdtd_x_um[-1]:.3f}] um",
        y_label=f"y: [{fdtd_y_um[0]:.3f}, {fdtd_y_um[-1]:.3f}] um",
        quantity=quantity,
        vmin=vmin,
        vmax=vmax,
    )
    right_panel = _make_panel(
        diffraction_plane_yx,
        value_label=f"{title_prefix}Diffraction h-y | {value_label}",
        x_label=f"h: [{diffraction_h_um[0]:.3f}, {diffraction_h_um[-1]:.3f}] um",
        y_label=f"y: [{diffraction_y_um[0]:.3f}, {diffraction_y_um[-1]:.3f}] um",
        quantity=quantity,
        vmin=vmin,
        vmax=vmax,
    )

    canvas = Image.new(
        "RGB",
        (left_panel.width + right_panel.width + 24, max(left_panel.height, right_panel.height)),
        color="white",
    )
    canvas.paste(left_panel, (0, 0))
    canvas.paste(right_panel, (left_panel.width + 24, 0))
    canvas.save(output_path)
    return output_path


def generate_hybrid_xy_visualization(
    *,
    fdtd,
    Ey_1d,
    layer_index: int,
    span_um: float,
    wavelength_um: float,
    n_slab: float,
    output_dir: str | Path,
    xy_monitor_name: str = "hybrid_xy_monitor",
    field_component: str = "Ey",
    quantity: str = "abs",
    pad_factor: int = 4,
    propagation_mode: str = "full",
    ky_cut_ratio: float = 1.0,
    coarse_factor: float = 1.0,
    smooth_rolloff: float = 0.0,
) -> dict[str, Any]:
    xy_fields = get_fields(
        fdtd,
        monitor_name=xy_monitor_name,
        field_result_name=f"{xy_monitor_name}_fields",
        get_eps=False,
        get_D=False,
        get_H=False,
        nointerpolation=False,
        unfold_symmetry=False,
    )
    fdtd_xy = extract_xy_plane_at_wavelength(
        xy_fields,
        component=field_component,
        wavelength_um=float(wavelength_um),
        selection="nearest",
    )
    diffraction_xy = compute_diffraction_xy_map(
        Ey_1d,
        span_um=float(span_um),
        x_positions_um=fdtd_xy["x_um"],
        wavelength_um=float(wavelength_um),
        n_slab=float(n_slab),
        pad_factor=int(max(1, pad_factor)),
        mode=propagation_mode,
        ky_cut_ratio=float(ky_cut_ratio),
        coarse_factor=float(coarse_factor),
        smooth_rolloff=float(smooth_rolloff),
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"layer{int(layer_index)}_{field_component.lower()}_xy"
    plot_path = output_dir / f"{stem}.png"
    data_path = output_dir / f"{stem}.npz"

    save_side_by_side_field_plot(
        fdtd_plane_yx=fdtd_xy["plane_yx"],
        fdtd_x_um=fdtd_xy["x_um"],
        fdtd_y_um=fdtd_xy["y_um"],
        diffraction_plane_yx=diffraction_xy["plane_yx"],
        diffraction_h_um=diffraction_xy["x_um"],
        diffraction_y_um=diffraction_xy["y_um"],
        output_path=plot_path,
        quantity=quantity,
        title_prefix=f"Layer {int(layer_index)} ",
    )
    np.savez_compressed(
        data_path,
        fdtd_plane_yx=np.asarray(fdtd_xy["plane_yx"], dtype=np.complex128),
        fdtd_x_um=np.asarray(fdtd_xy["x_um"], dtype=float),
        fdtd_y_um=np.asarray(fdtd_xy["y_um"], dtype=float),
        diffraction_plane_yx=np.asarray(diffraction_xy["plane_yx"], dtype=np.complex128),
        diffraction_h_um=np.asarray(diffraction_xy["x_um"], dtype=float),
        diffraction_y_um=np.asarray(diffraction_xy["y_um"], dtype=float),
        selected_wavelength_um=float(fdtd_xy["selected_wavelength_um"]),
    )
    return {
        "plot_path": plot_path,
        "data_path": data_path,
        "monitor_name": xy_monitor_name,
        "field_component": field_component,
        "selected_wavelength_um": float(fdtd_xy["selected_wavelength_um"]),
    }


# ---------------------------------------------------------------------------
# Hybrid 3D FDTD setup and run
# ---------------------------------------------------------------------------.


def _ensure_lumapi():
    if lumapi is None:
        raise ImportError(
            "lumapi is required for hybrid FDTD; install Lumerical Python API and ensure DLLs are on PATH"
        ) from _lumapi_import_error


def import_gds_into_fdtd(
    fdtd: Any,
    *,
    gds_path: str | Path,
    cell_name: str = "TOP",
    layer_name: str = "1:0",
    material: str = "etch",
    z_span_um: float = 0.22,
    x_offset_um: float = 0.0,
    y_offset_um: float = 0.0,
    z_offset_um: float = 0.0,
    index_value: float | None = None,
) -> None:
    path = Path(gds_path).resolve().as_posix()
    fdtd.eval("redrawoff;")
    fdtd.eval(f"gdsimport('{path}','{cell_name}','{layer_name}');")
    fdtd.eval(f"select('GDS_LAYER_{layer_name}');")
    fdtd.eval(f"set('z span',{float(z_span_um)}e-6);")
    if index_value is not None:
        fdtd.eval(f"set('index',{float(index_value)});")
    fdtd.eval(f"set('material','{material}');")
    fdtd.eval(f"set('x',{float(x_offset_um)}e-6);")
    fdtd.eval(f"set('y',{float(y_offset_um)}e-6);")
    fdtd.eval(f"set('z',{float(z_offset_um)}e-6);")
    fdtd.eval("redrawon;")
    fdtd.eval("redraw;")


def init_hybrid_fdtd(
    *,
    working_dir: str | Path,
    num_layers: int,
    total_width_um: float,
    z_span_um: float,
    background_material: str = "SiO2 (Glass) - Palik",
    sub_material: str = "Si (Silicon) - Palik",
    before_um: float = 0.1,
    after_um: float = 3,
    mesh_accuracy: int = 6,
    hide_fdtd_cad: bool = False,
) -> list:
    """
    Build simple hybrid FDTD projects (one per diffraction layer) with:
      - 3D FDTD region, slab, imported source named 'import_src'
      - linear-Y power monitor named 'hybrid_monitor'
      - z=0 x-y monitor named 'hybrid_xy_monitor'
    Returns list of lumapi.FDTD handles.
    """
    _ensure_lumapi()

    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    x_span_um = before_um + after_um + 1
    y_span_um = total_width_um
    z_span_um = float(z_span_um)

    handles = []
    num_cells = max(num_layers - 2, 1)
    for idx in range(num_cells):
        fdtd = lumapi.FDTD(hide=hide_fdtd_cad)
        fdtd.cd(str(working_dir))

        script_lines = [
            "switchtolayout;",
            "selectall; delete;",
            "redrawoff;",
            "addfdtd;",
            "set('dimension','3D');",
            "set('background material','<Object defined dielectric>');",
            "set('index',1);",
            f"set('mesh accuracy',{mesh_accuracy});",
            f"set('x span',{x_span_um}e-6);",
            f"set('y span',{y_span_um}e-6+1e-6);",
            f"set('z span',5*{z_span_um}e-6);",
            "setdevice('GPU');",
            "set('z min bc','PML');",
            "set('z max bc','PML');",
            f"set('x',0.5*{after_um}e-6); set('y',0); set('z',0);",
            "addrect;",
            "set('name','slab');",
            f"set('material','{sub_material}');",
            f"set('x span',5*{x_span_um}e-6);",
            f"set('y span',5*{y_span_um}e-6);",
            f"set('z span',{z_span_um}e-6);",
            f"set('x',0.5*{after_um}e-6); set('y',0); set('z',0);",
            "addrect;",
            "set('name','slab2');",
            f"set('material','{background_material}');",
            f"set('x span',5*{x_span_um}e-6);",
            f"set('y span',5*{y_span_um}e-6);",
            f"set('z span',5*{z_span_um}e-6);",
            f"set('x',0.5*{after_um}e-6); set('y',0); set('z',-3*{z_span_um}e-6);",
            "addimportedsource;",
            "set('name','import_src');",
            "set('injection axis','x');",
            "set('direction','Forward');",
            f"set('x',-{before_um}e-6);",
            "set('y',0); set('z',0);",
            f"set('y span',{total_width_um}e-6);",
            "set('z span',3e-6);",
            "addpower;",
            "set('name','hybrid_monitor');",
            "set('monitor type','Linear Y');",
            f"set('x',{after_um}e-6);",
            "set('y',0); set('z',0);",
            f"set('y span',{total_width_um}e-6);",
            "addpower;",
            "set('name','hybrid_xy_monitor');",
            "set('monitor type','2D Z-normal');",
            f"set('x',{0.5 * (after_um - before_um)}e-6);",
            f"set('x span',{before_um + after_um}e-6);",
            "set('y',0);",
            f"set('y span',{total_width_um}e-6);",
            "set('z',0);",
            "set('override global monitor settings',1);",
            "set('use source limits',1);",
            "set('frequency points',1);",
            "redrawon;",
        ]
        fdtd.eval("\n".join(script_lines))
        project_path = working_dir / f"hybrid_layer_{idx}.fsp"
        fdtd.save(str(project_path))
        handles.append(fdtd)
    return handles


def _generate_tapers_lsf(
    lsf_path: Path,
    *,
    num_tapers: int,
    mesh_accuracy=3,
    lane_w_um: float,
    taper_L_um: float,
    wavelength_um: float,
    final_field_mat: str | Path | None,
) -> Path:
    """
    鐢熸垚涓€涓彲鐩存帴 runscript 鐨?LSF锛?      import_src -> 姣忚矾锛氬乏渚у鐩存尝瀵?lane_w) -> poly taper(lane_w -> 0.5) -> 鍙充晶绐勭洿娉㈠(0.5)
               -> 姣忚矾锛歰ut_det_i(Linear Y) + out_mode_i(mode expansion)
    """
    lsf_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- fixed geometry (um) ----
    left_wg_L_um = 2.0
    right_wg_L_um = 2.0
    before_um = 2
    after_um = 0

    taper_w_in_um = float(lane_w_um)
    taper_w_out_um = 0.5

    z_span_um = 0.22
    mesh_accuracy = 2

    core_material = "Si (Silicon) - Palik"
    sub_material = "SiO2 (Glass) - Palik"

    total_w_um = float(num_tapers) * float(lane_w_um)

    # x positions (um)
    x0 = 0.0
    x1 = x0 + left_wg_L_um
    x2 = x1 + float(taper_L_um)
    x3 = x2 + right_wg_L_um

    # region coverage
    x_left_um = -before_um
    x_right_um = x3
    x_span_um = x_right_um - x_left_um
    x_center_um = 0.5 * (x_left_um + x_right_um)

    y_span_um = total_w_um + 4.0
    lam_m = float(wavelength_um) * 1e-6

    # monitor span around 0.5um wg
    mon_span_um = max(1.5, 3.0 * taper_w_out_um)

    # optional import field
    import_field_lines: list[str] = []
    if final_field_mat is not None:
        p = Path(final_field_mat).expanduser().resolve().as_posix()
        # Optional import: assume MAT stores a Lumerical dataset.
        import_field_lines = [
            f"__fields = load('{p}');",
            "importdataset('import_src', __fields);",
            "",
        ]

    lines: list[str] = []
    lines += [
        "# auto-generated by Python",
        "newproject;",
        "switchtolayout;",
        "deleteall;",
        "redrawoff;",
        "",
        f"num_tapers = {int(num_tapers)};",
        f"lane_w = {lane_w_um}e-6;",
        f"taper_L = {taper_L_um}e-6;",
        f"w_in = {taper_w_in_um}e-6;",
        f"w_out = {taper_w_out_um}e-6;",
        f"L_left = {left_wg_L_um}e-6;",
        f"L_right = {right_wg_L_um}e-6;",
        f"before = {before_um}e-6;",
        f"after = {after_um}e-6;",
        f"z_span = {z_span_um}e-6;",
        f"mesh_acc = {mesh_accuracy};",
        f"lam = {lam_m};",
        f"total_w = {total_w_um}e-6;",
        "",
        f"x0 = {x0}e-6;",
        f"x1 = {x1}e-6;",
        f"x2 = {x2}e-6;",
        f"x3 = {x3}e-6;",
        "",
        f"x_left = {x_left_um}e-6;",
        f"x_right = {x_right_um}e-6;",
        f"x_span = {x_span_um}e-6;",
        f"x_center = {x_center_um}e-6;",
        f"y_span = {y_span_um}e-6;",
        "",
        f"core_mat = '{core_material}';",
        f"sub_mat = '{sub_material}';",
        "",
        "# ---- FDTD region ----",
        "addfdtd;",
        "set('dimension','2D');",
        "set('background material','<Object defined dielectric>');",
        "set('index',1);",
        "set('mesh accuracy', mesh_acc);",
        "set('x', x_center-2e-6);",
        "set('x span', x_span-2e-6);",
        "set('y', 0);",
        "set('y span', y_span);",
        "set('z span', 5*z_span);",
        "setdevice('CPU');",
        "",
        "# ---- substrate ----",
        "addrect;",
        "set('name','sub');",
        "set('material', sub_mat);",
        "set('x', x_center);",
        "set('x span', 1.2*x_span);",
        "set('y', 0);",
        "set('y span', 1.2*y_span);",
        "set('z', -0.5*(1+0.22)*1e-6);",
        "set('z span', 1e-6);",
        "",
        "addrect;",
        "set('name','sub2');",
        "set('material', core_mat);",
        "set('x', -4e-6);",
        "set('x span', 4e-6);",
        "set('y', 0);",
        "set('y span', 1.2*y_span);",
        "set('z', 0);",
        "set('z span', 0.22e-6);",
        "",
        "# ---- imported source ----",
        "addimportedsource;",
        "set('name','import_src');",
        "set('injection axis','x');",
        "set('direction','Forward');",
        "set('x', -0.1e-6-2e-6);",
        "set('y', 0);",
        "set('z', 0);",
        "set('y span', total_w);",
        "set('z span', 3e-6);",
        "set('wavelength start', lam);",
        "set('wavelength stop', lam);",
        "",
    ]
    lines += import_field_lines
    lines += [
        "# ---- groups ----",
        "addgroup; set('name','tapers');",
        "addgroup; set('name','wg_right');",
        "addgroup; set('name','monitors');",
        "",
        f"mon_span = {mon_span_um}e-6;",
        "",
        "# ---- lanes ----",
        "for(i=0; i<num_tapers; i=i+1){",
        "  yc = -0.5*total_w + (i+0.5)*lane_w;",
        "",
        "  # poly taper (lane_w -> 0.5um)",
        "  addpoly;",
        "  set('name','taper_'+num2str(i));",
        "  set('material',core_mat);",
        "  set('z',0); set('z span', z_span);",
        "  pts=[ x1, yc + 0.5*w_in;",
        "        x2, yc + 0.5*w_out;",
        "        x2, yc - 0.5*w_out;",
        "        x1, yc - 0.5*w_in ];",
        "  set('vertices', pts);",
        "  addtogroup('tapers');",
        "",
        "  # right narrow straight (0.5um)",
        "  addrect;",
        "  set('name','wgR_'+num2str(i));",
        "  set('material',core_mat);",
        "  set('z',0); set('z span', z_span);",
        "  set('x', x2 + 0.5*L_right+0.5e-6);",
        "  set('x span', L_right+5e-6);",
        "  set('y', yc);",
        "  set('y span', w_out);",
        "  addtogroup('wg_right');",
        "",
        "  # detector monitor (Linear Y)",
        "  det = 'out_det_'+num2str(i);",
        "  adddftmonitor;",
        "  set('name', det);",
        "  set('monitor type','Linear Y');",
        "  set('x', x2 + 0.5*L_right);",
        "  set('y', yc);",
        "  set('y span', mon_span);",
        "  set('z', 0);",
        "  addtogroup('monitors');",
        "",
        "  # mode expansion monitor",
        "  md = 'out_mode_'+num2str(i);",
        "  addmodeexpansion;",
        "  set('name', md);",
        "  set('monitor type','2D X-normal');",
        "  setexpansion(md, 'monitors::'+det);",
        "  set('x', x2 + 0.5*L_right);",
        "  set('y', yc);",
        "  set('y span', mon_span);",
        "  set('z', 0);",
        "  set('override global monitor settings',1);",
        "  set('use source limits',1);",
        "  set('frequency points',1);",
        "  set('wavelength center', lam);",
        "  set('mode selection','fundamental mode');",
        "  addtogroup('monitors');",
        "}",
        " select('tapers');",
        " set('x',-4e-6);",
        " set('y',6e-6);",
        " select('wg_right');",
        " set('x',-4e-6);",
        " set('y',6e-6);",
        " select('monitors');",
        " set('x',-4e-6);",
        " set('y',6e-6);",
        "",
        "redrawon;",
        "",
    ]

    lsf_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lsf_path


def build_tapers(
    working_dir: str = "runs/three_parallel_tapers",
    *,
    num_tapers: int = 50,
    lane_w_um: float = 12.0,
    taper_L_um: float = 100.0,
    final_field_mat: str | Path | None = None,
    wavelength_um: float = 1.55,
    hide: bool = False,
) -> lumapi.FDTD:
    """
    鍏堢敓鎴?LSF 鏂囦欢锛屽啀鐢?eval(runscript) 鎵ц锛屼究浜庝綘鐩存帴鎵撳紑 .lsf 鎺掓煡銆?"""
    wd = Path(working_dir)
    wd.mkdir(parents=True, exist_ok=True)

    lsf_path = wd / f"tapers_poly_{int(num_tapers)}.lsf"
    _generate_tapers_lsf(
        lsf_path,
        num_tapers=num_tapers,
        lane_w_um=lane_w_um,
        taper_L_um=taper_L_um,
        wavelength_um=wavelength_um,
        final_field_mat=final_field_mat,
    )

    fdtd = lumapi.FDTD(hide=hide)
    fdtd.cd(str(wd))

    # 鍏抽敭锛氭墽琛屽垰鐢熸垚鐨?LSF锛堢敤鐩稿璺緞锛岄伩鍏?Windows 鍙嶆枩鏉犺浆涔夐棶棰橈級
    fdtd.eval(f"tapers_poly_{int(num_tapers)};")
    project_path = working_dir / f"apers_poly_{int(num_tapers)}.fsp"
    fdtd.save(str(project_path))

    return fdtd


def _to_numpy_complex(a) -> np.ndarray:
    if torch is not None and isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=np.complex128).squeeze()


def run_hybrid_fdtd(
    Ey_1d,
    *,
    mode_path: str | Path,
    fdtd_handles: Sequence,
    layer_index: int = 0,
    hybrid_zlink: float = 0.0,
    monitor_name: str = "hybrid_monitor",
    workdir: str | Path | None = None,
    wavelength_um: float = 1.55,
    span_um: float | None = None,
    n_slab: float | None = None,
    save_xy_plot: bool = False,
    xy_monitor_name: str = "hybrid_xy_monitor",
    plot_quantity: str = "abs",
    plot_pad_factor: int = 4,
    propagation_mode: str = "full",
    ky_cut_ratio: float = 1.0,
    coarse_factor: float = 1.0,
    smooth_rolloff: float = 0.0,
) -> dict:
    """
    Bridge a 1D Ey field into 3D FDTD (import source) and read back resampled Ey.
    Returns dict with ey_resampled, y_resampled, raw_y, raw_E, mat_path.
    """
    _ensure_lumapi()

    ey_1d = _to_numpy_complex(Ey_1d)
    if ey_1d.ndim > 1:
        ey_1d = ey_1d.reshape(-1)
    N_target = ey_1d.shape[0]

    mode = load_lumerical_modeE(Path(mode_path))
    z_m = np.asarray(mode["z"]).reshape(-1)
    Ex_z = np.asarray(mode["Ex"]).reshape(-1)
    Ey_z = np.asarray(mode["Ey"]).reshape(-1)
    Ez_z = np.asarray(mode["Ez"]).reshape(-1)

    if span_um is None:
        span_um = float(
            N_target  # caller should map physical span to sampling; fallback to unit grid
        )
    y_um = np.linspace(-0.5 * float(span_um), 0.5 * float(span_um), N_target)
    y_m = y_um * 1e-6

    Ex_yz, Ey_yz, Ez_yz = build_yz_vector_zlink0(
        ey_1d, y_m, Ex_z, Ey_z, Ez_z, z_m, zlink=hybrid_zlink
    )

    workdir = Path(workdir or _project_runs_dir("hybrid_fdtd"))
    workdir.mkdir(parents=True, exist_ok=True)

    mat_path = workdir / f"layer{layer_index}_Ey_yz.mat"
    save_mat_v73_for_lumerical_h5py(
        mat_path,
        x0=0.0,
        y=y_m,
        z=z_m,
        Ex_yz=Ex_yz,
        Ey_yz=Ey_yz,
        Ez_yz=Ez_yz,
        f0=mode.get("f0", None) or float(wavelength_um),
    )

    fdtd = fdtd_handles[layer_index]
    lambda_m = float(wavelength_um) * 1e-6
    script = f"""
matlabload('{mat_path.as_posix()}');
if (length(x)==0) {{ x=0; }}
xv = x; yv = y; zv = z;
Ex_i = permute(Ex, [3,2,1]);
Ey_i = permute(Ey, [3,2,1]);
Ez_i = permute(Ez, [3,2,1]);
EM = rectilineardataset('EM fields', xv, yv, zv);
lambda0 = {lambda_m};
f0 = c/lambda0;
EM.addparameter('lambda', lambda0, 'f', f0);
EM.addattribute('E', Ex_i, Ey_i, Ez_i);
switchtolayout;
select('import_src');
importdataset(EM);
run;
"""
    fdtd.eval(script)

    fields = fdtd.getresult(monitor_name, "fields")
    if fields is None or "E" not in fields:
        raise RuntimeError(f"Failed to read fields from monitor '{monitor_name}'")

    e = np.squeeze(fields["E"])
    if e.ndim == 1:
        ey_out = e
    elif e.ndim == 2 and e.shape[1] == 3:
        ey_out = e[:, 1]
    elif e.ndim == 3 and e.shape[-1] == 3:
        ey_out = e[e.shape[0] // 2, :, 1]
    else:
        raise ValueError(f"Unexpected E shape from hybrid monitor: {e.shape}")

    y_old = fields.get("y", None)
    if y_old is None or np.asarray(y_old).size != ey_out.shape[0]:
        y_old = np.linspace(0.0, 1.0, ey_out.shape[0])
    else:
        y_old = np.asarray(y_old).reshape(-1)

    y_target = np.linspace(y_old.min(), y_old.max(), N_target)
    ey_resampled = interp_complex(ey_out, y_old, y_target, fill=0.0)

    result = {
        "ey_resampled": ey_resampled,
        "y_resampled": y_target,
        "raw_y": y_old,
        "raw_E": e,
        "mat_path": mat_path,
    }
    if save_xy_plot:
        if n_slab is None:
            raise ValueError("n_slab must be provided when save_xy_plot=True")
        plot_dir = workdir / "xy_field_plots"
        result["xy_visualization"] = generate_hybrid_xy_visualization(
            fdtd=fdtd,
            Ey_1d=ey_1d,
            layer_index=int(layer_index),
            span_um=float(span_um),
            wavelength_um=float(wavelength_um),
            n_slab=float(n_slab),
            output_dir=plot_dir,
            xy_monitor_name=xy_monitor_name,
            quantity=plot_quantity,
            pad_factor=int(max(1, plot_pad_factor)),
            propagation_mode=propagation_mode,
            ky_cut_ratio=float(ky_cut_ratio),
            coarse_factor=float(coarse_factor),
            smooth_rolloff=float(smooth_rolloff),
        )
    return result


def _load_complex_field(path: str | Path) -> np.ndarray:
    field_path = Path(path)
    suffix = field_path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(
            np.load(field_path, allow_pickle=False), dtype=np.complex128
        ).squeeze()
    if suffix == ".mat":
        with h5py.File(field_path, "r") as handle:
            if "Ey" in handle:
                raw = handle["Ey"][()]
            else:
                keys = [key for key in handle.keys() if not key.startswith("#")]
                if not keys:
                    raise KeyError(f"No readable dataset found in {field_path}")
                raw = handle[keys[0]][()]
        if (
            getattr(raw, "dtype", None) is not None
            and raw.dtype.fields
            and {"real", "imag"}.issubset(raw.dtype.fields)
        ):
            raw = raw["real"] + 1j * raw["imag"]
        return np.asarray(raw, dtype=np.complex128).squeeze()
    raise ValueError(f"Unsupported field file suffix for {field_path}")


def run_hybrid_pipeline(
    hybrid_cfg: Mapping[str, Any],
    *,
    run_dir: str | Path,
) -> dict[str, Any]:
    if torch is None:
        raise ImportError("torch is required for hybrid FDTD runs")

    field_paths = hybrid_cfg.get("input_fields") or hybrid_cfg.get("field_paths")
    if not field_paths:
        raise KeyError("hybrid config requires input_fields")

    working_dir = Path(
        hybrid_cfg.get("working_dir") or Path(run_dir) / "hybrid_projects"
    )
    working_dir.mkdir(parents=True, exist_ok=True)

    handles = init_hybrid_fdtd(
        working_dir=working_dir,
        num_layers=int(hybrid_cfg["num_layers"]),
        total_width_um=float(hybrid_cfg["total_width_um"]),
        z_span_um=float(hybrid_cfg["z_span_um"]),
        background_material=str(
            hybrid_cfg.get("background_material", "SiO2 (Glass) - Palik")
        ),
        sub_material=str(hybrid_cfg.get("sub_material", "Si (Silicon) - Palik")),
        before_um=float(hybrid_cfg.get("before_um", 0.1)),
        after_um=float(hybrid_cfg.get("after_um", 3.0)),
        mesh_accuracy=int(hybrid_cfg.get("mesh_accuracy", 6)),
        hide_fdtd_cad=bool(hybrid_cfg.get("hide_fdtd_cad", False)),
    )

    for job in hybrid_cfg.get("gds_imports", []):
        if not isinstance(job, Mapping):
            raise TypeError("hybrid.gds_imports entries must be mappings")
        handle_idx = int(job.get("handle_index", 0))
        import_gds_into_fdtd(
            handles[handle_idx],
            gds_path=job["gds_path"],
            cell_name=str(job.get("cell_name", "TOP")),
            layer_name=str(job.get("layer_name", "1:0")),
            material=str(job.get("material", "etch")),
            z_span_um=float(job.get("z_span_um", 0.22)),
            x_offset_um=float(job.get("x_offset_um", 0.0)),
            y_offset_um=float(job.get("y_offset_um", 0.0)),
            z_offset_um=float(job.get("z_offset_um", 0.0)),
            index_value=(
                None if job.get("index_value") is None else float(job["index_value"])
            ),
        )

    complex_fields = [_load_complex_field(path) for path in field_paths]
    runner = PackedHybrid3DFDTD(
        complex_data_matrices=complex_fields,
        layer_spacing_hybride=hybrid_cfg["layer_spacing_um"],
        positions_span=hybrid_cfg["positions_span_um"],
        wavelength_um=float(hybrid_cfg.get("wavelength_um", 1.55)),
        n_slab=float(hybrid_cfg["n_slab"]),
        mode_path=hybrid_cfg["mode_path"],
        fdtd_handles=handles,
        hybrid_zlink=float(hybrid_cfg.get("hybrid_zlink", 0.0)),
        hybrid_monitor_name=str(
            hybrid_cfg.get("hybrid_monitor_name", "hybrid_monitor")
        ),
        hybrid_xy_monitor_name=str(
            hybrid_cfg.get("hybrid_xy_monitor_name", "hybrid_xy_monitor")
        ),
        waveguide_maping_span=float(hybrid_cfg.get("waveguide_mapping_span_um", 5.0)),
        device=str(hybrid_cfg.get("device", "cuda")),
        save_xy_plots=bool(hybrid_cfg.get("save_xy_plots", False)),
        xy_plot_dir=hybrid_cfg.get("xy_plot_dir"),
        xy_plot_quantity=str(hybrid_cfg.get("xy_plot_quantity", "abs")),
        xy_plot_pad_factor=int(hybrid_cfg.get("xy_plot_pad_factor", 4)),
        propagation_mode=str(hybrid_cfg.get("propagation_mode", "full")),
        ky_cut_ratio=float(hybrid_cfg.get("ky_cut_ratio", 1.0)),
        coarse_factor=float(hybrid_cfg.get("coarse_factor", 1.0)),
        smooth_rolloff=float(hybrid_cfg.get("smooth_rolloff", 0.0)),
    )
    runner.setuphs_hybride(
        mode=str(hybrid_cfg.get("propagation_mode", "full")),
        ky_cut_ratio=float(hybrid_cfg.get("ky_cut_ratio", 1.0)),
        coarse_factor=float(hybrid_cfg.get("coarse_factor", 1.0)),
        smooth_rolloff=float(hybrid_cfg.get("smooth_rolloff", 0.0)),
        pad_factor=int(hybrid_cfg.get("pad_factor", 16)),
    )

    outputs = runner.forward_hybride()
    arrays = [
        np.asarray(out.detach().cpu().numpy() if hasattr(out, "detach") else out)
        for out in outputs
    ]
    output_path = Path(run_dir) / "hybrid_fields.npy"
    np.save(output_path, np.asarray(arrays))
    return {
        "status": "completed",
        "field_count": len(arrays),
        "output_file": output_path,
        "working_dir": working_dir,
        "xy_plot_files": [record["plot_path"] for record in runner.xy_plot_records],
        "xy_plot_data_files": [record["data_path"] for record in runner.xy_plot_records],
    }


__all__ = [
    "load_lumerical_modeE",
    "build_yz_vector_zlink0",
    "save_mat_v73_for_lumerical_h5py",
    "interp_complex",
    "extract_xy_plane_at_wavelength",
    "compute_diffraction_xy_map",
    "save_side_by_side_field_plot",
    "generate_hybrid_xy_visualization",
    "init_hybrid_fdtd",
    "import_gds_into_fdtd",
    "run_hybrid_fdtd",
    "run_hybrid_pipeline",
    "getportfield",
]


# ---------------------------------------------------------------------------
# Minimal end-to-end hybrid propagator (torch-based 1D -> FDTD -> 1D)
# ---------------------------------------------------------------------------.


def getportfeild(
    taper,
    u,
    *,
    positions_span_um: float = 600.0,
    wavelength_um: float = 1.55,
    workdir: str | Path | None = None,
    num_ports: int | None = None,
):
    """
    2D FDTD 鐨?import source 鎺ュ彈 1D 绾垮満锛氱敤 (Ny,1) 琛ㄧず锛堝彟涓€涓淮搴︿负 1锛夛紝
    LSF 閲屽啀 reshape 鎴?(1,Ny,1) 瀵煎叆 rectilineardataset銆?    杩斿洖 (summary_text, fields_dict).
    """
    _ensure_lumapi()

    # --- 1D Ey(y) ---
    ey_1d = _to_numpy_complex(u)
    if ey_1d.ndim > 1:
        ey_1d = ey_1d.reshape(-1)
    Ny = int(ey_1d.size)

    span_um = float(positions_span_um)
    y_m = (np.linspace(-0.5 * span_um, 0.5 * span_um, Ny) * 1e-6).astype(np.float64)
    z_m = np.array([0.0], dtype=np.float64)  # Nz=1

    # Store the 1D field as a degenerate (Ny, 1) rectilinear dataset for the import source.
    Ey_yz = ey_1d.astype(np.complex128)[:, None]
    zeros_yz = np.zeros((Ny, 1), dtype=np.complex128)

    workdir = Path(workdir) if workdir is not None else _project_runs_dir("taper_eval")
    workdir.mkdir(parents=True, exist_ok=True)
    mat_path = workdir / "taper_import.mat"

    # 浠嶆部鐢ㄤ綘鐜版湁鐨?mat 淇濆瓨鍑芥暟锛堝彉閲忓悕 Ex/Ey/Ez/y/z 闇€涓庡畠涓€鑷达級
    save_mat_v73_for_lumerical_h5py(
        mat_path,
        x0=0.0,  # 濡傛灉浣犵殑淇濆瓨鍑芥暟鍐欏嚭 x锛屽垯鍙繚鐣欙紱LSF 涓嶄緷璧栧畠
        y=y_m,
        z=z_m,
        Ex_yz=zeros_yz,
        Ey_yz=Ey_yz,
        Ez_yz=zeros_yz,
        # f0 濡傛灉浣犱繚瀛樺嚱鏁伴渶瑕佸氨鐣欑潃锛涗笉瑕佹妸瀹冨綋棰戠巼鐢?        f0=float(wavelength_um),
    )

    lambda_m = float(wavelength_um) * 1e-6

    # 淇濇寔涓庨粯璁や繚瀛樻牸寮忎竴鑷达細鐩存帴 permute 鍒?(Nx,Ny,Nz)=(1,Ny,1)
    script = f"""
matlabload('{mat_path.as_posix()}');
if (length(x)==0) {{ x=0; }}
xv = x;
yv = y;
zv = z;
Ex_i = permute(Ex, [3,2,1]);
Ey_i = permute(Ey, [3,2,1]);
Ez_i = permute(Ez, [3,2,1]);
EM = rectilineardataset('EM fields', xv, yv, zv);
lambda0 = {lambda_m};
f0 = c/lambda0;
EM.addparameter('lambda', lambda0, 'f', f0);
EM.addattribute('E', Ex_i, Ey_i, Ez_i);
switchtolayout;
select('import_src');
importdataset(EM);
run;
"""
    taper.eval(script)

    # --- collect results ---
    if num_ports is None:
        try:
            num_ports = int(np.asarray(taper.getv("num_tapers")).reshape(-1)[0])
        except Exception:
            num_ports = 50

    mode_powers: list[float | None] = []
    expansion_results: dict[int, dict] = {}
    detector_fields: dict[int, dict] = {}
    summary_lines: list[str] = []

    for idx in range(int(num_ports)):
        power = None
        md_name = f"out_mode_{idx}"
        det_name = f"out_det_{idx}"

        exp_res = taper.getresult(md_name, "expansion")

        if isinstance(exp_res, dict):
            expansion_results[idx] = exp_res
            T = exp_res.get("T", None)
            if T is not None:
                t_arr = np.asarray(T).reshape(-1)
                if t_arr.size:
                    power = float(np.real(t_arr[0]))

        det_res = taper.getresult(det_name, "fields")

        detector_fields[idx] = det_res

        mode_powers.append(power)
        summary_lines.append(
            f"port {idx}: power={power if power is not None else 'nan'}"
        )

    return "\n".join(summary_lines), {
        "mode_powers": mode_powers,
        "expansion_results": expansion_results,
        "detector_fields": detector_fields,
    }


def getportfield(*args: Any, **kwargs: Any):
    return getportfeild(*args, **kwargs)


class PackedHybrid3DFDTD:
    """
    Lightweight class that mirrors the original forward_hybride/setuphs_hybride/hybridsimlink
    behavior in a self-contained way.
    """

    def __init__(
        self,
        *,
        complex_data_matrices: Sequence[np.ndarray],
        layer_spacing_hybride: Sequence[float],
        positions_span: Sequence[float],
        wavelength_um: float,
        n_slab: float,
        mode_path: str | Path,
        fdtd_handles: Sequence,
        hybrid_zlink: float = 0.0,
        hybrid_monitor_name: str = "hybrid_monitor",
        hybrid_xy_monitor_name: str = "hybrid_xy_monitor",
        waveguide_maping_span: float = 5.0,
        device: str = "cuda",
        save_xy_plots: bool = False,
        xy_plot_dir: str | Path | None = None,
        xy_plot_quantity: str = "abs",
        xy_plot_pad_factor: int = 4,
        propagation_mode: str = "full",
        ky_cut_ratio: float = 1.0,
        coarse_factor: float = 1.0,
        smooth_rolloff: float = 0.0,
    ):
        if torch is None:
            raise ImportError(
                "torch is required for PackedHybrid3DFDTD forward simulation"
            )

        self.complex_data_matrices = [np.asarray(m) for m in complex_data_matrices]
        self.layer_spacing_hybride = list(layer_spacing_hybride)
        self.positions_span = list(positions_span)

        self.wavelength = float(wavelength_um)
        self.n_slab = float(n_slab)
        self.mode_path = Path(mode_path)
        self.fdtd_handles = list(fdtd_handles)
        self.hybrid_zlink = float(hybrid_zlink)
        self.hybrid_monitor_name = str(hybrid_monitor_name)
        self.hybrid_xy_monitor_name = str(hybrid_xy_monitor_name)
        self.waveguide_maping_span = float(waveguide_maping_span)
        self.device = torch.device(
            device if torch.cuda.is_available() or device == "cpu" else "cpu"
        )
        self.save_xy_plots = bool(save_xy_plots)
        self.xy_plot_dir = (
            Path(xy_plot_dir).resolve()
            if xy_plot_dir is not None
            else (_project_runs_dir("hybrid_xy_plots") if self.save_xy_plots else None)
        )
        self.xy_plot_quantity = str(xy_plot_quantity)
        self.xy_plot_pad_factor = int(max(1, xy_plot_pad_factor))
        self.propagation_mode = str(propagation_mode)
        self.ky_cut_ratio = float(ky_cut_ratio)
        self.coarse_factor = float(coarse_factor)
        self.smooth_rolloff = float(smooth_rolloff)
        self.xy_plot_records: list[dict[str, Any]] = []

        self.num_layers = len(self.complex_data_matrices)
        if len(self.layer_spacing_hybride) != self.num_layers - 1:
            raise ValueError("layer_spacing_hybride length must be num_layers - 1")

        self.setuphs_hybride()

    # ---- copied/trimmed from original setuphs_hybride ----
    def setuphs_hybride(
        self,
        mode: str = "full",
        ky_cut_ratio: float = 1.0,
        coarse_factor: float = 1.0,
        smooth_rolloff: float = 0.0,
        pad_factor: int = 16,
    ):
        base = np.asarray(self.complex_data_matrices[0])
        N0 = int(base.shape[-1])
        pad_factor = int(max(1, pad_factor))
        Np = int(pad_factor * N0)

        self.N0 = N0
        self.Np = Np
        self.pad_start = (Np - N0) // 2

        span_um = float(self.positions_span[0])
        dy = span_um / N0
        self.dy = dy
        k = 2 * np.pi * float(self.n_slab) / float(self.wavelength)

        ky = 2 * np.pi * np.fft.fftfreq(Np, d=dy)
        ky_abs = np.abs(ky)

        ky_nyq = np.pi / dy
        ky_nyq_eff = ky_nyq / float(coarse_factor) if mode == "coarse" else ky_nyq
        ky_cut = min(float(ky_cut_ratio) * k, ky_nyq_eff)

        if smooth_rolloff and smooth_rolloff > 0:
            r = float(smooth_rolloff)
            k1 = ky_cut * (1.0 - r)
            k2 = ky_cut
            filt = np.ones_like(ky_abs, dtype=np.float64)
            filt[ky_abs >= k2] = 0.0
            mid = (ky_abs >= k1) & (ky_abs < k2)
            filt[mid] = 0.5 * (1.0 + np.cos(np.pi * (ky_abs[mid] - k1) / (k2 - k1)))
        else:
            filt = (ky_abs <= ky_cut).astype(np.float64)

        Hs_H = []
        for i in range(self.num_layers - 1):
            d = float(self.layer_spacing_hybride[i])

            if mode in ("bandlimit", "coarse"):
                kx = np.sqrt(np.maximum(k * k - ky * ky, 0.0))
                H = np.exp(1j * kx * d) * filt
            elif mode == "full":
                kx = np.sqrt((k * k - ky * ky).astype(np.complex128))
                H = np.exp(1j * kx * d)
            else:
                raise ValueError(f"Unknown mode={mode}")

            Hs_H.append(torch.as_tensor(H, dtype=torch.complex64, device=self.device))

        self.Hs_H = torch.stack(Hs_H, dim=0)

    # ---- copied/trimmed from original forward_hybride ----
    def forward_hybride(self):
        with torch.cuda.amp.autocast(enabled=False):
            dev = self.device

            base = torch.as_tensor(
                self.complex_data_matrices[0], device=dev, dtype=torch.complex64
            )
            N = base.shape[-1]

            N0 = int(getattr(self, "N0", N))
            Np = int(getattr(self, "Np", N))
            ps = int(getattr(self, "pad_start", 0))
            if N != N0 or self.Hs_H.shape[-1] != Np:
                raise RuntimeError("Call setuphs_hybride before forward_hybride")

            def pad_center(u, fill=0.0):
                if u.shape[-1] == Np:
                    return u
                if u.shape[-1] != N:
                    raise ValueError(
                        f"Unexpected width for pad_center: got {u.shape[-1]}, expected {N} or {Np}"
                    )
                fillv = torch.as_tensor(fill, device=u.device, dtype=u.dtype)
                out = torch.full(
                    (u.shape[0], Np), fillv, dtype=u.dtype, device=u.device
                )
                out[:, ps : ps + N] = u
                return out

            def crop_center(u):
                if Np == N:
                    return u
                return u[:, ps : ps + N]

            self.us_H = []
            for idx in range(self.num_layers):
                src = self.complex_data_matrices[
                    min(idx, len(self.complex_data_matrices) - 1)
                ]
                a_t = torch.as_tensor(src, device=dev, dtype=torch.complex64)
                if a_t.ndim == 1:
                    a_t = a_t.unsqueeze(0)
                self.us_H.append(a_t)

            Hs_H = self.Hs_H.to(dev)
            u = pad_center(self.us_H[0], fill=0.0)
            for layer_idx in range(1, self.num_layers):
                U = torch.fft.fft(torch.fft.ifftshift(u, dim=-1), dim=-1)
                U = U * Hs_H[layer_idx - 1]
                u = torch.fft.fftshift(torch.fft.ifft(U, dim=-1), dim=-1)
                self.us_H[layer_idx] = u
                u = self.hybridsimlink(crop_center(u), layer_idx)

                u = pad_center(u, fill=0.0)

            return self.us_H

    # ---- copied/trimmed from original hybridsimlink ----
    def hybridsimlink(self, u, layer_idx):
        dev = u.device
        u_np = u.detach().cpu().numpy()
        ey_1d = np.squeeze(u_np)
        if ey_1d.ndim > 1:
            ey_1d = ey_1d.reshape(-1)
        ey_1d = np.asarray(ey_1d, dtype=np.complex128)
        N_target = ey_1d.shape[0]

        span_um = float(self.positions_span[0])
        y_um = np.linspace(-0.5 * span_um, 0.5 * span_um, N_target)
        y_m = y_um * 1e-6

        mode = load_lumerical_modeE(Path(self.mode_path))
        z_m = np.asarray(mode["z"]).reshape(-1)
        Ex_z = np.asarray(mode["Ex"]).reshape(-1)
        Ey_z = np.asarray(mode["Ey"]).reshape(-1)
        Ez_z = np.asarray(mode["Ez"]).reshape(-1)

        Ex_yz, Ey_yz, Ez_yz = build_yz_vector_zlink0(
            ey_1d, y_m, Ex_z, Ey_z, Ez_z, z_m, zlink=float(self.hybrid_zlink)
        )

        workdir = Path(getattr(self, "hybrid_workdir", Path(self.mode_path).parent))
        workdir.mkdir(parents=True, exist_ok=True)
        mat_path = workdir / f"layer{layer_idx}_Ey_yz.mat"
        save_mat_v73_for_lumerical_h5py(
            mat_path,
            x0=0.0,
            y=y_m,
            z=z_m,
            Ex_yz=Ex_yz,
            Ey_yz=Ey_yz,
            Ez_yz=Ez_yz,
            f0=mode.get("f0", None) or float(self.wavelength),
        )

        if not self.fdtd_handles:
            return u
        idx = max(0, layer_idx - 1)
        if idx >= len(self.fdtd_handles):
            return u
        fdtd = self.fdtd_handles[idx]

        lambda_m = float(self.wavelength) * 1e-6
        script = f"""
matlabload('{mat_path.as_posix()}');
if (length(x)==0) {{ x=0; }}
xv = x; yv = y; zv = z;
Ex_i = permute(Ex, [3,2,1]);
Ey_i = permute(Ey, [3,2,1]);
Ez_i = permute(Ez, [3,2,1]);
EM = rectilineardataset('EM fields', xv, yv, zv);
lambda0 = {lambda_m};
f0 = c/lambda0;
EM.addparameter('lambda', lambda0, 'f', f0);
EM.addattribute('E', Ex_i, Ey_i, Ez_i);
switchtolayout;
select('import_src');
importdataset(EM);

"""
        fdtd.eval(script)
        fdtd.eval("run;")
        if self.save_xy_plots:
            plot_dir = self.xy_plot_dir if self.xy_plot_dir is not None else (workdir / "xy_field_plots")
            record = generate_hybrid_xy_visualization(
                fdtd=fdtd,
                Ey_1d=ey_1d,
                layer_index=int(layer_idx),
                span_um=float(span_um),
                wavelength_um=float(self.wavelength),
                n_slab=float(self.n_slab),
                output_dir=plot_dir,
                xy_monitor_name=self.hybrid_xy_monitor_name,
                quantity=self.xy_plot_quantity,
                pad_factor=self.xy_plot_pad_factor,
                propagation_mode=self.propagation_mode,
                ky_cut_ratio=self.ky_cut_ratio,
                coarse_factor=self.coarse_factor,
                smooth_rolloff=self.smooth_rolloff,
            )
            self.xy_plot_records.append(record)
        fields = get_fields(
            fdtd,
            monitor_name=self.hybrid_monitor_name,
            field_result_name="fields",
            get_eps=False,
            get_D=False,
            get_H=False,
            nointerpolation=False,
            unfold_symmetry=False,
        )
        picked = extract_ey_1d_at_wavelength(
            fields,
            wavelength_um=float(self.wavelength),
            selection="nearest",
        )
        ey_out = np.asarray(picked["ey"])
        y_old = np.asarray(picked["y"]).reshape(-1)
        if y_old.size == 0:
            y_old = np.linspace(0.0, 1.0, ey_out.shape[0])

        y_target = np.linspace(y_old.min(), y_old.max(), N_target)
        ey_resampled = interp_complex(ey_out, y_old, y_target, fill=0.0)

        out_tensor = torch.as_tensor(ey_resampled, device=dev, dtype=u.dtype)
        if u.ndim == 2:
            out_tensor = out_tensor.unsqueeze(0).expand(u.shape[0], -1)
        return out_tensor


__all__.append("PackedHybrid3DFDTD")
__all__.append("getportfeild")


# ---------------------------------------------------------------------------
# Legacy quick-start scratch runner kept in-module for local experimentation only
# ---------------------------------------------------------------------------.


def _demo_run(chipnum, portnum, first, handels=None):
    """
    Minimal end-to-end run. Edit the values in demo_cfg to match your project:
    - mode_path: path to Lumerical exported mode (MAT v7.3/HDF5)
    - working_dir: output folder for .fsp/.mat
    - geometry: widths/lengths/layer spacing/positions_span
    """
    chipnum = chipnum
    if torch is None:
        raise ImportError("torch is required for the demo run")
    if lumapi is None:
        raise ImportError(
            "lumapi is required for the demo run; ensure Lumerical API is installed"
        )
    BASE_DIR = _project_root()
    WORK_DIR = (BASE_DIR / "runs" / "hybrid_demo").resolve()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    demo_cfg = dict(
        working_dir=WORK_DIR,
        num_layers=4,
        total_width_um=600,
        z_span_um=0.22,
        background_material="SiO2 (Glass) - Palik",
        sub_material="Si (Silicon) - Palik",
        mode_path=(BASE_DIR / "mode1_E.mat").resolve(),
        n_slab=2.831267,
        layer_spacing=[220 - 0.1, 61.3 - 3 - 0.1, 218.7 - 3 - 0.1],
        positions_span=[600, 600, 600, 600],
        wavelength_um=1.55,
        pad_factor=16,
    )
    if first:
        # 1) Build FDTD projects
        handles = init_hybrid_fdtd(
            working_dir=demo_cfg["working_dir"],
            num_layers=demo_cfg["num_layers"],
            total_width_um=demo_cfg["total_width_um"],
            before_um=0.1,
            after_um=3,
            mesh_accuracy=4,
            z_span_um=demo_cfg["z_span_um"],
            background_material=demo_cfg["background_material"],
            sub_material=demo_cfg["sub_material"],
        )
        GDSpath = (BASE_DIR / f"chip{chipnum}.gds").resolve()
        handles[0].eval(f"redrawoff;")
        handles[0].eval(f"gdsimport('{GDSpath}','TOP','1:0');")  # fdtd.eval(f"")
        handles[0].eval(f"select('GDS_LAYER_1:0');")
        handles[0].eval(f"set('z span',0.22*1e-6);")
        handles[0].eval(f"set('index',1);")
        handles[0].eval(f"set('material','etch');")
        handles[0].eval(f"set('x',0);")
        handles[0].eval(f"set('x',0);")
        handles[0].eval(f"set('x',0);")
        handles[0].eval(f"redrawon;")
        handles[0].eval(f"redraw;")

        handles[1].eval(f"redrawoff;")
        handles[1].eval(f"gdsimport('{GDSpath}','TOP','1:0');")  # fdtd.eval(f"")
        handles[1].eval(f"select('GDS_LAYER_1:0');")
        handles[1].eval(f"set('z span',0.22*1e-6);")
        handles[1].eval(f"set('index',1);")
        handles[1].eval(f"set('material','etch');")
        handles[1].eval(f"set('x',-61.3e-6);")
        handles[1].eval(f"set('y',0);")
        handles[1].eval(f"set('z',0);")
        handles[1].eval(f"redrawon;")
        handles[1].eval(f"redraw;")
    # 2) Seed per-layer fields (load Ey_1.mat if available; fallback to uniform complex ones)
    ey_path = BASE_DIR / f"Ey_{portnum}.mat"
    if ey_path.exists():
        with h5py.File(ey_path, "r") as f:
            ey_ds = f["Ey"][()]
            if (
                getattr(ey_ds, "dtype", None) is not None
                and ey_ds.dtype.fields
                and {"real", "imag"}.issubset(ey_ds.dtype.fields)
            ):
                ey_data = ey_ds["real"] + 1j * ey_ds["imag"]
            else:
                ey_data = np.asarray(ey_ds)
        ey_1d = np.asarray(ey_data).squeeze().reshape(-1).astype(np.complex128)
        N = ey_1d.size
        monitors = [ey_1d.copy() for _ in range(demo_cfg["num_layers"])]

    runner = PackedHybrid3DFDTD(
        complex_data_matrices=monitors,
        layer_spacing_hybride=demo_cfg["layer_spacing"],
        positions_span=demo_cfg["positions_span"],
        wavelength_um=demo_cfg["wavelength_um"],
        n_slab=demo_cfg["n_slab"],
        mode_path=demo_cfg["mode_path"],
        fdtd_handles=handles,
        hybrid_zlink=0.0,
        save_xy_plots=True,
        xy_plot_dir=WORK_DIR / "xy_field_plots",
    )

    out = runner.forward_hybride()
    return out, handels


if __name__ == "__main__":
    raise SystemExit(
        "Use scripts/run_hybrid_fdtd.py with a YAML config. "
        "This module now provides reusable library functions only."
    )
