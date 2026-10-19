""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

import numpy as np
import scipy as sp
import scipy.constants
try:
    import lumapi
except ModuleNotFoundError:
    lumapi = None

try:
    from lumopt.utilities.fields import Fields, FieldsNoInterp
except ModuleNotFoundError:
    class Fields:  # pragma: no cover - lightweight fallback for training-only environments
        def __init__(self, x, y, z, wl, E, D=None, eps=None, H=None):
            self.x = np.asarray(x)
            self.y = np.asarray(y)
            self.z = np.asarray(z)
            self.wl = np.asarray(wl)
            self.E = np.asarray(E)
            self.D = None if D is None else np.asarray(D)
            self.eps = None if eps is None else np.asarray(eps)
            self.H = None if H is None else np.asarray(H)

    class FieldsNoInterp(Fields):  # pragma: no cover - lightweight fallback
        def __init__(self, x, y, z, wl, deltas, E, D=None, eps=None, H=None):
            super().__init__(x, y, z, wl, E, D=D, eps=eps, H=H)
            self.deltas = [np.asarray(delta) for delta in deltas]


def _require_lumerical_runtime() -> None:
    if lumapi is None:
        raise ModuleNotFoundError(
            "lumapi is required for live Lumerical/FDTD operations, but it is not installed in this Python environment"
        )


def _extract_e_array(fields_obj):
    if hasattr(fields_obj, "E"):
        return np.asarray(fields_obj.E)
    if isinstance(fields_obj, dict):
        if "E" in fields_obj and isinstance(fields_obj["E"], dict) and "E" in fields_obj["E"]:
            return np.asarray(fields_obj["E"]["E"])
        if "E" in fields_obj:
            return np.asarray(fields_obj["E"])
    raise ValueError("Cannot find electric field array `E` in fields object")


def _extract_lambda_array(fields_obj):
    if hasattr(fields_obj, "wl"):
        return np.asarray(getattr(fields_obj, "wl"))
    if hasattr(fields_obj, "lambda"):
        return np.asarray(getattr(fields_obj, "lambda"))
    if isinstance(fields_obj, dict):
        if "lambda_um" in fields_obj:
            return np.asarray(fields_obj["lambda_um"])
        if "E" in fields_obj and isinstance(fields_obj["E"], dict) and "lambda" in fields_obj["E"]:
            return np.asarray(fields_obj["E"]["lambda"])
        if "lambda" in fields_obj:
            return np.asarray(fields_obj["lambda"])
    return np.asarray([], dtype=float)


def wavelengths_to_um(wavelengths) -> np.ndarray:
    wl = np.asarray(wavelengths).reshape(-1)
    if wl.size == 0:
        return wl.astype(float)
    wl = wl.astype(float)
    if np.nanmax(np.abs(wl)) < 1e-3:
        return wl * 1e6
    return wl


def coordinates_to_um(values) -> np.ndarray:
    coords = np.asarray(values).reshape(-1)
    if coords.size == 0:
        return coords.astype(float)
    coords = coords.astype(float)
    if np.nanmax(np.abs(coords)) < 1e-3:
        return coords * 1e6
    return coords


def _detect_component_axis(shape):
    candidates = [i for i, s in enumerate(shape) if s == 3]
    return candidates[-1] if candidates else None


def _detect_wavelength_axis(shape, wl_count: int, component_axis: int | None):
    if wl_count <= 1:
        return None
    candidates = [i for i, s in enumerate(shape) if s == wl_count]
    if component_axis is not None and component_axis in candidates and len(candidates) > 1:
        candidates = [i for i in candidates if i != component_axis]
    return candidates[-1] if candidates else None


def _collapse_except_axis(arr: np.ndarray, keep_axis: int) -> np.ndarray:
    out = np.asarray(arr)
    axis = int(keep_axis)
    if axis < 0:
        axis += out.ndim
    for ax in reversed(range(out.ndim)):
        if ax == axis:
            continue
        out = np.take(out, indices=out.shape[ax] // 2, axis=ax)
    return np.asarray(out)


def select_field_component_at_wavelength(
    fields_obj,
    *,
    component: str | int = "Ey",
    wavelength_um: float | None = None,
    wavelength_index: int | None = None,
    selection: str = "nearest",
    squeeze: bool = True,
):
    """
    Select one field component at one wavelength from multi-wavelength monitor data.
    Returns dict with selected field and wavelength metadata.
    """
    e = _extract_e_array(fields_obj)
    wl_um = wavelengths_to_um(_extract_lambda_array(fields_obj))
    comp_map = {"Ex": 0, "Ey": 1, "Ez": 2, "ex": 0, "ey": 1, "ez": 2}
    comp_idx = int(component if isinstance(component, int) else comp_map[component])

    comp_axis = _detect_component_axis(e.shape)
    wl_axis = _detect_wavelength_axis(e.shape, wl_um.size, comp_axis)

    if wavelength_index is not None:
        idx = int(wavelength_index)
    elif wl_um.size <= 1:
        idx = 0
    elif wavelength_um is None:
        idx = wl_um.size // 2
    elif selection == "exact":
        hits = np.where(np.isclose(wl_um, float(wavelength_um), rtol=1e-9, atol=1e-12))[0]
        if hits.size == 0:
            raise ValueError(f"Wavelength {wavelength_um} um not found in available grid")
        idx = int(hits[0])
    else:
        idx = int(np.argmin(np.abs(wl_um - float(wavelength_um))))

    if wl_um.size > 0 and not (0 <= idx < wl_um.size):
        raise IndexError(f"wavelength_index out of range: {idx}, valid [0, {wl_um.size - 1}]")

    selected = np.asarray(e)
    if wl_axis is not None:
        selected = np.take(selected, indices=idx, axis=wl_axis)
        if comp_axis is not None and comp_axis > wl_axis:
            comp_axis -= 1

    if comp_axis is not None:
        selected = np.take(selected, indices=comp_idx, axis=comp_axis)
    elif comp_idx != 0:
        raise ValueError(
            f"Component axis not found in E shape={e.shape}; cannot select component index {comp_idx}"
        )

    if squeeze:
        selected = np.squeeze(selected)

    chosen_wl_um = float(wl_um[idx]) if wl_um.size > 0 else float(wavelength_um or np.nan)
    return {
        "field": np.asarray(selected),
        "selected_wavelength_um": chosen_wl_um,
        "selected_wavelength_index": int(idx),
        "wavelengths_um": np.asarray(wl_um, dtype=float),
        "wavelength_axis": wl_axis,
        "component_axis": _detect_component_axis(e.shape),
    }


def extract_ey_1d_at_wavelength(
    fields_obj,
    *,
    wavelength_um: float | None = None,
    wavelength_index: int | None = None,
    selection: str = "nearest",
):
    """
    Extract Ey(y) at a chosen wavelength.
    Any extra axes are collapsed by taking center index.
    """
    selected = select_field_component_at_wavelength(
        fields_obj,
        component="Ey",
        wavelength_um=wavelength_um,
        wavelength_index=wavelength_index,
        selection=selection,
        squeeze=False,
    )
    ey_nd = np.asarray(selected["field"])

    if isinstance(fields_obj, dict):
        if "y_um" in fields_obj:
            y = np.asarray(fields_obj["y_um"]).reshape(-1)
        elif "y" in fields_obj:
            y = np.asarray(fields_obj["y"]).reshape(-1)
        else:
            y = np.asarray([], dtype=float)
    else:
        y = np.asarray(getattr(fields_obj, "y", np.array([]))).reshape(-1)
    if y.size > 0 and y.size in ey_nd.shape:
        keep_axis = [i for i, s in enumerate(ey_nd.shape) if s == y.size][-1]
        ey = _collapse_except_axis(ey_nd, keep_axis)
    else:
        ey = np.squeeze(ey_nd)
        if ey.ndim > 1:
            ey = ey.reshape(-1)
        if y.size == 0:
            y = np.linspace(0.0, 1.0, ey.shape[0])

    ey = np.asarray(ey)
    if y.size != ey.shape[0]:
        y = np.linspace(
            y.min() if y.size else 0.0,
            y.max() if y.size else 1.0,
            ey.shape[0],
        )

    return {
        "ey": ey,
        "y": y,
        "selected_wavelength_um": selected["selected_wavelength_um"],
        "selected_wavelength_index": selected["selected_wavelength_index"],
        "wavelengths_um": selected["wavelengths_um"],
    }


def monitor_payload_from_fields(fields_obj, *, include_axes: bool = True):
    """
    Convert a Lumerical Fields object to a cache-friendly dict that preserves
    full multi-wavelength E-field data.
    """
    payload = {
        "E": _extract_e_array(fields_obj),
        "lambda_um": wavelengths_to_um(_extract_lambda_array(fields_obj)),
    }
    if include_axes:
        for axis_name in ("x", "y", "z"):
            if hasattr(fields_obj, axis_name):
                payload[f"{axis_name}_um"] = coordinates_to_um(getattr(fields_obj, axis_name))
    return payload


def index_payload_from_monitor(fdtd, monitor_name, *, unfold_symmetry: bool = True):
    fields_eps, x_vals, y_vals, z_vals, wavelengths = get_eps_from_sim(
        fdtd,
        monitor_name=monitor_name,
        unfold_symmetry=unfold_symmetry,
    )
    eps = np.asarray(fields_eps)
    if eps.ndim >= 1 and eps.shape[-1] == 3:
        scalar_eps = np.mean(np.real(eps), axis=-1)
    else:
        scalar_eps = np.real(eps)
    index = np.sqrt(np.maximum(scalar_eps, 0.0))
    return {
        "eps": eps,
        "index": index,
        "lambda_um": wavelengths_to_um(wavelengths),
        "x_um": coordinates_to_um(x_vals),
        "y_um": coordinates_to_um(y_vals),
        "z_um": coordinates_to_um(z_vals),
    }

def get_lambda_from_cad(fdtd, field_result_name):
    _require_lumerical_runtime()
    fdtd.eval("wl = {0}.E.lambda;".format(field_result_name))
    return np.array(lumapi.getVar(fdtd.handle, "wl")).flatten()

def get_fields_on_cad(fdtd, monitor_name, field_result_name, get_eps, get_D, get_H, nointerpolation, unfold_symmetry = True):
    _require_lumerical_runtime()
    unfold_symmetry_string = "true" if unfold_symmetry else "false"
    fdtd.eval("options=struct; options.unfold={0};".format(unfold_symmetry_string) + 
              "{0} = struct;".format(field_result_name) +
              "{0}.E = getresult('{1}','E',options);".format(field_result_name, monitor_name))

    if get_eps or get_D:
        index_monitor_name = monitor_name + '_index'
        fdtd.eval("{0}.index = getresult('{1}','index',options);".format(field_result_name, index_monitor_name))

    if get_H:
        fdtd.eval("{0}.H = getresult('{1}','H',options);".format(field_result_name, monitor_name))

    if nointerpolation:
        fdtd.eval("{0}.delta = struct;".format(field_result_name) +
                  "{0}.delta.x = getresult('{1}','delta_x',options);".format(field_result_name, monitor_name) +
                  "{0}.delta.y = getresult('{1}','delta_y',options);".format(field_result_name, monitor_name))
        monitor_dimension = fdtd.getresult(monitor_name, 'dimension')
        if monitor_dimension == 3:
            fdtd.eval("{0}.delta.z = getdata('{1}','delta_z');".format(field_result_name, monitor_name))
        else:
            fdtd.eval("{0}.delta.z = 0.0;".format(field_result_name))

def scale_fields_on_cad(fdtd, field_name, scaling_factors):
    _require_lumerical_runtime()
    lumapi.putMatrix(fdtd.handle, "scaling_factors", scaling_factors)

    fdtd.eval("scaling_matrix = zeros(length(scaling_factors),length(scaling_factors));" +
              "for(i=1:length(scaling_factors)) {scaling_matrix(i,i) = scaling_factors(i);}" +
              "cur_data = {}.E;".format(field_name)+
              "EE = cur_data.E;"+
              "sEE = size(EE);" +
              "EEx = cur_data.Ex;"+
              "EEy = cur_data.Ey;"+
              "EEz = cur_data.Ez;"+
              "EEx = reshape(EEx,[sEE(1)*sEE(2)*sEE(3),sEE(4)]);"+
              "EEy = reshape(EEy,[sEE(1)*sEE(2)*sEE(3),sEE(4)]);"+
              "EEz = reshape(EEz,[sEE(1)*sEE(2)*sEE(3),sEE(4)]);"+
              "EEx = reshape(mult(EEx,scaling_matrix),[sEE(1),sEE(2),sEE(3),sEE(4)]);"+
              "EEy = reshape(mult(EEy,scaling_matrix),[sEE(1),sEE(2),sEE(3),sEE(4)]);"+
              "EEz = reshape(mult(EEz,scaling_matrix),[sEE(1),sEE(2),sEE(3),sEE(4)]);"+
              "E2=rectilineardataset('E',cur_data.x,cur_data.y,cur_data.z);"+
              "E2.addparameter('lambda',cur_data.lambda,'f',cur_data.f);"+
              "E2.addattribute('E',EEx,EEy,EEz);"+
              "{}.E = E2;".format(field_name)+
              "clear(scaling_factors,scaling_matrix,cur_data,EE,sEE,EEx,EEy,EEz,E2);")

def get_fields(fdtd, monitor_name, field_result_name, get_eps, get_D, get_H, nointerpolation, unfold_symmetry = True, on_cad_only = False):
    _require_lumerical_runtime()

    get_fields_on_cad(fdtd, monitor_name, field_result_name, get_eps, get_D, get_H, nointerpolation=nointerpolation, unfold_symmetry=unfold_symmetry)

    ## If required, we now transfer the field data to Python and package it up 
    if not on_cad_only:
        fields_dict = lumapi.getVar(fdtd.handle, field_result_name)

    if get_eps:
        if fdtd.getnamednumber('varFDTD') == 1:
            if 'index_x' in fields_dict['index'] and 'index_y' in fields_dict['index'] and not 'index_z' in fields_dict['index']: # varFDTD TE simulation
                fields_dict['index']['index_z'] = fields_dict['index']['index_x']*0.0 + 1.0
            elif not 'index_x' in fields_dict['index'] and not 'index_y' in fields_dict['index'] and 'index_z' in fields_dict['index']: # varFDTD TM simulation
                fields_dict['index']['index_x'] = fields_dict['index']['index_z']*0.0 + 1.0
                fields_dict['index']['index_y'] = fields_dict['index']['index_x']
        assert 'index_x' in fields_dict['index'] and 'index_y' in fields_dict['index'] and 'index_z' in fields_dict['index']
        fields_eps = np.stack((np.power(fields_dict['index']['index_x'], 2), 
                               np.power(fields_dict['index']['index_y'], 2), 
                               np.power(fields_dict['index']['index_z'], 2)), 
                               axis = -1)
    else:
        fields_eps = None

    fields_D = fields_dict['E']['E'] * fields_eps * sp.constants.epsilon_0 if get_D else None

    fields_H = fields_dict['H']['H'] if get_H else None

    if nointerpolation:
        deltas = [fields_dict['delta']['x'], fields_dict['delta']['y'], fields_dict['delta']['z']]
        return FieldsNoInterp(fields_dict['E']['x'], fields_dict['E']['y'], fields_dict['E']['z'], fields_dict['E']['lambda'], deltas, fields_dict['E']['E'], fields_D, fields_eps, fields_H)
    else:
        return Fields(fields_dict['E']['x'], fields_dict['E']['y'], fields_dict['E']['z'], fields_dict['E']['lambda'], fields_dict['E']['E'], fields_D, fields_eps, fields_H)

def set_spatial_interp(fdtd,monitor_name,setting):
    script='select("{}");set("spatial interpolation","{}");'.format(monitor_name,setting)
    fdtd.eval(script)

def get_eps_from_sim(fdtd, monitor_name = 'opt_fields', unfold_symmetry = True):
    _require_lumerical_runtime()
    index_monitor_name = monitor_name + '_index'

    unfold_symmetry_string = "true" if unfold_symmetry else "false"
    fdtd.eval(('options=struct; options.unfold={0};'
               '{1}_result = getresult("{1}","index",options);'
               '{1}_eps_x = ({1}_result.index_x)^2;'
               '{1}_eps_y = ({1}_result.index_y)^2;'
               '{1}_eps_z = ({1}_result.index_z)^2;'
               '{1}_x = {1}_result.x;'
               '{1}_y = {1}_result.y;'
               '{1}_z = {1}_result.z;'
               '{1}_lambda = {1}_result.lambda;'
               ).format(unfold_symmetry_string, index_monitor_name))
    fields_eps_x = fdtd.getv('{0}_eps_x'.format(index_monitor_name))
    fields_eps_y = fdtd.getv('{0}_eps_y'.format(index_monitor_name))
    fields_eps_z = fdtd.getv('{0}_eps_z'.format(index_monitor_name))
    index_monitor_x      = fdtd.getv('{0}_x'.format(index_monitor_name))
    index_monitor_y      = fdtd.getv('{0}_y'.format(index_monitor_name))
    index_monitor_z      = fdtd.getv('{0}_z'.format(index_monitor_name))
    index_monitor_lambda = fdtd.getv('{0}_lambda'.format(index_monitor_name))

    ## This would be the more elegant and efficient way but it currently does not 
    ## work with unfolding options.
    # index_dict = fdtd.getresult(index_monitor_name, 'index')
    # fields_eps_x = np.power(index_dict['index_x'], 2)
    # fields_eps_y = np.power(index_dict['index_y'], 2)
    # fields_eps_z = np.power(index_dict['index_z'], 2)
    # index_monitor_x      = index_dict['x']
    # index_monitor_y      = index_dict['y']
    # index_monitor_z      = index_dict['z']
    # index_monitor_lambda = index_dict['lambda']


    fields_eps = np.stack((fields_eps_x, fields_eps_y, fields_eps_z), axis = -1)
    return fields_eps, index_monitor_x,index_monitor_y,index_monitor_z, index_monitor_lambda
