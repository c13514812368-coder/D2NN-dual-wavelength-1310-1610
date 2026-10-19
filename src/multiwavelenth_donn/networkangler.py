import math
import inspect
from pathlib import Path
from typing import List, Union

import h5py
import numpy as np
from scipy.io import loadmat
import torch
import torch.nn as nn

from .lumerical_scripts import extract_ey_1d_at_wavelength, get_fields
from .wavelength_tools import (
    canonical_wavelength_key,
    normalize_wavelength_dict,
    resolve_wavelength_choice,
    select_value_by_wavelength,
    sorted_wavelengths_from_dict,
)


LUMERICAL_OBJECT_DEFINED_DIELECTRIC = "<Object defined dielectric>"
LUMERICAL_SI_PALIK = "Si (Silicon) - Palik"
LUMERICAL_SIO2_PALIK = "SiO2 (Glass) - Palik"
_LUMERICAL_MATERIAL_NAME_ALIASES = {
    "<Object define dielectric>": LUMERICAL_OBJECT_DEFINED_DIELECTRIC,
    "Si": LUMERICAL_SI_PALIK,
    "SiO2": LUMERICAL_SIO2_PALIK,
}


def _resolve_torch_device(device: object) -> torch.device:
    try:
        resolved = torch.device(device)
    except Exception:
        return torch.device("cpu")
    if resolved.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return resolved


def normalize_lumerical_material_name(material: object) -> str:
    value = str(material).strip()
    return _LUMERICAL_MATERIAL_NAME_ALIASES.get(value, value)


def uses_object_defined_dielectric(material: object) -> bool:
    return normalize_lumerical_material_name(material) == LUMERICAL_OBJECT_DEFINED_DIELECTRIC


def make_monitor_positions(num_layers: int, spacings: Union[List[float], np.ndarray]) -> List[float]:
    """
    鏍规嵁灞傞棿璺濊绠楁瘡涓洃瑙嗗櫒鍦?x 鏂瑰悜鐨勪綅缃€?
    鍋囪绗竴涓洃瑙嗗櫒浣嶄簬娉㈠璧风偣锛屾€婚暱涓哄悇灞傞棿璺濈疮鍔犮€?
    """
    positions = [0.0]
    acc = 0.0
    for i in range(num_layers):
        acc += spacings[i] if i < len(spacings) else 0.0
        positions.append(acc)
    return positions

from typing import Any, Dict, List, Mapping, Optional, Sequence, Union


def _serialize_shape_config_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            str(key): _serialize_shape_config_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_serialize_shape_config_value(item) for item in value]
    return value


class NetworkShape:
    def __init__(
        self,
        # ---------------- 缃戠粶缁撴瀯浠ュ強鍩烘湰璁剧疆锛氱缁忕綉缁滅粨鏋?----------------
        num_layers: int,
        neurons_per_layer: List[int],
        total_length_um: float,
        total_width_um: float,
        inter_layer_spacings_um: Union[List[float], np.ndarray],
        wavelength_um: float = 1.55,
        layer_positions_um: Optional[Union[List[float], np.ndarray]] = None,

        # ---------------- 缃戠粶缁撴瀯浠ュ強鍩烘湰璁剧疆锛氫豢鐪熻缃?----------------
        mesh_accuracy: float = 3.0,
        dist_from_sub_to_fdtd_um: float = 0.0,
        monitor_positions_x_um: Optional[Union[List[float], np.ndarray]] = None,
        monitor_positions_z_um: Optional[Union[List[float], np.ndarray]] = None,
        use_var_fdtd: bool = True,

        # ---------------- 缃戠粶缁撴瀯浠ュ強鍩烘湰璁剧疆锛氭潗鏂欑粨鏋?----------------
        background_material: str = LUMERICAL_OBJECT_DEFINED_DIELECTRIC,
        background_material_index:float =1,

        propagation_material: str = LUMERICAL_SI_PALIK,
        propagation_material_index: float =1,
        propagation_thickness_um: float = 0.22,
        propagation_center_z_um: float = 0.0,
        propagation_span_x_um: Optional[float] = None,
        propagation_span_y_um: Optional[float] = None,

        sub_material: str = LUMERICAL_SIO2_PALIK,
        sub_material_index: float = 1,
        sub_thickness_um: float = 2.0,
        sub_span_x_um: Optional[float] = None,
        sub_span_y_um: Optional[float] = None,

        slot_material: Optional[str] = None,
        slot_material_index: float = 1,
        slot_thickness_um: Optional[float] = None,
        slot_center_z_um: Optional[float] = None,

        # ---------------- 杈撳叆杈撳嚭缁撴瀯 ----------------

        waveguide_width_um: float = 0.5,
        waveguide_length_um: float = 3,
        num_input_waveguides: int = 1,
        num_output_waveguides: int = 1,
        input_centers_y_um: Optional[Sequence[float]] = None,
        output_centers_y_um: Optional[Sequence[float]] = None,

        use_taper: bool = False,

        in_taper_length_um: float = 50,
        in_taper_width_um: float = 12.0,

        out_taper_length_um: float = 50.0,
        out_taper_width_um: float = 12.0,

        # ---------------- 琛嶅皠缁撴瀯 ----------------
        unit_spacing_um: float = 0.3,
        duty_cycle: float = 0.5,
        unit_span_um: float = 0.15,
        fill_material: str = LUMERICAL_SIO2_PALIK,
        repeats: int = 3,
        mapping_factor_class: Optional[Union[str, Dict[str, Any]]] = None,

        # ---------------- 鍏朵粬锛堝彲閫夛級 ----------------
        positions_span_um: Optional[Union[List[float], np.ndarray]] = None,

        monitor_positions: Optional[Sequence[float]] = None,
        local_map_enabled: bool = False,
        local_map_left_pad_um: float = 10.0,
        local_map_right_pad_um: float = 10.0,
        wavelength_start: float = 1.55,
        wavelength_stop: float = 1.55,
        wavelength_center: float = 1.55,
        wavelength_points: int = 1,


    ):
        # ========== 缃戠粶缁撴瀯浠ュ強鍩烘湰璁剧疆锛氱缁忕綉缁滅粨鏋?==========
        self.num_layers = int(num_layers)
        self.neurons_per_layer = [int(x) for x in neurons_per_layer]
        if self.num_layers <= 0:
            raise ValueError(f"shape.num_layers must be positive, got {self.num_layers}")
        if len(self.neurons_per_layer) != self.num_layers:
            raise ValueError(
                "shape.neurons_per_layer length must equal shape.num_layers: "
                f"num_layers={self.num_layers}, "
                f"len(neurons_per_layer)={len(self.neurons_per_layer)}, "
                f"neurons_per_layer={self.neurons_per_layer}"
            )
        if any(count <= 0 for count in self.neurons_per_layer):
            raise ValueError(
                "shape.neurons_per_layer entries must all be positive: "
                f"neurons_per_layer={self.neurons_per_layer}"
            )
        self.total_length_um = float(total_length_um)
        self.total_width_um = float(total_width_um)
        self.inter_layer_spacings_um = np.asarray(inter_layer_spacings_um, dtype=float)
        if self.inter_layer_spacings_um.size != self.num_layers - 1:
            raise ValueError(
                "shape.inter_layer_spacings_um length must equal shape.num_layers - 1: "
                f"num_layers={self.num_layers}, "
                f"expected={self.num_layers - 1}, "
                f"got={self.inter_layer_spacings_um.size}, "
                f"inter_layer_spacings_um={self.inter_layer_spacings_um.tolist()}"
            )
        self.wavelength_um = float(wavelength_um)
        self.layer_positions_um = None if layer_positions_um is None else np.asarray(layer_positions_um, dtype=float)
        if self.layer_positions_um is not None and self.layer_positions_um.size != self.num_layers:
            raise ValueError(
                "shape.layer_positions_um length must equal shape.num_layers: "
                f"num_layers={self.num_layers}, "
                f"got={self.layer_positions_um.size}, "
                f"layer_positions_um={self.layer_positions_um.tolist()}"
            )

        # ========== 缃戠粶缁撴瀯浠ュ強鍩烘湰璁剧疆锛氫豢鐪熻缃?==========
        self.use_var_fdtd = bool(use_var_fdtd)
        self.mesh_accuracy = float(mesh_accuracy)
        self.dist_from_sub_to_fdtd_um = float(dist_from_sub_to_fdtd_um)

        self.monitor_positions_x_um = None if monitor_positions_x_um is None else np.asarray(monitor_positions_x_um, dtype=float)
        self.monitor_positions_z_um = None if monitor_positions_z_um is None else np.asarray(monitor_positions_z_um, dtype=float)

        # ========== 缃戠粶缁撴瀯浠ュ強鍩烘湰璁剧疆锛氭潗鏂欑粨鏋?==========
        self.background_material = normalize_lumerical_material_name(background_material)
        self.background_material_index = float(background_material_index)

        self.propagation_material = normalize_lumerical_material_name(propagation_material)
        self.propagation_material_index = float(propagation_material_index)
        self.propagation_thickness_um = float(propagation_thickness_um)
        self.propagation_center_z_um = float(propagation_center_z_um)
        self.propagation_span_x_um = None if propagation_span_x_um is None else float(propagation_span_x_um)
        self.propagation_span_y_um = None if propagation_span_y_um is None else float(propagation_span_y_um)

        self.sub_material = normalize_lumerical_material_name(sub_material)
        self.sub_material_index = float(sub_material_index)
        self.sub_thickness_um = float(sub_thickness_um)
        self.sub_span_x_um = None if sub_span_x_um is None else float(sub_span_x_um)
        self.sub_span_y_um = None if sub_span_y_um is None else float(sub_span_y_um)



        self.slot_material = (
            None if slot_material is None else normalize_lumerical_material_name(slot_material)
        )
        self.slot_material_index = float(slot_material_index)
        self.slot_thickness_um = None if slot_thickness_um is None else float(slot_thickness_um)
        self.slot_center_z_um = None if slot_center_z_um is None else float(slot_center_z_um)


        # ========== 杈撳叆杈撳嚭缁撴瀯 ==========
        self.waveguide_width_um = float(waveguide_width_um)
        self.waveguide_length_um = float(waveguide_length_um)
        self.num_input_waveguides = int(num_input_waveguides)
        self.num_output_waveguides = int(num_output_waveguides)
        if self.num_input_waveguides <= 0:
            raise ValueError(
                f"shape.num_input_waveguides must be positive, got {self.num_input_waveguides}"
            )
        if self.num_output_waveguides <= 0:
            raise ValueError(
                f"shape.num_output_waveguides must be positive, got {self.num_output_waveguides}"
            )
        if self.neurons_per_layer[0] != self.num_input_waveguides:
            raise ValueError(
                "shape.neurons_per_layer[0] must equal shape.num_input_waveguides: "
                f"neurons_per_layer[0]={self.neurons_per_layer[0]}, "
                f"num_input_waveguides={self.num_input_waveguides}, "
                f"neurons_per_layer={self.neurons_per_layer}"
            )
        if self.neurons_per_layer[-1] != self.num_output_waveguides:
            raise ValueError(
                "shape.neurons_per_layer[-1] must equal shape.num_output_waveguides: "
                f"neurons_per_layer[-1]={self.neurons_per_layer[-1]}, "
                f"num_output_waveguides={self.num_output_waveguides}, "
                f"neurons_per_layer={self.neurons_per_layer}"
            )
        self.input_centers_y_um = input_centers_y_um 
        self.output_centers_y_um = output_centers_y_um 
        if self.input_centers_y_um is not None and len(self.input_centers_y_um) != self.num_input_waveguides:
            raise ValueError(
                "shape.input_centers_y_um length must equal shape.num_input_waveguides: "
                f"num_input_waveguides={self.num_input_waveguides}, "
                f"got={len(self.input_centers_y_um)}, "
                f"input_centers_y_um={list(self.input_centers_y_um)}"
            )
        if self.output_centers_y_um is not None and len(self.output_centers_y_um) != self.num_output_waveguides:
            raise ValueError(
                "shape.output_centers_y_um length must equal shape.num_output_waveguides: "
                f"num_output_waveguides={self.num_output_waveguides}, "
                f"got={len(self.output_centers_y_um)}, "
                f"output_centers_y_um={list(self.output_centers_y_um)}"
            )

        self.use_taper = bool(use_taper)

        self.in_taper_length_um = float(in_taper_length_um)
        self.in_taper_width_um = float(in_taper_width_um)
        self.out_taper_length_um = float(out_taper_length_um)
        self.out_taper_width_um = float(out_taper_width_um)



        # ========== 琛嶅皠缁撴瀯 ========== 
        self.unit_spacing_um = float(unit_spacing_um)
        self.duty_cycle = float(duty_cycle)
        self.unit_span_um = float(unit_span_um)
        self.fill_material = normalize_lumerical_material_name(fill_material)
        self.repeats = int(repeats)
        self.mapping_factor_class = mapping_factor_class
        if self.unit_spacing_um <= 0.0:
            raise ValueError(f"shape.unit_spacing_um must be positive, got {self.unit_spacing_um}")
        if self.unit_span_um <= 0.0:
            raise ValueError(f"shape.unit_span_um must be positive, got {self.unit_span_um}")
        if self.duty_cycle <= 0.0 or self.duty_cycle > 1.0:
            raise ValueError(
                f"shape.duty_cycle must be in (0, 1], got {self.duty_cycle}"
            )
        if self.repeats <= 0:
            raise ValueError(f"shape.repeats must be a positive integer, got {self.repeats}")

        inferred_slot_width_um = self.duty_cycle * self.unit_spacing_um
        if not np.isclose(
            self.unit_span_um,
            inferred_slot_width_um,
            rtol=1e-6,
            atol=1e-9,
        ):
            raise ValueError(
                "shape.unit_span_um must match shape.duty_cycle * shape.unit_spacing_um: "
                f"unit_span_um={self.unit_span_um}, "
                f"duty_cycle={self.duty_cycle}, "
                f"unit_spacing_um={self.unit_spacing_um}, "
                f"expected_unit_span_um={inferred_slot_width_um}"
            )

        slot_count_float = self.total_width_um / self.unit_spacing_um
        usable_slot_count = int(math.floor(slot_count_float + 1e-9))
        if usable_slot_count <= 0:
            raise ValueError(
                "shape.total_width_um must be large enough to place at least one slot center at the configured pitch: "
                f"total_width_um={self.total_width_um}, "
                f"unit_spacing_um={self.unit_spacing_um}, "
                f"slot_count_float={slot_count_float}"
            )

        parameter_count_float = self.total_width_um / (self.unit_spacing_um * float(self.repeats))
        expected_internal_neurons = int(math.floor(parameter_count_float + 1e-9))
        if expected_internal_neurons <= 0:
            raise ValueError(
                "shape.total_width_um must be large enough to place at least one repeated trainable parameter: "
                f"total_width_um={self.total_width_um}, "
                f"unit_spacing_um={self.unit_spacing_um}, "
                f"repeats={self.repeats}, "
                f"parameter_count_float={parameter_count_float}"
            )

        self.usable_slot_count = usable_slot_count
        self.trainable_parameter_count = expected_internal_neurons
        self.unused_slot_count = int(usable_slot_count - expected_internal_neurons * self.repeats)
        self.unused_top_margin_um = float(
            self.total_width_um - usable_slot_count * self.unit_spacing_um
        )
        internal_neurons = self.neurons_per_layer[1:-1]
        if any(count != expected_internal_neurons for count in internal_neurons):
            raise ValueError(
                "Internal neurons_per_layer must match the geometry-derived trainable parameter count "
                "int(total_width_um / (unit_spacing_um * repeats)): "
                f"expected_internal_neurons={expected_internal_neurons}, "
                f"neurons_per_layer={self.neurons_per_layer}, "
                f"total_width_um={self.total_width_um}, "
                f"unit_spacing_um={self.unit_spacing_um}, "
                f"repeats={self.repeats}, "
                f"usable_slot_count={usable_slot_count}, "
                f"unused_slot_count={self.unused_slot_count}"
            )

        # ========== 鍏朵粬锛堝彲閫夛級 ==========
        self.positions_span_um = None if positions_span_um is None else np.asarray(positions_span_um, dtype=float)
        if self.positions_span_um is not None and self.positions_span_um.size != self.num_layers:
            raise ValueError(
                "shape.positions_span_um length must equal shape.num_layers: "
                f"num_layers={self.num_layers}, "
                f"got={self.positions_span_um.size}, "
                f"positions_span_um={self.positions_span_um.tolist()}"
            )


        self.input_ys=input_centers_y_um
        self.output_ys =output_centers_y_um

        self.monitor_positions =monitor_positions
        self.local_map_enabled = bool(local_map_enabled)
        self.local_map_left_pad_um = float(local_map_left_pad_um)
        self.local_map_right_pad_um = float(local_map_right_pad_um)
        if self.local_map_left_pad_um < 0.0:
            raise ValueError(
                f"shape.local_map_left_pad_um must be >= 0, got {self.local_map_left_pad_um}"
            )
        if self.local_map_right_pad_um < 0.0:
            raise ValueError(
                f"shape.local_map_right_pad_um must be >= 0, got {self.local_map_right_pad_um}"
            )


        self.wavelength_start = float(wavelength_start)
        self.wavelength_stop = float(wavelength_stop)
        self.wavelength_center = float(wavelength_center)
        self.wavelength_points = max(1, int(wavelength_points))
        if abs(self.wavelength_stop - self.wavelength_start) <= 0.0:
            self.wavelength_points = 1

    def resolve_wavelength_spec(self) -> Dict[str, Any]:
        start_um = float(getattr(self, "wavelength_start", getattr(self, "wavelength_um", 1.55)))
        stop_um = float(getattr(self, "wavelength_stop", start_um))
        center_um = float(getattr(self, "wavelength_center", 0.5 * (start_um + stop_um)))
        points = max(1, int(getattr(self, "wavelength_points", 1)))
        if abs(stop_um - start_um) <= 0.0:
            points = 1
        return {
            "start_um": start_um,
            "stop_um": stop_um,
            "center_um": center_um,
            "points": points,
        }

    def wavelength_grid_um(self) -> np.ndarray:
        spec = self.resolve_wavelength_spec()
        if spec["points"] <= 1:
            return np.asarray([spec["center_um"]], dtype=float)
        return np.linspace(spec["start_um"], spec["stop_um"], int(spec["points"]), dtype=float)

    def to_config_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for name in inspect.signature(type(self).__init__).parameters:
            if name == "self":
                continue
            payload[name] = _serialize_shape_config_value(getattr(self, name))
        return payload

    def to_cache_payload(self) -> Dict[str, Any]:
        def _to_list_float(x):
            if x is None:
                return None
            return np.asarray(x, dtype=float).tolist()

        def _to_float_optional(x):
            return None if x is None else float(x)

        return {
            # 缃戠粶缁撴瀯浠ュ強鍩烘湰璁剧疆锛氱缁忕綉缁滅粨鏋?
            "num_layers": int(self.num_layers),
            "neurons_per_layer": [int(x) for x in self.neurons_per_layer],
            "total_length_um": float(self.total_length_um),
            "total_width_um": float(self.total_width_um),
            "inter_layer_spacings_um": _to_list_float(self.inter_layer_spacings_um),
            "layer_positions_um": _to_list_float(self.layer_positions_um),
            "wavelength_um": float(self.wavelength_um),
            "wavelength_start": float(self.wavelength_start),
            "wavelength_stop": float(self.wavelength_stop),
            "wavelength_points": int(getattr(self, "wavelength_points", 1)),

            # 缃戠粶缁撴瀯浠ュ強鍩烘湰璁剧疆锛氫豢鐪熻缃?
            "use_var_fdtd": bool(self.use_var_fdtd),
            "mesh_accuracy": float(self.mesh_accuracy),
            "slab_center_y_um": _to_float_optional(getattr(self, "slab_center_y_um", None)),
            "fdtd_center_x_um": _to_float_optional(getattr(self, "fdtd_center_x_um", None)),
            "fdtd_center_y_um": _to_float_optional(getattr(self, "fdtd_center_y_um", None)),
            "fdtd_center_z_um": _to_float_optional(getattr(self, "fdtd_center_z_um", None)),
            "dist_from_sub_to_fdtd_um": float(self.dist_from_sub_to_fdtd_um),
            "monitor_positions_x_um": _to_list_float(self.monitor_positions_x_um),
            "monitor_positions_z_um": _to_list_float(self.monitor_positions_z_um),
            "monitor_positions": _to_list_float(getattr(self, "monitor_positions", None)),
            "local_map_enabled": bool(getattr(self, "local_map_enabled", False)),
            "local_map_left_pad_um": float(getattr(self, "local_map_left_pad_um", 10.0)),
            "local_map_right_pad_um": float(getattr(self, "local_map_right_pad_um", 10.0)),

            # 缃戠粶缁撴瀯浠ュ強鍩烘湰璁剧疆锛氭潗鏂欑粨鏋?
            "background_material": str(self.background_material),
            "background_material_index": float(self.background_material_index),


            "propagation_material": str(self.propagation_material),
            "propagation_material_index": float(self.propagation_material_index),
            "propagation_thickness_um": float(self.propagation_thickness_um),
            "propagation_center_z_um": float(self.propagation_center_z_um),
            "propagation_span_x_um": None if self.propagation_span_x_um is None else float(self.propagation_span_x_um),
            "propagation_span_y_um": None if self.propagation_span_y_um is None else float(self.propagation_span_y_um),

            "sub_material": str(self.sub_material),
            "sub_material_index": float(self.sub_material_index),
            "sub_thickness_um": float(self.sub_thickness_um),
            "sub_center_z_um": _to_float_optional(getattr(self, "sub_center_z_um", None)),
            "sub_span_x_um": None if self.sub_span_x_um is None else float(self.sub_span_x_um),
            "sub_span_y_um": None if self.sub_span_y_um is None else float(self.sub_span_y_um),


            "slot_material": None if self.slot_material is None else str(self.slot_material),
            "slot_material_index": float(self.slot_material_index),
            "slot_thickness_um": _to_float_optional(self.slot_thickness_um),
            "slot_center_z_um": _to_float_optional(self.slot_center_z_um),
            "slot_span_x_um": _to_float_optional(getattr(self, "slot_span_x_um", None)),
            "slot_span_y_um": _to_float_optional(getattr(self, "slot_span_y_um", None)),

            # 杈撳叆杈撳嚭缁撴瀯
            "waveguide_width_um": float(self.waveguide_width_um),
            "waveguide_length_um": float(self.waveguide_length_um),
            "num_input_waveguides": int(self.num_input_waveguides),
            "num_output_waveguides": int(self.num_output_waveguides),
            "input_centers_y_um": _to_list_float(self.input_centers_y_um),
            "output_centers_y_um": _to_list_float(self.output_centers_y_um),
            "input_ys": _to_list_float(getattr(self, "input_ys", None)),
            "output_ys": _to_list_float(getattr(self, "output_ys", None)),

            "use_taper": bool(self.use_taper),
            "in_taper_length_um": float(self.in_taper_length_um),
            "in_taper_width_um": float(self.in_taper_width_um),
            "out_taper_length_um": float(self.out_taper_length_um),
            "out_taper_width_um": float(self.out_taper_width_um),



            # 琛嶅皠缁撴瀯
            "unit_spacing_um": float(self.unit_spacing_um),
            "duty_cycle": float(self.duty_cycle),
            "unit_span_um": float(self.unit_span_um),
            "fill_material": str(self.fill_material),
            "repeats": int(self.repeats),
            "mapping_factor_class": self.mapping_factor_class,

            "positions_span_um": _to_list_float(self.positions_span_um),
            "wavelength_center": float(self.wavelength_center),
        }

def load_lumerical_modeE(path: Path | str) -> dict:
    """
    Load a 1D mode field exported by Lumerical (MATLAB v7.3/HDF5).
    Returns a dict with Ex, Ey, Ez, z grid and metadata.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mode field file does not exist: {path}")

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

    def _load_matlab_v5_mode() -> dict:
        payload = loadmat(str(path), squeeze_me=True, struct_as_record=False)
        keys = {key for key in payload.keys() if not key.startswith("__")}
        if "E" in payload:
            top_local = payload["E"]
            if hasattr(top_local, "_fieldnames"):
                top_local = {name: getattr(top_local, name) for name in top_local._fieldnames}
        else:
            top_local = {
                "Ex": payload.get("Ex"),
                "Ey": payload.get("Ey"),
                "Ez": payload.get("Ez"),
                "x": payload.get("x", 0.0),
                "y": payload.get("y", 0.0),
                "z": payload.get("z", []),
                "lambda": payload.get("lambda"),
                "f": payload.get("f"),
            }
        if not isinstance(top_local, dict):
            raise RuntimeError(
                "MAT mode file does not contain a readable E struct or Ex/Ey/Ez arrays: "
                f"path={path}, keys={sorted(keys)}"
            )
        return top_local

    try:
        with h5py.File(str(path), "r") as f:
            if "E" in f:
                top = _read_node(f, f["E"])
            else:
                # fallback: direct datasets (Ex/Ey/Ez/x/y/z)
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
    except OSError:
        top = _load_matlab_v5_mode()

    if isinstance(top, dict):
        # prefer assembled E if present; otherwise stack Ex/Ey/Ez
        if "E" in top:
            E = _to_complex(top.get("E", top))
        else:
            Exd = _to_complex(top.get("Ex", None))
            Eyd = _to_complex(top.get("Ey", None))
            Ezd = _to_complex(top.get("Ez", None))
            # expect shape (1, Ny, Nz) or (Ny, Nz); build E -> (1,3,Nz)
            if Exd is None or Eyd is None or Ezd is None:
                raise RuntimeError("Missing Ex/Ey/Ez datasets in mode file")
            Exd = np.squeeze(Exd)
            Eyd = np.squeeze(Eyd)
            Ezd = np.squeeze(Ezd)
            def _collapse(arr):
                if arr.ndim == 3:  # (1,Ny,Nz)
                    return arr
                if arr.ndim == 2:  # (Ny,Nz)
                    return arr[np.newaxis, ...]
                if arr.ndim == 1:  # (Nz,)
                    return arr[np.newaxis, np.newaxis, ...]
                raise RuntimeError(f"Unsupported mode field shape: {arr.shape}")
            Exd = _collapse(Exd)
            Eyd = _collapse(Eyd)
            Ezd = _collapse(Ezd)
            # align Nz dimension
            Nz = Exd.shape[-1]
            if not (Eyd.shape[-1] == Ezd.shape[-1] == Nz):
                raise RuntimeError("Ex/Ey/Ez length mismatch in mode file")
            E = np.stack([Exd[0], Eyd[0], Ezd[0]], axis=0)  # (3,Ny,Nz) -> we will reshape later

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
        raise RuntimeError(f"E length {Ex.shape[-1]} does not match z length {z.shape[0]}")

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
):
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
    path: Path | str,
    *,
    x0: float,
    y,
    z,
    Ex_yz,
    Ey_yz,
    Ez_yz,
    f0: float | None = None,
):
    """
    Save rectilinear Ex/Ey/Ez(y,z) data into a MATLAB v7.3 file that Lumerical import source can read.
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
        return np.asfortranarray(a)

    def _save_real(fh, name, a):
        a = _as_fortran(np.asarray(a, np.float64))
        d = fh.create_dataset(name, data=a)
        _matlab_attrs(d, b"double", is_complex=False)

    def _save_complex(fh, name, a):
        a = np.asarray(a, np.complex128)
        comp = np.empty(a.shape, dtype=[("real", np.float64), ("imag", np.float64)], order="F")
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


class OpticalDONN(nn.Module):
    def __init__(
        self,
        inputposition,
        outputposition,
        positions_span,
        num_layers,
        num_neurons_each_layer,
        wavelength,
        n_slab,
        mapping_factor,
        layer_spacing,
        complex_data_matrices,
        gradornotgrad,
        overlap,
        device="cuda",
        dtype=torch.complex128,
        initcanshu=0,
        *,
        wavelengths_um=None,
        n_slab_by_wavelength_um=None,
        mapping_factor_by_wavelength_um=None,
        monitor_payloads=None,
        wavelength_reduction: str = "mean_power",
    ):
        super(OpticalDONN, self).__init__()
        self.inputposition = inputposition
        self.outputposition = outputposition
        self.num_layers = num_layers
        self.neurons = num_neurons_each_layer
        self.layer_spacing = layer_spacing
        self.positions_span = positions_span
        self.complex_data_matrices = [np.asarray(m).copy() for m in complex_data_matrices]
        self.monitor_payloads = [
            self._copy_monitor_payload(payload)
            for payload in (monitor_payloads if monitor_payloads is not None else complex_data_matrices)
        ]
        self.initcanshu = initcanshu
        self.gradornotgrad = gradornotgrad
        self.overlap = overlap
        self.wavelength_reduction = str(wavelength_reduction)
        self.parameter_semantics = "physical_length_um"
        # default loss type; keep alias for backward compatibility
        self.loss_type = "direct_mse"
        self.device = _resolve_torch_device(device)
        self.dtype = dtype

        requested_selected_wl = float(wavelength)
        wl_grid = (
            np.asarray(wavelengths_um, dtype=float).reshape(-1)
            if wavelengths_um is not None
            else np.asarray([], dtype=float)
        )
        n_slab_by = normalize_wavelength_dict(dict(n_slab_by_wavelength_um or {}))
        mapping_by = normalize_wavelength_dict(dict(mapping_factor_by_wavelength_um or {}))
        if wl_grid.size == 0:
            if n_slab_by:
                wl_grid = sorted_wavelengths_from_dict(n_slab_by)
            elif mapping_by:
                wl_grid = sorted_wavelengths_from_dict(mapping_by)
            else:
                wl_grid = np.asarray([requested_selected_wl], dtype=float)
        if wl_grid.size == 0:
            raise ValueError("OpticalDONN requires at least one wavelength")
        for wl_um in wl_grid.tolist():
            key = canonical_wavelength_key(wl_um)
            n_slab_by.setdefault(key, float(n_slab))
            mapping_by.setdefault(key, float(mapping_factor))
        selected = resolve_wavelength_choice(
            wl_grid,
            wavelength_um=requested_selected_wl,
            selection="nearest",
        )
        self.available_wavelengths_um = [float(v) for v in np.asarray(wl_grid, dtype=float).tolist()]
        self.selected_wavelength_um = float(selected["selected_wavelength_um"])
        self.selected_wavelength_index = int(selected["selected_wavelength_index"])
        self.wavelength = float(self.selected_wavelength_um)
        self.n_slab_by_wavelength_um = dict(n_slab_by)
        self.mapping_factor_by_wavelength_um = dict(mapping_by)
        self.n_slab = float(
            self._n_slab_for_wavelength(wavelength_um=self.selected_wavelength_um)
        )
        self.mapping_factor = float(
            self._mapping_factor_for_wavelength(wavelength_um=self.selected_wavelength_um)
        )
        self.last_output_by_wavelength = {}
        self.last_wavelength_contexts = []
        self.us_by_wavelength = {}
        self.hybrid_last_output_by_wavelength = {}
        self.hybrid_wavelength_contexts = []
        self.hybrid_us_by_wavelength = {}
        self.hybrid_requested_wavelengths_um = []
        self.hybrid_output_wavelengths_um = []

        self.init()
        if len(gradornotgrad) != num_layers:
            raise ValueError("The length of gradornotgrad must match num_layers.")
        if gradornotgrad[0] != 0 or gradornotgrad[-1] != 0:
            raise ValueError("The first and last elements of gradornotgrad must be 0.")

    @staticmethod
    def _copy_monitor_payload(payload):
        if isinstance(payload, dict):
            return {
                str(key): np.asarray(value).copy()
                for key, value in payload.items()
                if value is not None
            }
        return np.asarray(payload).copy()

    @staticmethod
    def _phase_pi_from_physical_length_um(length_um, mapping_factor_um: float):
        mapping = float(mapping_factor_um)
        if mapping == 0.0:
            raise ZeroDivisionError("mapping_factor must be non-zero when converting length to phase")
        return torch.as_tensor(length_um, dtype=torch.float32) / mapping

    @staticmethod
    def _legacy_phase_pi_to_physical_length_um(phase_pi, mapping_factor_um: float):
        mapping = float(mapping_factor_um)
        if mapping == 0.0:
            raise ZeroDivisionError("mapping_factor must be non-zero when converting phase to length")
        phase_tensor = torch.as_tensor(phase_pi, dtype=torch.float32)
        if mapping < 0.0:
            positive_mask = phase_tensor > 0.0
            if torch.any(positive_mask):
                wrapped = phase_tensor[positive_mask]
                phase_tensor = phase_tensor.clone()
                phase_tensor[positive_mask] = wrapped - 2.0 * torch.ceil(wrapped / 2.0)
        else:
            negative_mask = phase_tensor < 0.0
            if torch.any(negative_mask):
                phase_tensor = torch.remainder(phase_tensor, 2.0)
        return phase_tensor * mapping

    def _resolve_wavelength_context(
        self,
        *,
        wavelength_um: float | None = None,
        wavelength_index: int | None = None,
    ) -> Dict[str, float | int]:
        chosen = resolve_wavelength_choice(
            self.available_wavelengths_um,
            wavelength_um=self.selected_wavelength_um if wavelength_um is None else float(wavelength_um),
            wavelength_index=wavelength_index,
            selection="nearest",
        )
        selected_wl_um = float(chosen["selected_wavelength_um"])
        return {
            "selected_wavelength_um": selected_wl_um,
            "selected_wavelength_index": int(chosen["selected_wavelength_index"]),
            "mapping_factor": float(
                self._mapping_factor_for_wavelength(wavelength_um=selected_wl_um)
            ),
            "n_slab": float(self._n_slab_for_wavelength(wavelength_um=selected_wl_um)),
        }

    def _resolve_forward_wavelength_contexts(
        self,
        *,
        wavelength_um: float | None = None,
        wavelength_index: int | None = None,
    ) -> List[Dict[str, float | int]]:
        if wavelength_um is not None or wavelength_index is not None or len(self.available_wavelengths_um) <= 1:
            return [
                self._resolve_wavelength_context(
                    wavelength_um=wavelength_um,
                    wavelength_index=wavelength_index,
                )
            ]
        return [
            self._resolve_wavelength_context(wavelength_index=int(idx))
            for idx in range(len(self.available_wavelengths_um))
        ]

    def _mapping_factor_for_wavelength(
        self,
        *,
        wavelength_um: float | None = None,
        wavelength_index: int | None = None,
    ) -> float:
        return float(
            select_value_by_wavelength(
                self.mapping_factor_by_wavelength_um,
                wavelength_um=self.selected_wavelength_um if wavelength_um is None else float(wavelength_um),
                wavelength_index=wavelength_index,
                selection="nearest",
            )["value"]
        )

    def _n_slab_for_wavelength(
        self,
        *,
        wavelength_um: float | None = None,
        wavelength_index: int | None = None,
    ) -> float:
        return float(
            select_value_by_wavelength(
                self.n_slab_by_wavelength_um,
                wavelength_um=self.selected_wavelength_um if wavelength_um is None else float(wavelength_um),
                wavelength_index=wavelength_index,
                selection="nearest",
            )["value"]
        )

    def _selected_parameter_length_scale_um(self) -> float:
        scale = abs(self._mapping_factor_for_wavelength(wavelength_um=self.selected_wavelength_um))
        if scale <= 0.0:
            raise ValueError(
                "Selected mapping_factor must be non-zero to initialize physical length parameters"
            )
        return float(scale)

    @property
    def slot_length_um(self):
        return self._slot_length_um

    @property
    def delta_phi(self):
        return self.get_delta_phi_by_wavelength(wavelength_um=self.selected_wavelength_um)

    # keep old attribute name as alias
    @property
    def lootype(self):
        return self.loss_type

    @lootype.setter
    def lootype(self, value):
        self.loss_type = value

    def get_delta_phi_by_wavelength(
        self,
        *,
        wavelength_um: float | None = None,
        wavelength_index: int | None = None,
    ) -> List[torch.Tensor]:
        mapping_factor = self._mapping_factor_for_wavelength(
            wavelength_um=wavelength_um,
            wavelength_index=wavelength_index,
        )
        return [
            torch.as_tensor(param) / float(mapping_factor)
            for param in self.slot_length_um
        ]

    def check_all_params(self):
        for idx, param in enumerate(self.slot_length_um):
            print(f"Physical length parameter ({idx}) [um]: {param.data}")
        return self.slot_length_um

    def getcentermean(self,a):
        return np.mean(a[int(len(a))//3:2*int(len(a)//3)])#[int(len(a))//3:2*int(len(a)//3)]

    def getcaiyangdian(self):
        self.caiyangdian=self.complex_data_matrices[0].shape
        print(self.caiyangdian)

    def init_parameters(self):
        length_scale_um = self._selected_parameter_length_scale_um()

        if self.initcanshu == 0:
            init_function = lambda n, dtype=torch.float32: torch.zeros(n, dtype=dtype)
        elif self.initcanshu == 1:
            init_function = lambda n, dtype=torch.float32: torch.full((n,), length_scale_um, dtype=dtype)
        elif self.initcanshu == 0.5:
            init_function = lambda n, dtype=torch.float32: 2.0 * length_scale_um * torch.rand(n, dtype=dtype)
        elif self.initcanshu == "half_split":
            def init_function(n, dtype=torch.float32):
                half = n // 2
                active = length_scale_um * torch.rand(half, dtype=dtype)
                zeros = torch.zeros(n - half, dtype=dtype)
                return torch.cat((active, zeros), dim=0)
        else:
            raise ValueError("initcanshu must be 0, 0.5, 1, or 'half_split'")

        if hasattr(self, "_slot_length_um") and len(getattr(self, "_slot_length_um", [])) > 0:
            param_device = self._slot_length_um[0].device
        else:
            try:
                param_device = torch.device(self.device)
            except Exception:
                param_device = torch.device("cpu")
        dtype = torch.float32

        self._slot_length_um = nn.ParameterList([
            nn.Parameter(torch.zeros(n, dtype=dtype, device=param_device), requires_grad=bool(grad))
            if i == 0 or i == len(self.neurons) - 1 else
            nn.Parameter(init_function(n, dtype=dtype).to(param_device), requires_grad=bool(grad))
            for i, (n, grad) in enumerate(zip(self.neurons, self.gradornotgrad))
        ])

    def init(self):
        print("donn inited")
        self.getcaiyangdian()
        self.setuplocations()
        self.setuphs(mode="bandlimit",pad_factor=4)
        self.init_parameters()

    def get_parameters(self):
        params = {}
        for i, param in enumerate(self.slot_length_um):
            params[f"layer_{i}_slot_length_um"] = param.clone().detach()
            params[f"layer_{i}_delta_phi"] = self.delta_phi[i].clone().detach()
        return params

    def set_parameters(self, params):
        selected_mapping = self._mapping_factor_for_wavelength(wavelength_um=self.selected_wavelength_um)
        for i, param in enumerate(self.slot_length_um):
            slot_key = f"layer_{i}_slot_length_um"
            phase_key = f"layer_{i}_delta_phi"
            if slot_key in params:
                new_param_data = params[slot_key]
            elif phase_key in params:
                new_param_data = self._legacy_phase_pi_to_physical_length_um(
                    params[phase_key],
                    selected_mapping,
                )
            else:
                continue
            if not isinstance(new_param_data, torch.Tensor):
                raise TypeError("Expected torch.Tensor types in the input parameter dictionary.")
            param.data = new_param_data.clone().detach().to(device=param.device, dtype=param.dtype)
    def setuplocations(self):
        positions = []
        for span in self.positions_span:
            x = np.linspace(-0.5 * span, 0.5 * span, np.array(self.caiyangdian)[0])
            positions.append(x)
        self.positions = positions
    def setupWs(self):
        Ws = []
        for i, layer_spacing in enumerate(self.layer_spacing):
            W0   = self.getW(self.positions[i], self.positions[i+1], layer_spacing, 1, 0)
            beta = self.get_beta(W0, i, i+1)
            dert = self.get_derta(W0, i, i+1)
            W    = self.getW(self.positions[i], self.positions[i+1], layer_spacing, beta, dert)
            Ws.append(torch.as_tensor(W, dtype=torch.complex64))   # 缁熶竴澶嶆暟绮惧害
        Ws = torch.stack(Ws, dim=0)                                # [num_layers-1, N, N]
        self.register_buffer("Ws", Ws)     



    def transmission_coefficient(
        self,
        layer_idx,
        *,
        wavelength_um: float | None = None,
        wavelength_index: int | None = None,
    ):
        mapping_factor = self._mapping_factor_for_wavelength(
            wavelength_um=wavelength_um,
            wavelength_index=wavelength_index,
        )
        phase_pi = self._phase_pi_from_physical_length_um(
            self.slot_length_um[layer_idx],
            mapping_factor,
        ).to(device=self.slot_length_um[layer_idx].device, dtype=torch.float32)
        phase_shifts = torch.exp(1j * torch.pi * phase_pi)

        repeat_times =int( self.caiyangdian [0]// len(self.slot_length_um[layer_idx]))

        # 浣跨敤repeat_interleave鏉ラ噸澶嶅厓绱?
        extended_phase_shifts = phase_shifts.repeat_interleave(repeat_times)

        # 濡傛灉extended_phase_shifts鐨勯暱搴﹀ぇ浜巆aiyangdian锛岃繘琛岃皟鏁?
        if len(extended_phase_shifts) > self.caiyangdian[0]:
            # 鍘绘帀澶氫綑鐨勯儴鍒嗭紝閫氬父浠庡紑澶存垨缁撳熬鍒囩墖
            extended_phase_shifts = extended_phase_shifts[:self.caiyangdian[0]]
        elif len(extended_phase_shifts) < self.caiyangdian[0]:
            # 濡傛灉闀垮害涓嶈冻锛岃ˉ鍏呮渶鍚庝竴涓厓绱犵洿鍒拌揪鍒癱aiyangdian鐨勯暱搴?
            deficit = self.caiyangdian[0] - len(extended_phase_shifts)
            last_element = extended_phase_shifts[-1].unsqueeze(0).repeat(deficit)
            extended_phase_shifts = torch.cat((extended_phase_shifts, last_element), dim=0)

        return extended_phase_shifts


    def WnptoWtorch(self):
        torch_matrices = []
        for W in self.Ws:
            # 灏?NumPy 鐨勫鏁扮煩闃佃浆鎹负 PyTorch 澶嶆暟寮犻噺
            torch_matrix = torch.tensor(W, dtype=torch.complex64)
            torch_matrices.append(torch_matrix)
        self.Ws=torch_matrices
    def build_phase_mask(self):
        """鎶婇€愰€氶亾鐩镐綅鍐欏叆鏀规垚涓€娆?GEMM锛歱hase = exp(1j * (X*pi) @ mask)"""
        S = len(self.inputposition)
        N = self.complex_data_matrices[0].shape[0]
        mask = torch.zeros(S, N, dtype=torch.float32)
        span_um = float(self.positions_span[0])
        dy_um = span_um / float(self.caiyangdian[0])
        input_positions = self._positions_in_internal_axis(self.inputposition, span_um)
        input_mode = "centered"
        for i in range(S):
            pos = self._pos_to_index(float(input_positions[i]), span_um, dy_um, N, input_mode)
            half = int(self.waveguide_maping_span * 0.5 * (self.caiyangdian[0] / span_um))
            s, e = max(0, pos - half), min(N, pos + half)
            if s < e:
                mask[i, s:e] = 1.0  
        self.register_buffer("phase_mask", mask)  # [S,N]

    def build_Ws_T(self):
        """棰勫瓨杞疆鐗堟湰锛屽噺灏戝墠鍚戞椂鐨勮浆缃?姝ュ箙淇寮€閿€"""
        assert hasattr(self, "Ws"), "call setupWs() first"
        self.register_buffer("Ws_T", self.Ws.transpose(-1, -2).contiguous())  # [L,N,N]

    def _infer_coord_mode(self, positions, L):
        """杩斿洖 'zero_based' 鎴?'centered'"""
        if len(positions) == 0:
            return "centered"
        arr = np.asarray(positions, dtype=np.float64).reshape(-1)
        tol = max(1.0e-9, 1.0e-6 * max(abs(float(L)), 1.0))
        mn = float(np.min(arr))
        mx = float(np.max(arr))
        if mn < -tol:
            return "centered"
        if mx > float(L) * 0.5 + tol:
            return "zero_based"
        centered_distance = float(np.mean(np.abs(arr)))
        zero_based_distance = float(np.mean(np.abs(arr - 0.5 * float(L))))
        if zero_based_distance < centered_distance:
            return "zero_based"
        return "centered"

    def _pos_to_index(self, pos, L, dy, N, mode):
        """鐗╃悊鍧愭爣 -> 鏍呮牸绱㈠紩锛?..N-1锛夛紝骞?clip"""
        p = float(pos)
        if mode == "zero_based":
            idx = int(round(p / dy))
        else:
            idx = int(round((p + 0.5 * L) / dy))
        if idx < 0: idx = 0
        if idx > N - 1: idx = N - 1
        return idx

    def _positions_in_internal_axis(self, positions, span_um):
        arr = np.asarray(positions, dtype=np.float64).reshape(-1)
        mode = self._infer_coord_mode(arr.tolist(), float(span_um))
        if mode == "zero_based":
            return arr - 0.5 * float(span_um)
        return arr.copy()

    def _set_buffer_tensor(self, name: str, value: torch.Tensor) -> None:
        if name in self._buffers:
            self._buffers[name] = value
        else:
            self.register_buffer(name, value)

    def _resolve_monitor_tensor(
        self,
        monitor_idx: int,
        *,
        wavelength_um: float,
        device: torch.device,
    ) -> torch.Tensor:
        ey, _ = self._resolve_monitor_line_and_axis(
            monitor_idx,
            wavelength_um=wavelength_um,
        )
        return torch.as_tensor(ey, device=device, dtype=torch.complex64)

    def _resolve_monitor_line_and_axis(
        self,
        monitor_idx: int,
        *,
        wavelength_um: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.monitor_payloads:
            raise ValueError("monitor_payloads is empty")
        payload = self.monitor_payloads[min(monitor_idx, len(self.monitor_payloads) - 1)]
        y_axis_um = np.asarray([], dtype=float)
        if isinstance(payload, dict):
            try:
                picked = extract_ey_1d_at_wavelength(
                    payload,
                    wavelength_um=float(wavelength_um),
                    selection="nearest",
                )
                ey = np.asarray(picked["ey"])
                y_axis_um = np.asarray(picked.get("y", []), dtype=float).reshape(-1)
            except Exception:
                if "ey" in payload:
                    ey = np.asarray(payload["ey"])
                elif "Ey" in payload:
                    ey = np.asarray(payload["Ey"])
                else:
                    raise
                if "y_um" in payload:
                    y_axis_um = np.asarray(payload["y_um"], dtype=float).reshape(-1)
                elif "y" in payload:
                    y_axis_um = np.asarray(payload["y"], dtype=float).reshape(-1)
        else:
            ey = np.asarray(payload)
        ey = np.asarray(ey).squeeze()
        if ey.ndim != 1:
            raise ValueError(
                f"Resolved monitor line must be 1D, got shape {ey.shape} at wavelength_um={wavelength_um}"
            )
        if y_axis_um.size != ey.shape[0]:
            span_um = float(self.positions_span[0])
            y_axis_um = np.linspace(-0.5 * span_um, 0.5 * span_um, ey.shape[0], dtype=float)
        return ey.reshape(-1).copy(), y_axis_um

    def _resolve_rectangular_phase_support(
        self,
        *,
        y_axis_um: np.ndarray,
        center_y_um: float,
    ) -> tuple[int, int]:
        y_axis_um = np.asarray(y_axis_um, dtype=float).reshape(-1)
        if y_axis_um.size == 0:
            raise ValueError("y_axis_um must not be empty")
        if y_axis_um.size == 1:
            return 0, 1
        dy_um = float(np.mean(np.abs(np.diff(y_axis_um))))
        if dy_um <= 0.0:
            return 0, y_axis_um.size
        half = int(round(0.5 * float(self.waveguide_maping_span) / dy_um))
        center_idx = int(np.argmin(np.abs(y_axis_um - float(center_y_um))))
        i0 = max(0, center_idx - half)
        i1 = min(y_axis_um.size, center_idx + half + 1)
        if i0 >= i1:
            return center_idx, min(center_idx + 1, y_axis_um.size)
        return i0, i1

    def _build_input_phase_mask(
        self,
        *,
        base_line: torch.Tensor | np.ndarray,
        y_axis_um: np.ndarray,
        x: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        y_axis = np.asarray(y_axis_um, dtype=float).reshape(-1)
        amplitude = np.abs(np.asarray(base_line).reshape(-1))
        if amplitude.ndim != 1:
            raise ValueError(f"base_line must be 1D, got shape={amplitude.shape}")
        if y_axis.size != amplitude.size:
            raise ValueError(
                f"y_axis_um size mismatch: y_axis={y_axis.size}, amplitude={amplitude.size}"
            )

        input_positions_raw = np.asarray(self.inputposition, dtype=np.float64).reshape(-1)
        input_positions = self._positions_in_internal_axis(
            input_positions_raw,
            float(self.positions_span[0]),
        )
        if input_positions.size != x.shape[1]:
            raise ValueError(
                f"inputposition size mismatch: positions={input_positions.size}, input_dim={x.shape[1]}"
            )

        support_threshold = float(getattr(self, "input_phase_support_threshold", 0.05))
        mask = torch.ones((x.shape[0], amplitude.size), device=device, dtype=torch.complex64)
        support_summary = [None] * int(input_positions.size)

        order = np.argsort(input_positions)
        sorted_positions = input_positions[order]
        axis_min = float(np.min(y_axis))
        axis_max = float(np.max(y_axis))

        for rank, input_idx in enumerate(order.tolist()):
            center_y_um = float(input_positions[input_idx])
            left_bound_um = axis_min if rank == 0 else 0.5 * float(sorted_positions[rank - 1] + center_y_um)
            right_bound_um = axis_max if rank == len(order) - 1 else 0.5 * float(center_y_um + sorted_positions[rank + 1])

            partition = np.where((y_axis >= left_bound_um) & (y_axis <= right_bound_um))[0]
            if partition.size == 0:
                i0, i1 = self._resolve_rectangular_phase_support(
                    y_axis_um=y_axis,
                    center_y_um=center_y_um,
                )
                source = "rectangular_fallback_empty_partition"
            else:
                local_amp = amplitude[partition]
                local_peak = float(np.max(local_amp))
                if local_peak <= 0.0:
                    i0, i1 = self._resolve_rectangular_phase_support(
                        y_axis_um=y_axis,
                        center_y_um=center_y_um,
                    )
                    source = "rectangular_fallback_zero_field"
                else:
                    center_local = int(np.argmin(np.abs(y_axis[partition] - center_y_um)))
                    active = local_amp >= (support_threshold * local_peak)
                    active_indices = np.flatnonzero(active)
                    if active_indices.size == 0:
                        i0, i1 = self._resolve_rectangular_phase_support(
                            y_axis_um=y_axis,
                            center_y_um=center_y_um,
                        )
                        source = "rectangular_fallback_below_threshold"
                    else:
                        if not bool(active[center_local]):
                            nearest_rank = int(np.argmin(np.abs(active_indices - center_local)))
                            center_local = int(active_indices[nearest_rank])
                        left_local = center_local
                        right_local = center_local
                        while left_local > 0 and active[left_local - 1]:
                            left_local -= 1
                        while right_local + 1 < active.size and active[right_local + 1]:
                            right_local += 1
                        i0 = int(partition[left_local])
                        i1 = int(partition[right_local]) + 1
                        source = "monitor_field_support"

            phase = torch.exp(1j * torch.pi * x[:, input_idx]).to(torch.complex64)
            mask[:, i0:i1] *= phase[:, None]
            support_summary[input_idx] = {
                "input_index": int(input_idx),
                "center_y_um": center_y_um,
                "center_y_absolute_um": float(input_positions_raw[input_idx]),
                "start_y_um": float(y_axis[i0]),
                "end_y_um": float(y_axis[max(i0, i1 - 1)]),
                "width_um": float(abs(y_axis[max(i0, i1 - 1)] - y_axis[i0])),
                "source": source,
                "threshold_ratio": support_threshold,
            }

        self.last_input_phase_support_summary = support_summary
        return mask

    def _get_hs_for_wavelength(
        self,
        *,
        wavelength_um: float | None = None,
        wavelength_index: int | None = None,
    ) -> torch.Tensor:
        context = self._resolve_wavelength_context(
            wavelength_um=wavelength_um,
            wavelength_index=wavelength_index,
        )
        return self.Hs_multi[int(context["selected_wavelength_index"])]

    def _get_hybrid_hs_for_wavelength(
        self,
        *,
        wavelength_um: float | None = None,
        wavelength_index: int | None = None,
    ) -> torch.Tensor:
        context = self._resolve_wavelength_context(
            wavelength_um=wavelength_um,
            wavelength_index=wavelength_index,
        )
        hs_multi = getattr(self, "Hs_H_multi", None)
        if hs_multi is None:
            hs_single = getattr(self, "Hs_H", None)
            if hs_single is None:
                raise AttributeError("Call setuphs_hybride before requesting hybrid propagators")
            return hs_single
        return hs_multi[int(context["selected_wavelength_index"])]

    def setuphs(self,
                mode="full",
                ky_cut_ratio=1.0,
                coarse_factor=1.0,
                smooth_rolloff=0.0,
                pad_factor=16):
        """
        pad_factor: 闆跺～鍏呭€嶆暟锛?=2锛屾帹鑽?4锛?
        """

        # ---------- N0锛氱墿鐞嗙獥鍙ｉ暱搴︼紙蹇呴』涓?forward 涓€鑷达級 ----------
        if hasattr(self, "complex_data_matrices") and len(self.complex_data_matrices) > 0:
            N0 = int(np.asarray(self.complex_data_matrices[0]).shape[-1])
        else:
            N0 = int(self.caiyangdian[0])

        pad_factor = int(max(1, pad_factor))
        Np = int(pad_factor * N0)


        self.N0 = N0
        self.Np = Np
        self.pad_start = (Np - N0) // 2
        print( "self.N0 = N0self.Np = Np self.pad_start", self.N0,self.Np ,self.pad_start)
        # ---------- 鍗曚綅缁熶竴锛氬亣璁?span/wavelength/layer_spacing 閮芥槸 um ----------
        span_um = float(self.positions_span[0])


        # FFT 鍛ㄦ湡缃戞牸锛歞y = span/N0锛堜笌 forward 鐨?index 鏄犲皠淇濇寔涓€鑷达級
        dy = span_um /(N0) 
        self.dy = dy
        print("dy",dy)

        ky = 2 * np.pi * np.fft.fftfreq(Np, d=dy)
        ky_abs = np.abs(ky)

        # device 鍏滃簳
        dev = getattr(self, "device", None)
        if dev is None:
            try:
                dev = next(self.parameters()).device
            except Exception:
                dev = torch.device("cpu")

        hs_multi = []
        contexts = self._resolve_forward_wavelength_contexts()
        ky_nyq = np.pi / dy
        ky_nyq_eff = ky_nyq / float(coarse_factor) if mode == "coarse" else ky_nyq
        for context in contexts:
            k = 2 * np.pi * float(context["n_slab"]) / float(context["selected_wavelength_um"])
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

            hs_single = []
            for i in range(self.num_layers - 1):
                d = float(self.layer_spacing[i])

                if mode in ("bandlimit", "coarse"):
                    kx = np.sqrt(np.maximum(k * k - ky * ky, 0.0))
                    H = np.exp(1j * kx * d) * filt
                elif mode == "full":
                    kx = np.sqrt((k * k - ky * ky).astype(np.complex128))
                    H = np.exp(1j * kx * d)
                else:
                    raise ValueError(f"Unknown mode={mode}")

                hs_single.append(torch.as_tensor(H, dtype=torch.complex64, device=dev))

            hs_multi.append(torch.stack(hs_single, dim=0))

        hs_multi_tensor = torch.stack(hs_multi, dim=0)
        self._set_buffer_tensor("Hs_multi", hs_multi_tensor)
        self._set_buffer_tensor(
            "Hs",
            hs_multi_tensor[int(self.selected_wavelength_index)].clone(),
        )
        self._set_buffer_tensor(
            "wavelength_grid_buffer_um",
            torch.as_tensor(self.available_wavelengths_um, dtype=torch.float64, device=dev),
        )

    def build_propagator_kernel(
        self,
        d_um: float,
        *,
        ky=None,
        wavelength_um: float | None = None,
        n_slab: float | None = None,
        mode: str = "full",
        ky_cut_ratio: float = 1.0,
        coarse_factor: float = 1.0,
        smooth_rolloff: float = 0.0,
        pad_factor: int = 1,
        return_ky: bool = False,
        as_tensor: bool = False,
        device=None,
        dtype=None,
    ):
        """
        Build a 1D angular-spectrum propagator H that mirrors setuphs().
        Useful for fine-step visualizations outside forward() while keeping
        the same bandlimit and evanescent handling as the model.
        """
        if ky is None:
            if hasattr(self, "complex_data_matrices") and len(self.complex_data_matrices) > 0:
                N0 = int(np.asarray(self.complex_data_matrices[0]).shape[-1])
            else:
                N0 = int(self.caiyangdian[0])

            pad_factor = int(max(1, pad_factor))
            Np = int(pad_factor * N0)
            if Np % 2 == 1:
                Np += 1

            span_um = float(self.positions_span[0])
            dy = (span_um * 1e-6) / N0
            ky = 2 * np.pi * np.fft.fftfreq(Np, d=dy)
        else:
            ky = np.asarray(ky, dtype=np.float64)
            Np = ky.size
            if Np < 2:
                raise ValueError("ky must have at least 2 samples")

            dk = float(np.abs(ky[1] - ky[0]))
            dy = np.pi / dk if dk > 0 else 0.0

        ky_abs = np.abs(ky)

        resolved_wavelength_um = float(
            self.selected_wavelength_um if wavelength_um is None else wavelength_um
        )
        resolved_n_slab = float(
            self._n_slab_for_wavelength(wavelength_um=resolved_wavelength_um)
            if n_slab is None
            else n_slab
        )
        k = 2 * np.pi * resolved_n_slab / (resolved_wavelength_um * 1e-6)
        ky_nyq = np.pi / dy if dy > 0 else ky_abs.max()
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

        d_m = float(d_um) * 1e-6
        if mode in ("bandlimit", "coarse"):
            kx = np.sqrt(np.maximum(k * k - ky * ky, 0.0))
            H = np.exp(1j * kx * d_m) * filt
        elif mode == "full":
            kx = np.sqrt((k * k - ky * ky).astype(np.complex128))
            H = np.exp(1j * kx * d_m)
            absH = np.abs(H)
            bad = absH > 1.0
            if np.any(bad):
                H[bad] /= absH[bad]
        else:
            raise ValueError(f"Unknown mode={mode}")

        if as_tensor:
            dev = device if device is not None else getattr(self, "device", torch.device("cpu"))
            dt = torch.complex64 if dtype is None else dtype
            H_out = torch.as_tensor(H, dtype=dt, device=dev)
            return (H_out, ky) if return_ky else H_out

        return (H, ky) if return_ky else H


    def plotallfeilds(self, step_um: float = 0.01, input_phases=None, device=None, show: bool = True):
        """
        Visualize angular-spectrum propagation between layers with fine z steps.
        Returns a dict with raw slices, maps, axes, and matplotlib figures.
        """
        import matplotlib.pyplot as plt

        dev = device if device is not None else self.slot_length_um[0].device

        if input_phases is None:
            x_vec = torch.zeros((1, len(self.inputposition)), dtype=torch.float32, device=dev)
        else:
            x_vec = torch.as_tensor(input_phases, dtype=torch.float32, device=dev)
            if x_vec.ndim == 1:
                x_vec = x_vec.unsqueeze(0)

        self.eval()
        with torch.no_grad():
            _ = self.forward(x_vec)

        if len(self.us) < 1:
            raise RuntimeError("forward() did not populate fields (us).")

        N = self.us[0].shape[-1]
        span_um = float(self.positions_span[0])
        dy = span_um * 1e-6 / N
        ky = 2 * np.pi * np.fft.fftfreq(N, d=dy)

        def build_H_local(d):
            return np.asarray(self.build_propagator_kernel(
                d,
                ky=ky,
                mode="full",
                ky_cut_ratio=1.0,
                smooth_rolloff=0.0,
                coarse_factor=1.0,
                pad_factor=1,
                as_tensor=False,
            ))

        slices = []
        z_positions = []
        current_z = 0.0
        layer_spacings = np.asarray(self.layer_spacing, dtype=float)

        for layer_idx in range(len(self.us) - 1):
            u_layer = self.us[layer_idx].detach().cpu().numpy()
            if u_layer.ndim == 2:
                u_layer = u_layer[0]
            T = self.transmission_coefficient(layer_idx).detach().cpu().numpy()
            u_layer = u_layer * T

            remaining = float(layer_spacings[layer_idx]) if layer_idx < layer_spacings.size else 0.0
            while remaining > 1e-9:
                d = min(step_um, remaining)
                H = build_H_local(d)
                u0 = np.fft.ifftshift(u_layer, axes=-1)
                U = np.fft.fft(u0, axis=-1)
                u1 = np.fft.ifft(U * H, axis=-1)
                u1 = np.fft.fftshift(u1, axes=-1)

                slices.append(u1.squeeze())
                current_z += d
                z_positions.append(current_z)

                u_layer = u1
                remaining -= d

        if not slices:
            raise RuntimeError("No slices produced; check layer_spacing or inputs.")

        slices_arr = np.stack(slices, axis=0)
        amp_map = np.abs(slices_arr)
        phase_map = np.angle(slices_arr)

        y_axis = np.linspace(-0.5 * span_um, 0.5 * span_um, N)

        fig_amp, ax_amp = plt.subplots(figsize=(10, 5))
        im1 = ax_amp.imshow(amp_map, aspect="auto", extent=[y_axis[0], y_axis[-1], 0, z_positions[-1]], origin="lower")
        fig_amp.colorbar(im1, ax=ax_amp, label="|E|")
        ax_amp.set_xlabel("y (um)")
        ax_amp.set_ylabel("z (um)")
        ax_amp.set_title(f"Angular-spectrum amplitude map (step={step_um} um)")
        fig_amp.tight_layout()

        fig_phase, ax_phase = plt.subplots(figsize=(10, 5))
        im2 = ax_phase.imshow(phase_map, aspect="auto", extent=[y_axis[0], y_axis[-1], 0, z_positions[-1]], origin="lower", cmap="twilight")
        fig_phase.colorbar(im2, ax=ax_phase, label="phase (rad)")
        ax_phase.set_xlabel("y (um)")
        ax_phase.set_ylabel("z (um)")
        ax_phase.set_title(f"Angular-spectrum phase map (step={step_um} um)")
        fig_phase.tight_layout()

        if show:
            plt.show(block=False)

        return None
    def _run_single_wavelength_forward(self, x: torch.Tensor, *, context: Mapping[str, float | int]):
        dev = self.slot_length_um[0].device
        x = torch.as_tensor(x, device=dev, dtype=torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        B = x.shape[0]

        wl_um = float(context["selected_wavelength_um"])
        base = self._resolve_monitor_tensor(0, wavelength_um=wl_um, device=dev)
        N = base.shape[-1]

        N0 = int(getattr(self, "N0", N))
        Np = int(getattr(self, "Np", N))
        ps = int(getattr(self, "pad_start", 0))
        assert N == N0, f"N mismatch: forward N={N}, setuphs N0={N0}, wavelength_um={wl_um}"
        assert self.Hs_multi.shape[-1] == Np, f"Hs length mismatch: Hs={self.Hs_multi.shape[-1]}, Np={Np}"

        def pad_center(u, fill=0.0):
            if Np == N:
                return u
            fillv = torch.as_tensor(fill, device=u.device, dtype=u.dtype)
            out = torch.full((u.shape[0], Np), fillv, dtype=u.dtype, device=u.device)
            out[:, ps:ps + N] = u
            return out

        def pad_center_1d(v, fill=1.0):
            if Np == N:
                return v
            fillv = torch.as_tensor(fill, device=v.device, dtype=v.dtype)
            out = torch.full((Np,), fillv, dtype=v.dtype, device=v.device)
            out[ps:ps + N] = v
            return out

        def crop_center(u):
            if Np == N:
                return u
            return u[:, ps:ps + N]

        _, base_y_axis_um = self._resolve_monitor_line_and_axis(
            0,
            wavelength_um=wl_um,
        )
        mask = self._build_input_phase_mask(
            base_line=base.detach().cpu().numpy(),
            y_axis_um=base_y_axis_um,
            x=x,
            device=dev,
        )

        fields = []
        for idx in range(self.num_layers):
            src = self._resolve_monitor_tensor(
                idx,
                wavelength_um=wl_um,
                device=dev,
            )
            if src.ndim == 1:
                src = src.unsqueeze(0).expand(B, -1)
            fields.append(src)

        fields[0] = base.unsqueeze(0).expand(B, -1) * mask
        hs = self._get_hs_for_wavelength(wavelength_um=wl_um).to(dev)
        u = pad_center(fields[0], fill=0.0)
        for l in range(1, self.num_layers):
            T_space = self.transmission_coefficient(
                l - 1,
                wavelength_um=wl_um,
            ).to(dev)
            T_pad = pad_center_1d(T_space, fill=1.0)
            u = u * T_pad
            U = torch.fft.fft(torch.fft.ifftshift(u, dim=-1), dim=-1)
            U = U * hs[l - 1]
            u = torch.fft.fftshift(torch.fft.ifft(U, dim=-1), dim=-1)
            fields[l] = crop_center(u)

        response, _ = self.coupled_output_overlap(
            overlap_obj=self.overlap,
            ey=fields[-1],
            wavelength_m=wl_um * 1e-6,
            n_medium=float(context["n_slab"]),
            origin="center",
        )
        return {
            "response": response,
            "fields": fields,
            "context": dict(context),
        }

    def forward(self, x, *, wavelength_um: float | None = None, wavelength_index: int | None = None):
        with torch.cuda.amp.autocast(enabled=False):
            dev = self.slot_length_um[0].device
            x = torch.as_tensor(x, device=dev, dtype=torch.float32)
            if x.ndim == 1:
                x = x.unsqueeze(0)

            contexts = self._resolve_forward_wavelength_contexts(
                wavelength_um=wavelength_um,
                wavelength_index=wavelength_index,
            )
            responses = []
            representative = None
            representative_key = canonical_wavelength_key(self.selected_wavelength_um)
            self.last_output_by_wavelength = {}
            self.last_wavelength_contexts = [dict(context) for context in contexts]
            self.us_by_wavelength = {}
            self.last_output_wavelengths_um = [
                float(context["selected_wavelength_um"]) for context in contexts
            ]

            for context in contexts:
                single = self._run_single_wavelength_forward(x, context=context)
                response = torch.as_tensor(single["response"], device=dev, dtype=torch.float32)
                if response.ndim == 1:
                    response = response.unsqueeze(0)
                responses.append(response)

                key = canonical_wavelength_key(float(context["selected_wavelength_um"]))
                self.last_output_by_wavelength[key] = response.detach().cpu()
                self.us_by_wavelength[key] = single["fields"]
                if representative is None or key == representative_key:
                    representative = single

            if representative is None:
                representative = single
            self.us = representative["fields"]
            self.us_H = self.us
            self.us_operating_point = representative["context"]

            response_stack = torch.stack(responses, dim=1)
            if response_stack.shape[1] == 1:
                single_response = response_stack[:, 0, :]
                if single_response.shape[0] == 1:
                    single_response = single_response.squeeze(0)
                return single_response
            return response_stack
   


    def _interp_complex_np(self,x_src, f_src, x_tgt):
        # x_src: (N,), f_src: (N,) complex, x_tgt: (M,)
        fr = np.interp(x_tgt, x_src, np.real(f_src))
        fi = np.interp(x_tgt, x_src, np.imag(f_src))
        return fr + 1j * fi

    def coupled_output_overlap(
        self,
        overlap_obj,
        *,
        ey=None,
        wavelength_m: float,
        n_medium: float,
        origin: str = "left",
    ):
        dev = self.slot_length_um[0].device
        if ey is None:
            ey = self.us[-1]
        ey = torch.as_tensor(ey, device=dev)
        if ey.ndim == 1:
            ey = ey.unsqueeze(0)
        B, N = ey.shape
        if N < 2:
            raise ValueError("Ey must have at least 2 samples")

        # --- 鍏ㄥ眬 y 杞达細瀵归綈 forward/setuphs锛歞y = span/N锛屽尯闂?[y0, y0+span) ---
        span_m = float(self.positions_span[-1]) * 1e-6  # um->m
        if origin == "left":
            y0 = 0.0
        elif origin == "center":
            y0 = -0.5 * span_m
        else:
            raise ValueError("origin must be 'left' or 'center'")
        dy = span_m / float(N)  # 鍏抽敭锛氫笉鏄?(N-1)

        # mode 缃戞牸锛堝眬閮ㄥ潗鏍囷紝鍗曚綅 m锛?
        y_mode = torch.as_tensor(overlap_obj.y_m, dtype=torch.float64, device=dev)          # (Nm,)
        Ey_mode = torch.as_tensor(overlap_obj.Ey_m, dtype=torch.complex128, device=dev)     # (Nm,)

        outpos_m = torch.as_tensor(
            self._positions_in_internal_axis(self.outputposition, float(self.positions_span[-1])),
            dtype=torch.float64,
            device=dev,
        ) * 1e-6  # (P,)

        def sample_field(Ey_batch: torch.Tensor, center_m: torch.Tensor) -> torch.Tensor:
            """
            Ey_batch: (B,N) complex
            return:   (B,Nm) complex锛屽彇鍏ㄥ眬鍦哄湪 y = center + y_mode 鐨勭嚎鎬ф彃鍊?
            """
            # 鏌ヨ鐐癸紙鍏ㄥ眬鍧愭爣锛?
            yq = center_m + y_mode  # (Nm,)

            # 杞垚鈥滄牱鏈储寮曞潗鏍団€濓細y = y0 + idx*dy
            idx_f = (yq - y0) / dy  # (Nm,)

            # 涓轰繚璇?idx1 涓嶈秺鐣岋細鎶?idx0 闄愬埗鍦?[0, N-2]
            idx0 = torch.floor(idx_f).to(torch.int64)
            idx0 = idx0.clamp(0, N - 2)
            idx1 = idx0 + 1

            # 鎻掑€兼潈閲嶏細w = idx_f - idx0锛堟敞鎰?idx0 鐢ㄦ湭 clamp 鍓嶇殑浼氬嚭浜嬶紝鎵€浠ョ敤 clamp 鍚庣殑 idx0_f锛?
            idx0_f = idx0.to(torch.float64)
            w = (idx_f - idx0_f).clamp(0.0, 1.0).to(Ey_batch.real.dtype)  # (Nm,)

            # gather
            Ey0 = Ey_batch[:, idx0]  # (B,Nm)
            Ey1 = Ey_batch[:, idx1]  # (B,Nm)
            return (1.0 - w.unsqueeze(0)) * Ey0 + w.unsqueeze(0) * Ey1

        Ey_batched = ey.to(torch.complex128)  # (B,N)

        p_cols = []
        # 绂佺敤 AMP锛岄伩鍏?complex FFT/鑳介噺褰掍竴鍖栧嚭闂
        with torch.cuda.amp.autocast(enabled=False):
            for center in outpos_m:
                Ey_port = sample_field(Ey_batched, center)  # (B,Nm)
                det = overlap_obj.eta_kspace_vec_angler(
                    y_mode, Ey_port, Ey_mode,
                    wavelength=wavelength_m,
                    n_medium=n_medium,
                )
                p_val = det.get("proj_power_tensor", None)
                if p_val is None:
                    p_val = torch.as_tensor(det["proj_power"], device=dev, dtype=torch.float64)
                p_cols.append(p_val.to(torch.float32))  # (B,)

        p = torch.stack(p_cols, dim=1)  # (B,P)
        #p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)
        if p.shape[0] == 1:
            p = p.squeeze(0)
        return p*1e6, None

  
        




    def coupled_output_real(self,ey=None,ex=None): 
       

        # 鑾峰彇杈撳叆鏁版嵁 (杩欓噷鍋囪self.us鏄竴涓寘鍚玡x, ey鐨勫厓缁?
        if ey is None:
            ex_, ey = self.us[-1]
            ey=ey.detach().cpu().numpy()
            jump=0
        else:
            ey=ey
            jump=1
        if ex is None:
            ex, ey_ = self.us[-1]
            ex=ex.detach().cpu().numpy()
            jump=0
        else:
            ex=ex
            jump=1    
        output_position = self._positions_in_internal_axis(
            self.outputposition,
            float(self.positions_span[-1]),
        )
        ey=ey.squeeze()
        ex=ex.squeeze()
        # 璁＄畻姣忎釜閲囨牱鐐圭殑鐗╃悊姝ラ暱 螖x
        span_um = float(self.positions_span[-1])
        dx = span_um / self.caiyangdian[0]  # 璁＄畻鐗╃悊姝ラ暱 螖x
        out_mapping_span = self.out_maping_span
        if jump !=1:  
            sliced_e = []
            
            for position in output_position:  # 閬嶅巻杈撳嚭浣嶇疆
                center_index = self._pos_to_index(float(position), span_um, dx, ey.shape[0], "centered")
                slice_start = center_index-int(0.5*out_mapping_span // dx)  # 鏍规嵁dx璁＄畻鍒囩墖鐨勮捣濮嬩綅缃?
                slice_end = center_index+int(0.5*out_mapping_span // dx)   # 璁＄畻鍒囩墖缁撴潫浣嶇疆
                sliced_e.append([ex[slice_start:slice_end],ey[slice_start:slice_end]])  # 鍒囩墖骞朵繚瀛樺埌sliced_ey

            # 瀵规瘡涓垏鐗囪皟鐢╡ta_kspace_Ey_weighted_from_mat_and_ein杩涜璁＄畻
            eta_results = []
            p=[]
            print()        
            for slice in sliced_e:
                # 鍋囪eta_kspace_Ey_weighted_from_mat_and_ein杩斿洖涓や釜鍊?(eta, det)
                rt, rdet = eta_real_from_mat_and_ein(
                    Ex=slice[0], Ey=slice[1],
                    span_m=out_mapping_span*1e-6, 
                    emode_mat_path=self.mode_path, 
                    origin='center', 
                    wavelength=float(self.wavelength) * 1e-6, 
                    n_medium=float(getattr(self, "neff", getattr(self, "n_slab", 1.0)))
                )
                Pin_calc = rdet.get('Pin', None)
                eta_results.append(rt)  # 鏀堕泦eta鍊?

                p.append(Pin_calc)  # 璁＄畻骞舵敹闆嗗姛鐜囧€?
        else:
            rt, rdet  = eta_real_from_mat_and_ein(
                    Ex=ex, Ey=ey,
                    span_m=out_mapping_span*1e-6, 
                    emode_mat_path=self.mode_path, 
                    origin='center', 
                    wavelength=float(self.wavelength) * 1e-6, 
                    n_medium=float(getattr(self, "neff", getattr(self, "n_slab", 1.0)))
                )
            eta_results=[rt]  # 鏀堕泦eta鍊?
            Pin_calc = rdet.get('Pin', None)
            p=[Pin_calc]  # 璁＄畻骞舵敹闆嗗姛鐜囧€?   

        # 杩斿洖鎵€鏈夎绠楀嚭鐨別ta鍊?
        return eta_results,p
    

    @torch.no_grad()
    def test_accuracy(self, X_test, y_test, flag: int = 0,get_lense_factor=False):
        # 缁熶竴璁惧
        dev = next(self.parameters(), None).device if any(True for _ in self.parameters()) else (
            self.Ws.device if hasattr(self, "Ws") else torch.device("cpu")
        )
        # loss type (new name loss_type; keep lootype alias for compatibility)
        loss_type = getattr(self, "loss_type", getattr(self, "lootype", None))

        if loss_type == "crossentropy":
            correct = 0
            for i in range(len(X_test)):
                out = self.forward(X_test[i]).reshape(-1).to(dev)                  # [C]
                pred = int(torch.argmax(out).item())
                y = torch.as_tensor(y_test[i], dtype=torch.float32, device=dev).reshape(-1)
                if flag:
                    y = torch.roll(y, shifts=flag, dims=0)                         # 绛変环 np.roll
                true_ = int(torch.argmax(y).item())
                if pred == true_:
                    correct += 1
            return correct / max(1, len(X_test))

        if loss_type == "MSE_":
            # 閫愭牱鏈細|softmax1 - softmax2| 鐨勫潎鍊硷紙浣犲師鏉ュ叾瀹炴槸L1鑰岄潪MSE锛岃繖閲屼繚鎸佷笉鍙橈級
            losses = []
            for i in range(len(X_test)):
                out = self.forward(X_test[i]).reshape(-1).to(dev)
                # 鎸変綘鍘熼€昏緫锛氬悜閲忔寜鎬诲拰褰掍竴鍖栵紝鑰岄潪鎸夎softmax锛涗繚鐣欎繚鎶?
                out = out / out.sum().abs().clamp_min(1e-12)
                y = torch.as_tensor(y_test[i], dtype=torch.float32, device=dev).reshape(-1)
                y = y / y.sum().abs().clamp_min(1e-12)
                losses.append(torch.mean(torch.abs(out - y)))
            return torch.stack(losses).mean().item()

        if loss_type == "MSE":
            # 鍏堟妸鎵€鏈夎緭鍑哄拰鏍囩鏀堕泦璧锋潵锛堜笌浣犲師閫昏緫涓€鑷达細鍙彇[:, 0]鍒楋紝鍐嶅仛鈥滄€诲拰褰掍竴*鏍锋湰鏁扳€濓級
            outs, ys = [], []
            for i in range(len(X_test)):
                out = self.forward(X_test[i]).reshape(-1).to(dev)          # [C]
                outs.append(out)
                ys.append(torch.as_tensor(y_test[i], dtype=torch.float32, device=dev).reshape(-1))
            outs = torch.stack(outs)                                               # [N, C]
            ys   = torch.stack(ys)                                                 # [N, C]

            out_col0 = outs[:, 0]                                                  # 绛変环 np.array(out_data)[:,0]
            y_col0   = ys[:, 0]                                                    # 绛変环 np.array(true_label)[:,0]

            out_col0 = len(out_col0) * out_col0 / out_col0.sum().abs().clamp_min(1e-12)
            y_col0   = len(y_col0)   * y_col0   / y_col0.sum().abs().clamp_min(1e-12)

            # 涓庝綘鍘熶唬鐮佷竴鑷达細閫愭牱鏈?L1锛屽啀骞冲潎
            loss = torch.mean(torch.abs(out_col0 - y_col0))
            return loss.item()

        if loss_type in ("dircte mse", "direct mse", "direct_mse"):
            print("direct_mse")
            losses = []
            y_true_sum = 0.0
            y_pred_sum = 0.0

            # 璁惧缁熶竴锛堣嫢澶栧眰娌¤缃甦ev锛屽彲鐢ㄤ笅闈㈣繖琛岋級
            dev = next(self.parameters(), None).device if any(True for _ in self.parameters()) else (
                self.Ws.device if hasattr(self, "Ws") else torch.device("cpu")
            )

            with torch.no_grad():
                for i in range(len(X_test)):
                    # 鍓嶅悜锛屽嵆褰撳墠杈撳嚭
                    out = self.forward(X_test[i]).reshape(-1).to(dev)

                    # 鐪熷疄鏍囩
                    y = torch.as_tensor(y_test[i], dtype=out.dtype, device=dev).reshape(-1)

                    # 閫愭牱鏈?MSE
                    losses.append(torch.mean((out - y) ** 2))

                    # 鐢ㄤ簬 lense_factor 鐨勭疮璁″拰
                    y_true_sum += float(torch.sum(y).item())
                    y_pred_sum += float(torch.sum(out).item())

            mean_mse = torch.stack(losses).mean().item()

            # 鍙€夎繑鍥?lense_factor = sum(y_true) / sum(y_pred)
            if get_lense_factor:
                eps = 1e-12
                lense_factor = y_true_sum / (y_pred_sum + eps)
                return mean_mse, lense_factor
            else:
                return mean_mse

        # 榛樿杩斿洖
        return 0.0
    def setuphs_hybride(
        self,
        mode="full",
        ky_cut_ratio=1.0,
        coarse_factor=1.0,
        smooth_rolloff=0.0,
        pad_factor=16,
    ):
        if hasattr(self, "complex_data_matrices") and len(self.complex_data_matrices) > 0:
            N0 = int(np.asarray(self.complex_data_matrices[0]).shape[-1])
        else:
            N0 = int(self.caiyangdian[0])

        pad_factor = int(max(1, pad_factor))
        Np = int(pad_factor * N0)

        self.N0 = N0
        self.Np = Np
        self.pad_start = (Np - N0) // 2

        span_um = float(self.positions_span[0])
        dy = span_um / N0
        self.dy = dy

        ky = 2 * np.pi * np.fft.fftfreq(Np, d=dy)
        ky_abs = np.abs(ky)

        dev = getattr(self, "device", None)
        if dev is None:
            try:
                dev = next(self.parameters()).device
            except Exception:
                dev = torch.device("cpu")

        hs_multi = []
        contexts = self._resolve_forward_wavelength_contexts()
        ky_nyq = np.pi / dy
        ky_nyq_eff = ky_nyq / float(coarse_factor) if mode == "coarse" else ky_nyq
        for context in contexts:
            k = 2 * np.pi * float(context["n_slab"]) / float(context["selected_wavelength_um"])
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

            hs_single = []
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

                hs_single.append(torch.as_tensor(H, dtype=torch.complex64, device=dev))

            hs_multi.append(torch.stack(hs_single, dim=0))

        hs_multi_tensor = torch.stack(hs_multi, dim=0)
        self._set_buffer_tensor("Hs_H_multi", hs_multi_tensor)
        self._set_buffer_tensor(
            "Hs_H",
            hs_multi_tensor[int(self.selected_wavelength_index)].clone(),
        )

    def _run_single_wavelength_hybrid_forward(
        self,
        x: torch.Tensor,
        *,
        context: Mapping[str, float | int],
    ):
        dev = self.slot_length_um[0].device
        x = torch.as_tensor(x, device=dev, dtype=torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        B = x.shape[0]

        wl_um = float(context["selected_wavelength_um"])
        base = self._resolve_monitor_tensor(0, wavelength_um=wl_um, device=dev)
        N = base.shape[-1]

        N0 = int(getattr(self, "N0", N))
        Np = int(getattr(self, "Np", N))
        ps = int(getattr(self, "pad_start", 0))
        assert N == N0, f"N mismatch: hybrid forward N={N}, setuphs N0={N0}, wavelength_um={wl_um}"
        assert self.Hs_H_multi.shape[-1] == Np, f"Hybrid Hs length mismatch: Hs={self.Hs_H_multi.shape[-1]}, Np={Np}"

        def pad_center(u, fill=0.0):
            if u.shape[-1] == Np:
                return u
            if u.shape[-1] != N:
                raise ValueError(f"Unexpected width for pad_center: got {u.shape[-1]}, expected {N} or {Np}")
            fillv = torch.as_tensor(fill, device=u.device, dtype=u.dtype)
            out = torch.full((u.shape[0], Np), fillv, dtype=u.dtype, device=u.device)
            out[:, ps:ps + N] = u
            return out

        def crop_center(u):
            if Np == N:
                return u
            return u[:, ps:ps + N]

        _, base_y_axis_um = self._resolve_monitor_line_and_axis(
            0,
            wavelength_um=wl_um,
        )
        mask = self._build_input_phase_mask(
            base_line=base.detach().cpu().numpy(),
            y_axis_um=base_y_axis_um,
            x=x,
            device=dev,
        )

        fields = []
        for idx in range(self.num_layers):
            src = self._resolve_monitor_tensor(
                idx,
                wavelength_um=wl_um,
                device=dev,
            )
            if src.ndim == 1:
                src = src.unsqueeze(0).expand(B, -1)
            fields.append(src)

        fields[0] = base.unsqueeze(0).expand(B, -1) * mask
        hs = self._get_hybrid_hs_for_wavelength(wavelength_um=wl_um).to(dev)
        u = pad_center(fields[0], fill=0.0)
        for layer_idx in range(1, self.num_layers):
            U = torch.fft.fft(torch.fft.ifftshift(u, dim=-1), dim=-1)
            U = U * hs[layer_idx - 1]
            u = torch.fft.fftshift(torch.fft.ifft(U, dim=-1), dim=-1)
            u = self.hybridsimlink(
                crop_center(u),
                layer_idx,
                wavelength_um=wl_um,
                wavelength_index=int(context["selected_wavelength_index"]),
                n_slab=float(context["n_slab"]),
            )
            fields[layer_idx] = u
            u = pad_center(u, fill=0.0)

        response, _ = self.coupled_output_overlap(
            overlap_obj=self.overlap,
            ey=fields[-1],
            wavelength_m=wl_um * 1e-6,
            n_medium=float(context["n_slab"]),
            origin="center",
        )
        return {
            "response": response,
            "fields": fields,
            "context": dict(context),
        }

    def forward_hybride(
        self,
        x,
        *,
        wavelength_um: float | None = None,
        wavelength_index: int | None = None,
    ):
        with torch.cuda.amp.autocast(enabled=False):
            dev = self.slot_length_um[0].device
            x = torch.as_tensor(x, device=dev, dtype=torch.float32)
            if x.ndim == 1:
                x = x.unsqueeze(0)

            contexts = self._resolve_forward_wavelength_contexts(
                wavelength_um=wavelength_um,
                wavelength_index=wavelength_index,
            )
            responses = []
            representative = None
            representative_key = canonical_wavelength_key(self.selected_wavelength_um)
            self.hybrid_last_output_by_wavelength = {}
            self.hybrid_wavelength_contexts = [dict(context) for context in contexts]
            self.hybrid_us_by_wavelength = {}
            self.hybrid_requested_wavelengths_um = [
                float(context["selected_wavelength_um"]) for context in contexts
            ]
            self.hybrid_output_wavelengths_um = []

            for context in contexts:
                single = self._run_single_wavelength_hybrid_forward(x, context=context)
                response = torch.as_tensor(single["response"], device=dev, dtype=torch.float32)
                if response.ndim == 1:
                    response = response.unsqueeze(0)
                responses.append(response)

                key = canonical_wavelength_key(float(context["selected_wavelength_um"]))
                self.hybrid_last_output_by_wavelength[key] = response.detach().cpu()
                self.hybrid_us_by_wavelength[key] = single["fields"]
                if representative is None or key == representative_key:
                    representative = single

            if representative is None:
                representative = single
            self.us_H = representative["fields"]
            self.hybrid_operating_point = representative["context"]

            response_stack = torch.stack(responses, dim=1)
            if response_stack.shape[1] == 1:
                single_response = response_stack[:, 0, :]
                if single_response.shape[0] == 1:
                    single_response = single_response.squeeze(0)
                return single_response
            return response_stack

    def hybridsimlink(
        self,
        u,
        l,
        *,
        wavelength_um: float | None = None,
        wavelength_index: int | None = None,
        n_slab: float | None = None,
    ):
        """
        Bridge 1D field to hybrid 3D FDTD and return the resampled output field.
        """
        dev = u.device
        u_np = u.detach().cpu().numpy()
        ey_1d = np.squeeze(u_np)
        if ey_1d.ndim > 1:
            ey_1d = ey_1d.reshape(-1)
        ey_1d = np.asarray(ey_1d, dtype=np.complex128)
        N_target = ey_1d.shape[0]

        if not hasattr(self, "hybrid_inputs"):
            self.hybrid_inputs = []
        self.hybrid_inputs.append(ey_1d)

        span_um = float(self.positions_span[0])
        y_um = np.linspace(-0.5 * span_um, 0.5 * span_um, N_target)
        y_m = y_um * 1e-6

        mode_path = getattr(self, "mode_path", None)
        if mode_path is None:
            raise ValueError("mode_path must be set before calling hybridsimlink")
        mode = load_lumerical_modeE(Path(mode_path))

        z_m = np.asarray(mode["z"]).reshape(-1)
        Ex_z = np.asarray(mode["Ex"]).reshape(-1)
        Ey_z = np.asarray(mode["Ey"]).reshape(-1)
        Ez_z = np.asarray(mode["Ez"]).reshape(-1)
        zlink = float(getattr(self, "hybrid_zlink", 0.0))

        Ex_yz, Ey_yz, Ez_yz = build_yz_vector_zlink0(
            ey_1d, y_m, Ex_z, Ey_z, Ez_z, z_m, zlink=zlink
        )

        workdir = Path(getattr(self, "hybrid_workdir", Path(mode_path).parent))
        workdir.mkdir(parents=True, exist_ok=True)

        try:
            from scipy.io import savemat

            ey1d_path = workdir / f"layer{l}_Ey1d.mat"
            savemat(str(ey1d_path), {"y_m": y_m, "ey1d": ey_1d}, format="5")
        except Exception:
            pass

        mat_path = workdir / f"layer{l}_Ey_yz.mat"
        save_mat_v73_for_lumerical_h5py(
            mat_path,
            x0=0.0,
            y=y_m,
            z=z_m,
            Ex_yz=Ex_yz,
            Ey_yz=Ey_yz,
            Ez_yz=Ez_yz,
            f0=mode.get("f0", None) or float(
                self.selected_wavelength_um if wavelength_um is None else wavelength_um
            ),
        )

        fdtd_handles = getattr(self.sim, "hybrid_fdtds", None)
        if not fdtd_handles:
            assert print("init fist")
        idx = max(0, l - 1)
        if idx >= len(fdtd_handles):
            return u
        fdtd = fdtd_handles[idx]

        monitor_name = getattr(self.sim, "hybrid_monitor_name", "hybrid_monitor")
        requested_context = self._resolve_wavelength_context(
            wavelength_um=wavelength_um,
            wavelength_index=wavelength_index,
        )
        wl_req_um = float(requested_context["selected_wavelength_um"])
        lambda_m = wl_req_um * 1e-6
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
        """
        fdtd.eval(script)
        fdtd.eval("run;")

        fields = get_fields(
            fdtd,
            monitor_name=monitor_name,
            field_result_name="fields",
            get_eps=False,
            get_D=False,
            get_H=False,
            nointerpolation=False,
            unfold_symmetry=False,
        )

        picked = extract_ey_1d_at_wavelength(
            fields,
            wavelength_um=wl_req_um,
            selection="nearest",
        )
        ey_out = np.asarray(picked["ey"])
        y_old = np.asarray(picked["y"]).reshape(-1)
        if not hasattr(self, "hybrid_output_wavelengths_um"):
            self.hybrid_output_wavelengths_um = []
        self.hybrid_output_wavelengths_um.append(float(picked["selected_wavelength_um"]))

        if y_old.size == 0:
            y_old = np.linspace(0.0, 1.0, ey_out.shape[0])
        if y_old.size != ey_out.shape[0]:
            y_old = np.linspace(
                y_old.min() if y_old.size else 0.0,
                y_old.max() if y_old.size else 1.0,
                ey_out.shape[0],
            )

        y_target = np.linspace(y_old.min(), y_old.max(), N_target)
        ey_resampled = interp_complex(ey_out, y_old, y_target, fill=0.0)

        if not hasattr(self, "hybrid_outputs"):
            self.hybrid_outputs = []
        self.hybrid_outputs.append(np.asarray(ey_resampled))

        out_tensor = torch.as_tensor(ey_resampled, device=dev, dtype=u.dtype)
        if u.ndim == 2:
            out_tensor = out_tensor.unsqueeze(0).expand(u.shape[0], -1)
        return out_tensor

