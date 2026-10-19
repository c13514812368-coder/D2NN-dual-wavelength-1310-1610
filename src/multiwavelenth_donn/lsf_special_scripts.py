from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .networkangler import (
    normalize_lumerical_material_name,
    uses_object_defined_dielectric,
)


def resolve_wavelength_spec_um(shape: Any) -> dict[str, float | int]:
    if hasattr(shape, "resolve_wavelength_spec"):
        spec = shape.resolve_wavelength_spec()
        start_um = float(spec.get("start_um", getattr(shape, "wavelength_um", 1.55)))
        stop_um = float(spec.get("stop_um", start_um))
        center_um = float(spec.get("center_um", 0.5 * (start_um + stop_um)))
        points = max(1, int(spec.get("points", getattr(shape, "wavelength_points", 1))))
    else:
        start_um = float(getattr(shape, "wavelength_start", getattr(shape, "wavelength_um", 1.55)))
        stop_um = float(getattr(shape, "wavelength_stop", start_um))
        center_um = float(getattr(shape, "wavelength_center", 0.5 * (start_um + stop_um)))
        points = max(1, int(getattr(shape, "wavelength_points", 1)))
    if abs(stop_um - start_um) <= 0.0:
        points = 1
    return {
        "start_um": start_um,
        "stop_um": stop_um,
        "center_um": center_um,
        "points": points,
    }


def _material_lines(material: object, index_val: object, *, background: bool = False) -> list[str]:
    material_name = normalize_lumerical_material_name(material)
    setter = "background material" if background else "material"
    lines = [f"set('{setter}', '{material_name}');"]
    if uses_object_defined_dielectric(material_name):
        lines.append(f"set('index', {float(index_val)});")
    return lines


def write_mapping_factor_lsf(shape: Any, output_path: str | Path) -> Path:
    wl_spec = resolve_wavelength_spec_um(shape)
    path = Path(output_path)
    x_span = 15
    num_copies = x_span / float(shape.unit_spacing_um)
    y_span = 10
    dist_fdtdtosub = 2
    yspan_step = 0.1
    use_var_fdtd = bool(getattr(shape, "use_var_fdtd", False))
    lines: list[str] = [
        "switchtolayout;",
        "selectall;",
        "delete;",
        "",
        "iteration_limit=20;",
        ("addvarfdtd;" if use_var_fdtd else "addfdtd;"),
        *([] if use_var_fdtd else ["set('dimension', '2D');"]),
        'set("x", 0);',
        'set("y", 0);',
        f'set("x span", {x_span-dist_fdtdtosub}e-6);',
        f'set("y span", {y_span-dist_fdtdtosub}e-6);',
        f'set("z span", 5*{shape.propagation_thickness_um}e-6);',
        f"set('mesh accuracy', {shape.mesh_accuracy}+5);",
        "",
    ]
    lines.extend(_material_lines(shape.background_material, getattr(shape, "background_material_index", 1.0), background=True))
    lines.extend(
        [
            'set("z", 0);',
            f'if (haveproperty("y0")) {{ set("y0", {-0.5*y_span+dist_fdtdtosub}e-6); }}',
            "addrect;",
            'set("name", "sub");',
        ]
    )
    lines.extend(_material_lines(shape.propagation_material, getattr(shape, "propagation_material_index", 1.0)))
    lines.extend(
        [
            'set("x", 0);',
            'set("y", 0);',
            f'set("x span", {x_span}e-6);',
            f'set("y span", {y_span}e-6);',
            f'set("z span", {shape.propagation_thickness_um}e-6);',
            'set("z", 0);',
            "",
            "addpower;",
            'set("name", "monitor");',
            'set("monitor type", "linear x");',
            f'set("x span",{x_span-dist_fdtdtosub}e-6);',
            'set("x", 0);',
            'set("z", 0);',
            f'set("y", {0.5*y_span-0.5*dist_fdtdtosub}e-6);',
            "",
            "addpower;",
            'set("name", "monitor_2");',
            'set("monitor type", "linear x");',
            f'set("x span",{x_span-dist_fdtdtosub}e-6);',
            'set("x", 0);',
            f'set("y", {-0.4*y_span+0.5*dist_fdtdtosub}e-6);',
            'set("z", 0);',
            "",
            "addrect;",
            'set("name", "hope");',
            'set("material", "etch");',
            f'set("x span", {shape.unit_span_um}e-6);',
            'set("y span", 0.1e-6);',
            f'set("z span", {shape.propagation_thickness_um}e-6);',
            f'set("x", -0.5*{x_span}*1e-6);',
            'set("y", 0);',
            'set("z", 0);',
            "",
            "addplane;",
            'set("name", "plane_wave");',
            'set("x", 0);',
            f'set("x span",{x_span}e-6);',
            f'set("y", {-0.5*y_span+0.5*dist_fdtdtosub}e-6);',
            'set("z", 0);',
            f"set('wavelength start',{float(wl_spec['start_um'])}e-6);",
            f"set('wavelength stop', {float(wl_spec['stop_um'])}e-6);",
        ]
    )
    if int(wl_spec["points"]) > 1:
        lines.append(
            "if (haveproperty('multifrequency beam calculation')) { set('multifrequency beam calculation', true); }"
        )
        lines.append(
            f"if (haveproperty('number of frequency points')) {{ set('number of frequency points', {int(wl_spec['points'])}); }}"
        )
    lines.extend(
        [
            "",
            "function process_monitor_data(monitor_name) {",
            "    select(monitor_name);",
            '    ex_data = getresult(monitor_name, "Hz");',
            "    ex_phase = angle(ex_data)/pi;",
            "    nx2 = size(ex_phase,1);",
            "    seg_ratio = 0.30;",
            "    seg_len = round(nx2*seg_ratio);",
            "    i1 = floor((nx2 - seg_len)/2) + 1;",
            "    i2 = i1 + seg_len - 1;",
            "    phase_segment = ex_phase(i1:i2, :, :, 1);",
            "    avg_phase = mean(phase_segment);",
            "",
            '    select(monitor_name+"_2");',
            '    ex_data_2 = getresult(monitor_name+"_2", "Hz");',
            "    ex_phase_2 = angle(ex_data_2)/pi;",
            "    phase_segment_2 = ex_phase_2(i1:i2, :, :, 1);",
            "    avg_phase_2 = mean(phase_segment_2);",
            "",
            "    avg_phase_deta = avg_phase - avg_phase_2;",
            "    return avg_phase_deta;",
            "}",
            "",
            "function add_rect_group(original_rectangle_name, group_name) {",
            "    addgroup;",
            '    set("name", group_name+"_1");',
            "",
            "    select(original_rectangle_name);",
            '    x_original = get("x");',
            '    y_original = get("y");',
            '    width = get("x span");',
            '    height = get("y span");',
            '    material = get("material");',
            "",
            f"    num_copies = {num_copies};",
            f"    spacing = {shape.unit_spacing_um}e-6;",
            f"    zspan = {shape.propagation_thickness_um}e-6;",
            "",
            "    for (i = 1:num_copies) {",
            "        x_new = x_original + i*spacing;",
            "        addrect;",
            '        set("name", original_rectangle_name + "_" + num2str(i));',
            '        set("x", x_new);',
            '        set("x span", width);',
            '        set("y", y_original);',
            '        set("y span", height);',
            '        set("z span",zspan);',
            '        set("z", 0);',
            '        set("material", material);',
            '        addtogroup(group_name+"_1");',
            "    }",
            "}",
            "",
            "run;",
            "avg_phase_pre = process_monitor_data('monitor');",
            "switchtolayout;",
            "",
            "for (j = 1:iteration_limit) {",
            "    add_rect_group('hope', 'group');",
            "    run;",
            "    avg_phase = process_monitor_data('monitor') - avg_phase_pre;",
            "    switchtolayout;",
            "",
            '    select("group"+"_1");',
            "    delete;",
            '    select("hope");',
            '    yspan = get("y span");',
            f'    set("y span", yspan + {yspan_step}*1e-6);',
            '    write("out_txt"+".txt", num2str(avg_phase) + ",");',
            "}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path.resolve()


def write_plain_wave_lsf(
    shape: Any,
    output_path: str | Path,
    *,
    add: bool = True,
    name: str = "plain_src",
    x_um: float | None = None,
    y_um: float = 0.0,
    y_span_um: float | None = None,
    wl_um: float | None = None,
    wl_start_um: float | None = None,
    wl_stop_um: float | None = None,
    wl_points: int | None = None,
) -> Path:
    spec = resolve_wavelength_spec_um(shape)
    if wl_um is not None:
        src_start_um = float(wl_um)
        src_stop_um = float(wl_um)
        src_points = 1
    else:
        src_start_um = float(spec["start_um"] if wl_start_um is None else wl_start_um)
        src_stop_um = float(spec["stop_um"] if wl_stop_um is None else wl_stop_um)
        src_points = max(1, int(spec["points"] if wl_points is None else wl_points))
        if abs(src_stop_um - src_start_um) <= 0.0:
            src_points = 1

    if y_span_um is None:
        y_span_um = float(getattr(shape, "total_width_um", 0.0))

    if x_um is None:
        spacings = np.asarray(getattr(shape, "inter_layer_spacings_um", []), dtype=float)
        if spacings.size > 0:
            first_gap = float(spacings[0])
        else:
            num_layers = int(getattr(shape, "num_layers", 2))
            first_gap = float(getattr(shape, "total_length_um", 0.0)) / max(num_layers - 1, 1)
        x_um = -0.5 * float(getattr(shape, "total_length_um", 0.0)) + 0.5 * first_gap

    lines = [
        "switchtolayout;",
        "redrawoff;",
    ]
    num_in = int(getattr(shape, "num_input_waveguides", 0))
    if add:
        for i in range(num_in):
            lines.append(
                f"if (getnamednumber('mode_src{i}')>0) {{ setnamed('mode_src{i}','enabled',false); }}"
            )
        lines.extend(
            [
                f"select('{name}'); delete;",
                "addplane;",
                f"set('name','{name}');",
                "set('injection axis','x');",
                f"set('x',{x_um}e-6);",
                f"set('y',{y_um}e-6);",
                f"set('y span',{y_span_um}e-6);",
                f"set('wavelength start',{src_start_um}e-6);",
                f"set('wavelength stop',{src_stop_um}e-6);",
            ]
        )
        if src_points > 1:
            lines.append(
                "if (haveproperty('multifrequency beam calculation')) { set('multifrequency beam calculation',true); }"
            )
            lines.append(
                f"if (haveproperty('number of frequency points')) {{ set('number of frequency points',{src_points}); }}"
            )
        lines.append("set('enabled',true);")
    else:
        lines.append(f"select('{name}'); delete;")
        for i in range(num_in):
            lines.append(
                f"if (getnamednumber('mode_src{i}')>0) {{ setnamed('mode_src{i}','enabled',true); }}"
            )
    lines.append("redrawon;")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path.resolve()


def write_add_silicon_slots_lsf(
    shape: Any,
    output_path: str | Path,
    *,
    params: Sequence[float],
    positions: Sequence[Sequence[float]],
    material: str = "SiO2 (Glass) - Palik",
    deactivate_threshold: int = 50000,
    only_update: bool = False,
    direction: str = "x",
    z_span_um: float | None = None,
    width_um: float | None = None,
    spacing_um: float | None = None,
    rect_prefix: str = "rect",
    mapping_factor: float | None = None,
) -> Path:
    def _m_literal_from_um(value_um: float) -> str:
        return repr(float(value_um) * 1e-6)

    if z_span_um is None:
        z_span_um = float(shape.propagation_thickness_um)
    if width_um is None:
        width_um = float(getattr(shape, "unit_span_um", 0.0))
    if spacing_um is None:
        spacing_um = float(getattr(shape, "unit_spacing_um", 0.0))
    repeats = max(int(getattr(shape, "repeats", 1)), 1)

    direction = str(direction).lower()
    if direction not in ("x", "x-", "y", "y-"):
        raise ValueError("direction must be one of 'x', 'x-', 'y', or 'y-'")
    neg = direction.endswith("-")

    per_layer = list(getattr(shape, "neurons_per_layer"))[1:-1]
    if len(positions) != len(per_layer):
        raise ValueError(
            f"positions count ({len(positions)}) does not match internal layer count ({len(per_layer)})"
        )
    cum = np.cumsum([0] + per_layer)

    lines = [
        "# --- Auto-generated: Add/Update silicon slots ---",
        "redrawoff;",
    ]
    for j, (x0_um, y0_um) in enumerate(positions):
        group_name = f"group{j}"
        seg = np.asarray(params[cum[j]:cum[j + 1]], dtype=float)
        if seg.size == 0:
            continue

        if not only_update:
            lines.extend(
                [
                    f"select('{group_name}');",
                    "delete;",
                    "addgroup;",
                    f"set('name','{group_name}');",
                ]
            )

        physical_slot_count = int(seg.size * repeats)
        for i, raw_length_um in enumerate(seg):
            length_um = float(raw_length_um)
            if mapping_factor is not None:
                length_um *= float(mapping_factor)

            for repeat_idx in range(repeats):
                physical_idx = i * repeats + repeat_idx
                step = -float(spacing_um) if neg else float(spacing_um)
                if direction.startswith("y"):
                    x_um = float(x0_um) + physical_idx * step
                    y_um = float(y0_um)
                    x_span_um = float(width_um)
                    y_span_um = float(length_um)
                else:
                    x_um = float(x0_um) + (0.5 * float(length_um) if not only_update else 0.0)
                    y_um = float(y0_um) + physical_idx * step
                    x_span_um = float(length_um)
                    y_span_um = float(width_um)

                rect_name = f"{group_name}_{rect_prefix}_{physical_idx:03d}"
                if not only_update:
                    lines.extend(
                        [
                            "addrect;",
                            f"set('name','{rect_name}');",
                            f"set('material','{material}');",
                            f"set('z span', {_m_literal_from_um(z_span_um)});",
                            f"set('x', {_m_literal_from_um(x_um)}); set('y', {_m_literal_from_um(y_um)});",
                            f"set('x span', {_m_literal_from_um(x_span_um)}); set('y span', {_m_literal_from_um(y_span_um)});",
                            'set("z", 0);',
                            f"addtogroup('{group_name}');",
                        ]
                    )
                else:
                    lines.extend(
                        [
                            f"setnamed('{rect_name}','x', {_m_literal_from_um(x_um)});",
                            f"setnamed('{rect_name}','y', {_m_literal_from_um(y_um)});",
                            f"setnamed('{rect_name}','x span', {_m_literal_from_um(x_span_um)});",
                            f"setnamed('{rect_name}','y span', {_m_literal_from_um(y_span_um)});",
                            f"setnamed('{rect_name}','z span', {_m_literal_from_um(z_span_um)});",
                            'set("z", 0);',
                            f"setnamed('{rect_name}','material','{material}');",
                        ]
                    )

        if not only_update and physical_slot_count > deactivate_threshold:
            lines.append(f"select('{group_name}'); set('enabled', true);")

    lines.append("redrawon;")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path.resolve()


def iter_hybrid_init_scripts(shape: Any, *, acc: int = 5) -> Iterable[str]:
    before_um = float(getattr(shape, "before_um", getattr(shape, "before", 0.1)))
    after_um = float(getattr(shape, "after_um", getattr(shape, "after", 3.0)))
    x_span_um = float(before_um + after_um + 1)
    y_span_um = float(shape.total_width_um + shape.dist_from_sub_to_fdtd_um)
    z_span_um = float(5 * shape.propagation_thickness_um)
    left_x_um = -before_um
    right_x_um = after_um
    inter_spacings = np.asarray(getattr(shape, "inter_layer_spacings_um", []), dtype=float)
    sim_time = (1e-15) * 1000 * float(np.sum(inter_spacings)) / 20

    bg_material = normalize_lumerical_material_name(shape.background_material)
    slab_material = normalize_lumerical_material_name(shape.propagation_material)
    for _ in range(max(int(shape.num_layers) - 2, 0)):
        lines = [
            "switchtolayout;",
            "selectall; delete;",
            "redrawoff;",
            "addfdtd;",
            "set('dimension','3D');",
            f"set('background material','{bg_material}');",
        ]
        if uses_object_defined_dielectric(bg_material):
            lines.append(f"set('index',{float(getattr(shape, 'background_material_index', 1.0))});")
        lines.extend(
            [
                f"set('mesh accuracy',{acc});",
                f"set('x span',{x_span_um}e-6);",
                f"set('y span',{y_span_um}e-6);",
                f"set('z span',{z_span_um}e-6);",
                "set('z min bc','PML');",
                "set('z max bc','PML');",
                f"set('simulation time',{sim_time});",
                "setdevice('GPU');",
                f"set('x',{0.5*after_um}e-6); set('y',0); set('z',0);",
                "addrect;",
                "set('name','slab');",
                f"set('material','{slab_material}');",
            ]
        )
        if uses_object_defined_dielectric(slab_material):
            lines.append(f"set('index',{float(getattr(shape, 'propagation_material_index', 1.0))});")
        lines.extend(
            [
                f"set('x span',5*{shape.total_length_um}e-6);",
                f"set('y span',5*{shape.total_width_um}e-6);",
                f"set('z span',{shape.propagation_thickness_um}e-6);",
                "set('x',0); set('y',0); set('z',0);",
                "addimportedsource;",
                "set('name','import_src');",
                "set('injection axis','x');",
                "set('direction','Forward');",
                f"set('x',{left_x_um}e-6);",
                "set('y',0); set('z',0);",
                f"set('y span',{shape.total_width_um}e-6);",
                "set('z span',3e-6);",
                "addpower;",
                "set('name','hybrid_monitor');",
                "set('monitor type','Linear Y');",
                f"set('x',{right_x_um}e-6);",
                "set('y',0); set('z',0);",
                f"set('y span',{shape.total_width_um}e-6);",
                "redrawon;",
            ]
        )
        yield "\n".join(lines)
