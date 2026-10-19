from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import lumapi
except ModuleNotFoundError:
    lumapi = None
import numpy as np

from .lsf_special_scripts import (
    iter_hybrid_init_scripts,
    write_add_silicon_slots_lsf,
    write_mapping_factor_lsf,
    write_plain_wave_lsf,
)


def _safe_close(handle: Any) -> None:
    try:
        handle.close()
    except Exception:
        pass


def _require_lumapi_runtime() -> None:
    if lumapi is None:
        raise ModuleNotFoundError(
            "lumapi is required for live Lumerical/FDTD simulations, but it is not installed in this Python environment"
        )


class Simulation(object):
    def __init__(self, workingDir, use_var_fdtd, hide_fdtd_cad):
        _require_lumapi_runtime()
        self.fdtd = lumapi.MODE(hide=hide_fdtd_cad) if use_var_fdtd else lumapi.FDTD(hide=hide_fdtd_cad)
        self.workingDir = os.path.abspath(workingDir)
        self.fdtd.cd(self.workingDir)
        self.use_var_fdtd = use_var_fdtd
        self.hide_fdtd_cad = hide_fdtd_cad
        self.passworkingDir = workingDir

    def save(self, name):
        self.fdtd.cd(self.workingDir)
        full_name = os.path.join(self.workingDir, name)
        self.fdtd.save(full_name)
        return full_name

    def load(self, name):
        full_name = os.path.join(self.workingDir, name)
        self.fdtd.load(full_name)

    def save_index_to_vtk(self, filename):
        if self.fdtd.getnamednumber("global_fields_index") > 0:
            script = (
                'idx = getresult("global_fields_index", "index");'
                'vtksave("{}.vtr", idx);'
                "clear(idx);"
            ).format(filename)
            self.fdtd.eval(script)

    def save_fields_to_vtk(self, filename):
        if self.fdtd.getnamednumber("global_fields") > 0:
            script = (
                'globalHz = getresult("global_fields", "H");'
                'vtksave("{}.vtr", globalHz);'
                "clear(globalHz);"
            ).format(filename)
            self.fdtd.eval(script)

    def remove_data_and_save(self):
        self.fdtd.switchtolayout()
        self.fdtd.save()

    def __del__(self):
        _safe_close(getattr(self, "fdtd", None))


class SimuData:
    def __init__(
        self,
        network_shape,
        output_dir: str,
        use_var_fdtd=True,
        hide_fdtd_cad=False,
        base_lsf_templates: dict[str, Path] | None = None,
    ):
        _require_lumapi_runtime()
        self.network_shape = network_shape
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.workingDir = os.path.abspath(output_dir)
        self.fdtd = lumapi.MODE(hide=hide_fdtd_cad) if use_var_fdtd else lumapi.FDTD(hide=hide_fdtd_cad)
        self.fdtd.cd(self.workingDir)
        self.neff_fdtd = lumapi.MODE(hide=hide_fdtd_cad)
        self.neff_fdtd.cd(self.workingDir)
        self.cuple_fdtd = lumapi.MODE(hide=hide_fdtd_cad) if use_var_fdtd else lumapi.FDTD(hide=hide_fdtd_cad)
        self.cuple_fdtd.cd(self.workingDir)
        self.use_var_fdtd = use_var_fdtd
        self.hide_fdtd_cad = hide_fdtd_cad
        self.passworkingDir = output_dir
        self.last_generated_lsf_path: str | None = None
        self.last_generated_lsf_name: str | None = None
        self.mapping_factor: float | None = None
        self.hybrid_fdtds: list[Any] = []
        self.hybrid_fdtd_paths: list[str] = []
        self.base_lsf_templates = {
            str(key): Path(value).resolve() for key, value in dict(base_lsf_templates or {}).items()
        }
        required = ("network_simulate", "network_bidir", "calc_neff_mode_xprop")
        missing = [name for name in required if name not in self.base_lsf_templates]
        if missing:
            raise FileNotFoundError(
                f"Missing required base LSF path(s): {missing}. "
                f"Configured keys={sorted(self.base_lsf_templates)}"
            )

    def __del__(self):
        _safe_close(getattr(self, "fdtd", None))
        _safe_close(getattr(self, "neff_fdtd", None))
        _safe_close(getattr(self, "cuple_fdtd", None))
        for handle in getattr(self, "hybrid_fdtds", []):
            _safe_close(handle)

    def _remember_lsf_path(self, lsf_path: str | os.PathLike[str]) -> str:
        resolved_path = os.path.abspath(str(lsf_path))
        self.last_generated_lsf_path = resolved_path
        self.last_generated_lsf_name = os.path.basename(resolved_path)
        return resolved_path

    def _resolve_saved_lsf_path(self, script_name_or_path: str | os.PathLike[str]) -> Path:
        path = Path(script_name_or_path)
        if path.suffix == "":
            path = path.with_suffix(".lsf")
        if not path.is_absolute():
            path = Path(self.workingDir) / path
        return path.resolve()

    def get_base_lsf_path(self, template_name: str) -> Path:
        return self.base_lsf_templates[template_name]

    @staticmethod
    def _extract_lumerical_error_line(error_text: str) -> int | None:
        match = re.search(r"line\s+(\d+)", error_text)
        if match is None:
            return None
        return int(match.group(1))

    @staticmethod
    def _format_lsf_context(script_path: Path, line_number: int | None, radius: int = 3) -> str:
        try:
            lines = script_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            return f"<unable to read script context: {exc}>"

        if not lines:
            return "<empty script>"

        if line_number is None:
            start = 1
            end = min(len(lines), 8)
        else:
            start = max(1, line_number - radius)
            end = min(len(lines), line_number + radius)

        context_lines = []
        for idx in range(start, end + 1):
            marker = ">>" if idx == line_number else "  "
            context_lines.append(f"{marker} {idx:4d}: {lines[idx - 1]}")
        return "\n".join(context_lines)

    def _raise_lumerical_script_error(
        self,
        *,
        label: str,
        script_path: Path,
        exc: Exception,
    ) -> None:
        raw_error = str(exc).strip() or repr(exc)
        line_number = self._extract_lumerical_error_line(raw_error)
        context = self._format_lsf_context(script_path, line_number)
        report_path = Path(self.workingDir) / f"{script_path.stem}.error.txt"
        report_parts = [
            f"Lumerical script failed while {label}.",
            f"script={script_path}",
        ]
        if line_number is not None:
            report_parts.append(f"line={line_number}")
        report_parts.extend(
            [
                f"error={raw_error}",
                "context:",
                context,
            ]
        )
        report = "\n".join(report_parts)
        report_path.write_text(report + "\n", encoding="utf-8")
        raise RuntimeError(f"{report}\nerror_report={report_path}") from exc

    def eval_saved_lsf(
        self,
        handle,
        script_name_or_path: str | os.PathLike[str],
        *,
        label: str | None = None,
    ) -> None:
        script_path = self._resolve_saved_lsf_path(script_name_or_path)
        script_text = script_path.read_text(encoding="utf-8", errors="replace")
        self.last_generated_lsf_path = str(script_path)
        self.last_generated_lsf_name = script_path.name
        marker_path = Path(self.workingDir) / "last_lumerical_script.txt"
        marker_path.write_text(
            "\n".join(
                [
                    f"timestamp={datetime.now().isoformat()}",
                    f"label={label or script_path.stem}",
                    f"script={script_path}",
                    f"working_dir={self.workingDir}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        try:
            handle.cd(self.workingDir)
        except Exception:
            pass
        try:
            handle.eval(script_text)
        except Exception as exc:
            self._raise_lumerical_script_error(
                label=(label or script_path.stem),
                script_path=script_path,
                exc=exc,
            )

    def save(self, name):
        self.fdtd.cd(self.workingDir)
        self.neff_fdtd.cd(self.workingDir)
        self.cuple_fdtd.cd(self.workingDir)
        full_name = os.path.join(self.workingDir, name)
        self.fdtd.save(full_name)
        self.neff_fdtd.save(full_name + "_neff")
        self.cuple_fdtd.save(full_name + "_cuple")
        return full_name

    def calculate_neff_mode_xprop(
        self,
        wavelength_um: float | None = None,
        core_height_um: float | None = None,
        lateral_span_um: float | None = None,
    ) -> float:
        wl_um = float(wavelength_um or getattr(self.network_shape, "wavelength_um", 1.55))
        self.eval_saved_lsf(
            self.neff_fdtd,
            self.get_base_lsf_path("calc_neff_mode_xprop"),
            label=f"solve effective index at {wl_um:.6g} um",
        )
        return self.neff_fdtd.getdata("mode1", "neff")

    def getneff(self, wavelength_um: float | None = None) -> float:
        return self.calculate_neff_mode_xprop(wavelength_um=wavelength_um)

    def get_mapping_factor(self):
        self.neff_fdtd.cd(self.workingDir)
        self.neff_fdtd.save("mapping_factor")
        path = write_mapping_factor_lsf(
            self.network_shape,
            Path(self.output_dir) / "mapping_factor.lsf",
        )
        return str(path)

    def load(self, name):
        full_name = os.path.join(self.workingDir, name)
        self.fdtd.load(full_name)

    def save_index_to_vtk(self, filename):
        if self.fdtd.getnamednumber("global_fields_index") > 0:
            script = (
                'idx = getresult("global_fields_index", "index");'
                'vtksave("{}.vtr", idx);'
                "clear(idx);"
            ).format(filename)
            self.fdtd.eval(script)

    def save_fields_to_vtk(self, filename):
        if self.fdtd.getnamednumber("global_fields") > 0:
            script = (
                'globalHz = getresult("global_fields", "H");'
                'vtksave("{}.vtr", globalHz);'
                "clear(globalHz);"
            ).format(filename)
            self.fdtd.eval(script)

    def use_plain_wave(
        self,
        add: bool = True,
        name: str = "plain_src",
        x_um: float | None = None,
        y_um: float = 0.0,
        y_span_um: float | None = None,
        wl_um: float | None = None,
        wl_start_um: float | None = None,
        wl_stop_um: float | None = None,
        wl_points: int | None = None,
    ):
        path = write_plain_wave_lsf(
            self.network_shape,
            Path(self.output_dir) / "plain_wave_toggle.lsf",
            add=add,
            name=name,
            x_um=x_um,
            y_um=y_um,
            y_span_um=y_span_um,
            wl_um=wl_um,
            wl_start_um=wl_start_um,
            wl_stop_um=wl_stop_um,
            wl_points=wl_points,
        )
        self.eval_saved_lsf(self.fdtd, path, label=f"toggle plane wave source {name}")
        return str(path)

    def generate_add_silicon_slots_lsf(
        self,
        params,
        positions,
        material: str = "SiO2 (Glass) - Palik",
        deactivate_threshold: int = 50000,
        only_update: bool = False,
        direction: str = "x",
        z_span_um: float | None = None,
        width_um: float | None = None,
        spacing_um: float | None = None,
        rect_prefix: str = "rect",
    ) -> str:
        path = write_add_silicon_slots_lsf(
            self.network_shape,
            Path(self.output_dir) / "add_silicon_slots.lsf",
            params=params,
            positions=positions,
            material=material,
            deactivate_threshold=deactivate_threshold,
            only_update=only_update,
            direction=direction,
            z_span_um=z_span_um,
            width_um=width_um,
            spacing_um=spacing_um,
            rect_prefix=rect_prefix,
            mapping_factor=self.mapping_factor,
        )
        return self._remember_lsf_path(path)

    def hybride_init(self, acc=5):
        fdtd_handles = []
        project_paths = []
        for idx, script in enumerate(iter_hybrid_init_scripts(self.network_shape, acc=acc)):
            fdtd = lumapi.FDTD(hide=self.hide_fdtd_cad)
            fdtd.cd(self.workingDir)
            fdtd.eval(script)
            project_name = os.path.join(self.workingDir, f"hybrid_layer_{idx}.fsp")
            fdtd.save(project_name)
            fdtd_handles.append(fdtd)
            project_paths.append(project_name)

        self.hybrid_fdtds = fdtd_handles
        self.hybrid_fdtd_paths = project_paths
        return project_paths
