"""Calibrate the original scene3 device at 780 nm using Lumerical v241.

The delivered 1560-nm project is read-only input. Solver files use an ASCII
temporary directory because some Lumerical engines cannot handle Unicode paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from datetime import datetime
from types import SimpleNamespace

import numpy as np

SOURCE = Path(r"D:\课题\D2NN_1560nm_scene3_complete_20261015\D2NN_1560nm_scene3_complete_20261015")
API = Path(r"D:\Program Files\Lumerical\v241\api\python")
ROOT = Path(__file__).resolve().parent
os.environ["PATH"] = str(API.parents[1] / "bin") + os.pathsep + os.environ["PATH"]
sys.path[:0] = [str(API), str(SOURCE / "src")]

import lumapi
from multiwavelenth_donn.lsf_sidecar_sync import _render_template, build_neff_sync_context
from multiwavelenth_donn.lsf_special_scripts import write_mapping_factor_lsf


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def prepare():
    source_spec = SOURCE / "map" / "shape_spec.json"
    shape_cfg = json.loads(source_spec.read_text(encoding="utf-8"))["shape_config"]
    for key in ("wavelength_um", "wavelength_start", "wavelength_stop", "wavelength_center"):
        shape_cfg[key] = 0.780
    shape_cfg["wavelength_points"] = 1
    # Local material/slot calibration does not instantiate the network: the
    # archived 500-um shape's 1500-neuron counts fail its newer geometry check.
    # Keep the archived dimensions verbatim instead of changing physical size.
    shape = SimpleNamespace(**shape_cfg)
    cfg = {
        "source_project": str(SOURCE),
        "source_shape_sha256": hashlib.sha256(source_spec.read_bytes()).hexdigest(),
        "lumerical_api": str(API / "lumapi.py"),
        "shape": shape_cfg,
        "calibration": {
            "wavelength_um": 0.780,
            "slot_lengths_um": np.round(np.arange(0, 2.01, 0.1), 8).tolist(),
            "mesh_accuracy": 7,
            "fde_mesh_cells_z": 2000,
            "fde_boundary": "Metal at +/-1 um; checked against 2D cross-section; 1D PML on v241 produced NaN fields",
            "fde_search_neff": 3.4,
            "fde_trial_modes": 6,
            "fit_method": "unwrap complex transmission phase, OLS",
            "analytic_fallback": False,
            "lateral_boundary": "Periodic, 0.30 um single pitch; local infinite-array calibration",
            "source_pulse_length_fs": 20,
            "source_pulse_offset_fs": 60,
            "simulation_time_fs": 500,
            "note": "Same physical dimensions/materials as original; substrate and SiO2 slot fill included. Complex amplitude retained."
        }
    }
    write_json(ROOT / "config.json", cfg)
    neff_lsf = _render_template("calc_neff_mode_xprop", build_neff_sync_context(shape))
    neff_lsf = neff_lsf.replace("set('z span', 2.8*core_h_m);", "set('z span', 2e-6);")
    neff_lsf = neff_lsf.replace("set('z min bc', 'PML');", "set('z min bc', 'Metal');")
    neff_lsf = neff_lsf.replace("set('z max bc', 'PML');", "set('z max bc', 'Metal');")
    neff_lsf = neff_lsf.replace("set('number of trial modes', 50);", "set('number of trial modes', 6);\nset('use max index',0);\nset('n',3.4);")
    (ROOT / "neff_780nm.lsf").write_text(neff_lsf, encoding="utf-8")
    mapping_path = write_mapping_factor_lsf(shape, ROOT / "mapping_780nm.lsf")
    text = mapping_path.read_text(encoding="utf-8")
    # The legacy calibration omitted the oxide substrate and used air etch for
    # slots, although the delivered shape specifies oxide fill. Match the device.
    text = text.replace('set("name", "sub");', 'set("name", "propagation");')
    substrate = "\n".join([
        "addrect;", "set('name','oxide_substrate');",
        "set('material','SiO2 (Glass) - Palik');",
        "set('x span',15e-6);", "set('y span',10e-6);",
        "set('z span',2e-6);", "set('z',-1.11e-6);", ""
    ])
    text = text.replace("addpower;", substrate + "addpower;", 1)
    text = text.replace('set("material", "etch");', 'set("material", "SiO2 (Glass) - Palik");')
    text = text.replace("set('mesh accuracy', 2.0+5);", "set('mesh accuracy', 7);")
    text = text.replace("set('mesh accuracy', 7);", "set('mesh accuracy', 7);\nset('simulation time',500e-15);\nset('auto shutoff min',1e-7);\nset('x min bc','Periodic');\nset('x max bc','Periodic');")
    text = text.replace('set("x span", 13e-6);', 'set("x span", 0.3e-6);')
    text = text.replace('set("x span",13e-6);', 'set("x span",0.3e-6);')
    text = text.replace('set("y", 4.0e-6);', 'set("y", 2.5e-6);')
    text = text.replace('set("y", -3.0e-6);', 'set("y", -2.5e-6);')
    text = text.replace('set("y", -4.0e-6);', 'set("y", -3.5e-6);')
    text = text.replace("set('wavelength start',", "set('override global source settings',1);\nset('set wavelength',1);\nset('wavelength start',", 1)
    text = text.replace("set('polarization angle',0);", "set('polarization','TE');")
    # Remove the old wrapped phase average and use the complex transfer between
    # two monitors. Save both phase and amplitude for every slot length.
    start = text.index("function process_monitor_data")
    end = text.index("function add_rect_group", start)
    text = text[:start] + """function complex_transfer() {
    h_out = pinch(getdata('monitor','Hz'));
    h_in = pinch(getdata('monitor_2','Hz'));
    xx = pinch(getdata('monitor','x'));
    ii = find(abs(xx)<1.95e-6);
    return sum(h_out(ii)*conj(h_in(ii)))/sum(abs(h_in(ii))^2);
}

""" + text[end:]
    start = text.index("run;\navg_phase_pre")
    text = text[:start] + """setglobalmonitor('use source limits',true);
setglobalmonitor('frequency points',1);
select('plane_wave');
set('injection axis','y-axis');
set('direction','Forward');
# varFDTD determines source polarization from its selected slab mode.
set('set time domain',1);
set('frequency',299792458/780e-9);
set('pulselength',20e-15);
set('offset',60e-15);
select('hope'); set('enabled',0);
save('mapping_780nm');
run;
sample_frequency_hz = getdata('monitor','f');
t0 = complex_transfer();
write('complex_transmission.txt','length_um,re,im,abs_transfer');
write('complex_transmission.txt','0,1,0,'+num2str(abs(t0)));
switchtolayout;
select('hope'); set('enabled',1);
for (j=1:iteration_limit) {
    add_rect_group('hope','group');
    select('hope'); set('enabled',0);
    run;
    tj = complex_transfer();
    relative_t = tj/t0;
    write('complex_transmission.txt',num2str(j*0.1)+','+num2str(real(relative_t))+','+num2str(imag(relative_t))+','+num2str(abs(tj)));
    switchtolayout;
    select('group_1'); delete;
    select('hope'); set('enabled',1);
    set('y span',(j+1)*0.1e-6);
}
save('mapping_780nm');
"""
    mapping_path.write_text(text, encoding="utf-8")
    print("Prepared 780-nm FDE and 21-point varFDTD calibration", flush=True)
    return cfg


def probe(cfg, sim_dir):
    print(f"Opening hidden MODE session; solver_dir={sim_dir}", flush=True)
    with lumapi.MODE(hide=True) as mode:
        mode.cd(str(sim_dir))
        n_si = complex(np.asarray(mode.getindex("Si (Silicon) - Palik", 299792458 / 780e-9)).squeeze())
        mode.eval((ROOT / "neff_780nm.lsf").read_text(encoding="utf-8"))
        wl = float(np.asarray(mode.getnamed("FDE", "wavelength")).squeeze())
        if not np.isclose(wl, 780e-9, rtol=0, atol=1e-15):
            raise ValueError(f"FDE wavelength mismatch: {wl}")
        neff = complex(np.asarray(mode.getdata("FDE::data::mode1", "neff")).squeeze())
        field = np.asarray(mode.getresult("FDE::data::mode1", "E")["E"])
        if not np.isfinite(field).all() or np.max(np.abs(field)) <= 0:
            raise ValueError("780 nm FDE mode1 has invalid field; refusing to publish neff")
        loss = float(np.asarray(mode.getdata("FDE::data::mode1", "loss")).squeeze())
        mode.switchtolayout()
        mode.setnamed("FDE", "mesh cells z", 1000)
        mode.findmodes()
        coarse_neff = complex(np.asarray(mode.getdata("FDE::data::mode1", "neff")).squeeze())
        if abs(neff - coarse_neff) > 0.003:
            raise ValueError(f"FDE grid not converged: 2000 cells={neff}, 1000 cells={coarse_neff}")
        mode.switchtolayout()
        mode.setnamed("FDE", "mesh cells z", 2000)
        mode.findmodes()
        mode.save("neff_780nm")
        result = {
            "wavelength_nm": 780, "solver_wavelength_m": wl,
            "silicon_n_real": n_si.real, "silicon_n_imag": n_si.imag,
            "neff_real": neff.real, "neff_imag": neff.imag,
            "mode_loss_db_per_m": loss,
            "estimated_500um_material_mode_loss_db": loss * 500e-6,
            "solver_dir": str(sim_dir),
            "mode_field_finite": True,
            "neff_1000_cells_real": coarse_neff.real,
            "neff_1000_cells_imag": coarse_neff.imag,
            "neff_mesh_convergence_abs_delta": abs(neff-coarse_neff),
        }
        write_json(ROOT / "mode_probe.json", result)
        print(json.dumps(result, indent=2), flush=True)
    return result


def fit_results(sim_dir, probe_result):
    raw_path = sim_dir / "complex_transmission.txt"
    shutil.copy2(raw_path, ROOT / "complex_transmission.csv")
    frequencies = np.asarray(json.loads((ROOT / "solver_settings.json").read_text(encoding="utf-8"))["monitor_frequency_hz"])
    np.testing.assert_allclose(frequencies, 299792458 / 780e-9, rtol=1e-10)
    data = np.loadtxt(raw_path, delimiter=",", skiprows=1)
    if data.shape != (21, 4) or not np.all(np.isfinite(data)):
        raise ValueError(f"Expected finite 21x4 scan, got {data.shape}")
    lengths = data[:, 0]
    np.testing.assert_allclose(lengths, np.arange(21) * .1, atol=1e-10)
    trans = data[:, 1] + 1j * data[:, 2]
    baseline_expected = float(np.exp(-2*np.pi*probe_result["neff_imag"]/.780*5.0))
    baseline_relative_error = float(abs(data[0, 3]/baseline_expected - 1))
    if baseline_relative_error > .05:
        raise ValueError(f"Uniform-slab varFDTD/FDE amplitude disagreement: {baseline_relative_error}")
    phase_pi = np.unwrap(np.angle(trans)) / np.pi
    slope, intercept = np.polyfit(lengths, phase_pi, 1)
    fitted = slope * lengths + intercept
    ss = np.sum((phase_pi - phase_pi.mean())**2)
    r2 = 1 - np.sum((phase_pi - fitted)**2) / ss if ss > 0 else 0
    if not np.isfinite(slope) or abs(slope) < 1e-8:
        raise ValueError(f"Degenerate calibration slope: {slope}")
    result = {
        **probe_result,
        "completed_at": datetime.now().astimezone().isoformat(),
        "scan_solver_dir": str(sim_dir),
        "monitor_wavelength_nm": float(299792458 / np.asarray(frequencies).reshape(-1)[0] * 1e9),
        "mapping_factor_um_per_pi": float(1 / slope),
        "phase_slope_pi_per_um": float(slope),
        "phase_intercept_pi": float(intercept),
        "linear_fit_r2": float(r2),
        "max_phase_residual_pi": float(np.max(np.abs(phase_pi-fitted))),
        "relative_amplitude_min": float(np.abs(trans).min()),
        "relative_amplitude_max": float(np.abs(trans).max()),
        "absolute_monitor_transfer_min": float(data[:, 3].min()),
        "absolute_monitor_transfer_max": float(data[:, 3].max()),
        "phase_step_max_pi": float(np.max(np.abs(np.diff(phase_pi)))),
        "baseline_transfer_amplitude": float(data[0, 3]),
        "baseline_amplitude_from_complex_neff": baseline_expected,
        "baseline_relative_amplitude_error": baseline_relative_error,
        "local_linear_phase_fit_acceptable": bool(r2 >= .99 and np.max(np.abs(phase_pi-fitted)) <= .1),
        "linear_phase_only_model_suitable": bool(r2 >= .99 and np.max(np.abs(phase_pi-fitted)) <= .1 and np.abs(trans).min() >= .9 and np.abs(trans).max() <= 1.1),
        "original_500um_silicon_device_loss_acceptable": bool(probe_result["estimated_500um_material_mode_loss_db"] <= 10),
        "scope": "780nm slab mode and local slot response; full network map and training not run",
        "analytic_fallback_used": False,
    }
    write_json(ROOT / "calibration_results.json", result)
    np.savez(ROOT / "calibration_response.npz", wavelength_um=.780,
             slot_length_um=lengths, relative_complex_transmission=trans,
             phase_pi=phase_pi, absolute_transfer_amplitude=data[:, 3], fit_phase_pi=fitted)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(lengths, phase_pi, "o-", label="varFDTD")
    axes[0].plot(lengths, fitted, "--", label=f"Linear fit R2={r2:.4f}")
    axes[0].set_ylabel("Relative phase / pi")
    axes[0].legend()
    axes[1].plot(lengths, np.abs(trans), "o-")
    axes[1].set_ylabel("Relative field amplitude")
    axes[1].set_xlabel("Slot length (um)")
    fig.suptitle("780 nm: original 220 nm Si / SiO2 device")
    fig.tight_layout()
    fig.savefig(ROOT / "calibration_780nm.png", dpi=180)
    plt.close(fig)
    solver_files = ROOT / "solver_files"
    solver_files.mkdir(exist_ok=True)
    for folder, names in [
        (Path(probe_result["solver_dir"]), ["neff_780nm.lms", "mode1_E.mat"]),
        (sim_dir, ["mapping_780nm.lms", "mapping_780nm_p0.log", "complex_transmission.txt"]),
    ]:
        for name in names:
            path = folder / name
            if path.is_file():
                shutil.copy2(path, solver_files / name)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "probe", "calibrate", "scan", "analyze", "diagnose", "solver-check"])
    args = parser.parse_args()
    cfg = prepare()
    if args.stage == "prepare":
        return
    if args.stage == "analyze":
        settings = json.loads((ROOT / "solver_settings.json").read_text(encoding="utf-8"))
        mode_result = json.loads((ROOT / "mode_probe.json").read_text(encoding="utf-8"))
        fit_results(Path(settings["solver_dir"]), mode_result)
        return
    if args.stage == "solver-check":
        sim_dir = Path(tempfile.mkdtemp(prefix="d2nn_780nm_check_"))
        script = (ROOT / "neff_780nm.lsf").read_text(encoding="utf-8").split("findmodes;")[0]
        checks = []
        with lumapi.MODE(hide=True) as mode:
            mode.cd(str(sim_dir))
            for label, cells, boundary, dielectric, twod in [
                ("1D_400_PML",400,"PML",False,False),
                ("1D_400_metal",400,"Metal",False,False),
                ("1D_dielectric",400,"Metal",True,False),
                ("2D_Palik",200,"Metal",False,True),
                ("2D_dielectric",200,"Metal",True,True),
            ]:
                mode.eval(script)
                mode.setnamed("FDE","mesh cells z",cells)
                mode.setnamed("FDE","z min bc",boundary)
                mode.setnamed("FDE","z max bc",boundary)
                if dielectric:
                    mode.setnamed("core","material","<Object defined dielectric>")
                    mode.setnamed("core","index",3.70929)
                if twod:
                    mode.setnamed("FDE","solver type","2D X normal")
                    mode.setnamed("FDE","y span",1e-6)
                    mode.setnamed("FDE","mesh cells y",40)
                    mode.setnamed("FDE","y min bc","Metal")
                    mode.setnamed("FDE","y max bc","Metal")
                mode.findmodes()
                n = complex(np.asarray(mode.getdata("FDE::data::mode1","neff")).squeeze())
                field = np.asarray(mode.getresult("FDE::data::mode1","E")["E"])
                check = {"label":label,"neff_real":n.real,"neff_imag":n.imag,
                         "field_finite":bool(np.isfinite(field).all()),
                         "max_field":float(np.abs(field).max()) if np.isfinite(field).all() else None}
                checks.append(check)
                print(json.dumps(check), flush=True)
                mode.save(label)
                mode.switchtolayout()
            write_json(ROOT / "solver_checks.json", checks)
            # Report the exact inactive source setting and other errors.
            setup = (ROOT / "mapping_780nm.lsf").read_text(encoding="utf-8").split("function complex_transfer")[0]
            mode.eval("mapping_error='';try {" + setup + "} catch(mapping_error);")
            print("Mapping setup: " + str(mode.getv("mapping_error")), flush=True)
        return
    if args.stage == "diagnose":
        previous = json.loads((ROOT / "mode_probe.json").read_text(encoding="utf-8"))
        with lumapi.MODE(hide=True) as mode:
            mode.load(str(Path(previous["solver_dir"]) / "neff_780nm.lms"))
            mode.findmodes()
            count = int(np.asarray(mode.nummodes()).squeeze())
            print(f"FDE mode count: {count}", flush=True)
            modes = []
            for i in range(1, min(count, 12) + 1):
                name = f"FDE::data::mode{i}"
                n = complex(np.asarray(mode.getdata(name, "neff")).squeeze())
                e = mode.getresult(name, "E")
                z = np.asarray(e["z"]).reshape(-1)
                field = np.asarray(e["E"]).reshape(-1, 3)
                weight = np.sum(abs(field)**2, axis=1)
                modes.append({"mode": i, "neff_real": n.real, "neff_imag": n.imag,
                              "core_energy_fraction": float(weight[abs(z)<.11e-6].sum()/weight.sum()) if weight.sum() > 0 else None,
                              "field_max": float(np.max(np.abs(field))) if np.isfinite(field).all() else None,
                              "z_min": float(z.min()), "z_max": float(z.max())})
            print(json.dumps(modes, indent=2), flush=True)
            write_json(ROOT / "mode_diagnostics.json", modes)
            for name in ["FDE", "core", "cladding_top", "cladding_bottom"]:
                mode.select(name)
                print(name + " PROPERTIES: " + str(mode.get()), flush=True)
            mode.switchtolayout()
            mode.selectall()
            mode.delete()
            setup = (ROOT / "mapping_780nm.lsf").read_text(encoding="utf-8").split("function complex_transfer")[0]
            for line_no, line in enumerate(setup.splitlines(), 1):
                if not line.strip():
                    continue
                print(f"LSF {line_no}: {line}", flush=True)
                try:
                    mode.eval(line)
                except lumapi.LumApiError:
                    print("SELECTED OBJECT PROPERTIES: " + str(mode.get()), flush=True)
                    mode.eval("try { " + line + " } catch(lasterror);")
                    print("LSF ERROR: " + str(mode.getv("lasterror")), flush=True)
                    raise
        return
    sim_dir = Path(tempfile.mkdtemp(prefix="d2nn_780nm_"))
    write_json(ROOT / "active_run.json", {"solver_dir": str(sim_dir), "stage": args.stage, "status": "running"})
    mode_result = json.loads((ROOT / "mode_probe.json").read_text(encoding="utf-8")) if args.stage == "scan" else probe(cfg, sim_dir)
    if args.stage in ("calibrate", "scan"):
        print("Starting 0.0--2.0 um slot scan, 0.1 um steps, varFDTD mesh accuracy 7", flush=True)
        with lumapi.MODE(hide=True) as mode:
            mode.cd(str(sim_dir))
            script = (ROOT / "mapping_780nm.lsf").read_text(encoding="utf-8")
            prefix, functions = script.split("function complex_transfer", 1)
            functions, scan = ("function complex_transfer" + functions).split("setglobalmonitor('use source limits'", 1)
            scan = "setglobalmonitor('use source limits'" + scan
            for label, part in [("setup", prefix), ("functions", functions), ("scan", scan)]:
                print(f"LSF {label} start", flush=True)
                if label == "functions":
                    mode.eval(part)
                    continue
                mode.eval("calibration_error='';\ntry {\n" + part + "\n} catch(calibration_error);")
                error = mode.getv("calibration_error")
                if error:
                    (ROOT / "solver_error.txt").write_text(str(error), encoding="utf-8")
                    raise RuntimeError(f"Lumerical calibration {label}: {error}")
            write_json(ROOT / "solver_settings.json", {
                "monitor_frequency_hz": np.asarray(mode.getv("sample_frequency_hz")).reshape(-1).tolist(),
                "source_frequency_hz": float(np.asarray(mode.getnamed("plane_wave", "frequency")).squeeze()),
                "solver_dir": str(sim_dir),
            })
        fit_results(sim_dir, mode_result)
    write_json(ROOT / "active_run.json", {"solver_dir": str(sim_dir), "stage": args.stage, "status": "completed"})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        active = ROOT / "active_run.json"
        state = json.loads(active.read_text(encoding="utf-8")) if active.exists() else {}
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(active, state)
        raise
