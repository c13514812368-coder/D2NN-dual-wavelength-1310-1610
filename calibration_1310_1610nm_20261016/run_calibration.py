"""Independent 1310/1610 nm local TE calibration using Lumerical v241."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = Path(r"D:\课题\D2NN_1560nm_scene3_complete_20261015\D2NN_1560nm_scene3_complete_20261015")
API = Path(r"D:\Program Files\Lumerical\v241\api\python")
os.environ["PATH"] = str(API.parents[1] / "bin") + os.pathsep + os.environ["PATH"]
sys.path.insert(0, str(API))
C0 = 299792458.


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def scalar(value):
    return np.asarray(value).squeeze().item()


def evaluate(mode, script):
    mode.eval("cal_error='';try {\n" + script + "\n} catch(cal_error);")
    error = mode.getv("cal_error")
    if error:
        raise RuntimeError(str(error))


def fde_script(wavelength_nm):
    return f"""switchtolayout; selectall; delete;
addfde;
set('solver type','1D Z:X prop');
set('z span',3e-6); set('mesh cells z',2000);
set('z min bc','Metal'); set('z max bc','Metal');
set('wavelength',{wavelength_nm}e-9);
set('number of trial modes',6);
set('use max index',0); set('n',3.0);
addrect; set('name','substrate');
set('material','SiO2 (Glass) - Palik');
set('x span',2e-6); set('y span',2e-6);
set('z',-1.11e-6); set('z span',2e-6);
addrect; set('name','core');
set('material','Si (Silicon) - Palik');
set('x span',2e-6); set('y span',2e-6);
set('z',0); set('z span',.22e-6);
findmodes;
"""


def select_te(mode, bulk_index):
    modes = []
    for idx in range(1, int(scalar(mode.nummodes())) + 1):
        name = f"FDE::data::mode{idx}"
        n = complex(scalar(mode.getdata(name, "neff")))
        e = mode.getresult(name, "E")
        field = np.asarray(e["E"]).reshape(-1, 3)
        if not np.isfinite(field).all():
            continue
        power = np.sum(abs(field)**2, axis=0)
        if power.sum() == 0:
            continue
        frac = float(power[1]/power.sum())
        if frac > .8 and 1.45 < n.real < bulk_index.real:
            modes.append((n, name, frac, e))
    if not modes:
        raise ValueError("No finite TE guided slab mode found")
    return max(modes, key=lambda item: item[0].real)


def probe(mode, wl_nm, out):
    bulk = complex(scalar(mode.getindex("Si (Silicon) - Palik", C0/(wl_nm*1e-9))))
    script = fde_script(wl_nm)
    (out / "neff.lsf").write_text(script, encoding="utf-8")
    evaluate(mode, script)
    n, name, frac, fields = select_te(mode, bulk)
    loss = float(scalar(mode.getdata(name, "loss")))
    np.testing.assert_allclose(scalar(mode.getnamed("FDE", "wavelength")), wl_nm*1e-9, rtol=1e-12)
    mode.eval(f"mode_E=getresult('{name}','E');matlabsave('mode_E.mat',mode_E);")
    mode.save("neff")
    np.savez(out / "mode_field.npz", z=np.asarray(fields["z"]), E=np.asarray(fields["E"]))
    mode.switchtolayout()
    mode.setnamed("FDE", "mesh cells z", 1000)
    mode.findmodes()
    nc, _, _, _ = select_te(mode, bulk)
    mode.switchtolayout()
    mode.setnamed("FDE", "mesh cells z", 2000)
    mode.setnamed("FDE", "z span", 4e-6)
    mode.findmodes()
    nb, _, _, _ = select_te(mode, bulk)
    if abs(n-nc) > 1e-3 or abs(n-nb) > 1e-3:
        raise ValueError(f"FDE convergence failed: fine={n}, coarse={nc}, wider={nb}")
    result = dict(wavelength_nm=wl_nm, silicon_n_real=bulk.real, silicon_n_imag=bulk.imag,
                  neff_real=n.real, neff_imag=n.imag, TE_Ey_fraction=frac,
                  mode_loss_db_per_m=loss, estimated_500um_loss_db=loss*.0005,
                  mesh_1000_2000_neff_abs_delta=abs(n-nc),
                  boundary_3um_4um_neff_abs_delta=abs(n-nb), mode_field_finite=True)
    save_json(out / "mode_probe.json", result)
    print(json.dumps(result), flush=True)
    return result


def local_script(wl_nm):
    return f"""switchtolayout; selectall; delete;
addvarfdtd;
set('x',0); set('x span',.3e-6);
set('y',0); set('y span',8e-6);
set('z',0); set('z span',1.1e-6);
set('x min bc','Periodic'); set('x max bc','Periodic');
set('mesh accuracy',7); set('simulation time',1e-12);
set('auto shutoff min',1e-7);
if (haveproperty('y0')) {{ set('y0',-3e-6); }}
addrect; set('name','core');
set('material','Si (Silicon) - Palik');
set('x',0); set('x span',1e-6); set('y',0); set('y span',10e-6);
set('z',0); set('z span',.22e-6);
addrect; set('name','substrate');
set('material','SiO2 (Glass) - Palik');
set('x',0); set('x span',1e-6); set('y',0); set('y span',10e-6);
set('z',-1.11e-6); set('z span',2e-6);
addrect; set('name','slot');
set('material','SiO2 (Glass) - Palik');
set('override mesh order from material database',1); set('mesh order',1);
set('x',0); set('x span',.15e-6); set('y',0); set('y span',.1e-6);
set('z',0); set('z span',.22e-6); set('enabled',0);
addpower; set('name','out'); set('monitor type','linear x');
set('x',0); set('x span',.3e-6); set('y',2.5e-6); set('z',0);
addpower; set('name','in'); set('monitor type','linear x');
set('x',0); set('x span',.3e-6); set('y',-2.5e-6); set('z',0);
addplane; set('name','source'); set('injection axis','y-axis'); set('direction','Forward');
set('x',0); set('x span',1e-6); set('y',-3.5e-6); set('z',0);
set('override global source settings',1); set('set wavelength',1);
set('wavelength start',{wl_nm}e-9); set('wavelength stop',{wl_nm}e-9);
set('set time domain',1); set('frequency',{C0/(wl_nm*1e-9):.16g});
set('pulselength',40e-15); set('offset',120e-15);
setglobalmonitor('use source limits',1); setglobalmonitor('frequency points',1);
save('local_scan');
"""


def sample(mode, wl_nm, length, mesh=7, duration=1e-12):
    mode.switchtolayout()
    mode.setnamed("varFDTD", "mesh accuracy", mesh)
    mode.setnamed("varFDTD", "simulation time", duration)
    mode.setnamed("slot", "enabled", int(length > 0))
    mode.setnamed("slot", "y span", max(length, .1)*1e-6)
    mode.run()
    f = np.asarray(mode.getdata("out", "f")).reshape(-1)
    np.testing.assert_allclose(f, [C0/(wl_nm*1e-9)], rtol=1e-11)
    ho = np.asarray(mode.getdata("out", "Hz")).reshape(-1)
    hi = np.asarray(mode.getdata("in", "Hz")).reshape(-1)
    if not np.isfinite(ho).all() or not np.isfinite(hi).all() or np.vdot(hi, hi).real < 1e-20:
        raise ValueError(f"Invalid monitor fields at {wl_nm} nm, length={length}")
    transfer = np.vdot(hi, ho)/np.vdot(hi, hi)
    # Normalize transmitted output against the no-slot output under the same
    # incident source. The total field at the upstream monitor includes a
    # reflected wave and must not be used as the incident amplitude per point.
    return complex(transfer), float(f[0]), complex(np.mean(ho))


def analyze(out):
    p = json.loads((out / "mode_probe.json").read_text())
    rows = json.loads((out / "scan_points.json").read_text())
    if len(rows) != 21:
        raise ValueError(f"Expected 21 points, got {len(rows)}")
    x = np.array([r["slot_length_um"] for r in rows])
    t = np.array([complex(r["relative_re"], r["relative_im"]) for r in rows])
    phase = np.unwrap(np.angle(t))/np.pi
    slope, intercept = np.polyfit(x, phase, 1)
    fit = slope*x+intercept
    residual = float(abs(phase-fit).max())
    r2 = float(1-np.sum((phase-fit)**2)/np.sum((phase-phase.mean())**2))
    amp = abs(t)
    checks = json.loads((out / "spot_checks.json").read_text())
    max_check_phase = max(abs(c["phase_difference_pi"]) for c in checks)
    max_check_amplitude = max(abs(c["amplitude_difference"]) for c in checks)
    baseline_predicted = float(np.exp(-2*np.pi*p["neff_imag"]/(p["wavelength_nm"]*.001)*5))
    baseline_error = float(abs(rows[0]["transfer_abs"]/baseline_predicted-1))
    if baseline_error > .05:
        raise ValueError(f"FDE/varFDTD baseline mismatch {baseline_error}")
    result = dict(**p, mapping_factor_um_per_pi=float(1/slope), phase_slope_pi_per_um=float(slope),
                  phase_intercept_pi=float(intercept), linear_fit_r2=r2, max_phase_residual_pi=residual,
                  relative_amplitude_min=float(amp.min()), relative_amplitude_max=float(amp.max()),
                  phase_step_max_pi=float(abs(np.diff(phase)).max()),
                  baseline_transfer_abs=rows[0]["transfer_abs"], baseline_predicted_abs=baseline_predicted,
                  baseline_relative_error=baseline_error,
                  monitor_wavelength_nm=C0/rows[0]["monitor_frequency_hz"]*1e9,
                  linear_phase_only_fit_acceptable=bool(r2 >= .99 and residual <= .1 and amp.min() >= .9 and amp.max() <= 1.1),
                  response_normalization="output_field / no_slot_output_field, identical source",
                  max_spot_check_phase_difference_pi=max_check_phase,
                  max_spot_check_amplitude_difference=max_check_amplitude,
                  spot_check_thresholds=dict(phase_pi=.02,amplitude=.03),
                  spot_checks_acceptable=bool(max_check_phase <= .02 and max_check_amplitude <= .03),
                  spot_checks=checks, analytic_fallback_used=False,
                  scope="TE slab mode and periodic local slot response; full network map/training not run",
                  completed_at=datetime.now().astimezone().isoformat())
    save_json(out / "calibration_results.json", result)
    np.savez(out / "calibration_response.npz", wavelength_nm=p["wavelength_nm"], slot_length_um=x,
             relative_complex_transmission=t, phase_pi=phase, fitted_phase_pi=fit)
    data = np.column_stack([x,t.real,t.imag,amp,phase,fit])
    np.savetxt(out / "complex_response.csv", data, delimiter=",", comments="",
               header="slot_length_um,relative_re,relative_im,relative_amplitude,phase_pi,fitted_phase_pi")
    return result


def run_wavelength(wl_nm):
    import lumapi
    out = ROOT / f"wl{wl_nm}nm"
    out.mkdir(exist_ok=True)
    if (out / "calibration_results.json").exists():
        print(f"{wl_nm} nm already complete; reusing saved result", flush=True)
        return analyze(out)
    if (out / "scan_points.json").exists():
        previous = out / ("previous_attempt_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        previous.mkdir()
        for file in out.iterdir():
            if file.is_file():
                shutil.copy2(file, previous / file.name)
    sim = Path(tempfile.mkdtemp(prefix=f"d2nn_{wl_nm}nm_"))
    save_json(out / "status.json", dict(status="running", solver_dir=str(sim)))
    print(f"{wl_nm} nm solver_dir={sim}", flush=True)
    try:
        with lumapi.MODE(hide=True) as mode:
            mode.cd(str(sim))
            probe(mode, wl_nm, out)
            script = local_script(wl_nm)
            (out / "local_scan.lsf").write_text(script, encoding="utf-8")
            evaluate(mode, script)
            rows = []
            for length in np.round(np.arange(21)*.1, 8):
                transfer, freq, output = sample(mode, wl_nm, float(length))
                if length == 0:
                    base_output = output
                t = output/base_output
                rows.append(dict(slot_length_um=float(length), relative_re=t.real, relative_im=t.imag,
                                 transfer_abs=abs(transfer), monitor_frequency_hz=freq,
                                 output_re=output.real, output_im=output.imag,
                                 normalization="output_field / no_slot_output_field, identical source"))
                save_json(out / "scan_points.json", rows)
                print(f"{wl_nm} nm {len(rows)}/21 length={length:.1f} um |t/t0|={abs(t):.6f}", flush=True)
            checks = []
            for label, mesh, duration, length in [("mesh8_L0.5",8,1e-12,.5), ("mesh8_L2",8,1e-12,2.),
                                                 ("duration2ps_L2",7,2e-12,2.)]:
                _, _, baseline = sample(mode, wl_nm, 0, mesh, duration)
                _, _, check = sample(mode, wl_nm, length, mesh, duration)
                t = check/baseline
                original = rows[int(round(length*10))]
                original_t = complex(original["relative_re"], original["relative_im"])
                checks.append(dict(label=label, relative_re=t.real, relative_im=t.imag,
                                   amplitude_difference=float(abs(t)-abs(original_t)),
                                   phase_difference_pi=float(np.angle(t/original_t)/np.pi)))
                save_json(out / "spot_checks.json", checks)
                print(f"{wl_nm} nm check {label}: {checks[-1]}", flush=True)
            mode.switchtolayout()
            mode.save("local_scan")
        saved = out / "solver_files"
        saved.mkdir(exist_ok=True)
        for file in sim.iterdir():
            if file.suffix in (".lms", ".mat", ".log"):
                shutil.copy2(file, saved / file.name)
        result = analyze(out)
        save_json(out / "status.json", dict(status="completed", solver_dir=str(sim)))
        return result
    except Exception as exc:
        save_json(out / "status.json", dict(status="failed", solver_dir=str(sim), error=str(exc)))
        raise


def summarize(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,1,figsize=(9,8),sharex=True)
    for r in results:
        wl = r["wavelength_nm"]
        color = {1310:"#1f77b4",1610:"#e67e22"}[wl]
        data = np.load(ROOT / f"wl{wl}nm" / "calibration_response.npz")
        x = data["slot_length_um"]
        axes[0].plot(x,data["phase_pi"],"o-",color=color,label=f"{wl} nm, R2={r['linear_fit_r2']:.5f}")
        axes[0].plot(x,data["fitted_phase_pi"],"--",color=color,alpha=.65)
        axes[1].plot(x,abs(data["relative_complex_transmission"]),"o-",color=color,label=f"{wl} nm")
    axes[0].set_ylabel("Relative phase / pi"); axes[0].legend()
    axes[1].set_ylabel("Relative field amplitude"); axes[1].set_xlabel("Slot length (um)"); axes[1].legend()
    fig.suptitle("1310 / 1610 nm: 220 nm Si, SiO2-filled slots, 300 nm pitch")
    fig.tight_layout(); fig.savefig(ROOT / "calibration_comparison.png",dpi=180); plt.close(fig)
    save_json(ROOT / "calibration_comparison.json", results)
    save_json(ROOT / "calibration_parameters.json", dict(
        wavelengths_um=[r["wavelength_nm"]*.001 for r in results],
        n_slab_by_wavelength_um={f"{r['wavelength_nm']*.001:.2f}":r["neff_real"] for r in results},
        neff_imag_by_wavelength_um={f"{r['wavelength_nm']*.001:.2f}":r["neff_imag"] for r in results},
        mapping_factor_by_wavelength_um={f"{r['wavelength_nm']*.001:.2f}":r["mapping_factor_um_per_pi"] for r in results},
        response_files={f"{r['wavelength_nm']*.001:.2f}":f"wl{r['wavelength_nm']}nm/calibration_response.npz" for r in results},
        note="Linear mapping factors are fit summaries; use measured complex responses when amplitude variation matters. This is not a full-network mapped package."))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["prepare","run","analyze"])
    parser.add_argument("--wavelength-nm",type=int,choices=[1310,1610],default=None)
    args=parser.parse_args()
    shape_path=SOURCE / "map" / "shape_spec.json"
    shape=json.loads(shape_path.read_text(encoding="utf-8"))["shape_config"]
    assert shape["propagation_thickness_um"] == .22 and shape["unit_spacing_um"] == .3
    assert shape["unit_span_um"] == .15 and shape["fill_material"] == "SiO2 (Glass) - Palik"
    save_json(ROOT / "config.json",dict(source_project=str(SOURCE),source_shape_sha256=hashlib.sha256(shape_path.read_bytes()).hexdigest(),
        original_shape=shape,wavelengths_nm=[1310,1610],slot_lengths_um=np.round(np.arange(21)*.1,8).tolist(),
        solver="Lumerical v241 MODE FDE + TE varFDTD",fde=dict(mesh_cells_z=2000,span_z_um=3,boundary="Metal",search_n=3.,trial_modes=6,
        checks="1000 vs 2000 cells; 3 vs 4 um z span"),local=dict(pitch_um=.3,slot_width_um=.15,mesh_order_slot=1,
        boundary_x="Periodic",span_y_um=8,monitor_y_um=[-2.5,2.5],source_y_um=-3.5,mesh_accuracy=7,
        pulse_fs=40,offset_fs=120,duration_fs=1000,auto_shutoff=1e-7,
        checks="mesh 8 at 0.5 and 2 um; 2 ps at 2 um; each check uses its own baseline"),analytic_fallback=False))
    if args.stage=="prepare":
        for wl in [1310,1610]:
            out=ROOT/f"wl{wl}nm"; out.mkdir(exist_ok=True)
            (out/"neff.lsf").write_text(fde_script(wl),encoding="utf-8")
            (out/"local_scan.lsf").write_text(local_script(wl),encoding="utf-8")
        print("Prepared independent 1310/1610 nm configurations and scripts")
        return
    wavelengths=[args.wavelength_nm] if args.wavelength_nm else [1310,1610]
    results=[run_wavelength(wl) if args.stage=="run" else analyze(ROOT/f"wl{wl}nm") for wl in wavelengths]
    if len(results)==2:
        summarize(results)
    print(json.dumps(results,indent=2),flush=True)


if __name__=="__main__":
    main()
