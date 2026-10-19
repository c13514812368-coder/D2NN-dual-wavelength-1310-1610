"""Actual varFDTD Hz fields of the same 0.15 x 0.5 um periodic slot.

Reuses the independently validated calibration setup without modifying it.
Stores raw fields, corresponding baseline and independent solver files.
"""
from pathlib import Path
import json
import shutil
import sys
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_comparison import BASE, CAL, save_json
sys.path.insert(0, str(CAL))
import run_calibration as calibration
import lumapi

OUT = BASE / "local_slot_fields"


def run():
    OUT.mkdir(exist_ok=True)
    results = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
    for row, wl in enumerate((1310, 1610)):
        dest = OUT / f"wl{wl}nm"
        dest.mkdir(exist_ok=True)
        cache = dest / "fields.npz"
        if not cache.exists():
            sim = Path(tempfile.mkdtemp(prefix=f"donn_fields_{wl}_"))
            script = calibration.local_script(wl) + """
addpower; set('name','field_map'); set('monitor type','2D Z-normal');
set('x',0); set('x span',.3e-6); set('y',0); set('y span',6e-6); set('z',0);
"""
            (dest / "geometry.lsf").write_text(script, encoding="utf-8")
            with lumapi.MODE(hide=True) as mode:
                mode.cd(str(sim))
                calibration.evaluate(mode, script)
                _, freq, baseline_out = calibration.sample(mode, wl, 0.)
                x = np.asarray(mode.getdata("field_map", "x")).ravel()
                y = np.asarray(mode.getdata("field_map", "y")).ravel()
                h0 = np.asarray(mode.getdata("field_map", "Hz")).squeeze()
                _, _, slotted_out = calibration.sample(mode, wl, .5)
                hs = np.asarray(mode.getdata("field_map", "Hz")).squeeze()
                assert h0.shape == hs.shape == (len(x), len(y))
                assert np.isfinite(h0).all() and np.isfinite(hs).all()
                mode.save("slot_fields")
                np.savez_compressed(cache, x_um=x*1e6, propagation_um=y*1e6,
                                    baseline_Hz=h0, slot_Hz=hs, baseline_out=baseline_out,
                                    slot_out=slotted_out, wavelength_nm=wl, monitor_frequency_hz=freq)
            for file in sim.iterdir():
                if file.suffix in (".lms", ".log"):
                    shutil.copy2(file, dest / file.name)
        data = np.load(cache)
        t = complex(data["slot_out"] / data["baseline_out"])
        lut = np.load(CAL / f"wl{wl}nm/calibration_response.npz")
        expected = complex(lut["relative_complex_transmission"][5])
        phase_delta_pi = float(np.angle(t/expected)/np.pi)
        amplitude_delta = float(abs(t)-abs(expected))
        if abs(phase_delta_pi) > .01 or abs(amplitude_delta) > .01:
            raise ValueError("New field monitor run does not match local calibration")
        x, y = data["x_um"], data["propagation_um"]
        h0, hs = data["baseline_Hz"], data["slot_Hz"]
        ref = float(np.mean(abs(h0[:, (y < -1.) & (y > -2.)])**2))
        relative_field = hs / h0
        for col in range(2):
            ax = axes[row, col]
            if col == 0:
                values = abs(hs)**2/ref
                im_amp = ax.pcolormesh(y, x, values, shading="auto", cmap="magma", vmin=0, vmax=2)
                title = "|Hz|^2 / incident baseline"
            else:
                values = np.angle(relative_field)/np.pi
                im_phase = ax.pcolormesh(y, x, values, shading="auto", cmap="twilight", vmin=-1, vmax=1)
                title = "Phase of Hz / baseline (wrapped / pi)"
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle((-.25, -.075), .5, .15, fill=False, edgecolor="cyan", lw=1.2))
            ax.axvline(-2.5, color="white", ls=":", lw=.6)
            ax.axvline(2.5, color="white", ls=":", lw=.6)
            ax.set(title=f"{wl} nm: {title}", xlabel="Propagation y (um)", ylabel="Transverse x (um)")
        results.append(dict(wavelength_nm=wl, slot_length_um=.5, slot_width_um=.15, periodic_pitch_um=.3,
                            transmitted_relative_amplitude=abs(t), transmitted_relative_wrapped_phase_pi=float(np.angle(t)/np.pi),
                            calibration_unwrapped_phase_pi=float(lut["phase_pi"][5]),
                            calibration_amplitude_difference=amplitude_delta, calibration_phase_difference_pi=phase_delta_pi,
                            field_shape=list(hs.shape), solver="Lumerical v241 MODE varFDTD", whole_device_fdtd=False,
                            note="Periodic identical-slot array, Hz field component. Intensity is a field-component diagnostic, not total power efficiency. Reflected/standing fields exist upstream; incident normalization uses no-slot baseline."))
        print(json.dumps(results[-1]), flush=True)
    fig.suptitle("Actual local varFDTD: same SiO2-filled slot in 220 nm Si", fontsize=13)
    fig.subplots_adjust(left=.08, right=.88, bottom=.09, top=.89, wspace=.3, hspace=.35)
    fig.colorbar(im_amp, cax=fig.add_axes([.91, .57, .016, .29]), label="Relative |Hz|^2")
    fig.colorbar(im_phase, cax=fig.add_axes([.91, .12, .016, .29]), label="Relative phase / pi")
    fig.savefig(OUT / "local_slot_field_comparison.png", dpi=180)
    plt.close(fig)
    save_json(OUT / "summary.json", results)


if __name__ == "__main__":
    run()
