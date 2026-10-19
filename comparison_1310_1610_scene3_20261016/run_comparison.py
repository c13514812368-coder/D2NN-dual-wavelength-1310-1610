"""Matched calibration-informed scalar DONN experiments, not whole-device FDTD.

Run with Python 3.13 and torch/numpy/scipy/scikit-learn/matplotlib/joblib.
All lengths are in micrometres. Original data and results are read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix, classification_report

BASE = Path(__file__).resolve().parent
HERE = BASE / "resolved_grid10000"
PROJECT = BASE.parent
CAL = PROJECT / "calibration_1310_1610nm_20261016"
COARSE = ["natural_landform", "water_coast", "built_environment"]
FINE = ["n09193705_alp", "n09468604_valley", "n09472597_volcano",
        "n09256479_coral_reef", "n09332890_lakeside", "n09428293_seashore",
        "n03028079_church", "n02980441_castle", "n03661043_library"]
CONFIG = dict(model="calibrated scalar angular spectrum; analytic cosine apertures",
              whole_device_fdtd=False, wavelengths_nm=[1310, 1610], width_um=500.,
              gap_um=125., gaps=4, trainable_layers=3, groups_per_layer=555,
              group_pitch_um=.9, slot_pitch_um=.3, repeats=3, slot_width_um=.15,
              group_coverage_um=499.5, edge_margin_um=.25,
              grid_n=10000, dy_um=.05, padding_factor=2,
              input_centers_um=list(np.arange(-90., 91., 12.)), input_width_um=10.,
              output_centers_um=list(np.arange(-90., 91., 20.)), output_width_um=12.,
              slot_bounds_um=[0., 2.], interpolation="linear amplitude and unwrapped phase",
              seed=42, epochs=20, batch_size=64, cpu_threads=4,
              optimizer="Adam", learning_rate=.0008, weight_decay=.0001,
              cosine_eta_min=.00003, gradient_clip=1., input_noise_pi=.005,
              initialization="0.5 um + Normal(0,0.02 um), matched across wavelengths",
              selection="minimum validation CE among epochs with maximum validation accuracy",
              train_samples=7200, val_samples=1260, independent_test=False,
              caveats=["Periodic uniform-cell calibration used as a local thin transmission screen; ignores nonlocal coupling/back-reflections and angle dependence.",
                       "Analytic input/output modes, no new wavelength-specific full taper/port FDTD map.",
                       "555 physical groups resolve inconsistent archived 1500-group/500um geometry; not directly comparable to archived 1560nm accuracy.",
                       "Frozen original 16-channel encoding is supervised on coarse labels for the main task.",
                       "Fine task uses a separate training-only 9-label encoder, shared between wavelengths.",
                       "Separate trained masks do not establish a shared dual-wavelength device."])


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fine_labels():
    info = json.loads((PROJECT / "feature_upgrade_20261011/diagnostics.json").read_text(encoding="utf-8"))
    ys = []
    for split in ("train", "val"):
        paths = info["path_manifest"][split]
        labels = []
        for old in paths:
            relative = old.split("/" + split + "/", 1)[1]
            assert (PROJECT / "data/imagenet_1k_sd_scene_3coarse_fixed" / split / relative).is_file()
            coarse, fine, _ = relative.split("/")
            label = FINE.index(fine)
            assert COARSE.index(coarse) == label // 3
            labels.append(label)
        ys.append(np.array(labels, dtype=np.int64))
    return ys


def dataset(task):
    original = np.load(PROJECT / "feature_upgrade_20261011/encoded16.npz")
    yt, yv = fine_labels()
    np.testing.assert_array_equal(yt // 3, original["y_train"])
    np.testing.assert_array_equal(yv // 3, original["y_eval"])
    if task == "coarse3":
        return (original["x_train"], original["x_eval"], original["y_train"], original["y_eval"]), COARSE
    cache = HERE / "fine_encoded16.npz"
    if not cache.exists():
        from sklearn.preprocessing import StandardScaler
        from sklearn.kernel_approximation import Nystroem
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.decomposition import PCA
        import joblib
        raw = np.load(PROJECT / "feature_upgrade_20261011/raw_features.npz")
        np.testing.assert_array_equal(raw["yt"], original["y_train"])
        np.testing.assert_array_equal(raw["yv"], original["y_eval"])
        scaler = StandardScaler().fit(raw["xt"])
        a, b = scaler.transform(raw["xt"]), scaler.transform(raw["xv"])
        kernel = Nystroem(kernel="rbf", gamma=.002, n_components=384, random_state=42).fit(a)
        lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto", n_components=8).fit(kernel.transform(a), yt)
        pca = PCA(n_components=8, random_state=42).fit(a)
        a = np.concatenate([lda.transform(kernel.transform(a)), pca.transform(a)], axis=1)
        b = np.concatenate([lda.transform(kernel.transform(b)), pca.transform(b)], axis=1)
        lo, hi = np.percentile(a, [.5, 99.5], axis=0)
        span = np.maximum(hi-lo, 1e-6)
        a = np.clip(.2 + 1.6*(a-lo)/span, .2, 1.8).astype(np.float32)
        b = np.clip(.2 + 1.6*(b-lo)/span, .2, 1.8).astype(np.float32)
        np.savez_compressed(cache, x_train=a, x_eval=b, y_train=yt, y_eval=yv)
        joblib.dump(dict(scaler=scaler, kernel=kernel, lda=lda, pca=pca, low=lo, high=hi, classes=FINE), HERE / "fine_encoder.joblib")
        save_json(HERE / "fine_encoding.json", dict(fit="train_only", labels="nine fine subclasses", components="RBF384 -> LDA8 + standardized raw PCA8", phase_pi_range=[.2, 1.8], selection="predeclared; no validation hyperparameter search"))
    d = np.load(cache)
    return tuple(d[k] for k in ("x_train", "x_eval", "y_train", "y_eval")), FINE


class CalibratedDONN(nn.Module):
    def __init__(self, wl_nm, seed=42, n=10000, pad=2, slots=None):
        super().__init__()
        self.wl_nm, self.n, self.pad = int(wl_nm), n, pad
        self.dy = 500. / n
        self.start = (n * pad - n) // 2
        response = np.load(CAL / f"wl{wl_nm}nm/calibration_response.npz")
        probe = json.loads((CAL / f"wl{wl_nm}nm/mode_probe.json").read_text())
        self.neff = probe["neff_real"]
        self.register_buffer("lut_amp", torch.tensor(abs(response["relative_complex_transmission"]), dtype=torch.float32))
        self.register_buffer("lut_phase", torch.tensor(response["phase_pi"] * np.pi, dtype=torch.float32))
        np.testing.assert_allclose(response["slot_length_um"], np.arange(21) * .1, atol=1e-12)
        y = (np.arange(n) - n / 2 + .5) * self.dy
        group = np.floor((y + 249.75) / .9).astype(int)
        active = (y >= -249.75) & (y < 249.75)
        self.register_buffer("group_idx", torch.tensor(group.clip(0, 554), dtype=torch.long))
        self.register_buffer("group_active", torch.tensor(active))
        self.register_buffer("y", torch.tensor(y, dtype=torch.float32))
        def apertures(centers, width):
            delta = y[:, None] - np.asarray(centers)[None, :]
            m = np.where(abs(delta) < width / 2, np.cos(np.pi * delta / width), 0.)
            m /= np.sqrt((m*m).sum(axis=0, keepdims=True)*self.dy)
            return torch.tensor(m, dtype=torch.complex64)
        self.register_buffer("input_modes", apertures(np.arange(-90., 91., 12.), 10.))
        self.register_buffer("output_modes", apertures(np.arange(-90., 91., 20.), 12.))
        k = 2*np.pi*self.neff/(wl_nm*.001)
        ky = 2*np.pi*np.fft.fftfreq(n*pad, d=self.dy)
        kz = np.sqrt((k*k-ky*ky).astype(np.complex128))
        # Remove only the common carrier phase to improve complex64 precision.
        self.register_buffer("kz_relative", torch.tensor(kz-k, dtype=torch.complex128))
        self.register_buffer("H", torch.tensor(np.exp(1j*(kz-k)*125.), dtype=torch.complex64))
        generator = torch.Generator().manual_seed(seed)
        initial = .5 + .02*torch.randn(3, 555, generator=generator)
        self.lengths = nn.Parameter(initial if slots is None else torch.as_tensor(slots, dtype=torch.float32).clone())

    def masks(self, lengths=None):
        length = self.lengths if lengths is None else lengths
        q = length.clamp(0, 2) * 10
        idx = torch.floor(q).long().clamp(0, 19)
        t = q - idx
        amp = self.lut_amp[idx]*(1-t) + self.lut_amp[idx+1]*t
        phase = self.lut_phase[idx]*(1-t) + self.lut_phase[idx+1]*t
        transmission = torch.polar(amp, phase)[:, self.group_idx]
        transmission = torch.where(self.group_active[None, :], transmission, 1.)
        return F.pad(transmission, (self.start, self.n*self.pad-self.n-self.start), value=1.)

    def input_field(self, x):
        field = torch.polar(torch.ones_like(x), np.pi*x) @ self.input_modes.T / 4
        return F.pad(field, (self.start, self.n*self.pad-self.n-self.start))

    def propagate(self, u, distance=125.):
        h = self.H if distance == 125. else torch.exp(1j*self.kz_relative*distance).to(torch.complex64)
        return torch.fft.ifft(torch.fft.fft(u, dim=-1)*h, dim=-1)

    def ports(self, u):
        c = u[..., self.start:self.start+self.n] @ self.output_modes.conj() * self.dy
        return c.abs().square()

    def forward(self, x, trace=False, lengths=None):
        masks = self.masks(lengths)
        u = self.input_field(x)
        fields = [u] if trace else None
        for gap in range(4):
            if gap:
                u = u * masks[gap-1]
            u = self.propagate(u)
            if trace:
                fields.append(u)
        ports = self.ports(u)
        return (ports, fields) if trace else ports


def probabilities(ports, outputs):
    active = ports[:, :outputs]
    return active / active.sum(dim=1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def evaluate(model, x, y, outputs, lengths=None):
    ports = torch.cat([model(batch, lengths=lengths) for batch in x.split(64)]).numpy()
    p = ports[:, :outputs] / np.maximum(ports[:, :outputs].sum(1, keepdims=True), 1e-12)
    pred = p.argmax(1)
    truth = np.asarray(y)
    return dict(accuracy=float((pred == truth).mean()),
                ce=float(-np.log(np.maximum(p[np.arange(len(truth)), truth], 1e-12)).mean()),
                mean_active_collection_fraction=float(ports[:, :outputs].sum(1).mean()),
                mean_10port_collection_fraction=float(ports.sum(1).mean())), p, ports


def train(task, wl, seed, epochs=20):
    out = HERE / task / f"seed{seed}" / f"wl{wl}nm"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        print(f"SKIP completed {out}", flush=True)
        return
    arrays, names = dataset(task)
    xt, xv, yt, yv = [torch.tensor(a) for a in arrays]
    outputs = len(names)
    torch.manual_seed(seed)
    model = CalibratedDONN(wl, seed)
    opt = torch.optim.Adam(model.parameters(), lr=.0008, weight_decay=.0001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=.00003)
    generator = torch.Generator().manual_seed(seed+1000)
    history, best_key, best_epoch = [], (-1., -float("inf")), 0
    started = time.time()
    for epoch in range(1, epochs+1):
        model.train()
        permutation = torch.randperm(len(yt), generator=generator)
        total_loss, correct = 0., 0
        for indices in permutation.split(64):
            x = xt[indices] + torch.randn(xt[indices].shape, generator=generator)*.005
            opt.zero_grad(set_to_none=True)
            p = probabilities(model(x), outputs)
            loss = F.nll_loss(p.clamp_min(1e-12).log(), yt[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            if not torch.isfinite(model.lengths.grad).all():
                raise FloatingPointError("Nonfinite slot gradients")
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            with torch.no_grad():
                model.lengths.clamp_(0., 2.)
            total_loss += float(loss.detach())*len(indices)
            correct += int((p.argmax(1) == yt[indices]).sum())
        model.eval()
        metrics, _, _ = evaluate(model, xv, yv, outputs)
        row = dict(epoch=epoch, train_ce=total_loss/len(yt), train_noisy_accuracy=correct/len(yt),
                   val_accuracy=metrics["accuracy"], val_ce=metrics["ce"], learning_rate=opt.param_groups[0]["lr"], elapsed_s=time.time()-started)
        history.append(row)
        key = (metrics["accuracy"], -metrics["ce"])
        if key > best_key:
            best_key, best_epoch = key, epoch
            torch.save(dict(lengths_um=model.lengths.detach().clone(), wavelength_nm=wl, seed=seed, task=task,
                            epoch=epoch, config=CONFIG), out / "best_model.pt")
        save_json(out / "history.json", history)
        print(json.dumps(dict(task=task, wl=wl, seed=seed, **row)), flush=True)
        schedule.step()
    torch.save(dict(lengths_um=model.lengths.detach().clone(), wavelength_nm=wl, seed=seed, task=task, epoch=epochs, config=CONFIG), out / "final_model.pt")
    final, _, _ = evaluate(model, xv, yv, outputs)
    saved = torch.load(out / "best_model.pt", weights_only=False)
    with torch.no_grad():
        model.lengths.copy_(saved["lengths_um"])
    best, p, ports = evaluate(model, xv, yv, outputs)
    train_metrics, _, _ = evaluate(model, xt, yt, outputs)
    pred = p.argmax(1)
    np.savez_compressed(out / "validation_predictions.npz", probabilities=p, ports=ports, truth=np.asarray(yv), prediction=pred)
    np.save(out / "slot_lengths_um.npy", model.lengths.detach().numpy())
    summary = dict(task=task, wavelength_nm=wl, seed=seed, epochs=epochs, best_epoch=best_epoch,
                   best_validation=best, final_validation=final, best_training=train_metrics,
                   classes=names, confusion_matrix=confusion_matrix(yv, pred, labels=np.arange(outputs)).tolist(),
                   classification_report=classification_report(yv, pred, target_names=names, zero_division=0, output_dict=True),
                   slot_min_um=float(model.lengths.detach().min()), slot_max_um=float(model.lengths.detach().max()),
                   elapsed_s=time.time()-started, model_kind=CONFIG["model"], whole_device_fdtd=False)
    if outputs == 9:
        coarse_prob = p.reshape(-1, 3, 3).sum(2)
        summary["coarse_from_9ports_accuracy"] = float((coarse_prob.argmax(1) == np.asarray(yv)//3).mean())
        oracle = p.reshape(-1, 3, 3)[np.arange(len(p)), np.asarray(yv)//3].argmax(1)
        summary["within_true_coarse_fine_accuracy"] = float((oracle == np.asarray(yv)%3).mean())
        summary["within_true_coarse_fine_accuracy_per_group"] = {
            name: float((oracle[np.asarray(yv)//3 == c] == np.asarray(yv)[np.asarray(yv)//3 == c]%3).mean())
            for c, name in enumerate(COARSE)}
    save_json(out / "summary.json", summary)
    print("COMPLETED " + json.dumps({k: summary[k] for k in ("task", "wavelength_nm", "seed", "best_epoch", "best_validation", "elapsed_s")}), flush=True)


def preflight():
    torch.set_num_threads(4)
    records = {}
    for wl in (1310, 1610):
        m = CalibratedDONN(wl)
        x = torch.rand(4, 16)
        with torch.no_grad():
            incident = float((m.input_field(x).abs().square().sum(1)*m.dy).mean())
        loss = -probabilities(m(x), 3)[:, 0].clamp_min(1e-12).log().mean()
        loss.backward()
        assert torch.isfinite(m.lengths.grad).all() and float(m.lengths.grad.norm()) > 0
        with torch.no_grad():
            _, fields = m(x, trace=True)
            energy = [(u.abs().square().sum(1)*m.dy).numpy().tolist() for u in fields]
            assert incident < 1.00001 and incident > .99999
            assert np.all(np.diff(np.asarray(energy), axis=0) <= 1e-5)
            # Interpolator endpoints and length units: no extrapolation.
            assert float(m.lut_amp.max()) <= 1.00001
        records[str(wl)] = dict(input_energy=incident, plane_energies=energy, gradient_norm=float(m.lengths.grad.norm()))
    save_json(HERE / "preflight.json", records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["coarse3", "fine9"], choices=["coarse3", "fine9"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    HERE.mkdir(parents=True, exist_ok=True)
    config = dict(CONFIG, requested_seeds=args.seeds, requested_tasks=args.tasks, epochs=args.epochs)
    config["source_sha256"] = {str(p.relative_to(PROJECT)): sha(p) for p in [
        PROJECT / "feature_upgrade_20261011/encoded16.npz",
        PROJECT / "feature_upgrade_20261011/raw_features.npz",
        PROJECT / "feature_upgrade_20261011/diagnostics.json", PROJECT / "map/shape_spec.json",
        CAL / "wl1310nm/calibration_response.npz", CAL / "wl1610nm/calibration_response.npz"]}
    save_json(HERE / "config.json", config)
    preflight()
    if args.preflight_only:
        return
    for task in args.tasks:
        for seed in args.seeds:
            for wl in (1310, 1610):
                train(task, wl, seed, args.epochs)


if __name__ == "__main__":
    main()
