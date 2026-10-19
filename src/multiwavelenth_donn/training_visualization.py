from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _history_epochs(history: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([int(item["epoch"]) for item in history], dtype=int)


def _history_series(
    history: Sequence[Mapping[str, Any]],
    key: str,
    *,
    default: float = np.nan,
) -> np.ndarray:
    values: list[float] = []
    for item in history:
        if key in item:
            values.append(float(item[key]))
        else:
            values.append(float(default))
    return np.asarray(values, dtype=float)


def _resolve_model_y_axis_um(model: Any, field_size: int) -> np.ndarray:
    positions = getattr(model, "positions", None)
    if positions:
        axis = np.asarray(positions[0], dtype=float).reshape(-1)
        if axis.size == field_size:
            return axis
    span_um = float(getattr(model, "positions_span", [field_size])[0])
    return np.linspace(-0.5 * span_um, 0.5 * span_um, field_size, dtype=float)


def _save_loss_curve(
    history: Sequence[Mapping[str, Any]],
    *,
    out_dir: Path,
) -> Path:
    plt = _pyplot()
    epochs = _history_epochs(history)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    if "loss" in history[0]:
        loss = _history_series(history, "loss")
        ax.plot(epochs, loss, color="#2563eb", linewidth=1.5, label="loss")
        ax.set_ylabel("loss")
    else:
        train_loss = _history_series(history, "train_loss")
        eval_loss = _history_series(history, "eval_loss")
        ax.plot(epochs, train_loss, color="#2563eb", linewidth=1.5, label="train")
        ax.plot(epochs, eval_loss, color="#dc2626", linewidth=1.4, label="eval")
        ax.set_ylabel("loss")
        ax.legend(loc="best")

    ax.set_xlabel("epoch")
    ax.set_title("Training Loss Curve")
    ax.grid(alpha=0.2, linewidth=0.5)
    fig.tight_layout()

    out_path = out_dir / "loss_curve.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _save_accuracy_curve(
    history: Sequence[Mapping[str, Any]],
    *,
    out_dir: Path,
) -> Path | None:
    if not any("eval_accuracy" in item for item in history):
        return None

    plt = _pyplot()
    epochs = _history_epochs(history)
    train_accuracy = _history_series(history, "train_accuracy")
    eval_accuracy = _history_series(history, "eval_accuracy")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(epochs, train_accuracy, color="#2563eb", linewidth=1.5, label="train")
    ax.plot(epochs, eval_accuracy, color="#dc2626", linewidth=1.4, label="eval")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Training Accuracy Curve")
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.legend(loc="best")
    fig.tight_layout()

    out_path = out_dir / "accuracy_curve.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _save_output_trace_plots(
    history: Sequence[Mapping[str, Any]],
    *,
    out_dir: Path,
) -> Dict[str, Path]:
    if "transformed_output" not in history[0]:
        return {}

    plt = _pyplot()
    epochs = _history_epochs(history)
    transformed = np.asarray([item["transformed_output"] for item in history], dtype=float)
    raw = np.asarray([item["raw_output"] for item in history], dtype=float)
    outputs: Dict[str, Path] = {}

    wavelength_keys: list[str] = []
    output_labels = [f"port {idx}" for idx in range(transformed.shape[1])]
    final_xtick_labels = output_labels
    transformed_grouped: np.ndarray | None = None
    raw_grouped: np.ndarray | None = None
    num_ports_per_wavelength = 0
    prediction_by_wavelength = history[-1].get("reference_prediction_by_wavelength")
    if isinstance(prediction_by_wavelength, Mapping) and prediction_by_wavelength:
        wavelength_keys = [str(key) for key in prediction_by_wavelength.keys()]
        first_values = np.asarray(
            prediction_by_wavelength[wavelength_keys[0]],
            dtype=float,
        ).reshape(-1)
        num_ports_per_wavelength = int(first_values.size)
        expected = len(wavelength_keys) * num_ports_per_wavelength
        if expected == transformed.shape[1] and expected == raw.shape[1]:
            output_labels = [
                f"{wavelength_key} um P{port_idx}"
                for wavelength_key in wavelength_keys
                for port_idx in range(num_ports_per_wavelength)
            ]
            final_xtick_labels = [
                f"{wavelength_key}\nP{port_idx}"
                for wavelength_key in wavelength_keys
                for port_idx in range(num_ports_per_wavelength)
            ]
            transformed_grouped = transformed.reshape(
                transformed.shape[0],
                len(wavelength_keys),
                num_ports_per_wavelength,
            )
            raw_grouped = raw.reshape(
                raw.shape[0],
                len(wavelength_keys),
                num_ports_per_wavelength,
            )

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for port_idx in range(transformed.shape[1]):
        ax.plot(
            epochs,
            transformed[:, port_idx],
            linewidth=1.4,
            label=output_labels[port_idx],
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("normalized output")
    ax.set_title("Output Response During Training")
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    transformed_path = out_dir / "output_ports_vs_epoch.png"
    fig.savefig(transformed_path, dpi=180)
    plt.close(fig)
    outputs["output_ports_vs_epoch"] = transformed_path

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for port_idx in range(raw.shape[1]):
        ax.plot(
            epochs,
            raw[:, port_idx],
            linewidth=1.4,
            label=output_labels[port_idx],
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("raw output")
    ax.set_title("Raw Output Response During Training")
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    raw_path = out_dir / "raw_output_ports_vs_epoch.png"
    fig.savefig(raw_path, dpi=180)
    plt.close(fig)
    outputs["raw_output_ports_vs_epoch"] = raw_path

    final_values = transformed[-1]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(np.arange(final_values.size), final_values, color="#7c3aed")
    ax.set_xlabel("wavelength / physical output port")
    ax.set_ylabel("normalized output")
    ax.set_title("Final Output Response")
    ax.set_xticks(np.arange(final_values.size))
    ax.set_xticklabels(final_xtick_labels, rotation=0 if len(final_xtick_labels) <= 4 else 30)
    fig.tight_layout()
    final_path = out_dir / "final_output_ports.png"
    fig.savefig(final_path, dpi=180)
    plt.close(fig)
    outputs["final_output_ports"] = final_path

    if transformed_grouped is not None and raw_grouped is not None:
        fig, axes = plt.subplots(
            num_ports_per_wavelength,
            1,
            figsize=(9.0, 3.2 * num_ports_per_wavelength),
            squeeze=False,
            sharex=True,
        )
        for port_idx in range(num_ports_per_wavelength):
            ax = axes[port_idx, 0]
            for wl_idx, wavelength_key in enumerate(wavelength_keys):
                ax.plot(
                    epochs,
                    transformed_grouped[:, wl_idx, port_idx],
                    linewidth=1.4,
                    label=f"{wavelength_key} um",
                )
            ax.set_ylabel(f"P{port_idx}")
            ax.set_title(f"Physical Output Port {port_idx} Across Wavelengths")
            ax.grid(alpha=0.2, linewidth=0.5)
            ax.legend(loc="best")
        axes[-1, 0].set_xlabel("epoch")
        fig.tight_layout()
        grouped_trace_path = out_dir / "output_by_physical_port_vs_epoch.png"
        fig.savefig(grouped_trace_path, dpi=180)
        plt.close(fig)
        outputs["output_by_physical_port_vs_epoch"] = grouped_trace_path

        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        port_positions = np.arange(num_ports_per_wavelength)
        width = 0.8 / len(wavelength_keys)
        for wl_idx, wavelength_key in enumerate(wavelength_keys):
            ax.bar(
                port_positions - 0.4 + (wl_idx + 0.5) * width,
                transformed_grouped[-1, wl_idx, :],
                width=width,
                label=f"{wavelength_key} um",
            )
        ax.set_xlabel("physical output port")
        ax.set_ylabel("normalized output")
        ax.set_title("Final Output Grouped By Physical Port")
        ax.set_xticks(port_positions)
        ax.set_xticklabels([f"P{idx}" for idx in range(num_ports_per_wavelength)])
        ax.legend(loc="best")
        fig.tight_layout()
        grouped_final_path = out_dir / "final_output_grouped_by_physical_port.png"
        fig.savefig(grouped_final_path, dpi=180)
        plt.close(fig)
        outputs["final_output_grouped_by_physical_port"] = grouped_final_path
    return outputs


def _save_confusion_matrix_plot(
    history: Sequence[Mapping[str, Any]],
    *,
    out_dir: Path,
) -> Path | None:
    if "confusion_matrix" not in history[-1]:
        return None

    matrix = np.asarray(history[-1]["confusion_matrix"], dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title("Final Eval Confusion Matrix")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, int(matrix[row, col]), ha="center", va="center", color="#111827")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    out_path = out_dir / "confusion_matrix.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _save_delta_phi_plots(model: Any, *, out_dir: Path) -> Dict[str, Any]:
    plt = _pyplot()
    slot_length_um = list(getattr(model, "slot_length_um", []))
    delta_phi = list(getattr(model, "delta_phi", []))
    values_by_layer = slot_length_um if slot_length_um else delta_phi
    if not values_by_layer:
        raise ValueError("model.delta_phi is empty")
    label = "physical length (um)" if slot_length_um else "delta_phi"
    title = "Physical Length" if slot_length_um else "Delta Phi"

    line_paths: list[str] = []
    hist_paths: list[str] = []
    for layer_idx, param in enumerate(values_by_layer):
        values = torch.as_tensor(param).detach().cpu().numpy().reshape(-1)

        fig, ax = plt.subplots(figsize=(9.0, 4.2))
        ax.plot(np.arange(values.size), values, color="#0f766e", linewidth=1.1)
        ax.set_xlabel("neuron index")
        ax.set_ylabel(label)
        ax.set_title(f"Layer {layer_idx} {title}")
        ax.grid(alpha=0.2, linewidth=0.5)
        fig.tight_layout()
        line_path = out_dir / f"layer_{layer_idx}_delta_phi.png"
        fig.savefig(line_path, dpi=180)
        plt.close(fig)
        line_paths.append(str(line_path))

        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        ax.hist(values, bins=min(max(values.size // 2, 8), 40), color="#c2410c", alpha=0.85)
        ax.set_xlabel(label)
        ax.set_ylabel("count")
        ax.set_title(f"Layer {layer_idx} {title} Histogram")
        fig.tight_layout()
        hist_path = out_dir / f"layer_{layer_idx}_delta_phi_hist.png"
        fig.savefig(hist_path, dpi=180)
        plt.close(fig)
        hist_paths.append(str(hist_path))

    return {
        "delta_phi_lines": line_paths,
        "delta_phi_histograms": hist_paths,
    }


def _save_reference_field_plots(model: Any, *, out_dir: Path) -> Dict[str, Any]:
    plt = _pyplot()
    fields = getattr(model, "us_H", None)
    if fields is None:
        fields = getattr(model, "us", None)
    if fields is None:
        raise AttributeError("model does not expose propagated fields")

    amplitude_paths: list[str] = []
    phase_paths: list[str] = []
    for layer_idx, field in enumerate(fields):
        values = torch.as_tensor(field).detach().cpu().numpy().reshape(-1)
        y_axis_um = _resolve_model_y_axis_um(model, values.size)

        fig, ax = plt.subplots(figsize=(9.0, 4.2))
        ax.plot(y_axis_um, np.abs(values), color="#2563eb", linewidth=1.2)
        ax.set_xlabel("lateral y (um)")
        ax.set_ylabel("|E|")
        ax.set_title(f"Reference Layer {layer_idx} Amplitude")
        ax.grid(alpha=0.2, linewidth=0.5)
        fig.tight_layout()
        amp_path = out_dir / f"reference_layer_{layer_idx}_amplitude.png"
        fig.savefig(amp_path, dpi=180)
        plt.close(fig)
        amplitude_paths.append(str(amp_path))

        fig, ax = plt.subplots(figsize=(9.0, 4.2))
        ax.plot(y_axis_um, np.angle(values), color="#dc2626", linewidth=1.1)
        ax.set_xlabel("lateral y (um)")
        ax.set_ylabel("phase (rad)")
        ax.set_title(f"Reference Layer {layer_idx} Phase")
        ax.grid(alpha=0.2, linewidth=0.5)
        fig.tight_layout()
        phase_path = out_dir / f"reference_layer_{layer_idx}_phase.png"
        fig.savefig(phase_path, dpi=180)
        plt.close(fig)
        phase_paths.append(str(phase_path))

    return {
        "reference_field_amplitudes": amplitude_paths,
        "reference_field_phases": phase_paths,
    }


def save_training_visualizations(
    *,
    model: Any,
    history: Sequence[Mapping[str, Any]],
    out_dir: str | Path,
) -> Dict[str, Any]:
    figures_dir = Path(out_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    saved: Dict[str, Any] = {}
    saved["loss_curve"] = str(_save_loss_curve(history, out_dir=figures_dir))
    accuracy_curve = _save_accuracy_curve(history, out_dir=figures_dir)
    if accuracy_curve is not None:
        saved["accuracy_curve"] = str(accuracy_curve)
    for key, path in _save_output_trace_plots(history, out_dir=figures_dir).items():
        saved[key] = str(path)
    confusion_matrix_plot = _save_confusion_matrix_plot(history, out_dir=figures_dir)
    if confusion_matrix_plot is not None:
        saved["confusion_matrix"] = str(confusion_matrix_plot)
    saved.update(_save_delta_phi_plots(model, out_dir=figures_dir))
    saved.update(_save_reference_field_plots(model, out_dir=figures_dir))
    saved["figures_dir"] = str(figures_dir)
    return saved
