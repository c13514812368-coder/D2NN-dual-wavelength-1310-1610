from __future__ import annotations

import copy
import importlib
import importlib.util
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .experiment_artifacts import (
    build_trained_network_package,
    load_package,
    save_json,
    save_package,
)
from .config_loading import (
    get_seed,
    load_project_config,
)
from .config_sections import (
    resolve_train_config,
)
from .training_visualization import save_training_visualizations
from .workspace_layout import (
    TrainingJobLayout,
    resolve_training_config_path,
    resolve_training_job_layout,
)
from .project_paths import get_project_root


REQUIRED_TRAIN_HOOKS = (
    "build_training_state",
    "train_one_epoch",
    "evaluate_model",
)
REQUIRED_LOCAL_STATE_KEYS = (
    "model",
    "reference_input",
)


def _select_record_metric(record: Mapping[str, Any]) -> float:
    if "selection_metric" in record:
        return float(record["selection_metric"])
    for key in ("eval_loss", "loss", "train_loss"):
        if key in record:
            return float(record[key])
    raise KeyError(
        "Training record does not contain selection_metric, eval_loss, loss, or train_loss; "
        "cannot choose the best checkpoint."
    )


def _format_epoch_log(record: Mapping[str, Any]) -> str:
    parts = [f"epoch={int(record['epoch']):04d}"]
    for key in ("train_loss", "eval_loss", "loss", "train_accuracy", "eval_accuracy"):
        if key not in record:
            continue
        value = float(record[key])
        if "accuracy" in key:
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value:.6e}")
    if "best_eval_accuracy_so_far" in record:
        parts.append(
            "best_eval_accuracy_so_far="
            f"{float(record['best_eval_accuracy_so_far']):.4f}"
        )
    if "best_eval_accuracy_epoch_so_far" in record:
        parts.append(
            "best_eval_accuracy_epoch_so_far="
            f"{int(record['best_eval_accuracy_epoch_so_far'])}"
        )
    if "transformed_output" in record:
        parts.append(f"pred={np.asarray(record['transformed_output'])}")
    if "target_output" in record:
        parts.append(f"target={np.asarray(record['target_output'])}")
    return " ".join(parts)


def _coerce_reference_input(model: torch.nn.Module, value: Any) -> torch.Tensor:
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    reference_input = torch.as_tensor(value, dtype=torch.float32, device=device)
    if reference_input.ndim == 1:
        reference_input = reference_input.unsqueeze(0)
    return reference_input


def _dump_reference_fields(
    model: torch.nn.Module,
    reference_input: torch.Tensor,
    *,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        reference_output = model(reference_input)

    fields = getattr(model, "us_H", None)
    if fields is None:
        fields = getattr(model, "us", None)
    if fields is None:
        raise AttributeError("Model did not expose propagated fields after forward()")

    input_path = out_dir / "reference_input.npy"
    output_path = out_dir / "reference_output.npy"
    np.save(input_path, reference_input.detach().cpu().numpy())
    np.save(output_path, torch.as_tensor(reference_output).detach().cpu().numpy())

    field_dir = out_dir / "reference_fields"
    field_dir.mkdir(parents=True, exist_ok=True)
    field_paths: list[str] = []
    for layer_idx, field in enumerate(fields):
        layer_path = field_dir / f"layer_{layer_idx}.npy"
        np.save(layer_path, torch.as_tensor(field).detach().cpu().numpy())
        field_paths.append(str(layer_path))

    return {
        "reference_input_path": str(input_path),
        "reference_output_path": str(output_path),
        "reference_field_paths": field_paths,
    }


def _load_local_training_module(train_script_path: Path) -> Any:
    module_name = (
        f"local_train_{train_script_path.parent.name}_{abs(hash(str(train_script_path)))}"
    )
    spec = importlib.util.spec_from_file_location(module_name, train_script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load training module from {train_script_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise RuntimeError(
            f"Failed while importing local training script {train_script_path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    return _validate_training_module(module, source=str(train_script_path))


def _load_configured_training_module(module_name: str) -> Any:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed while importing configured training module {module_name}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return _validate_training_module(module, source=f"configured module {module_name}")


def _validate_training_module(module: Any, *, source: str) -> Any:
    missing_hooks = [name for name in REQUIRED_TRAIN_HOOKS if not hasattr(module, name)]
    if missing_hooks:
        raise AttributeError(
            f"{source} is missing required training hook(s): "
            f"{', '.join(missing_hooks)}"
        )
    return module


def _resolve_training_module(
    resolved_train_cfg: Mapping[str, Any],
    *,
    layout: TrainingJobLayout,
) -> tuple[Any, dict[str, Any]]:
    implementation_cfg = resolved_train_cfg.get("implementation")
    if implementation_cfg is not None and not isinstance(implementation_cfg, Mapping):
        raise TypeError("train.implementation must be a mapping")
    implementation_cfg = dict(implementation_cfg or {})

    impl_path = implementation_cfg.get("path")
    if impl_path is not None and not isinstance(impl_path, str):
        raise TypeError("train.implementation.path must be a string")
    if impl_path:
        path = Path(impl_path)
        if not path.is_absolute():
            path = (layout.train_bundle_dir / path).resolve()
        return _load_local_training_module(path), {
            "source": "config_path",
            "module": None,
            "path": str(path),
        }

    module_name = implementation_cfg.get("module")
    if module_name is not None and not isinstance(module_name, str):
        raise TypeError("train.implementation.module must be a string")
    if module_name:
        return _load_configured_training_module(module_name), {
            "source": "config_module",
            "module": module_name,
            "path": None,
        }

    if layout.train_script_path is not None:
        return _load_local_training_module(layout.train_script_path), {
            "source": "local_train_py",
            "module": None,
            "path": str(layout.train_script_path),
        }

    raise FileNotFoundError(
        f"Training bundle {layout.train_bundle_dir} must declare train.implementation.path "
        "or train.implementation.module in config.yaml, or provide a local train.py."
    )


def _prepare_job_dir(layout: TrainingJobLayout) -> Path | None:
    job_dir = layout.job_dir
    backup_dir: Path | None = None
    if job_dir.exists() and any(job_dir.iterdir()):
        archive_root = job_dir.parent / "artifacts_history"
        archive_root.mkdir(parents=True, exist_ok=True)
        backup_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = archive_root / backup_name
        suffix = 1
        while backup_dir.exists():
            backup_dir = archive_root / f"{backup_name}_{suffix:02d}"
            suffix += 1
        shutil.copytree(job_dir, backup_dir)

    job_dir.mkdir(parents=True, exist_ok=True)
    for dirname in ("checkpoints", "figures", "model_workspace", "reference_fields"):
        target = job_dir / dirname
        if target.exists():
            shutil.rmtree(target)
    for filename in (
        "manifest.json",
        "train.snapshot.yaml",
        "train_metrics.json",
        "eval_metrics.json",
        "visualization_manifest.json",
        "trained_network_package.pt",
        "reference_input.npy",
        "reference_output.npy",
    ):
        target = job_dir / filename
        if target.exists():
            target.unlink()
    job_dir.joinpath("train.snapshot.yaml").write_text(
        layout.train_config_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return backup_dir


def _build_training_state_for_bundle(
    module: Any,
    *,
    mapped_network_package: Mapping[str, Any],
    resolved_train_cfg: Mapping[str, Any],
    layout: TrainingJobLayout,
    seed: int,
) -> dict[str, Any]:
    state = module.build_training_state(
        mapped_network_package,
        resolved_train_cfg,
        train_dir=layout.job_dir,
        seed=seed,
    )
    if not isinstance(state, Mapping):
        raise TypeError(
            f"build_training_state(...) must return a mapping, got {type(state).__name__}"
        )
    state = dict(state)

    missing_state_keys = [key for key in REQUIRED_LOCAL_STATE_KEYS if key not in state]
    if missing_state_keys:
        raise KeyError(
            "build_training_state(...) result missing required keys: "
            + ", ".join(missing_state_keys)
        )
    if not isinstance(state["model"], torch.nn.Module):
        raise TypeError(
            "build_training_state(...)[\"model\"] must be a torch.nn.Module, got "
            f"{type(state['model']).__name__}"
        )
    if "mapped_network_package_for_training" in state and not isinstance(
        state["mapped_network_package_for_training"], Mapping
    ):
        raise TypeError(
            "build_training_state(...)[\"mapped_network_package_for_training\"] must be a mapping"
        )
    return state


def _resolve_loop_config(
    resolved_train_cfg: Mapping[str, Any],
    *,
    train_config_path: Path,
) -> tuple[int, int]:
    loop_cfg = dict(resolved_train_cfg.get("loop") or {})
    epochs = int(loop_cfg.get("epochs", 50))
    log_every = int(loop_cfg.get("log_every", 1))
    if epochs <= 0:
        raise ValueError(
            f"train.loop.epochs must be positive, got epochs={epochs} from {train_config_path}"
        )
    if log_every <= 0:
        raise ValueError(
            f"train.loop.log_every must be positive, got log_every={log_every} from {train_config_path}"
        )
    return epochs, log_every


def _run_training_loop(
    module: Any,
    state: Mapping[str, Any],
    *,
    epochs: int,
    log_every: int,
) -> dict[str, Any]:
    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_metric = float("inf")
    best_state_dict = copy.deepcopy(state["model"].state_dict())
    best_eval_accuracy = float("-inf")
    best_eval_accuracy_epoch = 0

    initial_eval_metrics = module.evaluate_model(state, 0)
    if not isinstance(initial_eval_metrics, Mapping):
        raise TypeError(
            f"evaluate_model(...) must return a mapping, got {type(initial_eval_metrics).__name__}"
        )

    initial_record = {"epoch": 0, **dict(initial_eval_metrics)}
    initial_metric = _select_record_metric(initial_record)
    if initial_metric <= best_metric:
        best_metric = initial_metric
        best_epoch = 0
        best_state_dict = copy.deepcopy(state["model"].state_dict())

    if "eval_accuracy_combined" in initial_record:
        initial_eval_accuracy = float(initial_record["eval_accuracy_combined"])
    elif "eval_accuracy" in initial_record:
        initial_eval_accuracy = float(initial_record["eval_accuracy"])
    else:
        initial_eval_accuracy = None

    if initial_eval_accuracy is not None and initial_eval_accuracy >= best_eval_accuracy:
        best_eval_accuracy = initial_eval_accuracy
        best_eval_accuracy_epoch = 0
    if initial_eval_accuracy is not None:
        initial_record["best_eval_accuracy_so_far"] = best_eval_accuracy
        initial_record["best_eval_accuracy_epoch_so_far"] = best_eval_accuracy_epoch

    history.append(initial_record)
    print(_format_epoch_log(initial_record))

    for epoch in range(1, epochs + 1):
        train_metrics = module.train_one_epoch(state, epoch)
        if not isinstance(train_metrics, Mapping):
            raise TypeError(
                f"train_one_epoch(...) must return a mapping, got {type(train_metrics).__name__}"
            )
        eval_metrics = module.evaluate_model(state, epoch)
        if not isinstance(eval_metrics, Mapping):
            raise TypeError(
                f"evaluate_model(...) must return a mapping, got {type(eval_metrics).__name__}"
            )

        record = {"epoch": epoch, **dict(train_metrics), **dict(eval_metrics)}

        metric = _select_record_metric(record)
        if metric <= best_metric:
            best_metric = metric
            best_epoch = epoch
            best_state_dict = copy.deepcopy(state["model"].state_dict())

        if "eval_accuracy_combined" in record:
            eval_accuracy = float(record["eval_accuracy_combined"])
        elif "eval_accuracy" in record:
            eval_accuracy = float(record["eval_accuracy"])
        else:
            eval_accuracy = None

        if eval_accuracy is not None and eval_accuracy >= best_eval_accuracy:
            best_eval_accuracy = eval_accuracy
            best_eval_accuracy_epoch = epoch
        if eval_accuracy is not None:
            record["best_eval_accuracy_so_far"] = best_eval_accuracy
            record["best_eval_accuracy_epoch_so_far"] = best_eval_accuracy_epoch

        history.append(record)

        if epoch == 1 or epoch % log_every == 0:
            print(_format_epoch_log(record))

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "best_state_dict": best_state_dict,
        "final_state_dict": copy.deepcopy(state["model"].state_dict()),
    }


def _resolve_reference_artifacts(
    module: Any,
    state: Mapping[str, Any],
    *,
    job_dir: Path,
) -> dict[str, Any]:
    if hasattr(module, "dump_reference_artifacts"):
        reference_artifacts = module.dump_reference_artifacts(state, train_dir=job_dir)
        if not isinstance(reference_artifacts, Mapping):
            raise TypeError(
                "dump_reference_artifacts(...) must return a mapping, got "
                f"{type(reference_artifacts).__name__}"
            )
        return dict(reference_artifacts)

    return _dump_reference_fields(
        state["model"],
        _coerce_reference_input(state["model"], state["reference_input"]),
        out_dir=job_dir,
    )


def _resolve_training_summary(
    module: Any,
    state: Mapping[str, Any],
    history: list[Mapping[str, Any]],
    *,
    best_epoch: int,
    best_metric: float,
) -> dict[str, Any]:
    if hasattr(module, "summarize_training"):
        summary = module.summarize_training(
            state,
            history,
            best_epoch=best_epoch,
            best_metric=best_metric,
        )
        if not isinstance(summary, Mapping):
            raise TypeError(
                f"summarize_training(...) must return a mapping, got {type(summary).__name__}"
            )
        return dict(summary)

    last = history[-1]
    summary: dict[str, Any] = {
        "epochs": int(len(history)),
        "best_epoch": int(best_epoch),
        "best_metric": float(best_metric),
    }
    if "task_kind" in state and "train_kind" not in state:
        summary["train_kind"] = state["task_kind"]
    for key in (
        "task_kind",
        "train_kind",
        "dataset_kind",
        "train_samples",
        "eval_samples",
        "input_dim",
        "output_dim",
        "reference_target",
    ):
        if key in state:
            summary[key] = state[key]
    for key in ("train_loss", "eval_loss", "loss", "transformed_output"):
        if key in last:
            summary[f"final_{key}"] = last[key]
    return summary


def _write_checkpoints(
    *,
    job_dir: Path,
    best_state_dict: Mapping[str, Any],
    final_state_dict: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> tuple[Path, Path]:
    checkpoints_dir = job_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = checkpoints_dir / "best_model.pt"
    final_checkpoint = checkpoints_dir / "final_model.pt"
    torch.save(
        {
            "state_dict": dict(best_state_dict),
            "metrics_summary": dict(summary),
        },
        best_checkpoint,
    )
    torch.save(
        {
            "state_dict": dict(final_state_dict),
            "metrics_summary": dict(summary),
        },
        final_checkpoint,
    )
    return best_checkpoint, final_checkpoint


def run_local_training_job(
    *,
    config_path: str | Path,
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = project_root or get_project_root()
    config_file = resolve_training_config_path(config_path)
    config = load_project_config(config_file)
    train_cfg = config.get("train")
    if not isinstance(train_cfg, dict) or not train_cfg:
        raise KeyError(f"Training job config requires a non-empty train section: {config_file}")

    layout = resolve_training_job_layout(config_file)
    if not layout.mapped_network_package_path.exists():
        raise FileNotFoundError(
            "Missing mapped network package for training job, expected "
            f"{layout.mapped_network_package_path}. Run the experiment config to the map stage first."
        )

    resolved_train_cfg = resolve_train_config(config, project_root=root)
    module, module_info = _resolve_training_module(
        resolved_train_cfg,
        layout=layout,
    )
    backup_dir = _prepare_job_dir(layout)
    mapped_network_package = load_package(layout.mapped_network_package_path)
    state = _build_training_state_for_bundle(
        module,
        mapped_network_package=mapped_network_package,
        resolved_train_cfg=resolved_train_cfg,
        layout=layout,
        seed=get_seed(config),
    )
    package_for_training = dict(
        state.get("mapped_network_package_for_training") or mapped_network_package
    )

    epochs, log_every = _resolve_loop_config(
        resolved_train_cfg,
        train_config_path=layout.train_config_path,
    )
    loop_result = _run_training_loop(
        module,
        state,
        epochs=epochs,
        log_every=log_every,
    )

    reference_artifacts = _resolve_reference_artifacts(
        module,
        state,
        job_dir=layout.job_dir,
    )
    summary = _resolve_training_summary(
        module,
        state,
        loop_result["history"],
        best_epoch=loop_result["best_epoch"],
        best_metric=loop_result["best_metric"],
    )
    best_checkpoint, final_checkpoint = _write_checkpoints(
        job_dir=layout.job_dir,
        best_state_dict=loop_result["best_state_dict"],
        final_state_dict=loop_result["final_state_dict"],
        summary=summary,
    )

    train_metrics_path = save_json(
        layout.job_dir / "train_metrics.json",
        {"history": loop_result["history"]},
    )
    eval_metrics_path = save_json(layout.job_dir / "eval_metrics.json", summary)
    visualization_manifest_path = save_json(
        layout.job_dir / "visualization_manifest.json",
        save_training_visualizations(
            model=state["model"],
            history=loop_result["history"],
            out_dir=layout.job_dir / "figures",
        ),
    )

    trained_network_package = build_trained_network_package(
        mapped_network_package=package_for_training,
        train_config=resolved_train_cfg,
        metrics_summary={
            **summary,
            "best_checkpoint": str(best_checkpoint),
            "final_checkpoint": str(final_checkpoint),
            "train_metrics_path": str(train_metrics_path),
            "eval_metrics_path": str(eval_metrics_path),
            "visualization_manifest_path": str(visualization_manifest_path),
        },
        best_state_dict=loop_result["best_state_dict"],
        final_state_dict=loop_result["final_state_dict"],
        reference_artifacts=reference_artifacts,
    )
    trained_network_package_path = save_package(
        layout.job_dir / "trained_network_package.pt",
        trained_network_package,
    )

    manifest_path = save_json(
        layout.job_dir / "manifest.json",
        {
            "train_config_path": str(layout.train_config_path),
            "train_script_path": (
                str(layout.train_script_path) if layout.train_script_path is not None else None
            ),
            "train_implementation_source": str(module_info["source"]),
            "train_implementation_module": module_info["module"],
            "train_implementation_path": module_info["path"],
            "train_bundle_dir": str(layout.train_bundle_dir),
            "workspace_dir": str(layout.workspace_dir),
            "map_dir": str(layout.map_dir),
            "mapped_network_package": str(layout.mapped_network_package_path),
            "previous_artifacts_backup": str(backup_dir) if backup_dir is not None else None,
            "mapped_network_package_used_for_training_signature": str(
                package_for_training.get("mapped_network_signature")
            ),
            "trained_network_package": str(trained_network_package_path),
            "best_checkpoint": str(best_checkpoint),
            "final_checkpoint": str(final_checkpoint),
            "train_metrics": str(train_metrics_path),
            "eval_metrics": str(eval_metrics_path),
            "visualization_manifest": str(visualization_manifest_path),
            "figures_dir": str(layout.job_dir / "figures"),
        },
    )
    return {
        "job_dir": str(layout.job_dir),
        "backup_dir": str(backup_dir) if backup_dir is not None else None,
        "manifest_path": str(manifest_path),
        "trained_network_package": str(trained_network_package_path),
    }


def run_local_training_bundle(
    bundle_dir: str | Path,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    return run_local_training_job(
        config_path=resolve_training_config_path(bundle_dir),
        project_root=project_root,
    )


__all__ = [
    "resolve_training_job_layout",
    "run_local_training_bundle",
    "run_local_training_job",
]
