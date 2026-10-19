from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .project_paths import resolve_config_workspace_dir


@dataclass(frozen=True)
class ExperimentWorkspaceLayout:
    config_path: Path
    workspace_dir: Path
    map_dir: Path


@dataclass(frozen=True)
class TrainingJobLayout:
    train_config_path: Path
    train_script_path: Path | None
    train_bundle_dir: Path
    workspace_dir: Path
    map_dir: Path
    job_dir: Path
    mapped_network_package_path: Path


@dataclass(frozen=True)
class ValidationJobLayout:
    validation_config_path: Path
    validation_script_path: Path | None
    validation_bundle_dir: Path
    train_bundle_dir: Path
    workspace_dir: Path
    map_dir: Path
    job_dir: Path
    train_artifacts_dir: Path
    trained_network_package_path: Path


def resolve_workspace_experiment_config(workspace_dir: str | Path) -> Path:
    workspace = Path(workspace_dir).resolve()
    if not workspace.exists():
        raise FileNotFoundError(f"Experiment workspace does not exist: {workspace}")
    if not workspace.is_dir():
        raise NotADirectoryError(f"Experiment workspace must be a directory, got {workspace}")

    preferred = workspace / f"{workspace.name}.yaml"
    if preferred.exists():
        return preferred

    yaml_files = sorted(workspace.glob("*.yaml")) + sorted(workspace.glob("*.yml"))
    if len(yaml_files) == 1:
        return yaml_files[0]
    if not yaml_files:
        raise FileNotFoundError(f"No experiment YAML config found in workspace {workspace}")
    raise ValueError(
        "Experiment workspace contains multiple YAML configs and no same-name config: "
        f"{workspace} -> {[path.name for path in yaml_files]}"
    )


def resolve_experiment_workspace(config_path: str | Path) -> ExperimentWorkspaceLayout:
    config_file = Path(config_path).resolve()
    workspace_dir = resolve_config_workspace_dir(config_file)
    return ExperimentWorkspaceLayout(
        config_path=config_file,
        workspace_dir=workspace_dir,
        map_dir=workspace_dir / "map",
    )


def resolve_training_config_path(config_path: str | Path) -> Path:
    candidate = Path(config_path).resolve()
    if candidate.is_dir():
        yaml_files = sorted(candidate.glob("*.yaml")) + sorted(candidate.glob("*.yml"))
        config_yaml = candidate / "config.yaml"
        if config_yaml.exists():
            return config_yaml.resolve()
        if len(yaml_files) == 1:
            return yaml_files[0].resolve()
        if not yaml_files:
            raise FileNotFoundError(
                f"Could not find a training YAML config in bundle directory {candidate}"
            )
        raise ValueError(
            "Training bundle directory contains multiple YAML configs and no config.yaml: "
            f"{candidate} -> {[path.name for path in yaml_files]}"
        )
    if not candidate.exists():
        raise FileNotFoundError(f"Training config path does not exist: {candidate}")
    if candidate.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"Training config must be a YAML file, got {candidate}")
    return candidate


def resolve_validation_config_path(config_path: str | Path) -> Path:
    candidate = Path(config_path).resolve()
    if candidate.is_dir():
        yaml_files = sorted(candidate.glob("*.yaml")) + sorted(candidate.glob("*.yml"))
        config_yaml = candidate / "config.yaml"
        validation_yaml = candidate / "validation.yaml"
        if config_yaml.exists():
            return config_yaml.resolve()
        if validation_yaml.exists():
            return validation_yaml.resolve()
        if len(yaml_files) == 1:
            return yaml_files[0].resolve()
        if not yaml_files:
            raise FileNotFoundError(
                f"Could not find a validation YAML config in bundle directory {candidate}"
            )
        raise ValueError(
            "Validation bundle directory contains multiple YAML configs and no config.yaml: "
            f"{candidate} -> {[path.name for path in yaml_files]}"
        )
    if not candidate.exists():
        raise FileNotFoundError(f"Validation config path does not exist: {candidate}")
    if candidate.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"Validation config must be a YAML file, got {candidate}")
    return candidate


def resolve_training_job_layout(config_path: str | Path) -> TrainingJobLayout:
    config_file = resolve_training_config_path(config_path)

    workspace_dir: Path | None = None
    for candidate in (config_file.parent, *config_file.parents):
        if (candidate / "map").exists():
            workspace_dir = candidate.resolve()
            break
    if workspace_dir is None:
        raise FileNotFoundError(
            f"Could not locate workspace for {config_file}. "
            "Each training config must live under an experiment folder that already contains map/."
        )

    bundle_dir = config_file.parent.resolve()
    train_script_path = bundle_dir / "train.py"
    map_dir = workspace_dir / "map"
    if not map_dir.exists():
        raise FileNotFoundError(
            f"Missing map directory {map_dir}. Run the experiment to the map stage first."
        )

    return TrainingJobLayout(
        train_config_path=config_file,
        train_script_path=train_script_path.resolve() if train_script_path.exists() else None,
        train_bundle_dir=bundle_dir,
        workspace_dir=workspace_dir,
        map_dir=map_dir,
        job_dir=bundle_dir / "artifacts",
        mapped_network_package_path=map_dir / "mapped_network_package.pt",
    )


def resolve_validation_job_layout(config_path: str | Path) -> ValidationJobLayout:
    config_file = resolve_validation_config_path(config_path)
    validation_bundle_dir = config_file.parent.resolve()

    workspace_dir: Path | None = None
    for candidate in (validation_bundle_dir, *validation_bundle_dir.parents):
        if (candidate / "map").exists():
            workspace_dir = candidate.resolve()
            break
    if workspace_dir is None:
        raise FileNotFoundError(
            f"Could not locate workspace for {config_file}. "
            "Each validation config must live under an experiment folder that already contains map/."
        )

    train_bundle_dir: Path | None = None
    for candidate in (validation_bundle_dir, *validation_bundle_dir.parents):
        if candidate == workspace_dir:
            break
        if (candidate / "train.py").exists() or (
            candidate / "artifacts" / "trained_network_package.pt"
        ).exists():
            train_bundle_dir = candidate.resolve()
            break
    if train_bundle_dir is None:
        raise FileNotFoundError(
            f"Could not locate train bundle for validation config {config_file}. "
            "Place validation configs inside a train bundle or one of its validation subfolders."
        )

    map_dir = workspace_dir / "map"
    if not map_dir.exists():
        raise FileNotFoundError(
            f"Missing map directory {map_dir}. Run the experiment to the map stage first."
        )

    validation_script_path = validation_bundle_dir / "validate.py"
    if validation_bundle_dir == train_bundle_dir:
        job_dir = train_bundle_dir / config_file.stem
    else:
        job_dir = validation_bundle_dir / "artifacts"

    train_artifacts_dir = train_bundle_dir / "artifacts"
    return ValidationJobLayout(
        validation_config_path=config_file,
        validation_script_path=(
            validation_script_path.resolve() if validation_script_path.exists() else None
        ),
        validation_bundle_dir=validation_bundle_dir,
        train_bundle_dir=train_bundle_dir,
        workspace_dir=workspace_dir,
        map_dir=map_dir,
        job_dir=job_dir.resolve(),
        train_artifacts_dir=train_artifacts_dir.resolve(),
        trained_network_package_path=(
            train_artifacts_dir / "trained_network_package.pt"
        ).resolve(),
    )


__all__ = [
    "ExperimentWorkspaceLayout",
    "TrainingJobLayout",
    "ValidationJobLayout",
    "resolve_experiment_workspace",
    "resolve_training_config_path",
    "resolve_training_job_layout",
    "resolve_validation_config_path",
    "resolve_validation_job_layout",
    "resolve_workspace_experiment_config",
]
