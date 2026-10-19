from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(
    path_value: str | Path,
    *,
    project_root: Path | None = None,
) -> Path:
    base = project_root or get_project_root()
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def repo_relative_path(
    path_value: str | Path,
    *,
    project_root: Path | None = None,
) -> str:
    base = (project_root or get_project_root()).resolve()
    path = resolve_repo_path(path_value, project_root=base)
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def resolve_config_workspace_dir(
    config_path: str | Path,
    *,
    project_root: Path | None = None,
) -> Path:
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = resolve_repo_path(cfg_path, project_root=project_root)
    cfg_path = cfg_path.resolve()
    workspace = cfg_path.parent
    if workspace.name != cfg_path.stem:
        raise ValueError(
            "Experiment config must live in its own folder named after the config stem, "
            f"expected .../{cfg_path.stem}/{cfg_path.name} but got {cfg_path}"
        )
    return workspace


def resolve_map_artifact_dir(
    config_path: str | Path,
    *,
    project_root: Path | None = None,
) -> Path:
    return resolve_config_workspace_dir(
        config_path,
        project_root=project_root,
    ) / "map"


__all__ = [
    "get_project_root",
    "repo_relative_path",
    "resolve_config_workspace_dir",
    "resolve_map_artifact_dir",
    "resolve_repo_path",
]
