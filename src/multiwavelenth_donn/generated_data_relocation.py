from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

TRAIN_BUNDLE_OUTPUT_MARKERS = (
    "trained_network_package.pt",
    "manifest.json",
    "train.snapshot.yaml",
    "train_metrics.json",
)


@dataclass(frozen=True)
class RelocationItem:
    category: str
    source: Path
    destination: Path


@dataclass(frozen=True)
class RelocationResult:
    category: str
    source: str
    destination: str
    status: str


def resolve_shortcut_target(shortcut_path: str | Path) -> Path:
    shortcut = Path(shortcut_path).resolve()
    if not shortcut.exists():
        raise FileNotFoundError(f"Shortcut file does not exist: {shortcut}")
    escaped = str(shortcut).replace("'", "''")
    command = (
        "$shell = New-Object -ComObject WScript.Shell; "
        f"$shortcut = $shell.CreateShortcut('{escaped}'); "
        "Write-Output $shortcut.TargetPath"
    )
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Failed to resolve shortcut target for {shortcut}: {proc.stderr.strip()}"
        )
    target = (proc.stdout or "").strip()
    if not target:
        raise RuntimeError(f"Shortcut {shortcut} does not define a target path")
    return Path(target).resolve()


def build_target_repo_root(project_root: str | Path, target_root: str | Path) -> Path:
    project = Path(project_root).resolve()
    target = Path(target_root).resolve()
    return target / project.name


def _is_train_bundle_output_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if path.name in {"artifacts", "artifacts_history", "__pycache__"}:
        return False
    return any((path / marker).exists() for marker in TRAIN_BUNDLE_OUTPUT_MARKERS)


def _iter_existing_generated_dirs(project_root: Path) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []

    simulation_root = project_root / "data" / "raw" / "simulation_data"
    if simulation_root.is_dir():
        items.append(("simulation_data", simulation_root))

    for runs_name in ("runs", "runs_smoke"):
        runs_root = project_root / runs_name
        if not runs_root.is_dir():
            continue
        for child in sorted(runs_root.iterdir(), key=lambda path: path.name):
            if child.is_dir():
                items.append(("managed_runs", child))

    experiment_root = project_root / "configs" / "experiment"
    if experiment_root.is_dir():
        for workspace in sorted(
            (path for path in experiment_root.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        ):
            map_dir = workspace / "map"
            if map_dir.is_dir():
                items.append(("workspace_map", map_dir))

            for bundle in sorted(
                (
                    path
                    for path in workspace.iterdir()
                    if path.is_dir() and path.name.startswith("train")
                ),
                key=lambda path: path.name,
            ):
                artifacts_dir = bundle / "artifacts"
                if artifacts_dir.is_dir():
                    items.append(("train_artifacts", artifacts_dir))
                artifacts_history_dir = bundle / "artifacts_history"
                if artifacts_history_dir.is_dir():
                    items.append(("train_artifacts_history", artifacts_history_dir))
                for child in sorted(
                    (path for path in bundle.iterdir() if _is_train_bundle_output_dir(path)),
                    key=lambda path: path.name,
                ):
                    items.append(("train_bundle_output", child))

    return items


def build_relocation_plan(
    project_root: str | Path,
    target_root: str | Path,
) -> list[RelocationItem]:
    project = Path(project_root).resolve()
    target_repo_root = build_target_repo_root(project, target_root)
    items: list[RelocationItem] = []
    seen_sources: set[Path] = set()

    for category, source in _iter_existing_generated_dirs(project):
        if source in seen_sources:
            continue
        destination = (target_repo_root / source.relative_to(project)).resolve()
        items.append(
            RelocationItem(
                category=category,
                source=source,
                destination=destination,
            )
        )
        seen_sources.add(source)
    return items


def _is_already_pointing_to_destination(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    try:
        return source.resolve() == destination.resolve()
    except OSError:
        return False


def _create_directory_junction(source: Path, destination: Path) -> None:
    if source.exists():
        raise FileExistsError(f"Cannot create junction because source still exists: {source}")
    source.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(source), str(destination)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to create directory junction: "
            f"source={source}, destination={destination}, stderr={proc.stderr.strip()}"
        )


def _get_conflicting_child_names(source: Path, destination: Path) -> list[str]:
    destination_names = {child.name for child in destination.iterdir()}
    return sorted(child.name for child in source.iterdir() if child.name in destination_names)


def _merge_directory_into_destination(source: Path, destination: Path) -> None:
    conflicts = _get_conflicting_child_names(source, destination)
    if conflicts:
        preview = ", ".join(conflicts[:5])
        raise FileExistsError(
            "Cannot merge relocation item because destination already contains child paths "
            f"with the same names: source={source}, destination={destination}, "
            f"conflicts={preview}"
        )

    for child in sorted(source.iterdir(), key=lambda path: path.name):
        shutil.move(str(child), str(destination / child.name))
    source.rmdir()


def execute_relocation_plan(
    plan: list[RelocationItem],
    *,
    link_back: bool = True,
    dry_run: bool = False,
) -> list[RelocationResult]:
    results: list[RelocationResult] = []
    for item in plan:
        if not item.source.exists():
            results.append(
                RelocationResult(
                    category=item.category,
                    source=str(item.source),
                    destination=str(item.destination),
                    status="missing",
                )
            )
            continue

        if _is_already_pointing_to_destination(item.source, item.destination):
            results.append(
                RelocationResult(
                    category=item.category,
                    source=str(item.source),
                    destination=str(item.destination),
                    status="already_linked",
                )
            )
            continue

        if item.destination.exists():
            if dry_run:
                conflicts = _get_conflicting_child_names(item.source, item.destination)
                status = (
                    "destination_exists_with_conflicts"
                    if conflicts
                    else "planned_merge_and_link"
                )
                results.append(
                    RelocationResult(
                        category=item.category,
                        source=str(item.source),
                        destination=str(item.destination),
                        status=status,
                    )
                )
                continue

            if not item.destination.is_dir():
                raise FileExistsError(
                    "Destination already exists for relocation item and is not a directory. "
                    f"source={item.source}, destination={item.destination}"
                )
            _merge_directory_into_destination(item.source, item.destination)
            if link_back:
                _create_directory_junction(item.source, item.destination)
                status = "merged_and_linked"
            else:
                status = "merged"
            results.append(
                RelocationResult(
                    category=item.category,
                    source=str(item.source),
                    destination=str(item.destination),
                    status=status,
                )
            )
            continue

        if dry_run:
            results.append(
                RelocationResult(
                    category=item.category,
                    source=str(item.source),
                    destination=str(item.destination),
                    status="planned",
                )
            )
            continue

        item.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(item.source), str(item.destination))
        if link_back:
            _create_directory_junction(item.source, item.destination)
            status = "moved_and_linked"
        else:
            status = "moved"

        results.append(
            RelocationResult(
                category=item.category,
                source=str(item.source),
                destination=str(item.destination),
                status=status,
            )
        )
    return results


def write_relocation_manifest(
    target_repo_root: str | Path,
    results: list[RelocationResult],
) -> Path:
    target = Path(target_repo_root).resolve()
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / "generated_data_relocation_manifest.json"
    payload = {
        "updated_at": datetime.now().isoformat(),
        "items": [asdict(result) for result in results],
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest_path


__all__ = [
    "RelocationItem",
    "RelocationResult",
    "build_relocation_plan",
    "build_target_repo_root",
    "execute_relocation_plan",
    "resolve_shortcut_target",
    "write_relocation_manifest",
]
