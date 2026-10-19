from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .project_paths import get_project_root, repo_relative_path, resolve_repo_path


@dataclass
class RunContext:
    pipeline: str
    run_dir: Path
    metrics_path: Path
    manifest_path: Path
    log_path: Path
    config_copy_path: Path
    git_path: Path
    env_path: Path


@dataclass
class ExperimentContext:
    config_path: Path
    run_dir: Path
    manifest_path: Path
    log_path: Path
    config_snapshot_path: Path
    git_path: Path
    env_path: Path
    logs_dir: Path


def _json_dump(path: Path, payload: Mapping[str, Any]) -> Path:
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return path


def _capture_cmd(args: list[str], cwd: Path) -> str:
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Command failed because the executable was not found: cmd={' '.join(args)!r}, cwd={cwd}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        stdout = (exc.stdout or "").strip()
        stderr = (exc.stderr or "").strip()
        details = []
        if stdout:
            details.append(f"stdout={stdout}")
        if stderr:
            details.append(f"stderr={stderr}")
        detail_text = "; ".join(details) if details else "no stdout/stderr captured"
        raise RuntimeError(
            f"Command failed: cmd={' '.join(args)!r}, cwd={cwd}, returncode={exc.returncode}. "
            f"{detail_text}"
        ) from exc
    return proc.stdout.strip()


def _capture_cmd_or_note(args: list[str], cwd: Path) -> str:
    try:
        return _capture_cmd(args, cwd=cwd)
    except RuntimeError as exc:
        return f"[command unavailable] {exc}"


def create_run_context(
    *,
    pipeline: str,
    config_path: str | Path,
    config: Mapping[str, Any],
    runs_root: str | Path = "runs",
    project_root: Path | None = None,
    timestamp: datetime | None = None,
) -> RunContext:
    base = project_root or get_project_root()
    config_file = resolve_repo_path(config_path, project_root=base)
    runs_dir = resolve_repo_path(runs_root, project_root=base)
    ts = timestamp or datetime.now()
    seed = int((config.get("meta") or {}).get("seed", 0))
    run_name = f"{ts:%Y-%m-%d_%H%M%S}_{config_file.stem}_seed{seed}"
    run_dir = runs_dir / pipeline / run_name
    suffix = 1
    while run_dir.exists():
        run_dir = runs_dir / pipeline / f"{run_name}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)

    ctx = RunContext(
        pipeline=pipeline,
        run_dir=run_dir,
        metrics_path=run_dir / "metrics.json",
        manifest_path=run_dir / "run_manifest.json",
        log_path=run_dir / "log.txt",
        config_copy_path=run_dir / "config.yaml",
        git_path=run_dir / "git.txt",
        env_path=run_dir / "env.txt",
    )

    ctx.config_copy_path.write_text(
        config_file.read_text(encoding="utf-8"), encoding="utf-8"
    )
    ctx.git_path.write_text(
        _capture_cmd_or_note(["git", "rev-parse", "HEAD"], cwd=base) + "\n",
        encoding="utf-8",
    )
    ctx.env_path.write_text(
        _capture_cmd_or_note([sys.executable, "-m", "pip", "freeze"], cwd=base) + "\n",
        encoding="utf-8",
    )
    ctx.log_path.write_text(
        f"[{datetime.now().isoformat()}] created run directory {ctx.run_dir}\n",
        encoding="utf-8",
    )

    write_metrics(ctx, {"status": "initialized"})
    write_manifest(
        ctx,
        {
            "config_path": repo_relative_path(config_file, project_root=base),
            "pipeline": pipeline,
            "seed": seed,
            "status": "initialized",
        },
    )
    return ctx


def _experiment_context_from_run_dir(
    run_dir: Path,
    *,
    config_path: Path,
) -> ExperimentContext:
    return ExperimentContext(
        config_path=config_path,
        run_dir=run_dir,
        manifest_path=run_dir / "manifest.json",
        log_path=run_dir / "log.txt",
        config_snapshot_path=run_dir / "config.snapshot.yaml",
        git_path=run_dir / "git.txt",
        env_path=run_dir / "env.txt",
        logs_dir=run_dir / "logs",
    )


def create_experiment_context(
    *,
    config_path: str | Path,
    config: Mapping[str, Any],
    runs_root: str | Path = "runs",
    project_root: Path | None = None,
    timestamp: datetime | None = None,
) -> ExperimentContext:
    base = project_root or get_project_root()
    config_file = resolve_repo_path(config_path, project_root=base)

    runs_dir = resolve_repo_path(runs_root, project_root=base) / "experiments"
    ts = timestamp or datetime.now()
    seed = int((config.get("meta") or {}).get("seed", 0))
    run_name = f"{ts:%Y-%m-%d_%H%M%S}_{config_file.stem}_seed{seed}"
    run_dir = runs_dir / run_name
    suffix = 1
    while run_dir.exists():
        run_dir = runs_dir / f"{run_name}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)

    ctx = _experiment_context_from_run_dir(run_dir, config_path=config_file)
    ctx.logs_dir.mkdir(parents=True, exist_ok=True)
    ctx.config_snapshot_path.write_text(
        config_file.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    ctx.git_path.write_text(
        _capture_cmd_or_note(["git", "rev-parse", "HEAD"], cwd=base) + "\n",
        encoding="utf-8",
    )
    ctx.env_path.write_text(
        _capture_cmd_or_note([sys.executable, "-m", "pip", "freeze"], cwd=base) + "\n",
        encoding="utf-8",
    )
    ctx.log_path.write_text(
        f"[{datetime.now().isoformat()}] created experiment directory {ctx.run_dir}\n",
        encoding="utf-8",
    )
    write_experiment_manifest(
        ctx,
        {
            "config_path": repo_relative_path(config_file, project_root=base),
            "experiment_id": run_dir.name,
            "run_dir": str(run_dir),
            "config_snapshot": str(ctx.config_snapshot_path),
            "git_file": str(ctx.git_path),
            "env_file": str(ctx.env_path),
            "log_file": str(ctx.log_path),
            "logs_dir": str(ctx.logs_dir),
            "status": "initialized",
            "stages": {},
        },
    )
    return ctx


def read_experiment_manifest(ctx: ExperimentContext) -> dict[str, Any]:
    if not ctx.manifest_path.exists():
        return {}
    try:
        return json.loads(ctx.manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Invalid experiment manifest JSON: "
            f"path={ctx.manifest_path}, line={exc.lineno}, column={exc.colno}, reason={exc.msg}"
        ) from exc


def write_experiment_manifest(
    ctx: ExperimentContext,
    manifest: Mapping[str, Any],
) -> Path:
    payload = dict(manifest)
    payload.setdefault("run_dir", str(ctx.run_dir))
    payload.setdefault("config_snapshot", str(ctx.config_snapshot_path))
    payload.setdefault("git_file", str(ctx.git_path))
    payload.setdefault("env_file", str(ctx.env_path))
    payload.setdefault("log_file", str(ctx.log_path))
    payload.setdefault("logs_dir", str(ctx.logs_dir))
    return _json_dump(ctx.manifest_path, payload)


def append_log(ctx: RunContext, message: str) -> None:
    with ctx.log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{datetime.now().isoformat()}] {message}\n")


def write_metrics(ctx: RunContext, metrics: Mapping[str, Any]) -> Path:
    return _json_dump(ctx.metrics_path, metrics)


def write_manifest(ctx: RunContext, manifest: Mapping[str, Any]) -> Path:
    payload = {
        "pipeline": ctx.pipeline,
        "run_dir": str(ctx.run_dir),
        "config_copy": str(ctx.config_copy_path),
        "git_file": str(ctx.git_path),
        "env_file": str(ctx.env_path),
        "metrics_file": str(ctx.metrics_path),
        "log_file": str(ctx.log_path),
    }
    payload.update(dict(manifest))
    return _json_dump(ctx.manifest_path, payload)
