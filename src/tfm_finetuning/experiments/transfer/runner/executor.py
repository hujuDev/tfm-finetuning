"""Run transfer stages in order and keep checkpoints next to their reports.

Example: ``execute_route(route, trainer=trainer, output=run_dir / "stages")``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from tfm_finetuning.experiments.transfer.spec import TransferRouteSpec, TransferStageSpec


@dataclass(frozen=True)
class StageProduction:
    """Selected weights and validation evidence returned by a model trainer."""

    checkpoint_path: Path
    report: Mapping[str, Any]
    view_checkpoints: Mapping[int, Path] = field(default_factory=dict)
    package_versions: Mapping[str, str] = field(default_factory=dict)
    reload_validator: Callable[[Path], Any] | None = None


class StageTrainer(Protocol):
    """Each call must create a fresh optimizer and realize the specified updates."""

    def train_stage(
        self, *, stage: TransferStageSpec, initial_checkpoint: Path | None, output_dir: Path
    ) -> StageProduction:
        """Train a stage and return its validation-selected weights."""


@dataclass(frozen=True)
class StageResult:
    """One completed stage and its local selected checkpoint."""

    stage: TransferStageSpec
    production: StageProduction


def require_checkpoint(path: str | Path) -> Path:
    """Reject absent or empty weights before any checkpoint consumer runs."""
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    files = [checkpoint] if checkpoint.is_file() else list(checkpoint.rglob("*"))
    if not any(item.is_file() and item.stat().st_size > 0 for item in files):
        raise ValueError(f"Checkpoint is empty: {checkpoint}")
    return checkpoint


def execute_route(
    route: TransferRouteSpec,
    *,
    trainer: StageTrainer,
    output: Path,
    initial_checkpoint: Path | None = None,
) -> tuple[StageResult, ...]:
    """Train every stage once, continuing from each selected checkpoint on disk."""
    if route.execution_mode != "train":
        raise ValueError("Checkpoint evaluation routes cannot execute training stages.")
    parent = require_checkpoint(initial_checkpoint) if initial_checkpoint is not None else None
    completed = []
    for index, stage in enumerate(route.stages):
        directory = (output / f"{index + 1:02d}_{stage.name}").resolve()
        directory.mkdir(parents=True, exist_ok=True)
        production = trainer.train_stage(stage=stage, initial_checkpoint=parent, output_dir=directory)
        selected = require_checkpoint(production.checkpoint_path)
        if not selected.is_relative_to(directory):
            raise ValueError("A stage must save its selected checkpoint in its own output directory.")
        missing = set(stage.view_boundaries).difference(production.view_checkpoints)
        if missing:
            raise ValueError(f"Stage {stage.name!r} did not save its shorter-budget checkpoints: {sorted(missing)}")
        for checkpoint in production.view_checkpoints.values():
            require_checkpoint(checkpoint)
        if production.reload_validator is not None:
            production.reload_validator(selected)
        (directory / "stage.json").write_text(json.dumps(dict(production.report), indent=2) + "\n")
        completed.append(StageResult(stage, production))
        parent = selected
    return tuple(completed)
