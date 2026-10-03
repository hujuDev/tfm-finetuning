"""Construct model transfer trainers and check their scientific output budgets."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from experiments.dataset_transfer.executor import StageProduction
from experiments.dataset_transfer.runner_data import TransferDatasetLoader
from experiments.dataset_transfer.spec import TransferRouteSpec, TransferStageSpec, TransferTrainingRecipe
from finetuning.models.base import DatasetTransferStageTrainerAdapter, DatasetTransferTrainerRequest
from finetuning.run.setup import FoundationRunSetup


class ValidatedStageTrainer:
    """Check exact balanced updates and validation-only checkpoint selection."""

    def __init__(self, delegate: Any, recipe: TransferTrainingRecipe) -> None:
        self.delegate = delegate
        self.recipe = recipe
        if TransferTrainingRecipe.from_common_config(delegate.common_config) != recipe:
            raise ValueError("The transfer trainer does not use the configured optimization recipe.")

    def train_stage(
        self, *, stage: TransferStageSpec, initial_checkpoint: Path | None, output_dir: Path
    ) -> StageProduction:
        """Train a stage and reject incomplete or scientifically different results."""
        if stage.training_recipe != self.recipe or stage.optimizer_reset != "fresh":
            raise ValueError("Every stage must use the configured recipe and a fresh optimizer.")
        production = self.delegate.train_stage(
            stage=stage, initial_checkpoint=initial_checkpoint, output_dir=output_dir
        )
        report = production.report
        if (
            report.get("allocated_optimizer_updates") != stage.update_budget
            or report.get("realized_optimizer_updates") != stage.update_budget
        ):
            raise ValueError(f"Stage {stage.name!r} did not realize its exact update budget.")
        counts = report.get("dataset_optimizer_updates", {})
        if (
            set(counts) != set(stage.datasets)
            or any(type(value) is not int or value < 0 for value in counts.values())
            or sum(counts.values()) != stage.update_budget
            or max(counts.values()) - min(counts.values()) > 1
        ):
            raise ValueError(f"Stage {stage.name!r} has an invalid per-dataset update allocation.")
        _validate_selection(report.get("checkpoint_selection"), stage, limit=stage.update_budget)
        for boundary in stage.view_boundaries:
            selection = report.get("checkpoint_view_selection", {}).get(str(boundary))
            _validate_selection(selection, stage, limit=boundary)
            payload = report.get("checkpoint_view_metrics", {}).get(str(boundary), {})
            _validate_scores(payload.get("metrics"))
            _validate_scores(payload.get("baseline_metrics"))
        if stage.role == "terminal":
            _validate_scores(report.get("metrics"))
            _validate_scores(report.get("zero-shot-metrics"))
        return production


def _validate_selection(payload: Any, stage: TransferStageSpec, *, limit: int) -> None:
    if not isinstance(payload, Mapping) or any(
        payload.get(key) != value
        for key, value in {"kind": stage.selector.kind, "datasets": list(stage.selector.datasets)}.items()
    ):
        raise ValueError(f"Stage {stage.name!r} did not apply its declared validation selector.")
    selected = payload.get("selected_update")
    if type(selected) is not int or not 0 < selected <= limit:
        raise ValueError(f"Stage {stage.name!r} selected a checkpoint outside its {limit}-update budget.")
    if limit in stage.view_boundaries and payload.get("max_update") != limit:
        raise ValueError("Shorter-budget checkpoints must be selected within their own update horizon.")


def _validate_scores(payload: Any) -> None:
    for split in ("val", "test"):
        value = payload.get(split, {}).get("score") if isinstance(payload, Mapping) else None
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
            raise ValueError("Transfer results need finite validation/test scores and matched ICL baselines.")


def build_stage_trainer(
    setup: FoundationRunSetup, *, route: TransferRouteSpec, dataset_loader: TransferDatasetLoader
) -> ValidatedStageTrainer:
    """Create a supported exact-update trainer; no epoch approximations are used."""
    if not isinstance(setup.adapter, DatasetTransferStageTrainerAdapter):
        raise ValueError(f"{setup.spec.model_id} has no dataset-transfer stage trainer.")
    trainer = setup.adapter.create_dataset_transfer_stage_trainer(
        DatasetTransferTrainerRequest(
            route=route,
            config=setup.config,
            common_config=setup.common_config,
            device=setup.device,
            dataset_loader=dataset_loader,
        )
    )
    missing = trainer.contract.missing_for(route)
    if missing:
        raise ValueError(f"{setup.spec.model_id} transfer trainer lacks: {', '.join(missing)}")
    return ValidatedStageTrainer(trainer, TransferTrainingRecipe.from_common_config(setup.common_config))
