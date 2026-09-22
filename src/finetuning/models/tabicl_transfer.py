"""Section 3 exact stage trainer for TabICLv2.

The orchestrator in this module is intentionally backend-injectable: production
uses :class:`TabICLTransferEngine`, while focused tests can verify scheduling,
selection, early stopping, prefix views, and report contracts without loading
the multi-gigabyte foundation checkpoint.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from experiments.dataset_transfer.executor import StageProduction
from experiments.dataset_transfer.pooled import build_balanced_episode_schedule
from experiments.dataset_transfer.selection import (
    CheckpointSelection,
    CheckpointValidation,
    ValidationMetric,
    select_source_mean_checkpoint,
    select_target_checkpoint,
)
from experiments.dataset_transfer.spec import TransferRouteSpec, TransferStageSpec
from finetuning.models.base import DatasetTransferTrainerContract
from finetuning.models.gpu_memory import (
    peak_gpu_memory_gib,
    reset_peak_gpu_memory,
)
from finetuning.models.tabicl_transfer_engine import TabICLTransferEngine

if TYPE_CHECKING:
    from finetuning.models.base import CommonFineTuneConfig
    from finetuning.models.tabicl import TabICLAdapter


class _TransferEngine(Protocol):
    gpu_time_seconds: float

    def train_update(self, dataset_name: str, *, rng: np.random.Generator) -> Any: ...

    def clone_state(self) -> Mapping[str, Any]: ...

    def evaluate_current(self, dataset_name: str, *, splits: tuple[str, ...]) -> Mapping[str, Any]: ...

    def evaluate_base(self, dataset_name: str, *, splits: tuple[str, ...]) -> Mapping[str, Any]: ...

    def evaluate_checkpoint(
        self,
        checkpoint_path: Path,
        dataset_name: str,
        *,
        splits: tuple[str, ...],
    ) -> Mapping[str, Any]: ...

    def export_state(
        self,
        state: Mapping[str, Any],
        output_dir: Path,
        *,
        update: int,
        metric: float,
    ) -> Path: ...

    def validate_reload(self, checkpoint_path: Path) -> bool: ...

    def diagnostics(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class _CandidateState:
    selection: CheckpointSelection
    state: Mapping[str, Any]


class TabICLDatasetTransferStageTrainer:
    """Train exact Section 3 stages with one optimizer session per stage."""

    contract = DatasetTransferTrainerContract.exact_balanced()

    def __init__(
        self,
        *,
        adapter: TabICLAdapter,
        route: TransferRouteSpec,
        config: Mapping[str, Any],
        common_config: CommonFineTuneConfig,
        device: str | None,
        dataset_loader: Callable[..., Any],
        engine_factory: Callable[..., _TransferEngine] = TabICLTransferEngine,
    ) -> None:
        self.adapter = adapter
        self.route = route
        self.config = config
        self.common_config = common_config
        self.device = device
        self.dataset_loader = dataset_loader
        self.engine_factory = engine_factory

    def train_stage(
        self,
        *,
        stage: TransferStageSpec,
        initial_checkpoint: Path | None,
        output_dir: Path,
    ) -> StageProduction:
        """Train, validation-select, export, and audit exactly one stage."""
        reset_peak_gpu_memory(self.device)
        if stage.model_id != "tabicl_v2":
            raise ValueError(f"TabICL transfer trainer received model_id={stage.model_id!r}.")
        if stage.strict_resource_semantics is not True:
            raise ValueError("TabICL Section 3 requires strict resource semantics.")
        dataset_count = len(stage.datasets)
        if stage.update_budget % dataset_count:
            raise ValueError("Balanced TabICL stages require update_budget divisible by dataset count.")
        if len(set(stage.sampling_weights)) != 1:
            raise ValueError("TabICL Section 3 currently supports uniform dataset sampling only.")
        if any(boundary % dataset_count for boundary in stage.view_boundaries):
            raise ValueError("Checkpoint views must align with complete balanced dataset rounds.")

        phase_offset = len(getattr(self.dataset_loader, "runtime_phases", ()))
        datasets = {
            name: self.dataset_loader(
                name,
                row_budget=stage.row_budget_per_dataset,
                row_budget_policy=stage.row_budget_policy,
                row_subset_seed=stage.row_subset_seed,
            )
            for name in stage.datasets
        }
        if stage.role == "terminal" and self.route.target_dataset not in datasets:
            datasets[self.route.target_dataset] = self.dataset_loader(
                self.route.target_dataset,
                row_budget=stage.row_budget_per_dataset,
                row_budget_policy=stage.row_budget_policy,
                row_subset_seed=stage.row_subset_seed,
            )
        engine = self.engine_factory(
            adapter=self.adapter,
            stage=stage,
            common_config=self.common_config,
            device=self.device,
            datasets=datasets,
            initial_checkpoint=initial_checkpoint,
        )

        baseline_by_dataset = {name: engine.evaluate_base(name, splits=("val",)) for name in stage.selector.datasets}
        schedule = build_balanced_episode_schedule(
            stage.datasets,
            updates_per_dataset=stage.update_budget // dataset_count,
            seed=int(stage.seed),
        )
        candidates: list[CheckpointValidation] = []
        selected: _CandidateState | None = None
        view_states: dict[int, _CandidateState] = {}
        dataset_updates: Counter[str] = Counter()
        dataset_examples: Counter[str] = Counter()
        history: list[dict[str, Any]] = []
        validation_seconds = 0.0
        patience_counter = 0
        early_stopping_best = -math.inf
        minimum_updates_before_stop = max(stage.view_boundaries, default=0)
        realized_updates = 0
        stage_started = time.monotonic()

        for global_index, dataset_name in enumerate(schedule):
            dataset_index = dataset_updates[dataset_name]
            update_result = engine.train_update(
                dataset_name,
                rng=_episode_rng(
                    seed=int(stage.seed),
                    dataset_name=dataset_name,
                    dataset_update_index=dataset_index,
                ),
            )
            dataset_updates[dataset_name] += 1
            dataset_examples[dataset_name] += int(update_result.examples)
            realized_updates = global_index + 1
            if realized_updates % dataset_count:
                continue

            validation_started = time.monotonic()
            candidate = _validation_candidate(
                engine,
                stage=stage,
                update=realized_updates,
                baselines=baseline_by_dataset,
            )
            candidates.append(candidate)
            next_selection = _select_candidate(candidates, stage=stage)
            if selected is None or next_selection.checkpoint_id != selected.selection.checkpoint_id:
                selected = _CandidateState(next_selection, engine.clone_state())
            validation_seconds += time.monotonic() - validation_started
            history.append(_history_row(candidate, selected.selection))

            for boundary in stage.view_boundaries:
                if boundary == realized_updates:
                    view_states[boundary] = selected

            current_score = _candidate_selection_score(candidate, stage=stage)
            if current_score > early_stopping_best + float(self.common_config.min_delta):
                early_stopping_best = current_score
                patience_counter = 0
            else:
                patience_counter += 1
            if (
                self.common_config.early_stopping
                and realized_updates >= minimum_updates_before_stop
                and patience_counter >= int(self.common_config.patience)
            ):
                break

        if selected is None:
            raise RuntimeError(f"TabICL stage {stage.name!r} produced no finite validation candidate.")
        missing_views = set(stage.view_boundaries).difference(view_states)
        if missing_views:
            raise RuntimeError(f"TabICL stage stopped before declared checkpoint views {sorted(missing_views)}.")

        checkpoint_path = engine.export_state(
            selected.state,
            output_dir / "selected",
            update=selected.selection.step,
            metric=selected.selection.selection_score,
        )
        view_checkpoints = {
            boundary: engine.export_state(
                candidate.state,
                output_dir / "views" / str(boundary),
                update=candidate.selection.step,
                metric=candidate.selection.selection_score,
            )
            for boundary, candidate in view_states.items()
        }
        report = self._build_report(
            stage=stage,
            datasets=datasets,
            engine=engine,
            selected=selected,
            view_states=view_states,
            checkpoint_path=checkpoint_path,
            view_checkpoints=view_checkpoints,
            baseline_by_dataset=baseline_by_dataset,
            dataset_updates=dataset_updates,
            dataset_examples=dataset_examples,
            history=history,
            validation_seconds=validation_seconds,
            realized_updates=realized_updates,
            stage_started=stage_started,
            phase_offset=phase_offset,
        )
        from experiments.dataset_transfer.runner_data import transfer_package_versions

        return StageProduction(
            checkpoint_path=checkpoint_path,
            report=report,
            view_checkpoints=view_checkpoints,
            package_versions=transfer_package_versions("tabicl_v2"),
            reload_validator=engine.validate_reload,
        )

    def _build_report(
        self,
        *,
        stage: TransferStageSpec,
        datasets: Mapping[str, Any],
        engine: _TransferEngine,
        selected: _CandidateState,
        view_states: Mapping[int, _CandidateState],
        checkpoint_path: Path,
        view_checkpoints: Mapping[int, Path],
        baseline_by_dataset: Mapping[str, Mapping[str, Any]],
        dataset_updates: Mapping[str, int],
        dataset_examples: Mapping[str, int],
        history: list[dict[str, Any]],
        validation_seconds: float,
        realized_updates: int,
        stage_started: float,
        phase_offset: int,
    ) -> dict[str, Any]:
        target = self.route.target_dataset
        terminal_metrics: Mapping[str, Any] = {}
        terminal_baseline: Mapping[str, Any] = {}
        view_metrics: dict[str, Any] = {}
        if stage.role == "terminal":
            terminal_metrics = engine.evaluate_checkpoint(
                checkpoint_path,
                target,
                splits=("val", "test"),
            )
            terminal_baseline = _merge_baseline_splits(
                engine,
                target,
                cached=baseline_by_dataset.get(target),
                required=("val", "test"),
            )
            for boundary, path in view_checkpoints.items():
                view_metrics[str(boundary)] = {
                    "metrics": engine.evaluate_checkpoint(path, target, splits=("val", "test")),
                    "baseline_metrics": terminal_baseline,
                }

        checkpoint_view_selection = {
            str(boundary): {
                **_selection_payload(candidate.selection, stage=stage),
                "max_update": int(boundary),
            }
            for boundary, candidate in view_states.items()
        }
        runtime_phases = list(getattr(self.dataset_loader, "runtime_phases", ()))[phase_offset:]
        report: dict[str, Any] = {
            "allocated_optimizer_updates": int(stage.update_budget),
            "realized_optimizer_updates": int(realized_updates),
            "dataset_optimizer_updates": {name: int(dataset_updates.get(name, 0)) for name in stage.datasets},
            "realized_training_examples": int(sum(dataset_examples.values())),
            "dataset_training_examples": {name: int(dataset_examples.get(name, 0)) for name in stage.datasets},
            "gpu_time_seconds": float(engine.gpu_time_seconds),
            "step_dependent_time_seconds": float(
                engine.gpu_time_seconds + validation_seconds
            ),
            "step_dependent_time_method": "training_plus_validation",
            "peak_gpu_memory_gib": peak_gpu_memory_gib(self.device),
            "validation_time_seconds": float(validation_seconds),
            "checkpoint_selection": _selection_payload(selected.selection, stage=stage),
            "checkpoint_view_selection": checkpoint_view_selection,
            "checkpoint_view_metrics": view_metrics,
            "validation_cadence": {
                "kind": "complete_balanced_dataset_round",
                "optimizer_updates_per_validation": len(stage.datasets),
                "minimum_updates_before_early_stopping": max(stage.view_boundaries, default=0),
            },
            "train_history": history,
            "runtime_phases": runtime_phases,
            "fit_events": [
                {
                    "stage": stage.name,
                    "training_mode": stage.training_mode,
                    "allocated_optimizer_updates": int(stage.update_budget),
                    "realized_optimizer_updates": int(realized_updates),
                    "seconds": float(time.monotonic() - stage_started),
                }
            ],
            "tabicl_transfer_diagnostics": dict(engine.diagnostics()),
        }
        if stage.role == "terminal":
            report.update(
                {
                    "score_metric": datasets[target].metric_name,
                    "score_metric_source": "foundation_model.metric",
                    "metrics": dict(terminal_metrics),
                    "zero-shot-metrics": dict(terminal_baseline),
                }
            )
        return report


def _episode_rng(*, seed: int, dataset_name: str, dataset_update_index: int) -> np.random.Generator:
    token = int.from_bytes(hashlib.sha256(dataset_name.encode("utf-8")).digest()[:4], "big")
    return np.random.default_rng(np.random.SeedSequence([int(seed), token, int(dataset_update_index)]))


def _validation_candidate(
    engine: _TransferEngine,
    *,
    stage: TransferStageSpec,
    update: int,
    baselines: Mapping[str, Mapping[str, Any]],
) -> CheckpointValidation:
    metrics: dict[str, ValidationMetric] = {}
    for dataset_name in stage.selector.datasets:
        current = engine.evaluate_current(dataset_name, splits=("val",))
        value = _split_score(current, "val", context=f"{dataset_name} checkpoint update {update}")
        baseline = _split_score(baselines[dataset_name], "val", context=f"{dataset_name} pure ICL")
        metrics[dataset_name] = ValidationMetric(value=value, baseline=baseline, higher_is_better=True)
    return CheckpointValidation(
        checkpoint_id=f"update-{int(update):08d}",
        step=int(update),
        metrics=metrics,
    )


def _select_candidate(
    candidates: list[CheckpointValidation],
    *,
    stage: TransferStageSpec,
) -> CheckpointSelection:
    if stage.selector.kind == "target_validation":
        return select_target_checkpoint(candidates, target_dataset=stage.selector.datasets[0])
    return select_source_mean_checkpoint(candidates, source_datasets=stage.selector.datasets)


def _candidate_selection_score(candidate: CheckpointValidation, *, stage: TransferStageSpec) -> float:
    selected = _select_candidate([candidate], stage=stage)
    return float(selected.selection_score)


def _selection_payload(selection: CheckpointSelection, *, stage: TransferStageSpec) -> dict[str, Any]:
    return {
        "kind": stage.selector.kind,
        "datasets": list(stage.selector.datasets),
        "selected_update": int(selection.step),
        "selection_score": float(selection.selection_score),
        "dataset_improvements": {name: float(value) for name, value in selection.dataset_improvements.items()},
    }


def _history_row(candidate: CheckpointValidation, selection: CheckpointSelection) -> dict[str, Any]:
    return {
        "update": int(candidate.step),
        "selection_score": float(_candidate_selection_score_from_metrics(candidate)),
        "selected_update": int(selection.step),
        "validation": {
            name: {
                "score": float(metric.value),
                "baseline_score": float(metric.baseline),
                "improvement": float(metric.improvement),
            }
            for name, metric in candidate.metrics.items()
        },
    }


def _candidate_selection_score_from_metrics(candidate: CheckpointValidation) -> float:
    improvements = [metric.improvement for metric in candidate.metrics.values()]
    return float(sum(improvements) / len(improvements))


def _merge_baseline_splits(
    engine: _TransferEngine,
    dataset_name: str,
    *,
    cached: Mapping[str, Any] | None,
    required: tuple[str, ...],
) -> dict[str, Any]:
    merged = dict(cached or {})
    missing = tuple(split for split in required if split not in merged)
    if missing:
        merged.update(engine.evaluate_base(dataset_name, splits=missing))
    return merged


def _split_score(payload: Mapping[str, Any], split: str, *, context: str) -> float:
    split_payload = payload.get(split)
    value = split_payload.get("score") if isinstance(split_payload, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise RuntimeError(f"{context} has no finite {split} score.")
    return float(value)


__all__ = [
    "TabICLDatasetTransferStageTrainer",
]
