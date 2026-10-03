"""Exact update-based Section 3 trainer for TabPFN v2.5.

The official TabPFN fine-tuning wrapper owns a complete epoch loop.  Section 3
instead needs one persistent optimizer while schema-native episodes from
different datasets are interleaved.  This module composes the wrapper's pinned
one-step primitives without changing the ordinary single-dataset ``fit`` path.
Vendor-private calls are isolated in :class:`TabPFNTransferRuntime` and checked
before any gradient update, so a future incompatible TabPFN release fails
closed rather than silently changing the experiment.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tfm_finetuning.experiments.transfer.pooled import build_balanced_episode_schedule
from tfm_finetuning.experiments.transfer.runner.data import transfer_package_versions
from tfm_finetuning.experiments.transfer.runner.executor import StageProduction
from tfm_finetuning.experiments.transfer.selection import (
    CheckpointValidation,
    ValidationMetric,
    select_source_mean_checkpoint,
    select_target_checkpoint,
)
from tfm_finetuning.experiments.transfer.spec import TransferRouteSpec, TransferStageSpec
from tfm_finetuning.finetuning.evaluation.core import (
    _evaluate_checkpoint,
    _evaluate_in_memory_estimator_on_data,
    _evaluate_zero_shot,
)
from tfm_finetuning.finetuning.models.base import (
    CommonFineTuneConfig,
    DatasetTransferTrainerContract,
)
from tfm_finetuning.finetuning.models.peak_memory import (
    peak_gpu_memory_gib,
    reset_peak_gpu_memory,
)
from tfm_finetuning.finetuning.models.tabpfn.transfer_runtime import (
    TabPFNTransferRuntime,
    dataset_episode_rng,
)


@dataclass(frozen=True)
class _SelectionSnapshot:
    """Best validation-selected state observed through one update prefix."""

    state: Mapping[str, Any]
    selected_update: int
    selection_score: float
    dataset_improvements: Mapping[str, float]


class TabPFNDatasetTransferStageTrainer:
    """Train exact Section 3 stages with a fresh stage-local optimizer.

    Example:
        The runner constructs this class through
        ``TabPFNAdapter.create_dataset_transfer_stage_trainer``.  Calling
        :meth:`train_stage` creates one optimizer, consumes at most the stage's
        declared update budget, selects checkpoints on the declared validation
        datasets, and returns a reload-validated weights-only package.
    """

    contract = DatasetTransferTrainerContract.exact_balanced()

    def __init__(
        self,
        *,
        adapter: Any,
        route: TransferRouteSpec,
        config: Mapping[str, Any],
        common_config: CommonFineTuneConfig,
        device: str | None,
        dataset_loader: Callable[..., Any],
        runtime_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.adapter = adapter
        self.route = route
        self.config = config
        self.common_config = common_config
        self.device = device
        self.dataset_loader = dataset_loader
        self.runtime_factory = runtime_factory or TabPFNTransferRuntime
        self._baseline_cache: dict[tuple[str, int, str], Mapping[str, Any]] = {}
        self._baseline_cache_hits = 0
        self._baseline_cache_misses = 0

    def train_stage(
        self,
        *,
        stage: TransferStageSpec,
        initial_checkpoint: Path | None,
        output_dir: Path,
    ) -> StageProduction:
        """Train and select one exact stage, never carrying optimizer state in."""
        reset_peak_gpu_memory(self.device)
        cache_hits_before = self._baseline_cache_hits
        cache_misses_before = self._baseline_cache_misses
        self._validate_stage(stage)
        output_dir.mkdir(parents=True, exist_ok=True)
        datasets = {
            name: self.dataset_loader(
                name,
                row_budget=stage.row_budget_per_dataset,
                row_budget_policy=stage.row_budget_policy,
                row_subset_seed=stage.row_subset_seed,
            )
            for name in stage.datasets
        }
        self._validate_dataset_tasks(stage, datasets)
        selector_data = {name: datasets[name] for name in stage.selector.datasets}
        baselines = self._validation_baselines(selector_data, seed=stage.seed)

        runtime = self.runtime_factory(
            adapter=self.adapter,
            common_config=self.common_config,
            task_type=next(iter(datasets.values())).task_type,
            seed=stage.seed,
            device=self.device,
            initial_checkpoint=initial_checkpoint,
            total_updates=stage.update_budget,
        )
        schedule = self._schedule(stage)
        candidates: list[CheckpointValidation] = []
        counts: Counter[str] = Counter()
        selected: _SelectionSnapshot | None = None
        prefix_snapshots: dict[int, _SelectionSnapshot] = {}
        realized_examples = 0
        validation_seconds = 0.0
        training_started = time.monotonic()
        best_early_score = -math.inf
        stale_updates = 0
        required_prefix = max(stage.view_boundaries, default=0)

        try:
            runtime.synchronize()
            gpu_started = time.monotonic()
            for zero_based_update, dataset_name in enumerate(schedule):
                dataset_update = counts[dataset_name]
                rng = dataset_episode_rng(stage.seed, dataset_name, dataset_update)
                episode = runtime.sample_episode(datasets[dataset_name], rng=rng)
                runtime.train_episode(episode)
                update = zero_based_update + 1
                counts[dataset_name] += 1
                realized_examples += episode.example_count

                # A checkpoint candidate is observed only after a complete
                # balanced dataset round. Single-dataset stages therefore
                # retain per-update validation while pooled candidates never
                # privilege whichever dataset happened to run first.
                if update % len(stage.datasets):
                    continue

                validation_started = time.monotonic()
                candidate = self._validation_candidate(
                    runtime,
                    selector_data=selector_data,
                    baselines=baselines,
                    update=update,
                )
                candidates.append(candidate)
                current_selection = _select_candidate(stage, candidates)
                validation_seconds += time.monotonic() - validation_started
                if selected is None or current_selection.checkpoint_id == candidate.checkpoint_id:
                    selected = _SelectionSnapshot(
                        state=runtime.clone_state(),
                        selected_update=current_selection.step,
                        selection_score=current_selection.selection_score,
                        dataset_improvements=dict(current_selection.dataset_improvements),
                    )
                if update in stage.view_boundaries:
                    if selected is None:
                        raise RuntimeError(f"No validation-selected state exists at update {update}.")
                    prefix_snapshots[update] = selected

                early_score = float(current_selection.selection_score)
                if early_score > best_early_score + float(self.common_config.min_delta):
                    best_early_score = early_score
                    stale_updates = 0
                else:
                    stale_updates += 1
                if (
                    self.common_config.early_stopping
                    and update >= required_prefix
                    and stale_updates >= int(self.common_config.patience)
                ):
                    break
            runtime.synchronize()
            gpu_seconds = time.monotonic() - gpu_started

            if selected is None:
                raise RuntimeError("TabPFN transfer stage completed no valid optimizer update.")
            realized_updates = sum(counts.values())
            if selected.selected_update > realized_updates:
                raise RuntimeError("Selected TabPFN checkpoint lies beyond the realized update prefix.")
            missing_views = set(stage.view_boundaries).difference(prefix_snapshots)
            if missing_views:
                raise RuntimeError(
                    f"TabPFN training stopped before required checkpoint views: {sorted(missing_views)}."
                )
            checkpoint_path = runtime.export_state(
                selected.state,
                output_dir,
                update=selected.selected_update,
                score=selected.selection_score,
                checkpoint_mode="best",
            )
            view_checkpoints = {
                boundary: runtime.export_state(
                    snapshot.state,
                    output_dir / f"view_{boundary}",
                    update=snapshot.selected_update,
                    score=snapshot.selection_score,
                    checkpoint_mode="final",
                )
                for boundary, snapshot in prefix_snapshots.items()
            }
        finally:
            runtime.close()

        evaluation_data = datasets.get(self.route.target_dataset)
        if stage.role == "terminal" and evaluation_data is None:
            evaluation_data = self.dataset_loader(
                self.route.target_dataset,
                row_budget=stage.row_budget_per_dataset,
                row_budget_policy=stage.row_budget_policy,
                row_subset_seed=stage.row_subset_seed,
            )
        if evaluation_data is None:
            evaluation_data = selector_data[stage.selector.datasets[0]]
        terminal_baseline: Mapping[str, Any] = {}
        terminal_metrics: Mapping[str, Any] = {}
        if stage.role == "terminal":
            terminal_baseline = self._evaluate_zero_shot_target(evaluation_data, seed=stage.seed)
            terminal_metrics = self._evaluate_checkpoint_target(
                checkpoint_path,
                evaluation_data,
                seed=stage.seed,
            )
        view_metrics = {
            str(boundary): {
                "metrics": self._evaluate_checkpoint_target(path, evaluation_data, seed=stage.seed),
                "baseline_metrics": terminal_baseline,
            }
            for boundary, path in view_checkpoints.items()
        }
        report = {
            "allocated_optimizer_updates": stage.update_budget,
            "realized_optimizer_updates": sum(counts.values()),
            "dataset_optimizer_updates": {name: int(counts[name]) for name in stage.datasets},
            "realized_training_examples": int(realized_examples),
            "amp_overflow_retry_count": int(getattr(runtime, "amp_overflow_retry_count", 0)),
            "unstratified_episode_count": int(getattr(runtime, "unstratified_episode_count", 0)),
            "gpu_time_seconds": float(gpu_seconds),
            "step_dependent_time_seconds": float(gpu_seconds),
            "step_dependent_time_method": "synchronized_training_and_validation_loop",
            "peak_gpu_memory_gib": peak_gpu_memory_gib(self.device),
            "validation_time_seconds": float(validation_seconds),
            "wall_time_seconds": float(time.monotonic() - training_started),
            "validation_cadence_updates": len(stage.datasets),
            "validation_candidate_updates": [candidate.step for candidate in candidates],
            "checkpoint_selection": {
                "kind": stage.selector.kind,
                "datasets": list(stage.selector.datasets),
                "max_update": int(sum(counts.values())),
                "selected_update": selected.selected_update,
                "selection_score": selected.selection_score,
                "dataset_improvements": dict(selected.dataset_improvements),
            },
            "score_metric": str(evaluation_data.metric_name),
            "metrics": terminal_metrics,
            "zero-shot-metrics": terminal_baseline,
            "checkpoint_view_metrics": view_metrics,
            "checkpoint_view_selection": {
                str(boundary): {
                    "kind": stage.selector.kind,
                    "datasets": list(stage.selector.datasets),
                    "max_update": boundary,
                    "selected_update": snapshot.selected_update,
                    "selection_score": snapshot.selection_score,
                    "dataset_improvements": dict(snapshot.dataset_improvements),
                }
                for boundary, snapshot in prefix_snapshots.items()
            },
            "validation_baseline_cache_hits": self._baseline_cache_hits - cache_hits_before,
            "validation_baseline_cache_misses": self._baseline_cache_misses - cache_misses_before,
        }
        return StageProduction(
            checkpoint_path=checkpoint_path,
            report=report,
            view_checkpoints=view_checkpoints,
            package_versions=transfer_package_versions(self.adapter.model_id),
            reload_validator=self._reload_validator(evaluation_data, seed=stage.seed),
        )

    def _validate_stage(self, stage: TransferStageSpec) -> None:
        if stage.model_id != self.adapter.model_id:
            raise ValueError(
                f"TabPFN transfer trainer received model {stage.model_id!r}, expected {self.adapter.model_id!r}."
            )
        if stage.task_family not in {"classification", "regression"}:
            raise ValueError(f"Unsupported Section 3 TabPFN task family: {stage.task_family!r}.")
        if stage.preprocessing != "shared":
            raise ValueError("TabPFN Section 3 requires independently applied shared finite-numeric preprocessing.")
        if stage.update_budget % len(stage.datasets):
            raise ValueError("A balanced TabPFN stage requires an update budget divisible by its dataset count.")
        if any(boundary % len(stage.datasets) for boundary in stage.view_boundaries):
            raise ValueError("TabPFN checkpoint views must coincide with complete balanced dataset rounds.")
        if len(set(stage.sampling_weights)) != 1:
            raise ValueError("TabPFN Section 3 currently requires equal dataset sampling weights.")
        if not stage.strict_resource_semantics:
            raise ValueError("TabPFN Section 3 requires strict resource semantics.")

    @staticmethod
    def _validate_dataset_tasks(stage: TransferStageSpec, datasets: Mapping[str, Any]) -> None:
        task_types = {data.task_type for data in datasets.values()}
        allowed = {"regression"} if stage.task_family == "regression" else {"binary", "multiclass"}
        if not task_types or not task_types.issubset(allowed):
            raise ValueError(
                f"Stage {stage.name!r} requires {stage.task_family} datasets, got {sorted(task_types)}."
            )

    @staticmethod
    def _schedule(stage: TransferStageSpec) -> tuple[str, ...]:
        return build_balanced_episode_schedule(
            stage.datasets,
            updates_per_dataset=stage.update_budget // len(stage.datasets),
            seed=stage.seed,
        )

    def _validation_baselines(self, datasets: Mapping[str, Any], *, seed: int) -> dict[str, float]:
        values: dict[str, float] = {}
        for name, data in datasets.items():
            metrics = self._zero_shot_splits(data, seed=seed, splits=("val",))
            values[name] = _score(metrics, split="val", context=f"baseline/{name}")
        return values

    def _validation_candidate(
        self,
        runtime: Any,
        *,
        selector_data: Mapping[str, Any],
        baselines: Mapping[str, float],
        update: int,
    ) -> CheckpointValidation:
        metrics: dict[str, ValidationMetric] = {}
        for name, data in selector_data.items():
            payload = _evaluate_in_memory_estimator_on_data(
                adapter=self.adapter,
                estimator=runtime.wrapper,
                data=data,
                device=self.device,
                common_config=self.common_config,
                timing_rows=None,
                phase="transfer_stage_validation",
                eval_source="live_weights",
                epoch=update,
                splits=("val",),
            )
            metrics[name] = ValidationMetric(
                value=_score(payload, split="val", context=f"update_{update}/{name}"),
                baseline=baselines[name],
                higher_is_better=True,
            )
        return CheckpointValidation(
            checkpoint_id=f"update_{update:06d}",
            step=update,
            metrics=metrics,
        )

    def _evaluate_zero_shot_target(self, data: Any, *, seed: int) -> Mapping[str, Any]:
        return self._zero_shot_splits(data, seed=seed, splits=("val", "test"))

    def _zero_shot_splits(
        self,
        data: Any,
        *,
        seed: int,
        splits: tuple[str, ...],
    ) -> dict[str, Any]:
        dataset_name = str(getattr(data, "name", self.route.target_dataset))
        result: dict[str, Any] = {}
        missing: list[str] = []
        for split in splits:
            key = (dataset_name, int(seed), split)
            cached = self._baseline_cache.get(key)
            if cached is None:
                self._baseline_cache_misses += 1
                missing.append(split)
            else:
                self._baseline_cache_hits += 1
                result[split] = cached
        if missing:
            evaluated = _evaluate_zero_shot(
                adapter=self.adapter,
                data=data,
                seed=seed,
                device=self.device,
                common_config=self.common_config,
                phase="transfer_stage_baseline",
                eval_source="base_pure_icl",
                splits=tuple(missing),
            )
            for split in missing:
                payload = evaluated[split]
                self._baseline_cache[(dataset_name, int(seed), split)] = payload
                result[split] = payload
        return result

    def _evaluate_checkpoint_target(
        self,
        checkpoint: Path,
        data: Any,
        *,
        seed: int,
    ) -> Mapping[str, Any]:
        return _evaluate_checkpoint(
            adapter=self.adapter,
            checkpoint_path=checkpoint,
            data=data,
            seed=int(seed),
            device=self.device,
            common_config=self.common_config,
            phase="transfer_selected_checkpoint",
            eval_source="selected_weights",
            splits=("val", "test"),
        )

    def _reload_validator(self, data: Any, *, seed: int) -> Callable[[Path], Any]:
        def _validate(path: Path) -> Any:
            estimator = self.adapter.load_checkpoint_for_inference(
                path,
                data.task_type,
                self.device,
                max(1, int(self.common_config.n_estimators_validation)),
                seed=int(seed),
            )
            context_X, context_y = _small_reload_context(
                data.val_context_X,
                data.val_context_y,
                classification=data.task_type != "regression",
            )
            fitted = self.adapter.fit_inference_context(estimator, context_X, context_y)
            query = data.val_X.iloc[:1] if hasattr(data.val_X, "iloc") else data.val_X[:1]
            predictions = np.asarray(self.adapter.predict_for_metrics(fitted, query, data.task_type))
            if predictions.size == 0 or not np.isfinite(predictions).all():
                raise RuntimeError("Reloaded TabPFN checkpoint produced invalid smoke-test predictions.")
            return True

        return _validate


def _select_candidate(stage: TransferStageSpec, candidates: list[CheckpointValidation]) -> Any:
    if stage.selector.kind == "target_validation":
        return select_target_checkpoint(candidates, target_dataset=stage.selector.datasets[0])
    if stage.selector.kind == "equal_mean_validation_improvement":
        return select_source_mean_checkpoint(candidates, source_datasets=stage.selector.datasets)
    raise ValueError(f"Unsupported transfer validation selector: {stage.selector.kind!r}.")


def _score(payload: Mapping[str, Any], *, split: str, context: str) -> float:
    split_payload = payload.get(split)
    value = split_payload.get("score") if isinstance(split_payload, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise RuntimeError(f"TabPFN transfer evaluation {context!r} produced invalid score {value!r}.")
    return float(value)


def _small_reload_context(
    X: Any,
    y: Any,
    *,
    classification: bool,
    maximum_rows: int = 128,
) -> tuple[Any, np.ndarray]:
    """Return a small deterministic context, class-complete when applicable."""
    labels = np.asarray(y)
    selected: list[int] = []
    if classification:
        for label in np.unique(labels):
            selected.append(int(np.flatnonzero(labels == label)[0]))
    selected.extend(index for index in range(min(len(labels), maximum_rows)) if index not in selected)
    indices = np.asarray(selected[:maximum_rows], dtype=int)
    frame = X.iloc[indices] if hasattr(X, "iloc") else X[indices]
    return frame, labels[indices]


__all__ = ["TabPFNDatasetTransferStageTrainer"]
