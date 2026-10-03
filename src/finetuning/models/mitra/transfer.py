"""Exact Section 3 stage trainer for the maintained Mitra configuration."""

from __future__ import annotations

import hashlib
import shutil
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from experiments.transfer.pooled import build_balanced_episode_schedule
from experiments.transfer.runner.executor import StageProduction
from experiments.transfer.spec import TransferRouteSpec, TransferStageSpec
from finetuning.models.base import DatasetTransferTrainerContract
from finetuning.models.mitra.transfer_evaluation import (
    MitraTransferCandidate,
    MitraTransferEvaluator,
)
from finetuning.models.mitra.transfer_session import (
    MitraDatasetEpisodeSource,
    MitraPersistentStageSession,
    assign_transfer_member_indices,
    exact_stage_schedulers,
    initialize_stage_estimator,
    prepare_stage_datasets,
)
from finetuning.models.peak_memory import (
    peak_gpu_memory_gib,
    reset_peak_gpu_memory,
)


class MitraDatasetTransferStageTrainer:
    """Train one reset-safe Mitra stage with exact logical update accounting."""

    contract = DatasetTransferTrainerContract.exact_balanced(
        logical_update_unit="dataset_episode_all_ensemble_members",
        physical_member_steps_reported=True,
    )

    def __init__(
        self,
        adapter: Any,
        *,
        route: TransferRouteSpec,
        config: Mapping[str, Any],
        common_config: Any,
        device: str | None,
        dataset_loader: Any,
    ) -> None:
        self.adapter = adapter
        self.route = route
        self.config = config
        self.common_config = common_config
        self.device = device
        self.dataset_loader = dataset_loader

    def train_stage(
        self,
        *,
        stage: TransferStageSpec,
        initial_checkpoint: Path | None,
        output_dir: Path,
    ) -> StageProduction:
        """Train, validation-select, and package one independent stage."""
        reset_peak_gpu_memory(self.device)
        started_at = time.monotonic()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if stage.model_id != "mitra":
            raise ValueError(f"Mitra transfer trainer received model_id={stage.model_id!r}.")
        if stage.task_family not in {"classification", "regression"}:
            raise ValueError(f"Unsupported Mitra Section 3 task family: {stage.task_family!r}.")
        if stage.preprocessing != "shared":
            raise ValueError("Mitra Section 3 requires independently applied shared finite-numeric preprocessing.")
        if stage.optimizer_reset != "fresh":
            raise ValueError("Mitra Section 3 stages must reconstruct a fresh optimizer at every stage boundary.")
        if stage.strict_resource_semantics is not True:
            raise ValueError("Mitra Section 3 stages require strict_resource_semantics=true.")
        sampling_weights = tuple(float(weight) for weight in stage.sampling_weights)
        if not all(np.isfinite(weight) for weight in sampling_weights) or len(set(sampling_weights)) != 1:
            raise ValueError("Mitra Section 3 stages currently require uniform dataset sampling weights.")
        if stage.update_budget % len(stage.datasets) != 0:
            raise ValueError("Mitra balanced stages require update_budget divisible by dataset count.")
        if any(boundary % len(stage.datasets) != 0 for boundary in stage.view_boundaries):
            raise ValueError("Mitra checkpoint views must coincide with complete balanced validation rounds.")
        evaluator = MitraTransferEvaluator(
            adapter=self.adapter,
            route=self.route,
            common_config=self.common_config,
            device=self.device,
            dataset_loader=self.dataset_loader,
        )

        raw_datasets = {
            name: self.dataset_loader(
                name,
                row_budget=stage.row_budget_per_dataset,
                row_budget_policy=stage.row_budget_policy,
                row_subset_seed=stage.row_subset_seed,
            )
            for name in stage.datasets
        }
        task_types = {str(data.task_type) for data in raw_datasets.values()}
        allowed = {"regression"} if stage.task_family == "regression" else {"binary", "multiclass"}
        if not task_types or not task_types.issubset(allowed):
            raise ValueError(
                f"Mitra Section 3 requires {stage.task_family} datasets, got {sorted(task_types)}."
            )
        representative_task = str(next(iter(raw_datasets.values())).task_type)
        estimator = initialize_stage_estimator(
            self.adapter,
            task_type=representative_task,
            common_config=self.common_config,
            seed=stage.seed,
            device=self.device,
            initial_checkpoint=initial_checkpoint,
        )
        trainers = tuple(estimator.trainers)
        assign_transfer_member_indices(trainers)
        prepared = prepare_stage_datasets(trainers, raw_datasets, seed=stage.seed)
        schedulers, diagnostics = exact_stage_schedulers(
            trainers,
            update_budget=stage.update_budget,
        )
        session = MitraPersistentStageSession(
            trainers=trainers,
            datasets=prepared,
            schedulers=schedulers,
            diagnostics=diagnostics,
            seed=stage.seed,
        )
        episode_sources = {
            name: MitraDatasetEpisodeSource(
                dataset,
                context_size=int(self.common_config.context_size),
                query_ratio=float(self.common_config.query_ratio),
            )
            for name, dataset in prepared.items()
        }
        baseline_splits = {
            name: (("val", "test") if stage.role == "terminal" and name == self.route.target_dataset else ("val",))
            for name in stage.selector.datasets
        }
        if stage.role == "terminal":
            baseline_splits[self.route.target_dataset] = ("val", "test")
        evaluation_target_data = raw_datasets.get(self.route.target_dataset)
        if stage.role == "terminal" and evaluation_target_data is None:
            evaluation_target_data = self.dataset_loader(
                self.route.target_dataset,
                row_budget=stage.row_budget_per_dataset,
                row_budget_policy=stage.row_budget_policy,
                row_subset_seed=stage.row_subset_seed,
            )
        evaluation_datasets = dict(raw_datasets)
        if evaluation_target_data is not None:
            evaluation_datasets[self.route.target_dataset] = evaluation_target_data
        baselines = {
            name: evaluator.evaluate_baseline(
                name,
                evaluation_datasets[name],
                splits=splits,
                stage=stage,
            )
            for name, splits in baseline_splits.items()
        }

        schedule = build_balanced_episode_schedule(
            stage.datasets,
            updates_per_dataset=stage.update_budget // len(stage.datasets),
            seed=stage.seed,
        )
        completed: Counter[str] = Counter()
        candidates: list[MitraTransferCandidate] = []
        boundary_candidates: dict[int, MitraTransferCandidate] = {}
        train_history: list[dict[str, Any]] = []
        training_seconds = 0.0
        training_time_method = "training_wall_time_proxy_excluding_validation"
        selection_seconds = 0.0
        best_early_stop_score: float | None = None
        stale_validation_rounds = 0
        for global_index, dataset_name in enumerate(schedule):
            dataset_update_index = completed[dataset_name]
            episode_rng = np.random.default_rng(
                np.random.SeedSequence([stage.seed, _dataset_token(dataset_name), dataset_update_index])
            )
            episode = episode_sources[dataset_name].sample_episode(rng=episode_rng)
            step_result, update_seconds, training_time_method = self._run_training_update(
                session,
                episode=episode,
                dataset_name=dataset_name,
                global_update_index=global_index,
                dataset_update_index=dataset_update_index,
            )
            training_seconds += update_seconds
            completed[dataset_name] += 1
            update = global_index + 1
            train_history.append(
                {
                    "train/step": update,
                    "train/loss": step_result["loss"],
                    "train/dataset": dataset_name,
                    "train/member_optimizer_steps": step_result["member_optimizer_steps"],
                }
            )
            if update % len(stage.datasets) != 0:
                continue
            selection_started = time.monotonic()
            candidate = evaluator.validation_candidate(
                estimator,
                stage=stage,
                update=update,
                output_dir=output_dir,
                raw_datasets=raw_datasets,
                baselines=baselines,
            )
            selection_seconds += time.monotonic() - selection_started
            candidates.append(candidate)
            train_history[-1]["train/validation_selection_score"] = candidate.selection.selection_score
            current_best = evaluator.select_candidate(stage, candidates)
            if update in stage.view_boundaries:
                view_path = output_dir / f"view-boundary-{update}.ckpt"
                shutil.copy2(current_best.path, view_path)
                boundary_candidates[update] = MitraTransferCandidate(
                    update=current_best.update,
                    path=view_path,
                    validation=current_best.validation,
                    selection=current_best.selection,
                )
            evaluator.prune_unselected_candidates(
                candidates,
                keep=current_best.path,
            )
            improved = (
                best_early_stop_score is None
                or candidate.selection.selection_score > best_early_stop_score + float(self.common_config.min_delta)
            )
            if improved:
                best_early_stop_score = candidate.selection.selection_score
                stale_validation_rounds = 0
            else:
                stale_validation_rounds += 1
            minimum_updates = max(stage.view_boundaries, default=0)
            if (
                bool(self.common_config.early_stopping)
                and update >= minimum_updates
                and stale_validation_rounds >= int(self.common_config.patience)
            ):
                break

        if not candidates:
            raise RuntimeError("Mitra transfer stage completed without a validation checkpoint candidate.")
        selected = evaluator.select_candidate(stage, candidates)
        view_checkpoints: dict[int, Path] = {}
        view_metrics: dict[str, Any] = {}
        view_selections: dict[str, Any] = {}
        metric_cache: dict[int, Mapping[str, Any]] = {}
        target_data = evaluation_datasets.get(self.route.target_dataset)
        target_baseline = baselines.get(self.route.target_dataset)
        if stage.role == "terminal" and target_baseline is None:
            target_baseline = evaluator.evaluate_baseline(
                self.route.target_dataset,
                target_data,
                splits=("val", "test"),
                stage=stage,
            )
        for boundary in stage.view_boundaries:
            try:
                boundary_selected = boundary_candidates[int(boundary)]
            except KeyError as exc:
                raise RuntimeError(
                    f"Mitra stage stopped before required checkpoint view boundary {boundary}."
                ) from exc
            view_checkpoints[int(boundary)] = boundary_selected.path
            metrics = evaluator.evaluate_selected_checkpoint(
                boundary_selected,
                data=target_data,
                stage=stage,
                cache=metric_cache,
            )
            view_metrics[str(int(boundary))] = {
                "metrics": metrics,
                "baseline_metrics": target_baseline,
            }
            view_selections[str(int(boundary))] = {
                "kind": stage.selector.kind,
                "datasets": list(stage.selector.datasets),
                "max_update": int(boundary),
                "selected_update": boundary_selected.update,
                "selection_score": boundary_selected.selection.selection_score,
            }

        terminal_metrics: Mapping[str, Any] = {}
        score_metric: str | None = None
        if stage.role == "terminal":
            terminal_metrics = evaluator.evaluate_selected_checkpoint(
                selected,
                data=target_data,
                stage=stage,
                cache=metric_cache,
            )
            score_metric = str(target_data.metric_name)
        realized = sum(completed.values())
        ensemble_size = len(trainers)
        expected_member_steps = realized * ensemble_size
        if session.member_optimizer_steps != [realized] * ensemble_size:
            raise RuntimeError(
                "Mitra transfer member-step accounting diverged from the completed logical updates: "
                f"logical={realized}, per_member={session.member_optimizer_steps}."
            )
        report = {
            "allocated_optimizer_updates": stage.update_budget,
            "realized_optimizer_updates": realized,
            "optimizer_update_unit": "logical_dataset_episode_all_ensemble_members",
            "allocated_member_optimizer_steps": stage.update_budget * ensemble_size,
            "realized_member_optimizer_steps": expected_member_steps,
            "member_optimizer_steps": {
                str(index): count for index, count in enumerate(session.member_optimizer_steps)
            },
            "dataset_optimizer_updates": {name: int(completed[name]) for name in stage.datasets},
            "dataset_member_optimizer_steps": {name: int(completed[name]) * ensemble_size for name in stage.datasets},
            "checkpoint_selection": {
                "kind": stage.selector.kind,
                "datasets": list(stage.selector.datasets),
                "selected_update": selected.update,
                "selection_score": selected.selection.selection_score,
                "dataset_improvements": dict(selected.selection.dataset_improvements),
                "validation_cadence": "complete_balanced_round",
            },
            "checkpoint_view_metrics": view_metrics,
            "checkpoint_view_selection": view_selections,
            "realized_training_examples": session.logical_examples,
            "realized_member_training_examples": session.member_examples,
            "gpu_time_seconds": training_seconds,
            "step_dependent_time_seconds": training_seconds + selection_seconds,
            "step_dependent_time_method": "training_plus_validation_selection",
            "peak_gpu_memory_gib": peak_gpu_memory_gib(self.device),
            "gpu_time_method": training_time_method,
            "validation_time_seconds": selection_seconds,
            "validation_rounds": len(candidates),
            "early_stopped": realized < stage.update_budget,
            "ensemble_size": ensemble_size,
            "score_metric": score_metric,
            "score_metric_source": "foundation_model.metric",
            "metrics": dict(terminal_metrics),
            "zero-shot-metrics": dict(target_baseline or {}),
            "train_history": train_history,
            "fit_events": [
                {
                    "stage": stage.name,
                    "logical_updates": realized,
                    "member_optimizer_steps": expected_member_steps,
                    "seconds": time.monotonic() - started_at,
                }
            ],
            "runtime_phases": [
                {"phase": "mitra_transfer_train", "seconds": training_seconds},
                {"phase": "mitra_transfer_validation_selection", "seconds": selection_seconds},
            ],
        }
        from experiments.transfer.runner.data import transfer_package_versions

        return StageProduction(
            checkpoint_path=selected.path,
            report=report,
            view_checkpoints=view_checkpoints,
            package_versions=transfer_package_versions("mitra"),
            reload_validator=evaluator.reload_validator(
                data=(target_data if stage.role == "terminal" else raw_datasets[stage.datasets[0]]),
                ensemble_size=ensemble_size,
                seed=stage.seed,
            ),
        )

    def _run_training_update(
        self,
        session: MitraPersistentStageSession,
        *,
        episode: Any,
        dataset_name: str,
        global_update_index: int,
        dataset_update_index: int,
    ) -> tuple[Mapping[str, Any], float, str]:
        trainer_device = str(getattr(session.trainers[0], "device", self.device) or "")
        if trainer_device.startswith("cuda"):
            import torch

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = session.train_episode(
                episode,
                dataset_name=dataset_name,
                global_update_index=global_update_index,
                dataset_update_index=dataset_update_index,
            )
            end.record()
            torch.cuda.synchronize(trainer_device)
            return result, float(start.elapsed_time(end)) / 1000.0, "cuda_events_per_logical_update"
        started = time.monotonic()
        result = session.train_episode(
            episode,
            dataset_name=dataset_name,
            global_update_index=global_update_index,
            dataset_update_index=dataset_update_index,
        )
        return result, time.monotonic() - started, "training_wall_time_proxy_excluding_validation"

def _dataset_token(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "big")


__all__ = [
    "MitraDatasetTransferStageTrainer",
]
