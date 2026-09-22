"""Validation selection, checkpoint evaluation, and reload audits for Mitra."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.dataset_transfer.selection import (
    CheckpointSelection,
    CheckpointValidation,
    ValidationMetric,
    select_source_mean_checkpoint,
    select_target_checkpoint,
)
from experiments.dataset_transfer.spec import (
    TransferRouteSpec,
    TransferStageSpec,
)
from finetuning.evaluation import (
    _evaluate_checkpoint,
    _evaluate_zero_shot,
)
from finetuning.models.mitra_checkpoints import (
    write_checkpoint_package,
)
from finetuning.models.mitra_transfer_session import (
    isolated_random_seed,
)


@dataclass(frozen=True)
class MitraTransferCandidate:
    """One portable checkpoint and its validation selection record."""

    update: int
    path: Path
    validation: CheckpointValidation
    selection: CheckpointSelection


class MitraTransferEvaluator:
    """Own target-only evaluation and validation selection for one route."""

    def __init__(
        self,
        *,
        adapter: Any,
        route: TransferRouteSpec,
        common_config: Any,
        device: str | None,
        dataset_loader: Any,
    ) -> None:
        self.adapter = adapter
        self.route = route
        self.common_config = common_config
        self.device = device
        self.dataset_loader = dataset_loader

    def validation_candidate(
        self,
        estimator: Any,
        *,
        stage: TransferStageSpec,
        update: int,
        output_dir: Path,
        raw_datasets: Mapping[str, Any],
        baselines: Mapping[str, Mapping[str, Any]],
    ) -> MitraTransferCandidate:
        """Persist and score one complete balanced-round candidate."""
        path = output_dir / f"candidate-update-{int(update)}.ckpt"
        write_checkpoint_package(estimator, path)
        validation_metrics: dict[str, ValidationMetric] = {}
        for name in stage.selector.datasets:
            data = raw_datasets.get(name) or self.dataset_loader(name)
            metrics = self.evaluate_checkpoint(
                path,
                data,
                splits=("val",),
                inference_seed=stage.seed,
                isolation_seed=evaluation_seed(stage.seed, name),
            )
            validation_metrics[name] = ValidationMetric(
                value=_score(metrics, split="val", dataset=name),
                baseline=_score(
                    baselines[name],
                    split="val",
                    dataset=name,
                ),
                higher_is_better=True,
            )
        validation = CheckpointValidation(
            checkpoint_id=path.name,
            step=int(update),
            metrics=validation_metrics,
        )
        selection = self.select_validations(stage, (validation,))
        return MitraTransferCandidate(
            update=int(update),
            path=path,
            validation=validation,
            selection=selection,
        )

    def select_candidate(
        self,
        stage: TransferStageSpec,
        candidates: list[MitraTransferCandidate],
    ) -> MitraTransferCandidate:
        """Select the declared target or equal-source-mean winner."""
        selection = self.select_validations(
            stage,
            tuple(candidate.validation for candidate in candidates),
        )
        return next(
            candidate
            for candidate in candidates
            if candidate.validation.checkpoint_id
            == selection.checkpoint_id
        )

    @staticmethod
    def select_validations(
        stage: TransferStageSpec,
        candidates: tuple[CheckpointValidation, ...],
    ) -> CheckpointSelection:
        """Apply the immutable stage selector."""
        if stage.selector.kind == "target_validation":
            return select_target_checkpoint(
                candidates,
                target_dataset=stage.selector.datasets[0],
            )
        if (
            stage.selector.kind
            == "equal_mean_validation_improvement"
        ):
            return select_source_mean_checkpoint(
                candidates,
                source_datasets=stage.selector.datasets,
            )
        raise ValueError(
            f"Unsupported Mitra transfer selector: "
            f"{stage.selector.kind!r}."
        )

    def evaluate_baseline(
        self,
        dataset_name: str,
        data: Any,
        *,
        splits: tuple[str, ...],
        stage: TransferStageSpec,
    ) -> Mapping[str, Any]:
        """Evaluate pure ICL on the same target-only context."""
        with isolated_random_seed(
            evaluation_seed(stage.seed, dataset_name)
        ):
            return _evaluate_zero_shot(
                adapter=self.adapter,
                data=data,
                seed=stage.seed,
                device=self.device,
                common_config=self.common_config,
                phase="mitra_transfer_base_icl",
                eval_source="base_pure_icl",
                splits=splits,
            )

    def evaluate_checkpoint(
        self,
        path: Path,
        data: Any,
        *,
        splits: tuple[str, ...],
        inference_seed: int,
        isolation_seed: int,
    ) -> Mapping[str, Any]:
        """Evaluate portable weights under an isolated random state."""
        with isolated_random_seed(isolation_seed):
            return _evaluate_checkpoint(
                adapter=self.adapter,
                checkpoint_path=path,
                data=data,
                seed=int(inference_seed),
                device=self.device,
                common_config=self.common_config,
                phase="mitra_transfer_checkpoint_selection",
                eval_source="portable_checkpoint",
                splits=splits,
            )

    def evaluate_selected_checkpoint(
        self,
        selected: MitraTransferCandidate,
        *,
        data: Any,
        stage: TransferStageSpec,
        cache: dict[int, Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        """Evaluate one selected checkpoint once on target val/test."""
        if selected.update not in cache:
            cache[selected.update] = self.evaluate_checkpoint(
                selected.path,
                data,
                splits=("val", "test"),
                inference_seed=stage.seed,
                isolation_seed=evaluation_seed(
                    stage.seed,
                    self.route.target_dataset,
                ),
            )
        return cache[selected.update]

    def reload_validator(
        self,
        *,
        data: Any,
        ensemble_size: int,
        seed: int,
    ) -> Callable[[Path], bool]:
        """Return the durable checkpoint reload-and-predict audit."""

        def validate(path: Path) -> bool:
            with isolated_random_seed(
                evaluation_seed(seed, "reload-validation")
            ):
                estimator = self.adapter.load_checkpoint_for_inference(
                    path,
                    data.task_type,
                    self.device,
                    ensemble_size,
                    seed=int(seed),
                )
                if (
                    len(getattr(estimator, "trainers", ()))
                    != ensemble_size
                ):
                    return False
                if hasattr(estimator, "classes_") or hasattr(
                    estimator,
                    "_foundation_mitra_feature_preprocessor",
                ):
                    raise RuntimeError(
                        "Mitra portable checkpoint retained source-bound "
                        "state before target fitting."
                    )
                context_X = getattr(
                    data,
                    "val_context_X",
                    data.train_X,
                )
                context_y = getattr(
                    data,
                    "val_context_y",
                    data.train_y,
                )
                query_X = getattr(
                    data,
                    "val_X",
                    data.train_X,
                ).iloc[:1]
                estimator = self.adapter.fit_inference_context(
                    estimator,
                    context_X,
                    context_y,
                )
                prediction = np.asarray(
                    self.adapter.predict_for_metrics(
                        estimator,
                        query_X,
                        data.task_type,
                    )
                )
                if (
                    prediction.shape[0] != 1
                    or not np.isfinite(prediction).all()
                ):
                    raise RuntimeError(
                        "Mitra reloaded checkpoint failed target-context "
                        "prediction smoke."
                    )
                return True

        return validate

    @staticmethod
    def prune_unselected_candidates(
        candidates: list[MitraTransferCandidate],
        *,
        keep: Path,
    ) -> None:
        """Delete old losers because they cannot become a later winner."""
        for candidate in candidates:
            if candidate.path != keep and candidate.path.exists():
                candidate.path.unlink()


def evaluation_seed(seed: int, dataset_name: str) -> int:
    """Derive deterministic isolated inference randomness per dataset."""
    token = int.from_bytes(
        hashlib.sha256(dataset_name.encode("utf-8")).digest()[:4],
        "big",
    )
    return int(
        np.random.SeedSequence(
            [int(seed), token, 0xE1A1]
        ).generate_state(1)[0]
    )


def _score(
    metrics: Mapping[str, Any],
    *,
    split: str,
    dataset: str,
) -> float:
    split_payload = metrics.get(split)
    value = (
        split_payload.get("score")
        if isinstance(split_payload, Mapping)
        else None
    )
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not np.isfinite(value)
    ):
        raise RuntimeError(
            f"Mitra transfer evaluation for {dataset!r} lacks finite "
            f"{split}/score."
        )
    return float(value)
