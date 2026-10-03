"""Validation checkpoint selectors for dataset-transfer stages."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from types import MappingProxyType
from typing import Literal

CheckpointSelector = Literal["target_only", "source_mean_improvement"]


@dataclass(frozen=True)
class ValidationMetric:
    """One raw validation value and its matched pure-ICL baseline."""

    value: float
    baseline: float
    higher_is_better: bool

    @property
    def improvement(self) -> float:
        """Return improvement in a common higher-is-better orientation."""
        value = _finite_float(self.value, field="value")
        baseline = _finite_float(self.baseline, field="baseline")
        if not isinstance(self.higher_is_better, bool):
            raise TypeError("higher_is_better must be a boolean.")
        direction = 1.0 if self.higher_is_better else -1.0
        return direction * (value - baseline)


@dataclass(frozen=True)
class CheckpointValidation:
    """Per-dataset validation evidence for one checkpoint candidate."""

    checkpoint_id: str
    step: int
    metrics: Mapping[str, ValidationMetric]


@dataclass(frozen=True)
class CheckpointSelection:
    """Selected checkpoint and the exact dataset-level effects used."""

    checkpoint_id: str
    step: int
    selector: CheckpointSelector
    selection_score: float
    dataset_improvements: Mapping[str, float]


def select_checkpoint(
    candidates: Sequence[CheckpointValidation],
    *,
    selector: CheckpointSelector,
    target_dataset: str | None = None,
    source_datasets: Sequence[str] = (),
) -> CheckpointSelection:
    """Select a checkpoint with target-only or equal-source-mean validation.

    ``target_only`` uses only the target dataset's orientation-corrected
    improvement. ``source_mean_improvement`` requires every declared source and
    computes their unweighted arithmetic mean; raw metrics are never averaged.
    Ties choose the earliest step and then the lexical checkpoint ID.
    """
    rows = tuple(candidates)
    if not rows:
        raise ValueError("candidates must contain at least one checkpoint.")
    _validate_candidates(rows)

    if selector == "target_only":
        datasets = (_dataset_name(target_dataset, field="target_dataset"),)
    elif selector == "source_mean_improvement":
        datasets = _dataset_names(source_datasets, field="source_datasets")
    else:
        raise ValueError(f"Unknown checkpoint selector: {selector!r}.")
    _validate_matched_baselines(rows, datasets=datasets)

    scored: list[tuple[float, int, str, dict[str, float]]] = []
    for candidate in rows:
        improvements = {dataset: _metric_for(candidate, dataset).improvement for dataset in datasets}
        score = improvements[datasets[0]] if selector == "target_only" else fmean(improvements.values())
        scored.append((score, candidate.step, candidate.checkpoint_id, improvements))

    score, step, checkpoint_id, improvements = min(
        scored,
        key=lambda item: (-item[0], item[1], item[2]),
    )
    return CheckpointSelection(
        checkpoint_id=checkpoint_id,
        step=step,
        selector=selector,
        selection_score=score,
        dataset_improvements=MappingProxyType(improvements),
    )


def select_target_checkpoint(
    candidates: Sequence[CheckpointValidation],
    *,
    target_dataset: str,
) -> CheckpointSelection:
    """Select using target validation only."""
    return select_checkpoint(candidates, selector="target_only", target_dataset=target_dataset)


def select_source_mean_checkpoint(
    candidates: Sequence[CheckpointValidation],
    *,
    source_datasets: Sequence[str],
) -> CheckpointSelection:
    """Select using the equal-weight mean of source validation improvements."""
    return select_checkpoint(
        candidates,
        selector="source_mean_improvement",
        source_datasets=source_datasets,
    )


def _validate_candidates(candidates: Sequence[CheckpointValidation]) -> None:
    ids: list[str] = []
    for candidate in candidates:
        checkpoint_id = _dataset_name(candidate.checkpoint_id, field="checkpoint_id")
        ids.append(checkpoint_id)
        if isinstance(candidate.step, bool) or int(candidate.step) != candidate.step or candidate.step < 0:
            raise ValueError(f"checkpoint step must be a non-negative integer, got {candidate.step!r}.")
    if len(set(ids)) != len(ids):
        raise ValueError("checkpoint IDs must be unique.")


def _validate_matched_baselines(
    candidates: Sequence[CheckpointValidation],
    *,
    datasets: Sequence[str],
) -> None:
    for dataset in datasets:
        baseline = _metric_for(candidates[0], dataset).baseline
        for candidate in candidates[1:]:
            other = _metric_for(candidate, dataset).baseline
            if not math.isclose(float(baseline), float(other), rel_tol=0.0, abs_tol=0.0):
                raise ValueError(f"Dataset {dataset!r} has unmatched pure-ICL baselines across checkpoint candidates.")


def _metric_for(candidate: CheckpointValidation, dataset: str) -> ValidationMetric:
    metric = candidate.metrics.get(dataset)
    if metric is None:
        raise ValueError(f"Checkpoint {candidate.checkpoint_id!r} lacks validation for dataset {dataset!r}.")
    if not isinstance(metric, ValidationMetric):
        raise TypeError(f"Checkpoint {candidate.checkpoint_id!r} validation for {dataset!r} must be ValidationMetric.")
    _ = metric.improvement
    return metric


def _dataset_names(values: Sequence[str], *, field: str) -> tuple[str, ...]:
    names = tuple(_dataset_name(value, field=field) for value in values)
    if not names:
        raise ValueError(f"{field} must contain at least one dataset.")
    if len(set(names)) != len(names):
        raise ValueError(f"{field} must not contain duplicate datasets.")
    return names


def _dataset_name(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be a non-empty string.")
    return text


def _finite_float(value: object, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be a finite number, got {value!r}.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be a finite number, got {value!r}.")
    return parsed


__all__ = [
    "CheckpointSelection",
    "CheckpointSelector",
    "CheckpointValidation",
    "ValidationMetric",
    "select_checkpoint",
    "select_source_mean_checkpoint",
    "select_target_checkpoint",
]
