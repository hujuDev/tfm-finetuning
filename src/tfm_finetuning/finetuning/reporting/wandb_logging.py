"""Canonical W&B keys shared by training and completed reports.

Example: ``flatten_score_metrics_for_wandb(metrics, prefix="baseline/")``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

CORE_METRICS = ("roc_auc", "accuracy", "log_loss", "rmse", "mse", "mae", "r2")
METRIC_NAMES = {
    "roc_auc_score": "roc_auc",
    "r2_score": "r2",
    "root_mean_squared_error": "rmse",
    "mean_squared_error": "mse",
    "mean_absolute_error": "mae",
}


def _finite(value: Any) -> float | None:
    if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def flatten_score_metrics_for_wandb(
    metrics: Mapping[str, Any],
    *,
    prefix: str,
    splits: tuple[str, ...] = ("val", "test"),
) -> dict[str, float]:
    """Expose only the higher-is-better selection score under each split's score key."""
    result = {}
    for split in splits:
        value = _finite(metrics.get(split, {}).get("score"))
        if value is not None:
            result[f"{prefix}{split}/score"] = value
    return result


def flatten_core_metrics_for_wandb(
    metrics: Mapping[str, Any],
    *,
    prefix: str,
    splits: tuple[str, ...] = ("val", "test"),
) -> dict[str, float]:
    """Expose conventional raw metrics, including MSE derived from RMSE when needed."""
    result = {}
    for split in splits:
        values = {METRIC_NAMES.get(name, name): _finite(value) for name, value in metrics.get(split, {}).items()}
        if values.get("rmse") is not None and values.get("mse") is None:
            values["mse"] = values["rmse"] ** 2
        result.update(
            {f"{prefix}{split}/{name}": values[name] for name in CORE_METRICS if values.get(name) is not None}
        )
    return result


def compute_score_deltas_for_wandb(
    *,
    baseline_metrics,
    final_metrics,
    prefix="delta/",
    splits=("val", "test"),
) -> dict[str, float]:
    """Subtract paired baseline scores from post-training scores."""
    baseline = flatten_score_metrics_for_wandb(baseline_metrics, prefix=prefix, splits=splits)
    final = flatten_score_metrics_for_wandb(final_metrics, prefix=prefix, splits=splits)
    return {key: value - baseline[key] for key, value in final.items() if key in baseline}


def compute_core_metric_deltas_for_wandb(
    *,
    baseline_metrics,
    final_metrics,
    prefix="delta/core/",
    splits=("val", "test"),
) -> dict[str, float]:
    """Subtract paired raw metrics; loss/error deltas retain their ordinary sign."""
    baseline = flatten_core_metrics_for_wandb(baseline_metrics, prefix=prefix, splits=splits)
    final = flatten_core_metrics_for_wandb(final_metrics, prefix=prefix, splits=splits)
    return {key: value - baseline[key] for key, value in final.items() if key in baseline}


def extract_class_report_details_for_wandb(metrics, *, phase, splits=("val", "test")) -> dict[str, float]:
    """Flatten per-class and aggregate precision, recall, F1, and support."""
    result = {}
    for split in splits:
        for label, payload in metrics.get(split, {}).get("classification_report", {}).items():
            if not isinstance(payload, Mapping):
                continue
            label = str(label).replace(" ", "_")
            if not (label.isdigit() or label in {"macro_avg", "weighted_avg"}):
                continue
            for name in ("precision", "recall", "f1-score", "support"):
                value = _finite(payload.get(name))
                if value is not None:
                    result[f"details/class_report/{phase}/{split}/{label}/{name}"] = value
    return result


def experiment_parameters(config: Mapping[str, Any]) -> dict[str, Any]:
    """Expose the runner's named sweep parameters for W&B grouping/filtering."""
    parameters = config.get("experiment", {}).get("key_param", {})
    return {f"experiment.key_param.{key}": value for key, value in parameters.items()}


def flatten_report_metrics_for_wandb(report: Mapping[str, Any]) -> dict[str, Any]:
    """Log baseline, selected checkpoint and last checkpoint distinctly."""
    result = {}
    phases = {
        "baseline": "zero_shot_metrics",
        "selected": "selected_checkpoint_metrics",
        "final": "final_checkpoint_metrics",
    }
    for phase, key in phases.items():
        metrics = report.get(key, {})
        result.update(flatten_score_metrics_for_wandb(metrics, prefix=f"{phase}/"))
        result.update(flatten_core_metrics_for_wandb(metrics, prefix=f"core/{phase}/"))
        result.update(extract_class_report_details_for_wandb(metrics, phase=phase))
    result.update(
        compute_score_deltas_for_wandb(
            baseline_metrics=report.get("zero_shot_metrics", {}),
            final_metrics=report.get("final_checkpoint_metrics", {}),
        )
    )
    result.update(
        compute_core_metric_deltas_for_wandb(
            baseline_metrics=report.get("zero_shot_metrics", {}),
            final_metrics=report.get("final_checkpoint_metrics", {}),
        )
    )
    for key in ("test_context_comparison", "distribution_shift_test_views"):
        result.update(flatten_test_context_comparison_for_wandb(report.get(key, {}), prefix=f"details/{key}/"))
    return result


def flatten_test_context_comparison_for_wandb(comparison_payload, *, prefix="details/test_context_comparison/"):
    """Keep transformed and original test views of the same checkpoint separately queryable."""
    result = {}
    for phase, key in {
        "baseline": "baseline_metrics",
        "selected": "selected_checkpoint_metrics",
        "final": "final_checkpoint_metrics",
    }.items():
        for policy, metrics in comparison_payload.get(key, {}).items():
            result.update(flatten_score_metrics_for_wandb(metrics, prefix=f"{prefix}{phase}/{policy}/"))
            result.update(flatten_core_metrics_for_wandb(metrics, prefix=f"{prefix}core/{phase}/{policy}/"))
    return result
