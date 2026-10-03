"""Raw metrics, paired gains, and normalized scores used by final analyses.

Raw values retain conventional metric orientation. Normalized scores compare a
raw value with the best raw value for the same dataset/preprocessing path.
Fine-tuning gains compare an adapted value with its matched ICL baseline.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

PRIMARY_METRIC = "primary_score"
LOWER_IS_BETTER_METRICS = frozenset({"log_loss", "cross_entropy", "rmse", "mse", "mae"})


def finite_number(value: object) -> float | None:
    """Return a finite float or ``None``."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def normalize_metric_key(value: object) -> str:
    """Normalize report metric aliases."""
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return {"cross_entropy": "log_loss", "rocauc": "roc_auc", "score": PRIMARY_METRIC}.get(text, text)


def metric_is_lower_better(metric: object) -> bool:
    """Return whether the conventional raw metric is minimized."""
    return normalize_metric_key(metric) in LOWER_IS_BETTER_METRICS


def raw_metric_from_score(value: object, metric: object) -> float | None:
    """Convert a higher-is-better report score to its conventional raw metric."""
    numeric = finite_number(value)
    if numeric is None:
        return None
    return -numeric if metric_is_lower_better(metric) else numeric


def split_raw_metric_value(
    metrics: Mapping[str, object],
    metric: object,
    *,
    score_metric: object = "",
) -> float | None:
    """Read a conventional raw metric from one report split mapping."""
    normalized = normalize_metric_key(metric)
    if normalized == PRIMARY_METRIC:
        concrete_metric = normalize_metric_key(score_metric)
        if concrete_metric and concrete_metric != PRIMARY_METRIC:
            explicit = _matching_metric_value(metrics, concrete_metric)
            if explicit is not None:
                return explicit
        score = _first_finite(metrics.get("score"), metrics.get(PRIMARY_METRIC))
        return raw_metric_from_score(score, concrete_metric) if score is not None else None
    return _matching_metric_value(metrics, normalized)


def fine_tuning_gain(baseline: object, adapted: object, metric: object) -> float | None:
    """Return the thesis-defined matched SFT gain over ICL."""
    baseline_value = finite_number(baseline)
    adapted_value = finite_number(adapted)
    if baseline_value is None or adapted_value is None:
        return None
    if metric_is_lower_better(metric):
        if baseline_value <= 0 or adapted_value <= 0:
            return None
        return finite_number(math.log(baseline_value / adapted_value))
    return finite_number(adapted_value - baseline_value)


def best_relative_normalized_score(value: object, best: object, metric: object) -> float | None:
    """Return the thesis-defined best-relative normalized score."""
    raw_value = finite_number(value)
    best_value = finite_number(best)
    if raw_value is None or best_value is None:
        return None
    if metric_is_lower_better(metric):
        if raw_value <= 0 or best_value <= 0:
            return None
        return finite_number(best_value / raw_value)
    if raw_value < 0 or best_value <= 0:
        return None
    return finite_number(raw_value / best_value)


def gain_quantity_key(metric: object) -> str:
    """Return the stable quantity token used in result provenance."""
    return "log_error_ratio" if metric_is_lower_better(metric) else "metric_difference"


def gain_label(value: object) -> str:
    """Render a compact signed gain without implying percentage units."""
    numeric = finite_number(value)
    if numeric is None:
        return "--"
    return f"{numeric:+.4g}"


def _matching_metric_value(metrics: Mapping[str, object], metric: str) -> float | None:
    for raw_key, raw_value in metrics.items():
        if normalize_metric_key(raw_key) == metric:
            return finite_number(raw_value)
    return None


def _first_finite(*values: object) -> float | None:
    for value in values:
        numeric = finite_number(value)
        if numeric is not None:
            return numeric
    return None


__all__ = [
    "LOWER_IS_BETTER_METRICS",
    "PRIMARY_METRIC",
    "best_relative_normalized_score",
    "fine_tuning_gain",
    "finite_number",
    "gain_label",
    "gain_quantity_key",
    "metric_is_lower_better",
    "normalize_metric_key",
    "raw_metric_from_score",
    "split_raw_metric_value",
]
