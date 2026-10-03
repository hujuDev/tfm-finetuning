"""Reusable statistics for overview dataset-pattern analysis.

Example:
    >>> bootstrap_mean_ci([0.0, 1.0], seed=0, n_resamples=100)[0]
    0.5
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_spearman_ci(
    x: Iterable[float],
    y: Iterable[float],
    *,
    seed: int,
    n_resamples: int = 5_000,
) -> tuple[float, float]:
    """Return a paired percentile-bootstrap 95% CI for Spearman's rho."""
    x_values = np.asarray(list(x), dtype=float)
    y_values = np.asarray(list(y), dtype=float)
    if x_values.shape != y_values.shape:
        raise ValueError("Paired bootstrap inputs must have the same shape.")
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[finite]
    y_values = y_values[finite]
    if x_values.size < 2 or n_resamples < 1:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, x_values.size, size=(n_resamples, x_values.size))
    x_ranks = stats.rankdata(x_values[indices], axis=1)
    y_ranks = stats.rankdata(y_values[indices], axis=1)
    x_centered = x_ranks - x_ranks.mean(axis=1, keepdims=True)
    y_centered = y_ranks - y_ranks.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(x_centered**2, axis=1) * np.sum(y_centered**2, axis=1))
    valid = denominator > 0
    if not np.any(valid):
        return np.nan, np.nan
    correlations = np.sum(x_centered[valid] * y_centered[valid], axis=1) / denominator[valid]
    low, high = np.percentile(correlations, [2.5, 97.5])
    return float(low), float(high)


def benjamini_hochberg(p_values: Iterable[float]) -> list[float]:
    """Return Benjamini--Hochberg adjusted p-values in input order."""
    values = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(values.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return adjusted.tolist()
    ordered = finite[np.argsort(values[finite])]
    ranked = values[ordered] * finite.size / np.arange(1, finite.size + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[ordered] = np.minimum(ranked, 1.0)
    return adjusted.tolist()


def holm_bonferroni(p_values: Iterable[float]) -> list[float]:
    """Return standard Holm--Bonferroni adjusted p-values in input order."""
    values = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(values.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return adjusted.tolist()
    ordered = finite[np.argsort(values[finite])]
    scaled = values[ordered] * (finite.size - np.arange(finite.size))
    adjusted[ordered] = np.minimum(np.maximum.accumulate(scaled), 1.0)
    return adjusted.tolist()


def fine_tuning_outcome_table(frame: pd.DataFrame, *, alpha: float = 0.05) -> pd.DataFrame:
    """Classify replicated matched gains using study-wide BH-adjusted t tests.

    ``effect`` and ``effect_se`` are the mean and standard error of the
    per-run matched gain. Rows with fewer than two run units remain explicitly
    untestable instead of being labelled as no change.
    """
    result = frame.copy()
    result["p_value"] = [
        _one_sample_t_p_value(mean, standard_error, run_count)
        for mean, standard_error, run_count in zip(
            result["effect"], result["effect_se"], result["run_count"], strict=True
        )
    ]
    result["q_value_bh"] = benjamini_hochberg(result["p_value"])
    detectable = result["q_value_bh"] < alpha
    result["outcome"] = np.select(
        [result["p_value"].isna(), detectable & (result["effect"] > 0), detectable & (result["effect"] < 0)],
        ["insufficient_replication", "help", "harm"],
        default="not_detected",
    )
    return result


def association_table(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    min_datasets: int = 5,
    bootstrap_seed: int = 0,
    bootstrap_resamples: int = 5_000,
) -> pd.DataFrame:
    """Compute associations, bootstrap CIs, LODO checks, BH, and Holm values."""
    records: list[dict[str, Any]] = []
    association_index = 0
    for (model, preprocessing, task_family), group in frame.groupby(
        ["model", "preprocessing", "task_family"], sort=True
    ):
        for feature in feature_columns:
            clean = group[["dataset", "effect", feature]].dropna()
            clean = clean[np.isfinite(clean["effect"]) & np.isfinite(clean[feature])]
            if len(clean) < min_datasets or clean[feature].nunique() < 3:
                continue
            rho, p_value = stats.spearmanr(clean[feature], clean["effect"])
            slope, intercept, slope_low, slope_high = stats.theilslopes(clean["effect"], clean[feature], alpha=0.95)
            lodo = _lodo_spearman(clean[feature].to_numpy(), clean["effect"].to_numpy())
            bootstrap_low, bootstrap_high = bootstrap_spearman_ci(
                clean[feature],
                clean["effect"],
                seed=bootstrap_seed + association_index,
                n_resamples=bootstrap_resamples,
            )
            association_index += 1
            records.append(
                {
                    "model": model,
                    "preprocessing": preprocessing,
                    "task_family": task_family,
                    "feature": feature,
                    "n_datasets": len(clean),
                    "spearman_rho": float(rho),
                    "p_value": float(p_value),
                    "theil_sen_slope": float(slope),
                    "theil_sen_intercept": float(intercept),
                    "slope_ci_low": float(slope_low),
                    "slope_ci_high": float(slope_high),
                    "bootstrap_ci_low": bootstrap_low,
                    "bootstrap_ci_high": bootstrap_high,
                    **lodo,
                }
            )
    result = pd.DataFrame(records)
    if not result.empty:
        result["q_value_bh"] = benjamini_hochberg(result["p_value"])
        result["p_value_holm"] = holm_bonferroni(result["p_value"])
    return result


def structural_hypothesis_association_table(
    frame: pd.DataFrame,
    *,
    hypotheses: list[dict[str, Any]],
    min_datasets: int = 5,
    bootstrap_seed: int = 0,
    bootstrap_resamples: int = 5_000,
) -> pd.DataFrame:
    """Test only predeclared structural associations and attach their rationale.

    Multiplicity corrections cover every estimable model-task-feature test in
    this compact family. P-values remain two-sided even though each hypothesis
    declares an expected direction, keeping the confirmatory gate conservative.
    """
    if not hypotheses:
        return pd.DataFrame()
    features = list(dict.fromkeys(str(item["feature"]) for item in hypotheses))
    result = association_table(
        frame,
        feature_columns=features,
        min_datasets=min_datasets,
        bootstrap_seed=bootstrap_seed,
        bootstrap_resamples=bootstrap_resamples,
    )
    if result.empty:
        return result
    declarations = pd.DataFrame(
        [
            {
                "feature": str(item["feature"]),
                "family": str(item["family"]),
                "expected_direction": str(item["expected_direction"]),
                "task_family": str(task),
                "hypothesis": str(item["hypothesis"]),
            }
            for item in hypotheses
            for task in item["tasks"]
        ]
    )
    result = result.merge(declarations, on=["feature", "task_family"], how="inner", validate="many_to_one")
    expected_sign = result["expected_direction"].map({"positive": 1.0, "negative": -1.0})
    result["direction_matches"] = np.sign(result["spearman_rho"]) == expected_sign
    result["bootstrap_excludes_zero"] = (result["bootstrap_ci_low"] > 0) | (result["bootstrap_ci_high"] < 0)
    return result


def _one_sample_t_p_value(mean: object, standard_error: object, run_count: object) -> float:
    """Return a two-sided one-sample t-test p-value from summary statistics."""
    try:
        mean_value = float(mean)
        standard_error_value = float(standard_error)
        count = int(run_count)
    except (TypeError, ValueError):
        return np.nan
    if count < 2 or not np.isfinite(mean_value) or not np.isfinite(standard_error_value):
        return np.nan
    if standard_error_value < 0:
        return np.nan
    if standard_error_value == 0:
        return 1.0 if mean_value == 0 else 0.0
    statistic = abs(mean_value / standard_error_value)
    return float(2.0 * stats.t.sf(statistic, df=count - 1))


def _lodo_spearman(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    values = [float(stats.spearmanr(np.delete(x, i), np.delete(y, i)).statistic) for i in range(len(x))]
    full = float(stats.spearmanr(x, y).statistic)
    full_sign = np.sign(full)
    stable = np.mean([np.sign(value) == full_sign for value in values]) if full_sign else np.nan
    return {
        "lodo_rho_min": float(np.nanmin(values)),
        "lodo_rho_max": float(np.nanmax(values)),
        "lodo_sign_stability": float(stable),
    }


__all__ = [
    "association_table",
    "benjamini_hochberg",
    "bootstrap_spearman_ci",
    "structural_hypothesis_association_table",
]
