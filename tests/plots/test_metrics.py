from __future__ import annotations

import math

import pytest

from plots.metrics import (
    best_relative_normalized_score,
    fine_tuning_gain,
    split_raw_metric_value,
)


def test_raw_metric_values_keep_conventional_orientation() -> None:
    """Raw quantities must expose positive errors instead of negated scores."""
    assert split_raw_metric_value(
        {"score": -0.5, "log_loss": 0.5},
        "primary_score",
        score_metric="log_loss",
    ) == pytest.approx(0.5)
    assert split_raw_metric_value(
        {"score": -0.5},
        "primary_score",
        score_metric="log_loss",
    ) == pytest.approx(0.5)
    assert split_raw_metric_value({"rmse": 2.5}, "rmse") == pytest.approx(2.5)


def test_fine_tuning_gain_matches_thesis_metric_semantics() -> None:
    """Bounded metrics use differences and positive errors use log ratios."""
    assert fine_tuning_gain(0.6, 0.72, "roc_auc") == pytest.approx(0.12)
    assert fine_tuning_gain(0.6, 0.65, "accuracy") == pytest.approx(0.05)
    assert fine_tuning_gain(0.6, 0.5, "log_loss") == pytest.approx(math.log(1.2))
    assert fine_tuning_gain(3.0, 2.0, "rmse") == pytest.approx(math.log(1.5))
    assert fine_tuning_gain(0.0, 0.5, "rmse") is None


def test_best_relative_normalization_matches_thesis_formula() -> None:
    """Normalization has one best-relative definition and no min-max fallback."""
    assert best_relative_normalized_score(0.8, 0.9, "roc_auc") == pytest.approx(0.8 / 0.9)
    assert best_relative_normalized_score(3.0, 2.0, "rmse") == pytest.approx(2.0 / 3.0)
    assert best_relative_normalized_score(-0.2, 0.1, "r2") is None
