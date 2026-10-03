from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfm_finetuning.data.dataset_meta_features import (
    compute_base_meta_features,
    compute_distribution_structure_features,
    compute_label_entropy_normalized,
    compute_local_target_structure_features,
    compute_target_p99_over_median,
    compute_target_robust_tail_index,
    compute_univariate_signal_top5_share,
)


def test_balanced_binary_labels_have_normalized_entropy_near_one() -> None:
    """Balanced binary labels should hit the maximum normalized entropy."""
    assert compute_label_entropy_normalized([0, 0, 1, 1]) == pytest.approx(1.0)


def test_imbalanced_binary_labels_have_lower_normalized_entropy() -> None:
    """Class imbalance should lower normalized entropy below the balanced case."""
    entropy = compute_label_entropy_normalized([0] * 95 + [1] * 5)

    assert entropy is not None
    assert 0.0 < entropy < 0.5


def test_regression_target_positive_median_computes_tail_ratio() -> None:
    """Regression target p99/median should use finite positive-median targets."""
    values = np.asarray([1.0, 2.0, 3.0, 4.0])

    assert compute_target_p99_over_median(values) == pytest.approx(float(np.percentile(values, 99)) / 2.5)


def test_regression_target_nonpositive_median_returns_none() -> None:
    """Regression target-tail severity is undefined for nonpositive medians."""
    assert compute_target_p99_over_median([-4.0, -1.0, 0.0, 10.0]) is None


def test_robust_target_tail_index_handles_zero_centred_targets() -> None:
    """The robust tail index should remain defined when p99/median is not."""
    values = [-2.0, -1.0, 0.0, 1.0, 2.0]

    assert compute_target_robust_tail_index(values) == pytest.approx(1.96)


def test_strong_univariate_signal_concentrates_in_top_features() -> None:
    """A single predictive feature should dominate the positive univariate signal."""
    y = np.asarray([0, 1] * 80)
    X = pd.DataFrame(
        {
            "signal": y,
            "constant_a": np.ones_like(y),
            "constant_b": np.zeros_like(y),
        }
    )

    share = compute_univariate_signal_top5_share(X, y, "binary")

    assert share == pytest.approx(1.0)


def test_constant_features_have_no_univariate_signal_share() -> None:
    """Constant/no-signal features should not fabricate a top-feature share."""
    y = np.asarray([0, 1] * 20)
    X = pd.DataFrame({"constant_a": np.ones_like(y), "constant_b": np.zeros_like(y)})

    assert compute_univariate_signal_top5_share(X, y, "binary") is None


def test_base_meta_features_include_dataset_scale_and_task_specific_fields() -> None:
    """The combined helper should expose scale, entropy, and signal descriptors."""
    y = np.asarray([0, 0, 1, 1])
    X = pd.DataFrame({"signal": y, "constant": [1, 1, np.nan, 1]})

    features = compute_base_meta_features(X, y, "binary")

    assert features["baseTrainRows"] == 4
    assert features["nFeaturesTotal"] == 2
    assert features["trainingTableCells"] == 8
    assert features["featureToRowRatio"] == pytest.approx(0.5)
    assert features["numericFeatureShare"] == pytest.approx(1.0)
    assert features["missingValueShare"] == pytest.approx(0.125)
    assert features["labelEntropyNormalized"] == pytest.approx(1.0)
    assert features["targetP99OverMedian"] is None
    assert features["targetRobustTailIndex"] is None


def test_categorical_cardinality_uses_nonmissing_raw_training_levels() -> None:
    """Cardinality should exclude missing values and numeric-typed columns."""
    levels = [f"level-{index}" for index in range(51)] + [None]
    X = pd.DataFrame(
        {
            "high_cardinality": levels,
            "low_cardinality": ["a", "b"] * 26,
            "binary": [True, False] * 26,
            "numeric_codes": range(52),
        }
    )

    features = compute_base_meta_features(X, [0, 1] * 26, "binary", include_univariate_signal=False)

    assert features["numHighCardinalityCategoricalFeatures"] == 1
    assert features["maxCategoricalCardinality"] == 51


def test_maximum_categorical_cardinality_is_unavailable_without_categorical_columns() -> None:
    """A zero high-cardinality count must not invent a categorical maximum."""
    X = pd.DataFrame({"numeric": [1.0, 2.0, 3.0, 4.0]})

    features = compute_base_meta_features(X, [0, 0, 1, 1], "binary", include_univariate_signal=False)

    assert features["numHighCardinalityCategoricalFeatures"] == 0
    assert features["maxCategoricalCardinality"] is None


def test_distribution_structure_features_capture_numeric_redundancy() -> None:
    X = pd.DataFrame(
        {
            "x": np.arange(20, dtype=float),
            "duplicate_x": np.arange(20, dtype=float),
        }
    )

    features = compute_distribution_structure_features(X)

    assert features["numericMeanAbsoluteSpearman"] == pytest.approx(1.0)
    assert features["numericEffectiveRankRatio"] == pytest.approx(0.5)
    assert features["duplicateRowShare"] == pytest.approx(0.0)


def test_distribution_structure_features_capture_schema_sparsity_and_missingness_patterns() -> None:
    category = ["common"] * 96 + ["rare-a"] * 2 + ["rare-b"] * 2
    X = pd.DataFrame(
        {
            "category": category,
            "missing_a": [np.nan, 0.0, np.nan, 0.0] * 25,
            "missing_b": [np.nan, np.nan, 0.0, 0.0] * 25,
        }
    )

    features = compute_distribution_structure_features(X)

    assert features["categoricalRareValueShare"] == pytest.approx(0.04)
    assert features["missingnessPatternEntropyNormalized"] == pytest.approx(np.log(4) / np.log(100))


def test_distribution_structure_features_capture_balanced_mixed_schema() -> None:
    X = pd.DataFrame({"numeric": [0.0, 1.0], "category": ["a", "b"]})

    features = compute_distribution_structure_features(X)

    assert features["schemaTypeEntropyNormalized"] == pytest.approx(1.0)


def test_local_class_disagreement_is_zero_for_separated_neighbourhoods() -> None:
    X = pd.DataFrame({"x": [0.0, 0.1, 1.0, 1.1]})

    features = compute_local_target_structure_features(X, [0, 0, 1, 1], "binary")

    assert features["localClassDisagreementRate"] == pytest.approx(0.0)
    assert features["localTargetVariationIqr"] is None


def test_local_regression_variation_is_normalized_by_target_iqr() -> None:
    X = pd.DataFrame({"x": [0.0, 0.1, 1.0, 1.1]})

    features = compute_local_target_structure_features(X, [0.0, 1.0, 10.0, 11.0], "regression")

    assert features["localClassDisagreementRate"] is None
    assert features["localTargetVariationIqr"] == pytest.approx(1.0 / 9.5)


def test_distribution_structure_features_report_median_numeric_tail_weight() -> None:
    values = np.asarray([-20.0, -2.0, -1.0, 0.0, 1.0, 2.0, 20.0])
    X = pd.DataFrame({"a": values, "b": values * 3})

    features = compute_distribution_structure_features(X)
    q01, q25, q75, q99 = np.percentile(values, [1, 25, 75, 99])

    assert features["numericTailWeightMedian"] == pytest.approx((q99 - q01) / (q75 - q25))


def test_marginal_shape_heterogeneity_increases_for_different_numeric_shapes() -> None:
    rng = np.random.default_rng(4)
    base = np.linspace(-2.0, 2.0, 200)
    similar = pd.DataFrame({"a": base, "b": base * 2.0})
    different = pd.DataFrame(
        {
            "continuous": base,
            "zero_heavy": np.r_[np.zeros(160), rng.lognormal(0.0, 1.2, 40)],
        }
    )

    similar_value = compute_distribution_structure_features(similar)["marginalShapeHeterogeneity"]
    different_value = compute_distribution_structure_features(different)["marginalShapeHeterogeneity"]

    assert similar_value == pytest.approx(0.0)
    assert different_value is not None
    assert different_value > similar_value


def test_sparse_region_outlier_mass_detects_isolated_rows() -> None:
    dense = pd.DataFrame({"x": np.linspace(0.0, 1.0, 100)})
    with_outlier = dense.copy()
    with_outlier.loc[90:99, "x"] = np.linspace(2.0, 6.0, 10)

    dense_value = compute_distribution_structure_features(dense)["sparseRegionOutlierMass"]
    outlier_value = compute_distribution_structure_features(with_outlier)["sparseRegionOutlierMass"]

    assert dense_value is not None
    assert outlier_value is not None
    assert outlier_value > dense_value


def test_categorical_cooccurrence_sparsity_requires_joint_patterns() -> None:
    common = pd.DataFrame(
        {
            "left": ["a"] * 50 + ["b"] * 50,
            "right": ["x"] * 50 + ["y"] * 50,
        }
    )
    sparse = common.copy()
    sparse.loc[:19, "right"] = [f"rare-{index}" for index in range(20)]

    common_value = compute_distribution_structure_features(common)["categoricalCooccurrenceSparsity"]
    sparse_value = compute_distribution_structure_features(sparse)["categoricalCooccurrenceSparsity"]

    assert common_value == pytest.approx(0.0)
    assert sparse_value == pytest.approx(0.2)
    assert compute_distribution_structure_features(common[["left"]])[
        "categoricalCooccurrenceSparsity"
    ] is None
