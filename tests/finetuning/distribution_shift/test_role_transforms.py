from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from finetuning.datasets.loading import SplitEvaluationData
from finetuning.distribution_shift import (
    apply_post_finite_distribution_shift_with_comparison,
    apply_pre_finite_distribution_shift,
    apply_test_context_majority_downsampling,
    inverse_transform_metric_prediction,
)
from finetuning.distribution_shift.fit_pool import combine_fit_pool, nested_selection_indices


def _classification_data() -> SplitEvaluationData:
    return SplitEvaluationData(
        task_type="binary",
        problem_type="binary",
        metric_name="roc_auc",
        train_X=pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [2.0, 1.0, 0.0]}),
        train_y=np.array([0, 1, 0]),
        val_context_X=pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [2.0, 1.0, 0.0]}),
        val_context_y=np.array([0, 1, 0]),
        val_X=pd.DataFrame({"a": [3.0, 4.0], "b": [0.0, 1.0]}),
        val_y=np.array([1, 0]),
        test_context_X=pd.DataFrame({"a": [0.0, 1.0, 3.0, 4.0], "b": [2.0, 1.0, 0.0, 1.0]}),
        test_context_y=np.array([0, 1, 1, 0]),
        test_X=pd.DataFrame({"a": [5.0, 6.0], "b": [2.0, 3.0]}),
        test_y=np.array([1, 0]),
        metadata={},
        train_ids=np.array(["row:0", "row:1", "row:2"], dtype=object),
        val_context_ids=np.array(["row:0", "row:1", "row:2"], dtype=object),
        val_ids=np.array(["row:3", "row:4"], dtype=object),
        test_context_ids=np.array(["row:0", "row:1", "row:3", "row:4"], dtype=object),
        test_ids=np.array(["test:0", "test:1"], dtype=object),
        val_context_components=np.array(["train_ft", "train_ft", "train_ft"], dtype=object),
        test_context_components=np.array(["train_ft", "train_ft", "val_ft", "val_ft"], dtype=object),
    )


def _regression_data() -> SplitEvaluationData:
    return SplitEvaluationData(
        task_type="regression",
        problem_type="regression",
        metric_name="rmse",
        train_X=pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [2.0, 1.0, 0.0]}),
        train_y=np.array([1.0, 10.0, 100.0]),
        val_context_X=pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [2.0, 1.0, 0.0]}),
        val_context_y=np.array([1.0, 10.0, 100.0]),
        val_X=pd.DataFrame({"a": [3.0, 4.0], "b": [0.0, 1.0]}),
        val_y=np.array([5.0, 50.0]),
        test_context_X=pd.DataFrame({"a": [0.0, 1.0, 3.0, 4.0], "b": [2.0, 1.0, 0.0, 1.0]}),
        test_context_y=np.array([1.0, 10.0, 5.0, 50.0]),
        test_X=pd.DataFrame({"a": [5.0, 6.0], "b": [2.0, 3.0]}),
        test_y=np.array([7.0, 70.0]),
        metadata={},
        train_ids=np.array(["row:0", "row:1", "row:2"], dtype=object),
        val_context_ids=np.array(["row:0", "row:1", "row:2"], dtype=object),
        val_ids=np.array(["row:3", "row:4"], dtype=object),
        test_context_ids=np.array(["row:0", "row:1", "row:3", "row:4"], dtype=object),
        test_ids=np.array(["test:0", "test:1"], dtype=object),
        val_context_components=np.array(["train_ft", "train_ft", "train_ft"], dtype=object),
        test_context_components=np.array(["train_ft", "train_ft", "val_ft", "val_ft"], dtype=object),
    )


def test_label_noise_keeps_overlapping_train_context_labels_identical() -> None:
    data = _classification_data()

    out = apply_pre_finite_distribution_shift(
        data,
        {"preprocess": {"steps": ["label_noise"], "label_noise": {"rate": 1.0}}},
        seed=0,
    )

    train_by_id = dict(zip(out.train_ids.tolist(), out.train_y.tolist(), strict=False))
    for row_id, label, component in zip(
        out.test_context_ids.tolist(),
        out.test_context_y.tolist(),
        out.test_context_components.tolist(),
        strict=False,
    ):
        if component == "train_ft":
            assert label == train_by_id[row_id]
    assert out.val_y.tolist() == data.val_y.tolist()
    assert out.test_y.tolist() == data.test_y.tolist()
    assert out.val_fit_y is not None
    assert out.val_fit_y.tolist() != data.val_y.tolist()
    report = out.metadata["distribution_shift"]["pre_finite"]["label_noise"]
    assert report["fit_pool_changed"] == 5
    assert report["validation_fit_changed"] == 2
    assert report["test_context_changed"] >= 2


def test_subsample_is_deterministic_and_preserves_train_context_row_ids() -> None:
    config = {"preprocess": {"steps": ["subsample_train"], "subsample_train": {"size": "1/2"}}}

    first = apply_pre_finite_distribution_shift(_classification_data(), config, seed=7)
    second = apply_pre_finite_distribution_shift(_classification_data(), config, seed=7)

    assert first.train_ids.tolist() == second.train_ids.tolist()
    assert set(first.val_context_ids.tolist()) == set(first.train_ids.tolist())
    assert set(first.test_context_ids[first.test_context_components == "train_ft"].tolist()).issubset(
        set(first.train_ids.tolist())
    )


def test_subsample_rebuilds_validation_and_context_roles_from_total_budget() -> None:
    config = {
        "preprocess": {
            "steps": ["subsample_train"],
            "subsample_train": {"size": "1/2"},
        }
    }

    out = apply_pre_finite_distribution_shift(_classification_data(), config, seed=7)

    assert len(out.train_X) == 1
    assert len(out.val_X) == 1
    assert set(out.test_context_ids[out.test_context_components == "val_ft"].tolist()) == set(
        out.val_ids.tolist()
    )
    report = out.metadata["distribution_shift"]["pre_finite"]["subsample_train"]
    assert report["strategy"] == "nested_stratified"
    assert report["fit_pool_size_before"] == 5
    assert report["fit_pool_size_after"] == 2
    assert report["validation_size_after"] == 1


def test_subsample_prefixes_are_nested_and_stratified_before_role_partitioning() -> None:
    pool = combine_fit_pool(_classification_data())

    smaller = nested_selection_indices(pool, target=2, task_type="binary", seed=11)
    larger = nested_selection_indices(pool, target=4, task_type="binary", seed=11)

    assert set(pool.ids[smaller]).issubset(set(pool.ids[larger]))
    assert set(pool.query_y[smaller].tolist()) == {0, 1}


def test_class_prior_shift_rejects_nonbinary_datasets() -> None:
    with pytest.raises(ValueError, match="binary-classification intervention"):
        apply_pre_finite_distribution_shift(
            _regression_data(),
            {
                "preprocess": {
                    "steps": ["rebalance_train"],
                    "rebalance_train": {
                        "target_pos_rate": 0.5,
                        "budget_policy": "common_feasible",
                        "target_rates": [0.25, 0.5],
                    },
                }
            },
            seed=0,
        )


def test_class_prior_conditions_share_one_budget_and_nested_class_orders() -> None:
    def transformed(target: object) -> SplitEvaluationData:
        return apply_pre_finite_distribution_shift(
            _classification_data(),
            {
                "preprocess": {
                    "steps": ["rebalance_train"],
                    "rebalance_train": {
                        "target_pos_rate": target,
                        "budget_policy": "common_feasible",
                        "target_rates": [0.25, 0.5],
                    },
                }
            },
            seed=9,
        )

    original = transformed("original")
    quarter = transformed(0.25)
    original_pool = combine_fit_pool(original)
    quarter_pool = combine_fit_pool(quarter)

    assert len(original_pool.X) == len(quarter_pool.X) == 4
    assert np.mean(original_pool.query_y == 1) == pytest.approx(0.5)
    assert np.mean(quarter_pool.query_y == 1) == pytest.approx(0.25)
    original_pos = set(original_pool.ids[original_pool.query_y == 1].tolist())
    original_neg = set(original_pool.ids[original_pool.query_y == 0].tolist())
    quarter_pos = set(quarter_pool.ids[quarter_pool.query_y == 1].tolist())
    quarter_neg = set(quarter_pool.ids[quarter_pool.query_y == 0].tolist())
    assert quarter_pos.issubset(original_pos)
    assert original_neg.issubset(quarter_neg)
    report = original.metadata["distribution_shift"]["pre_finite"]["rebalance_train"]
    assert report["family_reference"] is True
    assert report["common_fit_pool_budget"] == 4
    assert report["budget_policy"] == "common_feasible"


def test_target_transform_uses_dual_validation_targets_and_inverse_scoring() -> None:
    data = _regression_data()

    out = apply_pre_finite_distribution_shift(
        data,
        {"preprocess": {"steps": ["target_transform"], "target_transform": {"mode": "robust_log1p"}}},
        seed=0,
    )

    assert out.val_y.tolist() == data.val_y.tolist()
    assert out.test_y.tolist() == data.test_y.tolist()
    assert out.val_fit_y is not None
    assert not np.allclose(out.val_fit_y, out.val_y)
    prediction = np.asarray(out.val_fit_y)
    restored = inverse_transform_metric_prediction(
        prediction,
        task_type="regression",
        target_transform_state=out.target_transform_state,
    )
    assert np.allclose(restored, out.val_y)


def test_heavy_tail_contamination_keeps_query_targets_clean() -> None:
    data = _regression_data()

    out = apply_pre_finite_distribution_shift(
        data,
        {
            "preprocess": {
                "steps": ["heavy_tail_contamination"],
                "heavy_tail_contamination": {"rate": 1.0, "magnitude_iqr": 2.0},
            }
        },
        seed=1,
    )

    assert out.val_y.tolist() == data.val_y.tolist()
    assert out.test_y.tolist() == data.test_y.tolist()
    assert out.val_fit_y is not None
    assert np.all(out.val_fit_y > data.val_y)
    assert np.all(out.train_y > data.train_y)
    train_by_id = dict(zip(out.train_ids.tolist(), out.train_y.tolist(), strict=False))
    val_by_id = dict(zip(out.val_ids.tolist(), out.val_fit_y.tolist(), strict=False))
    for row_id, label, component in zip(
        out.test_context_ids.tolist(),
        out.test_context_y.tolist(),
        out.test_context_components.tolist(),
        strict=False,
    ):
        if component == "train_ft":
            assert label == pytest.approx(train_by_id[row_id])
        elif component == "val_ft":
            assert label == pytest.approx(val_by_id[row_id])
    report = out.metadata["distribution_shift"]["pre_finite"]["heavy_tail_contamination"]
    assert report["transformation_setting"] == "fit_side_transformation"
    assert report["validation_fit_changed"] == len(data.val_y)


def test_heavy_tail_resolves_calibrated_rate_before_fit_side_mitigation() -> None:
    data = dataclasses.replace(_regression_data(), metadata={"dataset_id": "demo-data"})

    out = apply_pre_finite_distribution_shift(
        data,
        {
            "preprocess": {
                "steps": ["heavy_tail_contamination", "target_transform"],
                "heavy_tail_contamination": {
                    "rate": "calibrated",
                    "rate_by_dataset": {"demo_data": 1},
                    "magnitude_iqr": 2,
                },
                "target_transform": {"mode": "robust_log1p"},
            }
        },
        seed=1,
    )

    restored_validation_target = inverse_transform_metric_prediction(
        np.asarray(out.val_fit_y),
        task_type="regression",
        target_transform_state=out.target_transform_state,
    )
    assert np.all(restored_validation_target > data.val_y)
    contamination_report = out.metadata["distribution_shift"]["pre_finite"]["heavy_tail_contamination"]
    assert contamination_report["rate"] == pytest.approx(1)


def test_contextual_scale_matched_condition_adds_consistent_unit_context() -> None:
    data = _regression_data()

    out = apply_pre_finite_distribution_shift(
        data,
        {
            "preprocess": {
                "steps": ["contextual_scale"],
                "contextual_scale": {
                    "condition": "scale_sx_with_unit_column",
                    "scale_spread": 10,
                    "feature_strategy": "first",
                },
            }
        },
        seed=3,
    )

    assert "__synthetic_unit__" in out.train_X.columns
    assert "__synthetic_unit__" in out.val_context_X.columns
    train_context = dict(zip(out.train_ids.tolist(), out.train_X["__synthetic_unit__"].tolist(), strict=False))
    train_values = dict(zip(out.train_ids.tolist(), out.train_X["a"].tolist(), strict=False))
    for row_id, unit, value in zip(
        out.val_context_ids.tolist(),
        out.val_context_X["__synthetic_unit__"].tolist(),
        out.val_context_X["a"].tolist(),
        strict=False,
    ):
        assert unit == train_context[row_id]
        assert value == pytest.approx(train_values[row_id])
    report = out.metadata["distribution_shift"]["pre_finite"]["contextual_scale"]
    assert report["condition"] == "scale_sx_with_unit_column"
    assert report["transformation_setting"] == "matched_transformation"
    assert report["fit_side_role"] == "transformed"
    assert report["test_side_role"] == "transformed"
    assert report["context_column_present"] is True
    assert report["selected_features"] == ["a"]
    assert report["scale_spread"] == pytest.approx(10.0)


def test_contextual_scale_matched_no_unit_condition_keeps_ambiguous_values_only() -> None:
    data = _regression_data()

    out = apply_pre_finite_distribution_shift(
        data,
        {
            "preprocess": {
                "steps": ["contextual_scale"],
                "contextual_scale": {
                    "condition": "scale_sx_no_unit_column",
                    "scale_spread": 10,
                    "feature_strategy": "first",
                },
            }
        },
        seed=3,
    )

    assert "__synthetic_unit__" not in out.train_X.columns
    assert list(out.train_X.columns) == list(data.train_X.columns)
    report = out.metadata["distribution_shift"]["pre_finite"]["contextual_scale"]
    assert report["context_column_present"] is False
    assert report["contextualized_feature_count"] == 1


def test_contextual_scale_1x_conditions_are_original_and_matched_controls() -> None:
    data = _regression_data()

    original = apply_pre_finite_distribution_shift(
        data,
        {
            "preprocess": {
                "steps": ["contextual_scale"],
                "contextual_scale": {"condition": "scale_1x_no_unit_column"},
            }
        },
        seed=3,
    )
    with_unit = apply_pre_finite_distribution_shift(
        data,
        {
            "preprocess": {
                "steps": ["contextual_scale"],
                "contextual_scale": {
                    "condition": "scale_1x_with_unit_column",
                    "feature_strategy": "first",
                },
            }
        },
        seed=3,
    )

    pd.testing.assert_frame_equal(original.train_X, data.train_X)
    original_report = original.metadata["distribution_shift"]["pre_finite"]["contextual_scale"]
    assert original_report["transformation_setting"] == "no_transformation"
    assert with_unit.train_X["__synthetic_unit__"].unique().tolist() == ["unit_0"]
    assert np.allclose(with_unit.train_X["a"], data.train_X["a"])
    with_unit_report = with_unit.metadata["distribution_shift"]["pre_finite"]["contextual_scale"]
    assert with_unit_report["transformation_setting"] == "matched_transformation"


def test_contextual_scale_resolves_calibrated_spread_by_dataset() -> None:
    data = _regression_data()
    data = dataclasses.replace(data, metadata={"dataset_id": "demo-data"})

    out = apply_pre_finite_distribution_shift(
        data,
        {
            "preprocess": {
                "steps": ["contextual_scale"],
                "contextual_scale": {
                    "condition": "scale_sx_no_unit_column",
                    "scale_spread": "calibrated",
                    "scale_spread_by_dataset": {"demo_data": 30},
                    "feature_strategy": "first",
                },
            }
        },
        seed=3,
    )

    report = out.metadata["distribution_shift"]["pre_finite"]["contextual_scale"]
    assert report["scale_spread"] == pytest.approx(30)


def test_contextual_scale_resolves_calibrated_feature_count_by_dataset() -> None:
    data = dataclasses.replace(_regression_data(), metadata={"dataset_id": "demo-data"})

    out = apply_pre_finite_distribution_shift(
        data,
        {
            "preprocess": {
                "steps": ["contextual_scale"],
                "contextual_scale": {
                    "condition": "scale_sx_no_unit_column",
                    "scale_spread": 10,
                    "feature_count": "calibrated",
                    "feature_count_by_dataset": {"demo_data": 2},
                    "feature_strategy": "first",
                },
            }
        },
        seed=3,
    )

    report = out.metadata["distribution_shift"]["pre_finite"]["contextual_scale"]
    assert report["contextualized_feature_count"] == 2


def test_feature_order_fit_state_returns_both_test_views_from_one_state() -> None:
    data = _regression_data()

    primary, alternate = apply_post_finite_distribution_shift_with_comparison(
        data,
        {
            "preprocess": {
                "steps": ["shuffle_feature_order"],
                "shuffle_feature_order": {"fit_state": "permuted", "record_permutation": True},
            }
        },
        seed=4,
    )

    assert alternate is not None
    assert primary.train_X.to_numpy().tolist() != data.train_X.to_numpy().tolist()
    assert primary.val_X.to_numpy().tolist() != data.val_X.to_numpy().tolist()
    assert primary.test_X.to_numpy().tolist() == data.test_X.to_numpy().tolist()
    assert alternate.test_X.to_numpy().tolist() != primary.test_X.to_numpy().tolist()
    comparison = primary.metadata["evaluation_comparison"]
    assert comparison == {
        "kind": "distribution_shift_test_view",
        "family": "feature_order",
        "fit_state": "permuted",
        "primary_view": "canonical",
        "alternate_view": "permuted",
    }


def test_feature_order_update_state_only_changes_gradient_update_rows() -> None:
    data = _regression_data()

    transformed, alternate = apply_post_finite_distribution_shift_with_comparison(
        data,
        {
            "preprocess": {
                "steps": ["shuffle_feature_order"],
                "shuffle_feature_order": {
                    "update_state": "fixed_permutation",
                    "record_permutation": True,
                },
            }
        },
        seed=4,
    )

    assert alternate is None
    assert transformed.train_X.to_numpy().tolist() != data.train_X.to_numpy().tolist()
    for role in ("val_context", "val", "test_context", "test"):
        assert getattr(transformed, f"{role}_X").to_numpy().tolist() == getattr(
            data, f"{role}_X"
        ).to_numpy().tolist()
    report = transformed.metadata["distribution_shift"]["post_finite"]["shuffle_feature_order"]
    assert report["gradient_update_roles"] == ["train"]
    assert report["same_icl_data_across_conditions"] is True
    assert report["paired_test_views"] is False


def test_numeric_scale_is_train_fitted_and_preserves_binary_columns() -> None:
    data = _regression_data()
    binary = np.array([0.0, 1.0, 0.0])
    data = dataclasses.replace(
        data,
        train_X=data.train_X.assign(binary=binary),
        val_context_X=data.val_context_X.assign(binary=binary),
        val_X=data.val_X.assign(binary=[1.0, 0.0]),
        test_context_X=data.test_context_X.assign(binary=[0.0, 1.0, 1.0, 0.0]),
        test_X=data.test_X.assign(binary=[1.0, 0.0]),
    )

    primary, alternate = apply_post_finite_distribution_shift_with_comparison(
        data,
        {"preprocess": {"steps": ["numeric_scale"], "numeric_scale": {"fit_state": "standardized"}}},
        seed=0,
    )

    assert alternate is not None
    assert np.allclose(primary.train_X[["a", "b"]].mean().to_numpy(), 0.0, atol=1e-6)
    assert primary.train_X["binary"].tolist() == data.train_X["binary"].tolist()
    assert primary.test_X.to_numpy().tolist() == data.test_X.to_numpy().tolist()
    assert alternate.test_X["binary"].tolist() == data.test_X["binary"].tolist()
    assert not np.allclose(alternate.test_X[["a", "b"]], primary.test_X[["a", "b"]])


def test_numeric_scale_update_state_only_changes_gradient_update_rows() -> None:
    data = _regression_data()

    transformed, alternate = apply_post_finite_distribution_shift_with_comparison(
        data,
        {
            "preprocess": {
                "steps": ["numeric_scale"],
                "numeric_scale": {"update_state": "standardized"},
            }
        },
        seed=0,
    )

    assert alternate is None
    assert np.allclose(transformed.train_X.mean().to_numpy(), 0.0, atol=1e-6)
    for role in ("val_context", "val", "test_context", "test"):
        pd.testing.assert_frame_equal(getattr(transformed, f"{role}_X"), getattr(data, f"{role}_X"))
    report = transformed.metadata["distribution_shift"]["post_finite"]["numeric_scale"]
    assert report["gradient_update_roles"] == ["train"]
    assert report["same_icl_data_across_conditions"] is True
    assert report["paired_test_views"] is False


def test_spurious_reference_keeps_same_neutral_schema_in_every_role() -> None:
    data = _classification_data()

    out = apply_pre_finite_distribution_shift(
        data,
        {"preprocess": {"steps": ["spurious_feature"], "spurious_feature": {"strength": 0}}},
        seed=3,
    )

    for role in ("train", "val_context", "val", "test_context", "test"):
        assert "__spurious_signal__" in getattr(out, f"{role}_X").columns
    report = out.metadata["distribution_shift"]["pre_finite"]["spurious_feature"]
    assert report["transformation_setting"] == "no_transformation"
    assert report["same_schema_control"] is True


def _retained_minority_config(target: object) -> dict[str, object]:
    return {
        "preprocess": {
            "steps": ["rebalance_train"],
            "rebalance_train": {
                "target_focal_rate": target,
                "focal_label_policy": "original_fit_majority",
                "majority_tie_break": "smallest_label_by_type_then_value_v1",
                "budget_policy": "retain_original_minority",
            },
        }
    }


def test_class_prior_majority_downsampling_retains_all_original_minority_rows() -> None:
    data = _classification_data()
    original = apply_pre_finite_distribution_shift(data, _retained_minority_config("original"), seed=9)
    balanced = apply_pre_finite_distribution_shift(data, _retained_minority_config(0.5), seed=9)
    original_pool = combine_fit_pool(original)
    balanced_pool = combine_fit_pool(balanced)

    original_minority_ids = set(original_pool.ids[original_pool.query_y == 1].tolist())
    balanced_minority_ids = set(balanced_pool.ids[balanced_pool.query_y == 1].tolist())
    assert len(original_pool.X) == 5
    assert len(balanced_pool.X) == 4
    assert balanced_minority_ids == original_minority_ids
    assert np.mean(balanced_pool.query_y == 0) == pytest.approx(0.5)

    original_report = original.metadata["distribution_shift"]["pre_finite"]["rebalance_train"]
    balanced_report = balanced.metadata["distribution_shift"]["pre_finite"]["rebalance_train"]
    assert original_report["original_full_pool_control"] is True
    assert original_report["fit_pool_size_after"] == original_report["fit_pool_size_before"]
    assert balanced_report["original_full_pool_control"] is False
    assert balanced_report["retained_original_minority_rows"] is True
    assert balanced_report["fixed_original_minority_count"] == 2
    assert balanced_report["class_counts_after"] == {"focal": 2, "other": 2}
    assert balanced_report["variable_fit_pool_budget"] == 4


def test_class_prior_majority_downsampling_is_deterministic() -> None:
    first = apply_pre_finite_distribution_shift(_classification_data(), _retained_minority_config(0.5), seed=17)
    second = apply_pre_finite_distribution_shift(_classification_data(), _retained_minority_config(0.5), seed=17)

    assert combine_fit_pool(first).ids.tolist() == combine_fit_pool(second).ids.tolist()


def test_test_context_majority_downsampling_changes_only_icl_context() -> None:
    data = dataclasses.replace(
        _classification_data(),
        test_context_X=pd.DataFrame({"a": range(6), "b": range(10, 16)}),
        test_context_y=np.array([0, 0, 0, 0, 1, 1]),
        test_context_ids=np.array([f"context:{idx}" for idx in range(6)], dtype=object),
        test_context_components=np.array(["train_ft"] * 4 + ["val_ft"] * 2, dtype=object),
    )

    balanced, report = apply_test_context_majority_downsampling(
        data,
        {
            "target_focal_rate": 0.5,
            "majority_tie_break": "smallest_label_by_type_then_value_v1",
        },
        seed=17,
    )

    assert len(balanced.test_context_X) == 4
    assert np.mean(balanced.test_context_y == 0) == pytest.approx(0.5)
    assert {"context:4", "context:5"}.issubset(set(balanced.test_context_ids.tolist()))
    pd.testing.assert_frame_equal(balanced.train_X, data.train_X)
    pd.testing.assert_frame_equal(balanced.val_X, data.val_X)
    pd.testing.assert_frame_equal(balanced.test_X, data.test_X)
    assert balanced.train_y.tolist() == data.train_y.tolist()
    assert balanced.val_y.tolist() == data.val_y.tolist()
    assert balanced.test_y.tolist() == data.test_y.tolist()
    assert report["transformation_setting"] == "test_icl_context_only"
    assert report["test_labels_used_for_orientation"] is False


def test_class_prior_majority_downsampling_rejects_increased_majority_rate() -> None:
    with pytest.raises(ValueError, match="cannot increase the original-majority rate"):
        apply_pre_finite_distribution_shift(_classification_data(), _retained_minority_config(0.9), seed=0)
