"""Tests for foundation model finetune: preprocessing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.support.model_finetune_support import (
    preprocessing_runner,
)
from tfm_finetuning.finetuning import config as config_runner
from tfm_finetuning.finetuning.datasets import loading as data_runner
from tfm_finetuning.finetuning.datasets.loading import (
    SplitEvaluationData,
)
from tfm_finetuning.finetuning.datasets.preprocessing_protocol import (
    PREPROCESSING_NATIVE,
)
from tfm_finetuning.finetuning.models.preprocessing import (
    feature_schema_metadata,
)
from tfm_finetuning.finetuning.models.registry import FoundationModelSpec


def test_feature_schema_metadata_marks_native_categorical_columns() -> None:
    schema = feature_schema_metadata(
        pd.DataFrame(
            {
                "month": pd.Series(["Mar", "Apr"], dtype="category"),
                "visits": [1.0, 2.0],
                "weekend": [True, False],
            }
        )
    )

    assert schema["columns"] == ["month", "visits", "weekend"]
    assert schema["categorical_columns"] == ["month", "weekend"]
    assert schema["categorical_feature_indices"] == [0, 2]
    assert schema["numeric_columns"] == ["visits"]

@pytest.mark.parametrize("protocol", ["native"])
def test_native_preprocessing_records_adapter_feature_schema(protocol: str) -> None:
    data = SplitEvaluationData(
        task_type="binary",
        problem_type="binary",
        metric_name="roc_auc",
        train_X=pd.DataFrame({"month": pd.Series(["Mar", "Apr"], dtype="category"), "x": [1.0, 2.0]}),
        train_y=np.array([0, 1]),
        val_context_X=pd.DataFrame({"month": pd.Series(["Mar", "Apr"], dtype="category"), "x": [1.0, 2.0]}),
        val_context_y=np.array([0, 1]),
        val_X=pd.DataFrame({"month": pd.Series(["May"], dtype="category"), "x": [3.0]}),
        val_y=np.array([1]),
        test_context_X=pd.DataFrame({"month": pd.Series(["Mar"], dtype="category"), "x": [1.0]}),
        test_context_y=np.array([0]),
        test_X=pd.DataFrame({"month": pd.Series(["Jun"], dtype="category"), "x": [4.0]}),
        test_y=np.array([1]),
        metadata={},
    )

    transformed = preprocessing_runner._apply_preprocessing_protocol(
        data,
        protocol=protocol,
        model_id="tabpfn_v2_5",
        seed=0,
    )
    metadata = transformed.metadata["feature_preprocessing"]
    refreshed = preprocessing_runner._refresh_adapter_feature_schema_metadata(transformed)
    refreshed_metadata = refreshed.metadata["feature_preprocessing"]

    assert transformed.train_X is data.train_X
    assert metadata["adapter_feature_schema"]["categorical_feature_indices"] == [0]
    assert metadata["categorical_feature_indices"] == [0]
    assert metadata["adapter_validation_encoding"] == "tabpfn_native_categorical_ordinal_bridge"
    assert metadata["adapter_validation_encoding_scope"] == "adapter_boundary_only"
    assert metadata["adapter_validation_encoded_columns"] == ["month"]
    assert metadata["adapter_validation_unknown_handling"] == "unknown_or_missing_category_to_nan"
    assert refreshed_metadata["adapter_feature_schema"]["categorical_feature_indices"] == [0]
    assert refreshed_metadata["categorical_feature_indices"] == [0]

def test_densify_sparse_feature_frames_materializes_pandas_sparse_columns() -> None:
    data = SplitEvaluationData(
        task_type="binary",
        problem_type="binary",
        metric_name="roc_auc",
        train_X=pd.DataFrame({"x": pd.arrays.SparseArray([0.0, 1.0]), "dense": [2.0, 3.0]}),
        train_y=np.asarray([0, 1]),
        val_context_X=pd.DataFrame({"x": pd.arrays.SparseArray([0.0, 1.0]), "dense": [2.0, 3.0]}),
        val_context_y=np.asarray([0, 1]),
        val_X=pd.DataFrame({"x": pd.arrays.SparseArray([1.0]), "dense": [3.0]}),
        val_y=np.asarray([1]),
        test_context_X=pd.DataFrame({"x": pd.arrays.SparseArray([0.0, 1.0]), "dense": [2.0, 3.0]}),
        test_context_y=np.asarray([0, 1]),
        test_X=pd.DataFrame({"x": pd.arrays.SparseArray([0.0]), "dense": [2.0]}),
        test_y=np.asarray([0]),
        metadata={},
    )

    dense = data_runner._densify_sparse_feature_frames(data)

    assert not isinstance(dense.train_X["x"].dtype, pd.SparseDtype)
    assert not isinstance(dense.val_context_X["x"].dtype, pd.SparseDtype)
    assert not isinstance(dense.val_X["x"].dtype, pd.SparseDtype)
    assert not isinstance(dense.test_context_X["x"].dtype, pd.SparseDtype)
    assert not isinstance(dense.test_X["x"].dtype, pd.SparseDtype)
    assert dense.train_X["dense"].tolist() == [2.0, 3.0]
    assert dense.metadata["sparse_feature_densification"] == {
        "applied": True,
        "sparse_columns_by_split": {
            "train": ["x"],
            "val_context": ["x"],
            "val_query": ["x"],
            "test_context": ["x"],
            "test_query": ["x"],
        },
    }

def test_load_split_evaluation_data_defaults_test_context_to_inner_train(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    (dataset_path / "info.json").write_text(json.dumps({"task_id": 42}), encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_load_bundle(**kwargs):
        captured.update(kwargs)
        return {
            "problem_type": "binary",
            "metric_name": "roc_auc",
            "task_id": kwargs["task_id"],
            "fold": 0,
            "repeat": 0,
            "sample": 0,
            "context_features": {
                "val": pd.DataFrame({"x": range(9)}),
                "test": pd.DataFrame({"x": range(10)}),
            },
            "context_labels": {
                "val": pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0]),
                "test": pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
            },
            "query_features": {
                "val": pd.DataFrame({"x": [10, 11]}),
                "test": pd.DataFrame({"x": [12, 13]}),
            },
            "query_labels": {
                "val": pd.Series([0, 1]),
                "test": pd.Series([1, 0]),
            },
        }

    monkeypatch.setattr(data_runner._pipeline_eval_shared, "load_tabarena_aligned_eval_bundle", fake_load_bundle)

    data = data_runner.load_split_evaluation_data(
        config={
            "data": {"path": str(dataset_path), "split": "fold0_repeat0_sample0"},
            "foundation_model": {"validation_split_ratio": 0.1, "validation_split_seed": 7},
        },
        repo_root=tmp_path,
    )

    assert captured["val_fraction"] == 0.1
    assert captured["val_seed"] == 7
    assert len(data.train_X) == 9
    assert len(data.test_context_X) == 9
    assert data.metadata["test_context_includes_validation_data"] is False
    assert data.metadata["test_context_source"] == "inner_train"

@pytest.mark.parametrize(
    "context_flag_config",
    [
        {"foundation_model": {"test_context_includes_validation_data": True}},
        {"foundation_model": {"test_context_includes_explicit_validation_data": True}},
    ],
)
def test_load_split_evaluation_data_validation_context_flag_selects_outer_train(
    tmp_path: Path,
    monkeypatch,
    context_flag_config: dict[str, Any],
) -> None:
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    (dataset_path / "info.json").write_text(json.dumps({"task_id": 42}), encoding="utf-8")

    def fake_load_bundle(**_kwargs):
        return {
            "problem_type": "binary",
            "metric_name": "roc_auc",
            "task_id": 42,
            "fold": 0,
            "repeat": 0,
            "sample": 0,
            "context_features": {
                "val": pd.DataFrame({"x": range(9)}),
                "test": pd.DataFrame({"x": range(10)}),
            },
            "context_labels": {
                "val": pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0]),
                "test": pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
            },
            "query_features": {
                "val": pd.DataFrame({"x": [10, 11]}),
                "test": pd.DataFrame({"x": [12, 13]}),
            },
            "query_labels": {
                "val": pd.Series([0, 1]),
                "test": pd.Series([1, 0]),
            },
        }

    monkeypatch.setattr(data_runner._pipeline_eval_shared, "load_tabarena_aligned_eval_bundle", fake_load_bundle)

    data = data_runner.load_split_evaluation_data(
        config={
            "data": {"path": str(dataset_path), "split": "fold0_repeat0_sample0"},
            **context_flag_config,
        },
        repo_root=tmp_path,
    )

    assert len(data.test_context_X) == 10
    assert data.metadata["test_context_includes_validation_data"] is True
    assert data.metadata["test_context_source"] == "outer_train"

def test_resolve_preprocessing_protocol_defaults_to_model_native() -> None:
    assert preprocessing_runner._resolve_preprocessing_protocol({}) == PREPROCESSING_NATIVE





def test_resolve_configured_foundation_model_spec_honors_checkpoint_metadata(tmp_path: Path) -> None:
    spec = FoundationModelSpec(
        model_id="tabicl_v2",
        display_name="TabICLv2",
        adapter_cls_path="tfm_finetuning.finetuning.models.tabicl.model.TabICLAdapter",
        supports_classification=True,
        supports_regression=True,
        classifier_checkpoint="old-classifier.ckpt",
        regressor_checkpoint="old-regressor.ckpt",
        source_path=tmp_path,
        version_payload={
            "source_available": True,
            "checkpoint_identifier_classifier": "old-classifier.ckpt",
            "checkpoint_identifier_regressor": "old-regressor.ckpt",
        },
    )

    resolved = config_runner.resolve_configured_foundation_model_spec(
        spec,
        {
            "foundation_model": {
                "classifier_checkpoint": "tabicl-classifier-v2-20260212.ckpt",
                "regressor_checkpoint": "tabicl-regressor-v2-20260212.ckpt",
            }
        },
    )

    assert resolved.classifier_checkpoint == "tabicl-classifier-v2-20260212.ckpt"
    assert resolved.regressor_checkpoint == "tabicl-regressor-v2-20260212.ckpt"
    assert resolved.version_payload["checkpoint_identifier_classifier"] == "tabicl-classifier-v2-20260212.ckpt"
    assert resolved.version_payload["checkpoint_identifier_regressor"] == "tabicl-regressor-v2-20260212.ckpt"
