"""Tests for model finetune: tabicl."""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tests.support.model_finetune_support import (
    preprocessing_runner,
)
from tfm_finetuning.finetuning.datasets.loading import (
    SplitEvaluationData,
)
from tfm_finetuning.finetuning.datasets.preprocessing_protocol import (
    PREPROCESSING_SHARED,
)
from tfm_finetuning.finetuning.models.base import CommonFineTuneConfig
from tfm_finetuning.finetuning.models.preprocessing import (
    SHARED_FEATURE_PREPROCESSING,
    FiniteFeaturePreprocessor,
)
from tfm_finetuning.finetuning.models.registry import ModelSpec
from tfm_finetuning.finetuning.models.tabicl.model import TabICLAdapter
from tfm_finetuning.finetuning.training import fit as fit_runner


def test_tabicl_finite_feature_preprocessor_returns_finite_float32_frames() -> None:
    train = pd.DataFrame(
        {
            "num": [1.0, np.nan, np.inf],
            "all_missing": [np.nan, np.nan, np.nan],
            "cat": ["b", None, "a"],
            "flag": [True, False, None],
        }
    )
    query = pd.DataFrame(
        {
            "num": [np.nan, -np.inf, 5.0],
            "all_missing": [7.0, np.nan, np.nan],
            "cat": ["new", None, "a"],
            "flag": [True, None, False],
        }
    )

    preprocessor = FiniteFeaturePreprocessor.fit(train)
    transformed_train = preprocessor.transform(train)
    transformed_query = preprocessor.transform(query)

    assert list(transformed_query.columns) == list(train.columns)
    assert {str(dtype) for dtype in transformed_query.dtypes} == {"float32"}
    assert np.isfinite(transformed_train.to_numpy()).all()
    assert np.isfinite(transformed_query.to_numpy()).all()
    assert transformed_train.loc[1, "num"] == 1.0
    assert transformed_train.loc[0, "all_missing"] == 0.0
    assert transformed_query.loc[0, "cat"] == 1.0
    assert transformed_query.loc[1, "cat"] == 0.0
    assert preprocessor.metadata()["method"] == SHARED_FEATURE_PREPROCESSING

def test_tabicl_finite_feature_preprocessor_handles_wide_frames_without_fragmentation_warning() -> None:
    train = pd.DataFrame({f"feature_{index}": [float(index), np.nan, np.inf] for index in range(150)})
    preprocessor = FiniteFeaturePreprocessor.fit(train)

    with warnings.catch_warnings(record=True) as warnings_record:
        warnings.simplefilter("always")
        transformed = preprocessor.transform(train)

    assert not any(isinstance(item.message, pd.errors.PerformanceWarning) for item in warnings_record)
    assert list(transformed.columns) == list(train.columns)
    assert {str(dtype) for dtype in transformed.dtypes} == {"float32"}
    assert np.isfinite(transformed.to_numpy()).all()


def test_apply_feature_preprocessing_records_tabicl_metadata() -> None:
    data = SplitEvaluationData(
        task_type="binary",
        problem_type="binary",
        metric_name="roc_auc",
        train_X=pd.DataFrame({"num": [1.0, np.nan], "cat": ["a", None]}),
        train_y=np.array([0, 1]),
        val_context_X=pd.DataFrame({"num": [1.0, np.nan], "cat": ["a", None]}),
        val_context_y=np.array([0, 1]),
        val_X=pd.DataFrame({"num": [np.nan], "cat": ["unseen"]}),
        val_y=np.array([1]),
        test_context_X=pd.DataFrame({"num": [np.inf], "cat": [None]}),
        test_context_y=np.array([0]),
        test_X=pd.DataFrame({"num": [-np.inf], "cat": ["a"]}),
        test_y=np.array([1]),
        metadata={"data_backend": "test"},
    )

    transformed = preprocessing_runner._apply_preprocessing_protocol(data, protocol="shared", model_id="tabicl_v2", seed=0)

    assert transformed.metadata["feature_preprocessing"]["method"] == SHARED_FEATURE_PREPROCESSING
    assert transformed.metadata["feature_preprocessing"]["protocol"] == PREPROCESSING_SHARED
    assert np.isfinite(transformed.train_X.to_numpy()).all()
    assert np.isfinite(transformed.val_X.to_numpy()).all()
    assert np.isfinite(transformed.test_context_X.to_numpy()).all()
    assert transformed.val_X.loc[0, "cat"] == 1.0
    assert transformed.metadata["feature_preprocessing"]["source_feature_schema"]["categorical_columns"] == ["cat"]
    assert transformed.metadata["feature_preprocessing"]["adapter_feature_schema"]["categorical_feature_indices"] == []
    assert "adapter_validation_encoding" not in transformed.metadata["feature_preprocessing"]

def test_fit_with_oom_retry_reduces_tabicl_context_and_cleans_partial_checkpoints(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeEstimator:
        def __init__(self, context_size: int) -> None:
            self.context_size = context_size
            self.save_interval = 1

    class FakeAdapter:
        def __init__(self) -> None:
            self.attempted_context_sizes: list[int] = []

        def resolve_finetune_estimator(self, _task_type, common_config, _seed, _device):
            return FakeEstimator(int(common_config.context_size))

        def fit_finetuned(self, estimator, _train_X, _train_y, _val_X, _val_y, output_dir: Path):
            self.attempted_context_sizes.append(estimator.context_size)
            if estimator.context_size == 10000:
                (output_dir / "partial.ckpt").write_text("stale", encoding="utf-8")
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
            return estimator

    monkeypatch.setattr(fit_runner, "_release_cuda_memory", lambda: None)
    output = tmp_path / "artifact"
    output.mkdir()
    adapter = FakeAdapter()

    result = fit_runner._fit_with_oom_retry(
        adapter=adapter,
        data=SplitEvaluationData(
            task_type="binary",
            problem_type="binary",
            metric_name="roc_auc",
            train_X=pd.DataFrame({"x": [0, 1]}),
            train_y=np.array([0, 1]),
            val_context_X=pd.DataFrame({"x": [0, 1]}),
            val_context_y=np.array([0, 1]),
            val_X=pd.DataFrame({"x": [2, 3]}),
            val_y=np.array([0, 1]),
            test_context_X=pd.DataFrame({"x": [0, 1]}),
            test_context_y=np.array([0, 1]),
            test_X=pd.DataFrame({"x": [4, 5]}),
            test_y=np.array([0, 1]),
            metadata={},
        ),
        spec=SimpleNamespace(model_id="tabicl_v2"),
        config={},
        common_config=CommonFineTuneConfig(context_size=10000),
        resolved_seed=0,
        device=None,
        output=output,
        checkpoint_mode="best",
        test_eval_interval=0,
        keep_intermediate_checkpoints=True,
        eval_timing_rows=[],
    )

    assert adapter.attempted_context_sizes == [10000, 5000]
    assert result.common_config.context_size == 5000
    assert result.oom_retry["configured_context_size"] == 10000
    assert result.oom_retry["effective_context_size"] == 5000
    assert result.oom_retry["resource_fallback_policy"] == "oom_context_retry"
    assert result.oom_retry["oom_retry_triggered"] is True
    assert result.oom_retry["context_size_fraction"] == 0.5
    assert result.oom_retry["successful_context_size"] == 5000
    assert result.oom_retry["retry_count"] == 1
    assert not (output / "partial.ckpt").exists()

def test_tabicl_native_finetune_converts_categorical_features_before_fit(tmp_path: Path) -> None:
    class FakeTabICLEstimator:
        def fit(self, X, y, *, X_val, y_val, output_dir: Path):
            self.X = np.asarray(X)
            self.X_val = np.asarray(X_val)
            self.y = np.asarray(y)
            self.y_val = np.asarray(y_val)
            self.output_dir = output_dir
            assert self.X.dtype.kind in {"f", "i", "u"}
            assert self.X_val.dtype.kind in {"f", "i", "u"}
            return self

    spec = ModelSpec(
        model_id="tabicl_v2",
        display_name="TabICLv2",
        adapter_cls_path="tfm_finetuning.finetuning.models.tabicl.model.TabICLAdapter",
        supports_classification=True,
        supports_regression=True,
        classifier_checkpoint="tabicl-classifier-v2-20260212.ckpt",
        regressor_checkpoint="tabicl-regressor-v2-20260212.ckpt",
        source_path=tmp_path,
        version_payload={"source_available": True},
    )
    adapter = TabICLAdapter(spec)

    fitted = adapter.fit_finetuned(
        FakeTabICLEstimator(),
        pd.DataFrame({"weekday": ["Mon", "Thu", None], "visits": [1.0, np.nan, 3.0]}),
        np.array([0, 1, 0]),
        pd.DataFrame({"weekday": ["Thu", "Sun"], "visits": [2.0, 4.0]}),
        np.array([1, 0]),
        tmp_path,
    )
    diagnostics = adapter.consume_finetune_diagnostics()

    assert fitted.X.shape == (3, 2)
    assert fitted.X_val.shape == (2, 2)
    assert np.isfinite(fitted.X).all()
    assert np.isfinite(fitted.X_val).all()
    assert diagnostics["tabicl_native_finetune_input_preprocessing"]["method"] == "tabicl_transform_to_numerical"

def test_tabicl_shared_finetune_keeps_shared_numeric_inputs(tmp_path: Path) -> None:
    class FakeTabICLEstimator:
        def fit(self, X, y, *, X_val, y_val, output_dir: Path):
            self.X = np.asarray(X)
            self.X_val = np.asarray(X_val)
            self.y = np.asarray(y)
            self.y_val = np.asarray(y_val)
            self.output_dir = output_dir
            return self

    spec = ModelSpec(
        model_id="tabicl_v2",
        display_name="TabICLv2",
        adapter_cls_path="tfm_finetuning.finetuning.models.tabicl.model.TabICLAdapter",
        supports_classification=True,
        supports_regression=True,
        classifier_checkpoint="tabicl-classifier-v2-20260212.ckpt",
        regressor_checkpoint="tabicl-regressor-v2-20260212.ckpt",
        source_path=tmp_path,
        version_payload={"source_available": True},
    )
    adapter = TabICLAdapter(spec)
    adapter.set_preprocessing_protocol(PREPROCESSING_SHARED)

    fitted = adapter.fit_finetuned(
        FakeTabICLEstimator(),
        pd.DataFrame({"feature_0": [1.0, np.nan], "feature_1": [2.0, np.inf]}, dtype=np.float32),
        np.array([0, 1]),
        pd.DataFrame({"feature_0": [3.0], "feature_1": [-np.inf]}, dtype=np.float32),
        np.array([1]),
        tmp_path,
    )
    diagnostics = adapter.consume_finetune_diagnostics()

    assert fitted.X.dtype == np.float32
    assert fitted.X_val.dtype == np.float32
    assert np.isfinite(fitted.X).all()
    assert np.isfinite(fitted.X_val).all()
    assert diagnostics["tabicl_native_finetune_input_preprocessing"]["method"] == (
        "shared_preprocessed_numeric_passthrough"
    )
    assert diagnostics["tabicl_native_finetune_input_preprocessing"]["protocol"] == (
        PREPROCESSING_SHARED
    )
