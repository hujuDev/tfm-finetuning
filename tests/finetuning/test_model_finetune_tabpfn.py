"""Tests for foundation model finetune: tabpfn."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from finetuning.datasets.loading import (
    SplitEvaluationData,
)
from finetuning.datasets.preprocessing_protocol import (
    PREPROCESSING_NATIVE,
    PREPROCESSING_SHARED,
)
from finetuning.models.base import CommonFineTuneConfig, InMemoryFinetuneLogger
from finetuning.models.registry import FoundationModelSpec
from finetuning.models.tabpfn.model import TabPFNAdapter
from finetuning.models.tabpfn.vendor import TabPFNBindings
from finetuning.training import fit as fit_runner
from finetuning.training import history as history_runner
from finetuning.training import resource_policy as policy_runner
from finetuning.training.in_memory import InMemoryFitController
from finetuning.training.runtime_probe import _FitRuntimeProbe
from tests.support.model_finetune_support import (
    FakeTorchTensor,
    _tabpfn_test_spec,
)


def test_fit_with_oom_retry_can_be_enabled_for_tabpfn_context(tmp_path: Path, monkeypatch) -> None:
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
                raise RuntimeError("torch.cuda.OutOfMemoryError: CUDA out of memory")
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
        spec=SimpleNamespace(model_id="tabpfn_v2_5"),
        config={"foundation_model": {"oom_retry_context_sizes": [5000, 2500]}},
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
    assert result.oom_retry["enabled"] is True
    assert result.oom_retry["candidate_context_sizes"] == [10000, 5000, 2500]
    assert result.oom_retry["retry_count"] == 1
    assert not (output / "partial.ckpt").exists()

def test_estimator_fallback_retries_tabpfn_wide_feature_count() -> None:
    class FakeAdapter:
        def minimum_required_estimators(self, feature_count: int) -> int:
            return 4 if feature_count > 500 else 1

    data = SplitEvaluationData(
        task_type="binary",
        problem_type="binary",
        metric_name="roc_auc",
        train_X=pd.DataFrame(np.zeros((2, 501), dtype=np.float32)),
        train_y=np.array([0, 1]),
        val_context_X=pd.DataFrame(np.zeros((2, 501), dtype=np.float32)),
        val_context_y=np.array([0, 1]),
        val_X=pd.DataFrame(np.zeros((2, 501), dtype=np.float32)),
        val_y=np.array([0, 1]),
        test_context_X=pd.DataFrame(np.zeros((2, 501), dtype=np.float32)),
        test_context_y=np.array([0, 1]),
        test_X=pd.DataFrame(np.zeros((2, 501), dtype=np.float32)),
        test_y=np.array([0, 1]),
        metadata={},
    )

    effective, metadata = policy_runner._apply_estimator_fallback_policy(
        adapter=FakeAdapter(),
        data=data,
        spec=SimpleNamespace(model_id="tabpfn_v2_5"),
        config={},
        common_config=CommonFineTuneConfig(
            n_estimators_finetune=2,
            n_estimators_validation=2,
            n_estimators_final_inference=8,
        ),
    )

    assert effective.n_estimators_finetune == 4
    assert effective.n_estimators_validation == 4
    assert effective.n_estimators_final_inference == 8
    assert metadata["estimator_fallback_policy"] == "estimator_retry"
    assert metadata["estimator_fallback_triggered"] is True
    assert metadata["required_min_n_estimators"] == 4
    assert metadata["configured_n_estimators"]["n_estimators_finetune"] == 2
    assert metadata["effective_n_estimators"]["n_estimators_finetune"] == 4

def test_estimator_fallback_strict_fails_loudly_for_tabpfn_wide_feature_count() -> None:
    class FakeAdapter:
        def minimum_required_estimators(self, feature_count: int) -> int:
            return 4 if feature_count > 500 else 1

    data = SplitEvaluationData(
        task_type="binary",
        problem_type="binary",
        metric_name="roc_auc",
        train_X=pd.DataFrame(np.zeros((2, 501), dtype=np.float32)),
        train_y=np.array([0, 1]),
        val_context_X=pd.DataFrame(np.zeros((2, 501), dtype=np.float32)),
        val_context_y=np.array([0, 1]),
        val_X=pd.DataFrame(np.zeros((2, 501), dtype=np.float32)),
        val_y=np.array([0, 1]),
        test_context_X=pd.DataFrame(np.zeros((2, 501), dtype=np.float32)),
        test_context_y=np.array([0, 1]),
        test_X=pd.DataFrame(np.zeros((2, 501), dtype=np.float32)),
        test_y=np.array([0, 1]),
        metadata={},
    )

    with pytest.raises(SystemExit, match="requires at least 4 estimator"):
        policy_runner._apply_estimator_fallback_policy(
            adapter=FakeAdapter(),
            data=data,
            spec=SimpleNamespace(model_id="tabpfn_v2_5"),
            config={"foundation_model": {"estimator_fallback_policy": "strict"}},
            common_config=CommonFineTuneConfig(n_estimators_finetune=2),
        )

def test_tabpfn_adapter_passes_feature_schema_categorical_indices(tmp_path: Path) -> None:
    class FakeTabPFNEstimator:
        def __init__(self, **kwargs) -> None:
            self.kwargs = dict(kwargs)

    bindings = TabPFNBindings(
        classifier=FakeTabPFNEstimator,
        regressor=FakeTabPFNEstimator,
        finetuned_classifier=FakeTabPFNEstimator,
        finetuned_regressor=FakeTabPFNEstimator,
        clone_model_for_evaluation=lambda *_args, **_kwargs: None,
        save_model=lambda *_args, **_kwargs: None,
    )
    adapter = TabPFNAdapter(_tabpfn_test_spec(tmp_path), bindings=bindings)
    adapter.set_feature_schema({"categorical_feature_indices": [0, 2]})
    adapter.set_common_config(CommonFineTuneConfig(ignore_pretraining_limits=True))

    zero_shot = adapter.resolve_zero_shot_estimator("binary", seed=0, device="cpu", n_estimators_inference=2)
    finetuner = adapter.resolve_finetune_estimator("binary", CommonFineTuneConfig(), seed=0, device="cpu")
    loaded = adapter.load_checkpoint_for_inference(
        tmp_path / "model.ckpt",
        "binary",
        "cpu",
        2,
        seed=0,
    )

    assert zero_shot.kwargs["categorical_features_indices"] == [0, 2]
    assert zero_shot.kwargs["ignore_pretraining_limits"] is True
    assert finetuner.kwargs["extra_classifier_kwargs"]["categorical_features_indices"] == [0, 2]
    assert finetuner.kwargs["extra_classifier_kwargs"]["ignore_pretraining_limits"] is True
    assert loaded.kwargs["categorical_features_indices"] == [0, 2]
    assert loaded.kwargs["ignore_pretraining_limits"] is True

def test_tabpfn_native_validation_encodes_pandas_categories_for_fit_and_predict(tmp_path: Path) -> None:
    class FakeEstimator:
        def fit(self, X, y):
            self.fit_X = X
            self.fit_y = y
            return self

        def predict_proba(self, X):
            self.predict_X = X
            return np.repeat([[0.4, 0.6]], len(X), axis=0)

    adapter = TabPFNAdapter(_tabpfn_test_spec(tmp_path))
    adapter.set_preprocessing_protocol(PREPROCESSING_NATIVE)
    adapter.set_feature_schema({"categorical_feature_indices": [0]})
    context_X = pd.DataFrame({"cat": pd.Series(["b", "a", None], dtype="category"), "x": [1.0, 2.0, 3.0]})
    query_X = pd.DataFrame({"cat": pd.Series(["a", "c", None], dtype="category"), "x": [4.0, 5.0, 6.0]})
    estimator = FakeEstimator()

    adapter.fit_inference_context(estimator, context_X, np.array([0, 1, 0]))
    adapter.predict_for_metrics(estimator, query_X, "binary")

    assert isinstance(context_X["cat"].dtype, pd.CategoricalDtype)
    assert estimator.fit_X is not context_X
    assert estimator.fit_X["cat"].dtype == np.float64
    np.testing.assert_allclose(estimator.fit_X["cat"].to_numpy()[:2], [1.0, 0.0])
    assert np.isnan(estimator.fit_X.loc[2, "cat"])
    np.testing.assert_allclose(estimator.predict_X["cat"].to_numpy()[:1], [0.0])
    assert np.isnan(estimator.predict_X.loc[1, "cat"])
    assert np.isnan(estimator.predict_X.loc[2, "cat"])

def test_tabpfn_shared_finite_numeric_does_not_use_native_category_encoder(tmp_path: Path) -> None:
    class FakeEstimator:
        def fit(self, X, y):
            self.fit_X = X
            self.fit_y = y
            return self

        def predict_proba(self, X):
            self.predict_X = X
            return np.repeat([[0.4, 0.6]], len(X), axis=0)

    adapter = TabPFNAdapter(_tabpfn_test_spec(tmp_path))
    adapter.set_preprocessing_protocol(PREPROCESSING_SHARED)
    adapter.set_feature_schema({"categorical_feature_indices": [0]})
    assert adapter._extra_tabpfn_kwargs() == {}
    context_X = pd.DataFrame({"cat": pd.Series(["b", "a"], dtype="category"), "x": [1.0, 2.0]})
    query_X = pd.DataFrame({"cat": pd.Series(["a"], dtype="category"), "x": [3.0]})
    estimator = FakeEstimator()

    adapter.fit_inference_context(estimator, context_X, np.array([0, 1]))
    adapter.predict_for_metrics(estimator, query_X, "binary")

    assert estimator.fit_X is context_X
    assert estimator.predict_X is query_X
    assert isinstance(estimator.fit_X["cat"].dtype, pd.CategoricalDtype)
    assert isinstance(estimator.predict_X["cat"].dtype, pd.CategoricalDtype)

def test_tabpfn_adapter_snapshots_evaluates_and_exports_in_memory_state(tmp_path: Path) -> None:
    class FakeTorchModel:
        def __init__(self, value: int) -> None:
            self.value = value

        def state_dict(self) -> dict[str, FakeTorchTensor]:
            return {"weight": FakeTorchTensor(self.value)}

        def load_state_dict(self, state_dict: dict[str, FakeTorchTensor]) -> None:
            self.value = state_dict["weight"].value

    class FakeEvalEstimator:
        def __init__(self, **kwargs) -> None:
            self.kwargs = dict(kwargs)

    inner = SimpleNamespace(
        models_=[FakeTorchModel(1)],
        configs_=["config"],
        inference_config_={"setting": True},
    )
    wrapper = SimpleNamespace(finetuned_estimator_=inner)
    cloned: dict[str, Any] = {}
    saved: dict[str, Any] = {}

    def fake_clone_model_for_evaluation(original_model, eval_init_args, model_class):
        cloned["original"] = original_model
        cloned["eval_init_args"] = dict(eval_init_args)
        return model_class(model_path="in-memory", **eval_init_args)

    def fake_save_tabpfn_model(*, model, save_path: Path, additional_fields: dict[str, Any]) -> None:
        saved["value"] = model.models_[0].value
        saved["additional_fields"] = dict(additional_fields)
        save_path.write_text("checkpoint", encoding="utf-8")

    adapter = TabPFNAdapter(
        _tabpfn_test_spec(tmp_path),
        bindings=TabPFNBindings(
            classifier=FakeEvalEstimator,
            regressor=FakeEvalEstimator,
            finetuned_classifier=FakeEvalEstimator,
            finetuned_regressor=FakeEvalEstimator,
            clone_model_for_evaluation=fake_clone_model_for_evaluation,
            save_model=fake_save_tabpfn_model,
        ),
    )
    adapter.set_common_config(CommonFineTuneConfig(ignore_pretraining_limits=True))

    best_state = adapter.clone_finetuned_state(wrapper)
    inner.models_[0].value = 3
    eval_estimator = adapter.build_in_memory_inference_estimator(wrapper, "binary", "cpu", 7)
    exported = adapter.export_finetuned_state(
        wrapper,
        best_state,
        tmp_path,
        epoch=4,
        metric=0.9,
        checkpoint_mode="best",
    )

    assert eval_estimator.kwargs["n_estimators"] == 7
    assert eval_estimator.kwargs["ignore_pretraining_limits"] is True
    assert cloned["original"] is inner
    assert exported == tmp_path / "best.pth"
    assert exported.read_text(encoding="utf-8") == "checkpoint"
    assert saved["value"] == 1
    assert saved["additional_fields"]["epoch"] == 4
    assert saved["additional_fields"]["best_metric"] == 0.9
    assert saved["additional_fields"]["checkpoint_export_mode"] == "best_in_memory_state"
    assert inner.models_[0].value == 3

def test_tabpfn_export_requires_best_checkpoint_but_falls_back_for_final(tmp_path: Path) -> None:
    class FakeTabPFNFinetuner:
        def __init__(self) -> None:
            self.finetuned_estimator_ = SimpleNamespace(name="trained-tabpfn")

        def fit(self, X, y, *, X_val, y_val, output_dir: Path):
            self.X = X
            self.y = y
            self.X_val = X_val
            self.y_val = y_val
            self.output_dir = output_dir
            return self

    saved: dict[str, Any] = {}

    def fake_save_tabpfn_model(*, model, save_path: Path, additional_fields: dict[str, Any]) -> None:
        saved["model"] = model
        saved["additional_fields"] = additional_fields
        save_path.write_text("saved", encoding="utf-8")

    spec = FoundationModelSpec(
        model_id="tabpfn_v2_5",
        display_name="TabPFNv2.5",
        adapter_cls_path="finetuning.models.tabpfn.model.TabPFNAdapter",
        supports_classification=True,
        supports_regression=True,
        classifier_checkpoint="models/tabpfn-v2.5-classifier.ckpt",
        regressor_checkpoint="models/tabpfn-v2.5-regressor.ckpt",
        source_path=tmp_path,
        version_payload={"source_available": True},
    )
    adapter = TabPFNAdapter(
        spec,
        bindings=TabPFNBindings(
            classifier=FakeTabPFNFinetuner,
            regressor=FakeTabPFNFinetuner,
            finetuned_classifier=FakeTabPFNFinetuner,
            finetuned_regressor=FakeTabPFNFinetuner,
            clone_model_for_evaluation=lambda *_args, **_kwargs: None,
            save_model=fake_save_tabpfn_model,
        ),
    )
    finetuner = FakeTabPFNFinetuner()

    adapter.fit_finetuned(
        finetuner,
        pd.DataFrame({"x": [0.0, 1.0]}),
        np.array([0, 1]),
        pd.DataFrame({"x": [2.0, 3.0]}),
        np.array([1, 0]),
        tmp_path,
    )
    with pytest.raises(FileNotFoundError, match="best-validation checkpoint"):
        adapter.export_checkpoint(tmp_path, checkpoint_mode="best")
    exported = adapter.export_checkpoint(tmp_path, checkpoint_mode="final")

    assert exported == tmp_path / "model.ckpt"
    assert exported.read_text(encoding="utf-8") == "saved"
    assert saved["model"] is finetuner.finetuned_estimator_
    assert saved["additional_fields"]["checkpoint_export_mode"] == "final_in_memory_fallback"

def test_fit_runtime_probe_streams_tabpfn_evaluate_model_callbacks() -> None:
    class FakeInnerEstimator:
        classes_ = np.array([0, 1])

        def __init__(self, kind: str) -> None:
            self.kind = kind

    class FakeModel:
        def __init__(self, owner) -> None:
            self.owner = owner
            self.training = True

        def state_dict(self) -> dict[str, str]:
            return {"state": self.owner.current_state}

        def eval(self) -> None:
            self.training = False

        def train(self, enabled: bool = True) -> None:
            self.training = enabled

    class FakeEstimator:
        def __init__(self) -> None:
            self.current_state = "baseline"
            self.current_primary = 0.5
            self.model_ = FakeModel(self)

        def _metric_improved(self, current: float, best: float) -> bool:
            return current > best

        def _evaluate_model(self, _eval_config, _train_X, _train_y, _val_X, _val_y):
            return SimpleNamespace(primary=self.current_primary)

    class FakeAdapter:
        def clone_finetuned_state(self, estimator):
            return dict(estimator.model_.state_dict())

        def apply_finetuned_state(self, estimator, state) -> None:
            estimator.current_state = state["state"]

        def build_in_memory_inference_estimator(self, estimator, _task_type, _device, _n_estimators):
            return FakeInnerEstimator(estimator.current_state)

        def fit_inference_context(self, estimator, _context_X, _context_y):
            return estimator

        def predict_for_metrics(self, estimator, query_X, _task_type):
            if estimator.kind == "epoch2":
                return np.tile(np.array([[0.55, 0.45]]), (len(query_X), 1))
            return np.array([[0.8, 0.2], [0.2, 0.8]])[: len(query_X)]

    data = SplitEvaluationData(
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
        metadata={"data_backend": "test"},
    )
    controller = InMemoryFitController(
        adapter=FakeAdapter(),
        data=data,
        common_config=CommonFineTuneConfig(epochs=2),
        device="cpu",
        checkpoint_mode="best",
        test_eval_interval=0,
        evaluate_estimator_on_split=fit_runner._evaluate_estimator_on_split,
        log_eval_timing=lambda _row: None,
    )
    estimator = FakeEstimator()
    _FitRuntimeProbe(controller=controller).install(estimator)

    estimator._evaluate_model({"device": "cpu"}, data.train_X, data.train_y, data.val_X, data.val_y)
    estimator.current_state = "epoch1"
    estimator.current_primary = 0.8
    estimator._evaluate_model({"device": "cpu"}, data.train_X, data.train_y, data.val_X, data.val_y)
    estimator.current_state = "epoch2"
    estimator.current_primary = 0.7
    estimator._evaluate_model({"device": "cpu"}, data.train_X, data.train_y, data.val_X, data.val_y)

    assert [row["epoch"] for row in controller.validation_rows] == [1, 2]
    assert controller.best_epoch == 1
    assert controller.final_epoch == 2
    assert controller.best_eval_count == 0
    assert controller.best_metrics is not None
    assert set(controller.best_metrics) == {"val"}

    controller.after_fit(estimator, validation_device="cpu")

    assert controller.best_eval_count == 1
    assert controller.best_metrics is not None
    assert controller.best_metrics["test"]["score"] == 1.0

def test_training_history_rows_accept_tabpfn_display_metric_keys() -> None:
    logger_obj = InMemoryFinetuneLogger()
    logger_obj.log_epoch({"train/epoch": 0, "val/MSE": 4.0, "train/mean_loss": 1.5}, step=1)
    data = SplitEvaluationData(
        task_type="regression",
        problem_type="regression",
        metric_name="rmse",
        train_X=pd.DataFrame({"x": [0, 1]}),
        train_y=np.array([0.0, 1.0]),
        val_context_X=pd.DataFrame({"x": [0, 1]}),
        val_context_y=np.array([0.0, 1.0]),
        val_X=pd.DataFrame({"x": [2]}),
        val_y=np.array([2.0]),
        test_context_X=pd.DataFrame({"x": [0, 1]}),
        test_context_y=np.array([0.0, 1.0]),
        test_X=pd.DataFrame({"x": [3]}),
        test_y=np.array([3.0]),
        metadata={},
    )

    rows = history_runner._training_history_rows(logger_obj=logger_obj, data=data, baseline_metrics={})

    assert rows[0]["train/val/score"] == -2.0
    assert rows[0]["core/train/val/mse"] == 4.0
