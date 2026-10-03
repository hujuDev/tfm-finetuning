"""Tests for model finetune: evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tests.support.model_finetune_support import (
    preprocessing_runner,
    run_data_runner,
)
from tfm_finetuning.finetuning import model_finetune as runner
from tfm_finetuning.finetuning.datasets import loading as data_runner
from tfm_finetuning.finetuning.datasets.loading import (
    TEST_CONTEXT_POLICY_BALANCED_ORIGINAL_MAJORITY,
    TEST_CONTEXT_POLICY_FULL_ORIGINAL,
    TEST_CONTEXT_POLICY_TRAIN_ONLY,
    TEST_CONTEXT_POLICY_TRAIN_PLUS_VALIDATION,
    SplitEvaluationData,
)
from tfm_finetuning.finetuning.datasets.preprocessing_protocol import (
    PREPROCESSING_NATIVE,
)
from tfm_finetuning.finetuning.evaluation import core as evaluation_runner
from tfm_finetuning.finetuning.models.base import CommonFineTuneConfig
from tfm_finetuning.finetuning.training import fit as fit_runner
from tfm_finetuning.finetuning.training import history as history_runner
from tfm_finetuning.finetuning.training import resource_policy as policy_runner
from tfm_finetuning.finetuning.training.in_memory import InMemoryFitController
from tfm_finetuning.utils.config import get_nested


def test_evaluation_splits_default_to_test_and_allow_validation_only_calibration() -> None:
    assert runner._resolve_evaluation_splits({}) == ("val", "test")
    assert runner._resolve_evaluation_splits({"finetune": {"evaluation_splits": ["val"]}}) == ("val",)

    with pytest.raises(ValueError, match="may contain only val and test"):
        runner._resolve_evaluation_splits({"finetune": {"evaluation_splits": ["train"]}})


def test_evaluate_estimator_uses_fitted_classes_for_sparse_multiclass_query() -> None:
    class FakeEstimator:
        classes_ = np.array(["1", "2", "3", "5", "U"])

    class FakeAdapter:
        def fit_inference_context(self, estimator, _context_X, _context_y):
            return estimator

        def predict_for_metrics(self, _estimator, query_X, _task_type):
            probs = np.array(
                [
                    [0.05, 0.80, 0.05, 0.05, 0.05],
                    [0.05, 0.05, 0.80, 0.05, 0.05],
                    [0.05, 0.05, 0.05, 0.80, 0.05],
                    [0.05, 0.05, 0.05, 0.05, 0.80],
                ]
            )
            return probs[: len(query_X)]

    payload = fit_runner._evaluate_estimator_on_split(
        adapter=FakeAdapter(),
        estimator=FakeEstimator(),
        context_X=pd.DataFrame({"x": [0, 1, 2, 3, 4]}),
        context_y=np.array(["1", "2", "3", "5", "U"]),
        query_X=pd.DataFrame({"x": [10, 11, 12, 13]}),
        query_y=np.array(["2", "3", "5", "U"]),
        task_type="multiclass",
    )

    assert payload["accuracy"] == 1.0
    assert payload["log_loss"] >= 0.0
    assert payload["score"] == -payload["log_loss"]


def test_evaluate_estimator_penalizes_query_class_missing_from_fitted_context() -> None:
    class FakeEstimator:
        classes_ = np.array(
            [
                "Cottonwood_Willow",
                "Douglas_fir",
                "Krummholz",
                "Lodgepole_Pine",
                "Ponderosa_Pine",
                "Spruce_Fir",
            ]
        )

    class FakeAdapter:
        def fit_inference_context(self, estimator, _context_X, _context_y):
            return estimator

        def predict_for_metrics(self, _estimator, query_X, _task_type):
            return np.full((len(query_X), 6), 1.0 / 6.0)

    payload = fit_runner._evaluate_estimator_on_split(
        adapter=FakeAdapter(),
        estimator=FakeEstimator(),
        context_X=pd.DataFrame({"x": range(6)}),
        context_y=FakeEstimator.classes_,
        query_X=pd.DataFrame({"x": range(10, 17)}),
        query_y=np.array(["Aspen", *FakeEstimator.classes_]),
        task_type="multiclass",
    )

    assert np.isfinite(payload["log_loss"])
    assert payload["log_loss"] > 5.0
    assert payload["classification_report"]["Aspen"]["support"] == 1.0


def test_resolve_oom_retry_context_sizes_can_disable_retry() -> None:
    common_config = CommonFineTuneConfig(context_size=10000)

    assert policy_runner._resolve_oom_retry_context_sizes(
        {"finetune": {"oom_retry_context_sizes": "off"}},
        model_id="tabicl_v2",
        common_config=common_config,
    ) == (10000,)


def test_apply_model_row_limits_caps_train_and_context_rows() -> None:
    data = SplitEvaluationData(
        task_type="binary",
        problem_type="binary",
        metric_name="roc_auc",
        train_X=pd.DataFrame({"x": range(100)}),
        train_y=np.asarray([0, 1] * 50),
        val_context_X=pd.DataFrame({"x": range(90)}),
        val_context_y=np.asarray([0, 1] * 45),
        val_X=pd.DataFrame({"x": range(40)}),
        val_y=np.asarray([0, 1] * 20),
        test_context_X=pd.DataFrame({"x": range(80)}),
        test_context_y=np.asarray([0, 1] * 40),
        test_X=pd.DataFrame({"x": range(30)}),
        test_y=np.asarray([0, 1] * 15),
        metadata={},
    )

    limited = data_runner._apply_model_row_limits(
        data,
        config={
            "preprocess": {"clamp_train_pool": {"max_size": 10}},
            "finetune": {
                "inference_context_max_rows": 12,
                "eval_query_max_rows": 7,
            },
        },
        seed=123,
    )
    repeated = data_runner._apply_model_row_limits(
        data,
        config={
            "preprocess": {"clamp_train_pool": {"max_size": 10}},
            "finetune": {
                "inference_context_max_rows": 12,
                "eval_query_max_rows": 7,
            },
        },
        seed=123,
    )

    assert len(limited.train_X) == 10
    assert len(limited.val_context_X) == 12
    assert len(limited.test_context_X) == 12
    assert len(limited.val_X) == 7
    assert len(limited.test_X) == 7
    assert limited.train_X["x"].tolist() == repeated.train_X["x"].tolist()
    assert limited.metadata["row_limits"]["raw_rows"]["train"] == 100
    assert limited.metadata["row_limits"]["effective_rows"] == {
        "train": 10,
        "val_context": 12,
        "val_query": 7,
        "test_context": 12,
        "test_query": 7,
    }


def test_finetune_baseline_only_runs_test_context_comparison_once_per_policy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeEstimator:
        classes_ = np.array([0, 1])

        def __init__(self) -> None:
            self.context_rows = 0

    class FakeAdapter:
        def __init__(self, _spec) -> None:
            pass

        def resolve_zero_shot_estimator(self, _task_type, _seed, _device, _n_estimators):
            return FakeEstimator()

        def fit_inference_context(self, estimator, context_X, _context_y):
            estimator.context_rows = len(context_X)
            return estimator

        def predict_for_metrics(self, estimator, query_X, _task_type):
            good = np.array([[0.1, 0.9], [0.9, 0.1]])
            bad = np.array([[0.9, 0.1], [0.1, 0.9]])
            return (good if estimator.context_rows > 2 else bad)[: len(query_X)]

        def resolved_model_params(self, _common_config):
            return {}

    def fake_load_split_evaluation_data(*, config, repo_root):
        _ = repo_root
        includes_validation = bool(get_nested(config, "finetune.test_context_includes_validation_data"))
        test_context_len = 3 if includes_validation else 2
        return SplitEvaluationData(
            task_type="binary",
            problem_type="binary",
            metric_name="roc_auc",
            train_X=pd.DataFrame({"x": [0, 1]}),
            train_y=np.array([0, 1]),
            val_context_X=pd.DataFrame({"x": [0, 1]}),
            val_context_y=np.array([0, 1]),
            val_X=pd.DataFrame({"x": [10, 11]}),
            val_y=np.array([1, 0]),
            test_context_X=pd.DataFrame({"x": range(test_context_len)}),
            test_context_y=np.array([0, 1, 0] if includes_validation else [0, 1]),
            test_X=pd.DataFrame({"x": [20, 21]}),
            test_y=np.array([1, 0]),
            metadata={
                "data_backend": "test",
                "test_context_includes_validation_data": includes_validation,
                "test_context_source": "outer_train" if includes_validation else "inner_train",
            },
        )

    fake_spec = SimpleNamespace(
        model_id="tabpfn_v2",
        display_name="TabPFNv2",
        version_payload={},
        adapter_cls=FakeAdapter,
        ensure_source_available=lambda: None,
    )
    monkeypatch.setattr(runner, "get_model_spec", lambda *_args, **_kwargs: fake_spec)
    monkeypatch.setattr(run_data_runner, "load_split_evaluation_data", fake_load_split_evaluation_data)
    monkeypatch.setattr(preprocessing_runner, "load_split_evaluation_data", fake_load_split_evaluation_data)

    report = runner.main(
        {
            "paths": {"results_root": str(tmp_path / "results")},
            "seed": 0,
            "finetune": {
                "model_id": "tabpfn_v2",
                "baseline_only": True,
                "test_context_includes_validation_data": False,
                "test_context_comparison": {"enabled": True},
            },
            "data": {"path": ":data/tabarena/churn", "split": "fold0_repeat0_sample0"},
            "wandb": {"mode": "disabled"},
        },
        tmp_path / "generated" / "comparison-smoke",
        force=True,
    )

    assert isinstance(report, dict)
    comparison = report["test_context_comparison"]
    assert comparison["primary_policy"] == TEST_CONTEXT_POLICY_TRAIN_ONLY
    assert comparison["alternate_policy"] == TEST_CONTEXT_POLICY_TRAIN_PLUS_VALIDATION
    assert comparison["policies"][TEST_CONTEXT_POLICY_TRAIN_ONLY]["test_context_rows"] == 2
    assert comparison["policies"][TEST_CONTEXT_POLICY_TRAIN_PLUS_VALIDATION]["test_context_rows"] == 3
    baseline = comparison["baseline_metrics"]
    assert baseline[TEST_CONTEXT_POLICY_TRAIN_ONLY]["test"]["score"] == pytest.approx(0.0)
    assert baseline[TEST_CONTEXT_POLICY_TRAIN_PLUS_VALIDATION]["test"]["score"] == pytest.approx(1.0)
    assert comparison["deltas"]["train_plus_validation_minus_train_only"]["baseline"]["test"]["score"] == (
        pytest.approx(1.0)
    )
    eval_rows = report["runtime"]["eval_splits"]
    comparison_rows = [row for row in eval_rows if row["phase"] == "zero_shot_test_context_comparison"]
    assert len(comparison_rows) == 1
    assert comparison_rows[0]["split"] == "test"
    assert comparison_rows[0]["context_rows"] == 3


def test_in_memory_controller_evaluates_test_only_after_fit_when_periodic_off() -> None:
    """Checkpoint search stays validation-only before one post-fit selected-state test."""

    class FakeInnerEstimator:
        classes_ = np.array([0, 1])

        def __init__(self, kind: str) -> None:
            self.kind = kind

    class FakeEstimator:
        def __init__(self) -> None:
            self.current_state = "epoch1"
            self.model_ = SimpleNamespace(
                state_dict=lambda: {"state": self.current_state},
                eval=lambda: None,
                train=lambda *_args: None,
            )

        def _metric_improved(self, current: float, best: float) -> bool:
            return current > best

    class FakeAdapter:
        def clone_finetuned_state(self, estimator) -> dict[str, str]:
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
    timing_rows: list[dict[str, object]] = []
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
    controller.bind_timing_rows(timing_rows)
    estimator = FakeEstimator()

    controller.after_validation(estimator, epoch=1, primary=0.8, validation_device="cpu")
    estimator.current_state = "epoch2"
    controller.after_validation(estimator, epoch=2, primary=0.7, validation_device="cpu")

    assert controller.best_epoch == 1
    assert controller.final_epoch == 2
    assert controller.periodic_rows == []
    assert [row["epoch"] for row in controller.validation_rows] == [1, 2]
    assert controller.best_metrics is not None
    assert set(controller.best_metrics) == {"val"}
    assert {row["phase"] for row in timing_rows} == {"validation_in_memory"}

    controller.after_fit(estimator, validation_device="cpu")

    assert controller.best_metrics is not None
    assert controller.best_metrics["test"]["score"] == 1.0
    assert {row["phase"] for row in timing_rows} == {"validation_in_memory", "selected_checkpoint"}
    assert {(row["phase"], row["split"]) for row in timing_rows} == {
        ("validation_in_memory", "val"),
        ("selected_checkpoint", "val"),
        ("selected_checkpoint", "test"),
    }


def test_in_memory_controller_streams_periodic_eval_during_validation() -> None:
    class FakeInnerEstimator:
        classes_ = np.array([0, 1])

        def __init__(self, kind: str) -> None:
            self.kind = kind

    class FakeEstimator:
        def __init__(self) -> None:
            self.current_state = "epoch1"

    class FakeAdapter:
        def clone_finetuned_state(self, estimator):
            return {"state": estimator.current_state}

        def apply_finetuned_state(self, estimator, state):
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
    timing_rows: list[dict[str, object]] = []
    controller = InMemoryFitController(
        adapter=FakeAdapter(),
        data=data,
        common_config=CommonFineTuneConfig(epochs=2),
        device="cpu",
        checkpoint_mode="best",
        test_eval_interval=1,
        evaluate_estimator_on_split=fit_runner._evaluate_estimator_on_split,
        log_eval_timing=lambda _row: None,
    )
    controller.bind_timing_rows(timing_rows)
    estimator = FakeEstimator()

    controller.after_validation(estimator, epoch=1, primary=None, validation_device="cpu")
    estimator.current_state = "epoch2"
    controller.after_validation(estimator, epoch=2, primary=None, validation_device="cpu")

    assert [row["epoch"] for row in controller.periodic_rows] == [1, 2]
    assert controller.periodic_rows[0]["metrics"]["test"]["score"] == 1.0
    assert controller.periodic_rows[1]["metrics"]["test"]["score"] == 0.5
    assert controller.validation_rows == []
    assert controller.best_metrics is not None
    assert controller.best_metrics["test"]["score"] == 1.0
    assert controller.best_epoch == 1
    assert controller.final_epoch == 2
    assert {row["phase"] for row in timing_rows} == {"periodic_in_memory"}
    assert {(row["epoch"], row["split"]) for row in timing_rows} == {
        (1, "val"),
        (1, "test"),
        (2, "val"),
        (2, "test"),
    }


@pytest.mark.parametrize("model_id", ["tabpfn_v2_5", "mitra"])
def test_finetune_main_writes_disk_checkpoint_history_contract(
    tmp_path: Path,
    monkeypatch,
    model_id: str,
) -> None:
    """Non-in-memory adapters should still expose selected and final checkpoint test rows."""

    class DiskCheckpointAdapter:
        def __init__(self, spec) -> None:
            self.model_id = spec.model_id

        def resolved_model_params(self, common_config: CommonFineTuneConfig) -> dict[str, object]:
            return {"epochs": common_config.epochs}

        def resolve_zero_shot_estimator(self, *_args, **_kwargs):
            return SimpleNamespace(kind="zero")

        def resolve_finetune_estimator(self, *_args, **_kwargs):
            return SimpleNamespace(kind="train")

        def fit_finetuned(self, estimator, _train_X, _train_y, _val_X, _val_y, output_dir: Path):
            (output_dir / "best.ckpt").write_text("epoch1", encoding="utf-8")
            (output_dir / "epoch1.ckpt").write_text("epoch1", encoding="utf-8")
            (output_dir / "epoch2.ckpt").write_text("epoch2", encoding="utf-8")
            estimator.is_fitted_ = True
            return estimator

        def load_checkpoint_for_inference(self, checkpoint_path: Path, *_args, **_kwargs):
            return SimpleNamespace(kind=checkpoint_path.read_text(encoding="utf-8"))

        def fit_inference_context(self, estimator, _context_X, _context_y):
            return estimator

        def predict_for_metrics(self, estimator, query_X, _task_type):
            if estimator.kind == "epoch1":
                return np.array([[0.8, 0.2], [0.2, 0.8]])[: len(query_X)]
            if estimator.kind == "epoch2":
                return np.tile(np.array([[0.55, 0.45]]), (len(query_X), 1))
            return np.tile(np.array([[0.7, 0.3]]), (len(query_X), 1))

    fake_spec = SimpleNamespace(
        model_id=model_id,
        display_name=model_id,
        version_payload={},
        adapter_cls=DiskCheckpointAdapter,
        ensure_source_available=lambda: None,
    )
    monkeypatch.setattr(runner, "get_model_spec", lambda *_args, **_kwargs: fake_spec)
    monkeypatch.setattr(
        run_data_runner,
        "load_split_evaluation_data",
        lambda **_kwargs: SplitEvaluationData(
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
        ),
    )
    monkeypatch.setenv("TABPFN_USE_ARTIFACTS", "1")
    monkeypatch.setenv("TABPFN_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))

    config = {
        "paths": {"results_root": str(tmp_path / "results")},
        "seed": 0,
        "finetune": {
            "model_id": model_id,
            "epochs": 2,
            "test_eval_interval_epochs": 1,
            "final_eval_checkpoint": "final",
            "save_best_val_checkpoint": True,
            "preprocessing_protocol": PREPROCESSING_NATIVE,
        },
        "data": {"path": ":data/tabarena/churn", "split": "fold0_repeat0_sample0"},
        "wandb": {"mode": "disabled"},
    }

    runner.main(config, tmp_path / "generated" / model_id, force=True)

    artifact_dir = tmp_path / "generated" / model_id
    report_payload = json.loads((artifact_dir / "finetune_report.json").read_text(encoding="utf-8"))
    assert (artifact_dir / "best.ckpt").exists()
    assert report_payload["finetune"]["model_id"] == model_id
    assert report_payload["checkpoint_available"] is True
    assert report_payload["checkpoint_retention"]["kept_files"] == 1
    assert report_payload["selected_checkpoint_path"] == str(artifact_dir / "best.ckpt")
    assert report_payload["checkpoint_policy"]["save_best_val_checkpoint"] is True
    assert report_payload["evaluation"]["selected_epoch"] == 1
    assert report_payload["evaluation"]["final_epoch"] == 2
    assert report_payload["selected_checkpoint_metrics"]["test"]["score"] == 1.0
    assert report_payload["final_checkpoint_metrics"]["test"]["score"] == 0.5
    epoch_rows = {(row["epoch"], row["split"], row["eval_source"]) for row in report_payload["epoch_evaluations"]}
    assert (1, "test", "periodic_checkpoint") in epoch_rows
    assert (2, "test", "final_checkpoint") in epoch_rows
    phase_names = {row["phase"] for row in report_payload["runtime"]["phases"]}
    assert {
        "row_limits",
        "periodic_validation_eval",
        "selected_checkpoint_eval",
        "final_checkpoint_eval",
    }.issubset(phase_names)


def test_epoch_evaluation_rows_require_regression_score_to_be_negative_rmse() -> None:
    rows = history_runner._epoch_evaluation_rows(
        periodic_rows=[
            {
                "epoch": 1,
                "eval_source": "in_memory",
                "metrics": {
                    "val": {
                        "score": -2.0,
                        "raw_metric_name": "rmse",
                        "raw_metric_value": 2.0,
                        "rmse": 2.0,
                        "mse": 4.0,
                    },
                    "test": {
                        "score": -3.0,
                        "raw_metric_name": "rmse",
                        "raw_metric_value": 3.0,
                        "rmse": 3.0,
                        "mse": 9.0,
                    },
                },
            }
        ],
        score_metric="rmse",
        timing_rows=[
            {"phase": "periodic_in_memory", "epoch": 1, "split": "val", "n_estimators": 2, "context_rows": 10},
            {"phase": "periodic_in_memory", "epoch": 1, "split": "test", "n_estimators": 8, "context_rows": 12},
        ],
    )

    assert [(row["split"], row["score"], row["raw_metric_value"]) for row in rows] == [
        ("val", -2.0, 2.0),
        ("test", -3.0, 3.0),
    ]
    assert rows[0]["n_estimators"] == 2
    assert rows[0]["context_rows"] == 10


def test_epoch_evaluation_rows_reject_regression_score_logged_as_negative_mse() -> None:
    with pytest.raises(ValueError, match="inconsistent RMSE score"):
        history_runner._epoch_evaluation_rows(
            periodic_rows=[
                {
                    "epoch": 1,
                    "metrics": {
                        "val": {"score": -4.0, "raw_metric_name": "rmse", "raw_metric_value": 2.0},
                        "test": {"score": -3.0, "raw_metric_name": "rmse", "raw_metric_value": 3.0},
                    },
                }
            ],
            score_metric="rmse",
            timing_rows=[],
        )


def test_original_majority_comparison_pairs_full_training_with_balanced_icl_context(tmp_path: Path) -> None:
    data = SplitEvaluationData(
        task_type="binary",
        problem_type="binary",
        metric_name="roc_auc",
        train_X=pd.DataFrame({"x": [0, 1, 2, 3]}),
        train_y=np.array([0, 0, 0, 1]),
        val_context_X=pd.DataFrame({"x": [0, 1, 2, 3]}),
        val_context_y=np.array([0, 0, 0, 1]),
        val_X=pd.DataFrame({"x": [4, 5]}),
        val_y=np.array([0, 1]),
        test_context_X=pd.DataFrame({"x": [0, 1, 2, 3, 4, 5]}),
        test_context_y=np.array([0, 0, 0, 0, 1, 1]),
        test_X=pd.DataFrame({"x": [6, 7]}),
        test_y=np.array([0, 1]),
        metadata={"test_context_includes_validation_data": True},
    )
    config = {
        "finetune": {
            "test_context_comparison": {
                "enabled": True,
                "mode": "original_majority_downsampling",
                "target_focal_rate": 0.5,
            }
        }
    }

    primary = preprocessing_runner._mark_primary_test_context_comparison_data(data, config)
    alternate = preprocessing_runner._prepare_alternate_test_context_data(
        config=config,
        repo_root=tmp_path,
        primary_data=primary,
        seed=11,
        preprocessing_protocol=PREPROCESSING_NATIVE,
        model_id="tabpfn_v2",
    )

    assert alternate is not None
    pd.testing.assert_frame_equal(alternate.train_X, primary.train_X)
    assert len(primary.test_context_X) == 6
    assert len(alternate.test_context_X) == 4
    assert np.mean(alternate.test_context_y == 0) == pytest.approx(0.5)
    payload = evaluation_runner._build_test_context_comparison_payload(
        primary_data=primary,
        alternate_data=alternate,
        baseline_metrics={"test": {"score": 0.7}},
        alternate_baseline_metrics={"test": {"score": 0.8}},
    )
    assert payload is not None
    assert payload["primary_policy"] == TEST_CONTEXT_POLICY_FULL_ORIGINAL
    assert payload["alternate_policy"] == TEST_CONTEXT_POLICY_BALANCED_ORIGINAL_MAJORITY
    delta_name = "balanced_original_majority_downsampled_minus_full_original"
    assert payload["deltas"][delta_name]["baseline"]["test"]["score"] == pytest.approx(0.1)
