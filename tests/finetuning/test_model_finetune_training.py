"""Tests for model finetune: training."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tests.support.model_finetune_support import (
    run_data_runner,
)
from tfm_finetuning.finetuning import model_finetune as runner
from tfm_finetuning.finetuning.datasets.loading import (
    SplitEvaluationData,
)
from tfm_finetuning.finetuning.datasets.preprocessing_protocol import (
    PREPROCESSING_NATIVE,
)
from tfm_finetuning.finetuning.models.base import CommonFineTuneConfig
from tfm_finetuning.finetuning.reporting import report as reporting_runner
from tfm_finetuning.finetuning.training import checkpoints as checkpoint_runner
from tfm_finetuning.finetuning.training import history as history_runner


def test_select_checkpoint_path_prefers_latest_epoch_for_final_mode(tmp_path: Path) -> None:
    (tmp_path / "best.ckpt").write_text("best", encoding="utf-8")
    (tmp_path / "epoch1.ckpt").write_text("epoch1", encoding="utf-8")
    (tmp_path / "epoch3.ckpt").write_text("epoch3", encoding="utf-8")
    (tmp_path / "checkpoint_2K_5.pth").write_text("tabpfn-epoch5", encoding="utf-8")
    (tmp_path / "checkpoint_2K_best.pth").write_text("tabpfn-best", encoding="utf-8")

    assert (
        checkpoint_runner.select_checkpoint_path(tmp_path, checkpoint_mode="final") == tmp_path / "checkpoint_2K_5.pth"
    )
    assert checkpoint_runner.select_checkpoint_path(tmp_path, checkpoint_mode="best") == tmp_path / "best.ckpt"


def test_finetune_main_validation_only_in_memory_exports_best_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeInnerEstimator:
        classes_ = np.array([0, 1])

        def __init__(self, kind: str) -> None:
            self.kind = kind

    class FakeModel:
        def __init__(self, owner) -> None:
            self.owner = owner
            self.mode = "train"

        def state_dict(self) -> dict[str, str]:
            return {"state": self.owner.current_state}

        def eval(self) -> None:
            self.mode = "eval"

        def train(self, enabled: bool = True) -> None:
            self.mode = "train" if enabled else "eval"

    class FakeFinetuneEstimator:
        save_interval = 1
        n_estimators_finetune = 2
        n_estimators_validation = 2
        n_estimators_inference = 2

        def __init__(self) -> None:
            self.current_state = "baseline"
            self.current_primary = 0.5
            self.model_ = FakeModel(self)
            self.is_fitted_ = False

        def _set_training_mode(self, enabled: bool) -> None:
            self.model_.train(enabled)

        def _metric_improved(self, current: float, best: float) -> bool:
            return current > best

        def _validate_current_model(self, _device, _train_X, _train_y, _val_X, _val_y):
            return SimpleNamespace(primary=self.current_primary)

    class FakeAdapter:
        model_id = "tabicl_v2"

        def __init__(self, _spec) -> None:
            pass

        def resolved_model_params(self, common_config: CommonFineTuneConfig) -> dict[str, object]:
            return {"epochs": common_config.epochs}

        def resolve_zero_shot_estimator(self, *_args, **_kwargs):
            return SimpleNamespace(kind="zero", classes_=np.array([0, 1]))

        def resolve_finetune_estimator(self, *_args, **_kwargs):
            return FakeFinetuneEstimator()

        def fit_finetuned(self, estimator, train_X, train_y, val_X, val_y, _output_dir: Path):
            estimator.current_state = "baseline"
            estimator.current_primary = 0.5
            estimator._validate_current_model(None, train_X, train_y, val_X, val_y)
            for epoch, primary in ((1, 0.1), (2, 0.2)):
                estimator.current_state = f"epoch{epoch}"
                estimator.current_primary = primary
                estimator._validate_current_model(None, train_X, train_y, val_X, val_y)
            estimator.is_fitted_ = True
            return estimator

        def clone_finetuned_state(self, estimator) -> dict[str, str]:
            return dict(estimator.model_.state_dict())

        def apply_finetuned_state(self, estimator, state) -> None:
            estimator.current_state = state["state"]

        def build_in_memory_inference_estimator(self, estimator, _task_type, _device, _n_estimators):
            return FakeInnerEstimator(kind=estimator.model_.state_dict()["state"])

        def export_finetuned_state(
            self,
            _estimator,
            state_dict: dict[str, str],
            output_dir: Path,
            *,
            epoch: int | None,
            metric: float | None,
            checkpoint_mode: str,
        ) -> Path:
            assert checkpoint_mode == "best"
            assert epoch == 1
            assert metric == 1.0
            checkpoint_path = output_dir / "best.ckpt"
            checkpoint_path.write_text(state_dict["state"], encoding="utf-8")
            return checkpoint_path

        def load_checkpoint_for_inference(self, *_args, **_kwargs):
            raise AssertionError("validation-only in-memory reports must not reload checkpoints")

        def fit_inference_context(self, estimator, _context_X, _context_y):
            return estimator

        def predict_for_metrics(self, estimator, query_X, _task_type):
            if estimator.kind == "zero":
                return np.tile(np.array([[0.7, 0.3]]), (len(query_X), 1))
            if estimator.kind == "epoch2":
                return np.tile(np.array([[0.55, 0.45]]), (len(query_X), 1))
            return np.array([[0.8, 0.2], [0.2, 0.8]])[: len(query_X)]

    fake_spec = SimpleNamespace(
        model_id="tabicl_v2",
        display_name="TabICLv2",
        version_payload={},
        adapter_cls=FakeAdapter,
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

    report = runner.main(
        {
            "paths": {"results_root": str(tmp_path / "results")},
            "seed": 0,
            "finetune": {
                "model_id": "tabicl_v2",
                "epochs": 2,
                "test_eval_interval_epochs": 0,
                "save_best_val_checkpoint": True,
                "final_eval_checkpoint": "best",
                "preprocessing_protocol": "shared",
            },
            "data": {"path": ":data/tabarena/churn", "split": "fold0_repeat0_sample0"},
            "wandb": {"mode": "disabled"},
        },
        tmp_path / "generated" / "validation-only",
        force=True,
    )

    artifact_dir = tmp_path / "generated" / "validation-only"
    report_payload = json.loads((artifact_dir / "finetune_report.json").read_text(encoding="utf-8"))

    assert isinstance(report, dict)
    assert (artifact_dir / "best.ckpt").read_text(encoding="utf-8") == "epoch1"
    assert report_payload["checkpoint_available"] is True
    assert report_payload["checkpoint_retention"]["kept_files"] == 1
    assert report_payload["selected_checkpoint_path"] == str(artifact_dir / "best.ckpt")
    assert report_payload["checkpoint_policy"]["save_best_val_checkpoint"] is True
    assert report_payload["checkpoint_policy"]["export_source"] == "in_memory_best_state"
    assert report_payload["checkpoint_policy"]["validation_eval_rows"] == 2
    assert report_payload["selected_checkpoint_metrics"]["test"]["score"] == 1.0
    assert report_payload["final_checkpoint_metrics"]["test"]["score"] == 0.5
    assert report_payload["evaluation"]["test_evaluation_policy"] == "selected_and_final_only"
    assert report_payload["evaluation"]["selected_epoch"] == 1
    assert report_payload["evaluation"]["selected_test_eval_source"] == "in_memory_best"
    epoch_rows = {(row["epoch"], row["split"], row["eval_source"]) for row in report_payload["epoch_evaluations"]}
    assert (1, "val", "in_memory_validation") in epoch_rows
    assert (2, "val", "in_memory_validation") in epoch_rows
    assert (1, "test", "in_memory_best") in epoch_rows
    assert (2, "test", "in_memory_final") in epoch_rows


@pytest.mark.parametrize(
    (
        "capture_best_state",
        "capture_best_epoch",
        "expected_selected_epoch",
        "expected_selected_source",
        "expected_selected_score",
    ),
    [
        (True, 1, 1, "in_memory_best", 1.0),
        (True, None, 2, "in_memory_best", 1.0),
        (False, None, 0, "zero_shot_initial", 0.5),
    ],
)
def test_finetune_main_uses_adapter_captured_in_memory_states_without_best_checkpoint(
    tmp_path: Path,
    monkeypatch,
    capture_best_state: bool,
    capture_best_epoch: int | None,
    expected_selected_epoch: int,
    expected_selected_source: str,
    expected_selected_score: float,
) -> None:
    class FakeInnerEstimator:
        classes_ = np.array([0, 1])

        def __init__(self, kind: str) -> None:
            self.kind = kind

    class FakeFinetuneEstimator:
        n_estimators_finetune = 2
        n_estimators_validation = 2
        n_estimators_inference = 2

        def __init__(self) -> None:
            self.current_state = "initial"
            self.is_fitted_ = False

    class FakeMitraLikeAdapter:
        model_id = "mitra"

        def __init__(self, _spec) -> None:
            pass

        def resolved_model_params(self, common_config: CommonFineTuneConfig) -> dict[str, object]:
            return {"epochs": common_config.epochs}

        def resolve_zero_shot_estimator(self, *_args, **_kwargs):
            return SimpleNamespace(kind="zero", classes_=np.array([0, 1]))

        def resolve_finetune_estimator(self, *_args, **_kwargs):
            return FakeFinetuneEstimator()

        def fit_finetuned(self, estimator, *_args, **_kwargs):
            logger_obj = getattr(estimator, "experiment_logger", None)
            logger_obj.log_epoch({"train/epoch": 0, "val/roc_auc": 0.4}, step=1)
            logger_obj.log_epoch({"train/epoch": 1, "val/roc_auc": 0.9 if capture_best_state else 0.3}, step=2)
            estimator.current_state = "epoch1"
            if capture_best_state:
                estimator._captured_best_state = {"state": "epoch1"}
            estimator.current_state = "epoch2"
            estimator._captured_final_state = {"state": "epoch2"}
            estimator.current_state = "epoch1"
            estimator.is_fitted_ = True
            return estimator

        def capture_finetuned_states(self, estimator):
            payload = {
                "final": {"state": estimator._captured_final_state, "epoch": 2, "metric": 0.4},
            }
            if hasattr(estimator, "_captured_best_state"):
                payload["best"] = {"state": estimator._captured_best_state, "epoch": capture_best_epoch, "metric": 0.9}
            return payload

        def clone_finetuned_state(self, estimator):
            return {"state": estimator.current_state}

        def apply_finetuned_state(self, estimator, state):
            estimator.current_state = state["state"]

        def build_in_memory_inference_estimator(self, estimator, _task_type, _device, _n_estimators):
            return FakeInnerEstimator(kind=estimator.current_state)

        def export_checkpoint(self, *_args, **_kwargs):
            raise AssertionError("best/final checkpoint export must stay disabled")

        def export_finetuned_state(self, *_args, **_kwargs):
            raise AssertionError("save_best_val_checkpoint=false must not export captured state")

        def load_checkpoint_for_inference(self, *_args, **_kwargs):
            raise AssertionError("no checkpoint should be loaded for in-memory metrics")

        def fit_inference_context(self, estimator, _context_X, _context_y):
            return estimator

        def predict_for_metrics(self, estimator, query_X, _task_type):
            if estimator.kind == "zero":
                return np.tile(np.array([[0.7, 0.3]]), (len(query_X), 1))
            if estimator.kind == "epoch2":
                return np.tile(np.array([[0.55, 0.45]]), (len(query_X), 1))
            return np.array([[0.8, 0.2], [0.2, 0.8]])[: len(query_X)]

    fake_spec = SimpleNamespace(
        model_id="mitra",
        display_name="Mitra",
        version_payload={"checkpoint_identifier_classifier": "autogluon/mitra-classifier"},
        adapter_cls=FakeMitraLikeAdapter,
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

    report = runner.main(
        {
            "paths": {"results_root": str(tmp_path / "results")},
            "seed": 0,
            "finetune": {
                "model_id": "mitra",
                "epochs": 2,
                "test_eval_interval_epochs": 1,
                "save_best_val_checkpoint": False,
                "final_eval_checkpoint": "final",
                "preprocessing_protocol": PREPROCESSING_NATIVE,
            },
            "data": {"path": ":data/tabarena/churn", "split": "fold0_repeat0_sample0"},
            "wandb": {"mode": "disabled"},
        },
        tmp_path / "generated" / "mitra",
        force=True,
    )

    artifact_dir = tmp_path / "generated" / "mitra"
    report_payload = json.loads((artifact_dir / "finetune_report.json").read_text(encoding="utf-8"))

    assert isinstance(report, dict)
    assert not (artifact_dir / "model.ckpt").exists()
    assert not (artifact_dir / "best.ckpt").exists()
    assert not list(artifact_dir.glob("epoch*.ckpt"))
    assert report_payload["checkpoint_available"] is False
    assert report_payload["selected_checkpoint_path"] is None
    assert report_payload["final_checkpoint_path"] is None
    assert report_payload["selected_checkpoint_metrics"]["test"]["score"] == expected_selected_score
    assert report_payload["final_checkpoint_metrics"]["test"]["score"] == 0.5
    assert report_payload["evaluation"]["selected_epoch"] == expected_selected_epoch
    assert report_payload["evaluation"]["final_epoch"] == 2
    assert report_payload["evaluation"]["selected_test_eval_source"] == expected_selected_source
    assert report_payload["evaluation"]["final_test_eval_source"] == "in_memory_final"
    epoch_rows = {(row["epoch"], row["split"], row["eval_source"]) for row in report_payload["epoch_evaluations"]}
    assert (expected_selected_epoch, "test", expected_selected_source) in epoch_rows
    assert (2, "test", "in_memory_final") in epoch_rows
    assert (2, "test", "final_checkpoint") not in epoch_rows


def test_finetune_main_allows_zero_shot_initial_best_with_saved_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Disk-backed adapters may legitimately select the epoch-0 baseline as best validation."""
    baseline_checkpoint = tmp_path / "base.ckpt"
    baseline_checkpoint.write_text("zero", encoding="utf-8")

    class DiskCheckpointAdapter:
        model_id = "tabpfn_v2_5"

        def __init__(self, _spec) -> None:
            pass

        def resolved_model_params(self, common_config: CommonFineTuneConfig) -> dict[str, object]:
            return {"epochs": common_config.epochs}

        def baseline_checkpoint_path(self, _task_type):
            return baseline_checkpoint

        def resolve_zero_shot_estimator(self, *_args, **_kwargs):
            return SimpleNamespace(kind="zero")

        def resolve_finetune_estimator(self, *_args, **_kwargs):
            return SimpleNamespace(kind="train")

        def fit_finetuned(self, estimator, _train_X, _train_y, _val_X, _val_y, output_dir: Path):
            logger_obj = getattr(estimator, "experiment_logger", None)
            logger_obj.log_epoch({"train/epoch": 1, "val/roc_auc": 0.4}, step=1)
            logger_obj.log_epoch({"train/epoch": 2, "val/roc_auc": 0.3}, step=2)
            (output_dir / "epoch1.ckpt").write_text("epoch1", encoding="utf-8")
            (output_dir / "epoch2.ckpt").write_text("epoch2", encoding="utf-8")
            estimator.is_fitted_ = True
            return estimator

        def load_checkpoint_for_inference(self, checkpoint_path: Path, *_args, **_kwargs):
            return SimpleNamespace(kind=checkpoint_path.read_text(encoding="utf-8"))

        def fit_inference_context(self, estimator, _context_X, _context_y):
            return estimator

        def predict_for_metrics(self, estimator, query_X, _task_type):
            if estimator.kind == "zero":
                return np.array([[0.8, 0.2], [0.2, 0.8]])[: len(query_X)]
            return np.tile(np.array([[0.55, 0.45]]), (len(query_X), 1))

    fake_spec = SimpleNamespace(
        model_id="tabpfn_v2_5",
        display_name="TabPFNv2.5",
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

    runner.main(
        {
            "paths": {"results_root": str(tmp_path / "results")},
            "seed": 0,
            "finetune": {
                "model_id": "tabpfn_v2_5",
                "epochs": 2,
                "test_eval_interval_epochs": 0,
                "final_eval_checkpoint": "best",
                "save_best_val_checkpoint": True,
                "preprocessing_protocol": PREPROCESSING_NATIVE,
            },
            "data": {"path": ":data/tabarena/churn", "split": "fold0_repeat0_sample0"},
            "wandb": {"mode": "disabled"},
        },
        tmp_path / "generated" / "tabpfn",
        force=True,
    )

    artifact_dir = tmp_path / "generated" / "tabpfn"
    report_payload = json.loads((artifact_dir / "finetune_report.json").read_text(encoding="utf-8"))
    assert (artifact_dir / "model.ckpt").read_text(encoding="utf-8") == "zero"
    assert not (artifact_dir / "best.ckpt").exists()
    assert report_payload["checkpoint_available"] is True
    assert report_payload["checkpoint_retention"]["kept_files"] == 1
    assert report_payload["checkpoint_retention"]["deleted_files"] == 2
    assert report_payload["checkpoint_path"] == str(artifact_dir / "model.ckpt")
    assert report_payload["selected_checkpoint_path"] == str(baseline_checkpoint)
    assert report_payload["selected_checkpoint_metrics"]["test"]["score"] == 1.0
    assert report_payload["final_checkpoint_metrics"]["test"]["score"] == 0.5
    assert report_payload["evaluation"]["selected_epoch"] == 0
    assert report_payload["evaluation"]["selected_test_eval_source"] == "zero_shot_initial"
    epoch_rows = {(row["epoch"], row["split"], row["eval_source"]) for row in report_payload["epoch_evaluations"]}
    assert (0, "val", "zero_shot_initial") in epoch_rows
    assert (0, "test", "zero_shot_initial") in epoch_rows
    assert (report_payload["evaluation"]["final_epoch"], "test", "final_checkpoint") in epoch_rows


def test_fallback_periodic_rows_alias_final_checkpoint_timing() -> None:
    timing_rows = [
        {"phase": "final_checkpoint", "split": "val", "seconds": 0.4, "score": 0.8},
        {"phase": "final_checkpoint", "split": "test", "seconds": 0.6, "score": 0.7},
    ]

    rows = checkpoint_runner._fallback_periodic_rows_from_final_checkpoint(
        final_metrics={
            "val": {"score": 0.8, "roc_auc": 0.8},
            "test": {"score": 0.7, "roc_auc": 0.7},
        },
        final_epoch=2,
        final_checkpoint_path=Path("best.ckpt"),
        interval=1,
        timing_rows=timing_rows,
    )

    assert rows == [
        {
            "epoch": 2,
            "checkpoint_path": "best.ckpt",
            "eval_source": "final_checkpoint",
            "runtime_seconds": pytest.approx(1.0),
            "metrics": {
                "val": {"score": 0.8, "roc_auc": 0.8},
                "test": {"score": 0.7, "roc_auc": 0.7},
            },
            "fallback_source": "final_checkpoint",
        }
    ]
    aliases = [row for row in timing_rows if row["phase"] == "periodic_final_checkpoint"]
    assert len(aliases) == 2
    assert {row["split"] for row in aliases} == {"val", "test"}
    assert all(row["epoch"] == 2 for row in aliases)
    assert all(row["alias_of"] == "final_checkpoint" for row in aliases)


def test_canonical_epoch_rows_preserve_selected_and_final_sources_at_same_epoch() -> None:
    rows = history_runner._canonical_epoch_evaluation_rows(
        train_history=[],
        periodic_rows=[],
        selected_metrics={
            "val": {"score": 0.8, "roc_auc": 0.8},
            "test": {"score": 0.7, "roc_auc": 0.7},
        },
        final_metrics={
            "val": {"score": 0.6, "roc_auc": 0.6},
            "test": {"score": 0.5, "roc_auc": 0.5},
        },
        best_epoch=2,
        final_epoch=2,
        score_metric="roc_auc",
        timing_rows=[],
        selected_checkpoint_path=None,
        final_checkpoint_path=None,
        selected_eval_source="in_memory_best",
        final_eval_source="in_memory_final",
        selected_timing_phase="selected_checkpoint",
        final_timing_phase="final_in_memory",
    )

    row_keys = {(row["epoch"], row["split"], row["eval_source"]) for row in rows}
    assert row_keys == {
        (2, "val", "in_memory_best"),
        (2, "test", "in_memory_best"),
        (2, "val", "in_memory_final"),
        (2, "test", "in_memory_final"),
    }


def test_canonical_epoch_rows_prefer_runner_owned_validation_rows() -> None:
    rows = history_runner._canonical_epoch_evaluation_rows(
        train_history=[{"train/epoch": 1.0, "train/val/score": 0.1}],
        validation_rows=[
            {
                "epoch": 1,
                "eval_source": "in_memory_validation",
                "metrics": {"val": {"score": 0.8, "roc_auc": 0.8}},
            }
        ],
        periodic_rows=[],
        selected_metrics=None,
        final_metrics=None,
        best_epoch=None,
        final_epoch=None,
        score_metric="roc_auc",
        timing_rows=[
            {
                "phase": "validation_in_memory",
                "epoch": 1,
                "split": "val",
                "n_estimators": 2,
                "context_rows": 5,
                "query_rows": 3,
            }
        ],
        selected_checkpoint_path=None,
        final_checkpoint_path=None,
    )

    assert rows == [
        {
            "epoch": 1,
            "split": "val",
            "score_metric": "roc_auc",
            "score": 0.8,
            "raw_metric_name": "roc_auc",
            "raw_metric_value": 0.8,
            "eval_source": "in_memory_validation",
            "n_estimators": 2,
            "context_rows": 5,
            "query_rows": 3,
            "roc_auc": 0.8,
        }
    ]


def test_log_wandb_summary_merges_periodic_rows_into_epoch_steps(tmp_path: Path) -> None:
    class FakeRun:
        def __init__(self) -> None:
            self.summary: dict[str, object] = {}
            self.logged: list[tuple[int, dict[str, object]]] = []

        def log(self, payload: dict[str, object]) -> None:
            self.logged.append((payload["train/epoch"], dict(payload)))

    fake_run = FakeRun()

    reporting_runner._log_wandb_summary(
        fake_run,
        report={
            "score_metric": "roc_auc",
            "score_metric_source": "finetune.metric",
            "finetune_test_eval_interval_epochs": 1,
            "final_train_epoch": 2,
            "fit_diagnostics": {
                "tabicl_meta_batch_split": {
                    "fallback_count": 3,
                    "dense_label_remap_count": 2,
                    "singleton_chunk_count": 3,
                    "query_size_clamped_count": 1,
                }
            },
        },
        config={},
        train_history=[
            {"step": 10, "train/epoch": 1.0, "train/loss": 0.5, "train/val/score": 0.6},
            {"step": 20, "train/epoch": 2.0, "train/loss": 0.4, "train/val/score": 0.7},
        ],
        periodic_rows=[
            {"epoch": 1, "metrics": {"test": {"score": 0.61, "roc_auc": 0.61}}},
            {"epoch": 2, "metrics": {"test": {"score": 0.72, "roc_auc": 0.72}}},
        ],
        output=tmp_path,
    )

    assert [step for step, _payload in fake_run.logged] == [1, 2]
    assert fake_run.logged[0][1]["train/test/score"] == 0.61
    assert fake_run.logged[1][1]["train/test/score"] == 0.72
