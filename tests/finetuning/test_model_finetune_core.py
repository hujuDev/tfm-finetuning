"""Tests for foundation model finetune: core."""

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
    PREPROCESSING_SHARED,
)
from tfm_finetuning.finetuning.models.base import CommonFineTuneConfig
from tfm_finetuning.finetuning.models.preprocessing import (
    SHARED_FEATURE_PREPROCESSING,
)
from tfm_finetuning.finetuning.training import resource_policy as policy_runner


def test_estimator_fallback_treats_none_minimum_as_no_requirement() -> None:
    class FakeAdapter:
        def minimum_required_estimators(self, _feature_count: int) -> None:
            return None

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
        spec=SimpleNamespace(model_id="tabicl_v2"),
        config={},
        common_config=CommonFineTuneConfig(n_estimators_finetune=2),
    )

    assert effective.n_estimators_finetune == 2
    assert metadata["required_min_n_estimators"] == 1
    assert metadata["estimator_fallback_triggered"] is False

@pytest.mark.parametrize(
    "checkpoint_mode",
    ["final", "best"],
)
def test_foundation_model_main_writes_direct_run_artifacts(
    tmp_path: Path,
    monkeypatch,
    checkpoint_mode: str,
) -> None:
    loaded_checkpoint_states: list[str] = []

    class FakeInnerEstimator:
        classes_ = np.array([0, 1])

        def __init__(self, kind: str = "inner") -> None:
            self.kind = kind

        def fit(self, _X, _y):
            return self

        def predict_proba(self, X):
            return np.tile(np.array([[0.6, 0.4]]), (len(X), 1))

    class FakeModel:
        def __init__(self, owner) -> None:
            self.owner = owner
            self.mode = "train"

        def state_dict(self) -> dict[str, str]:
            return {"state": self.owner.current_state}

        def eval(self) -> None:
            self.mode = "eval"

        def train(self) -> None:
            self.mode = "train"

    class FakeFinetuneEstimator:
        save_interval = 1
        n_estimators_finetune = 2
        n_estimators_validation = 2
        n_estimators_inference = 8

        def __init__(self) -> None:
            self.current_state = "baseline"
            self.current_primary = 0.5
            self.model_ = FakeModel(self)
            self.model_config_ = {"fake": True}
            self.is_fitted_ = False

        def _set_training_mode(self, enabled: bool) -> None:
            if enabled:
                self.model_.train()
            else:
                self.model_.eval()

        def _build_inner_estimator(self, *_args, **_kwargs):
            return FakeInnerEstimator(kind=self.current_state)

        def _validate_current_model(self, _device, train_X, train_y, val_X, _val_y):
            inner = self._build_inner_estimator(None, self.n_estimators_validation, None)
            inner.fit(train_X, train_y)
            inner.predict_proba(val_X)
            return SimpleNamespace(primary=self.current_primary, secondary={"roc_auc": self.current_primary})

        def _save_checkpoint(
            self,
            *,
            output_dir: Path,
            epoch: int,
            optimizer=None,
            scheduler=None,
            best_metric=0.0,
            is_best: bool = False,
            save_interval: bool = False,
        ):
            _ = optimizer, scheduler, best_metric
            if is_best:
                (output_dir / "best.ckpt").write_text("best", encoding="utf-8")
            if save_interval:
                (output_dir / f"epoch{epoch}.ckpt").write_text(f"epoch{epoch}", encoding="utf-8")

    class FakeAdapter:
        model_id = "tabicl_v2"

        def __init__(self, _spec) -> None:
            pass

        def resolved_model_params(self, common_config: CommonFineTuneConfig) -> dict[str, object]:
            return {"epochs": common_config.epochs}

        def resolve_zero_shot_estimator(self, *_args, **_kwargs):
            return SimpleNamespace(kind="zero")

        def resolve_finetune_estimator(self, *_args, **_kwargs):
            return FakeFinetuneEstimator()

        def fit_finetuned(self, estimator, train_X, train_y, val_X, val_y, output_dir: Path):
            estimator.current_state = "baseline"
            estimator.current_primary = 0.5
            estimator._validate_current_model(None, train_X, train_y, val_X, val_y)
            for epoch, primary in ((1, 0.7), (2, 0.4)):
                estimator.current_state = f"epoch{epoch}"
                estimator.current_primary = primary
                estimator._validate_current_model(None, train_X, train_y, val_X, val_y)
                estimator._save_checkpoint(
                    output_dir=output_dir,
                    epoch=epoch,
                    is_best=True,
                    save_interval=True,
                )
            estimator.is_fitted_ = True
            return estimator

        def consume_finetune_diagnostics(self) -> dict[str, object]:
            return {
                "tabicl_meta_batch_split": {
                    "fallback_count": 2,
                    "dense_label_remap_count": 1,
                    "singleton_chunk_count": 2,
                    "query_size_clamped_count": 0,
                }
            }

        def export_checkpoint(self, output_dir: Path, _checkpoint_mode: str) -> Path:
            _ = output_dir
            raise AssertionError("default in-memory runs must not call adapter.export_checkpoint")

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
            _ = state_dict, output_dir, epoch, metric, checkpoint_mode
            raise AssertionError("default in-memory runs must not export finetuned state")

        def load_checkpoint_for_inference(self, checkpoint_path: Path, *_args, **_kwargs):
            checkpoint_state = checkpoint_path.read_text(encoding="utf-8")
            loaded_checkpoint_states.append(checkpoint_state)
            return SimpleNamespace(kind=checkpoint_state)

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
        version_payload={"checkpoint_identifier_classifier": "tabicl-classifier-v2-20260212.ckpt"},
        adapter_cls=FakeAdapter,
        ensure_source_available=lambda: None,
    )
    monkeypatch.setattr(runner, "get_foundation_model_spec", lambda *_args, **_kwargs: fake_spec)
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
    monkeypatch.setenv("WANDB_RUN_ID", "run-1")

    config = {
        "paths": {"results_root": str(tmp_path / "results")},
        "seed": 0,
        "foundation_model": {
            "model_id": "tabicl_v2",
            "epochs": 2,
            "preprocessing_protocol": "shared",
            "test_eval_interval_epochs": 1,
            "final_eval_checkpoint": checkpoint_mode,
        },
        "data": {"path": ":data/tabarena/churn", "split": "fold0_repeat0_sample0"},
        "wandb": {"mode": "disabled"},
    }

    report = runner.main(config, tmp_path / "generated" / "finetune", force=True)

    assert isinstance(report, dict)
    artifact_dir = tmp_path / "generated" / "finetune"
    assert not (tmp_path / "artifacts").exists()
    assert (artifact_dir / "resolved_config.toml").exists()
    assert not (artifact_dir / "model.ckpt").exists()
    assert not (artifact_dir / "best.ckpt").exists()
    assert not list(artifact_dir.glob("epoch*.ckpt"))
    report_payload = json.loads((artifact_dir / "finetune_report.json").read_text(encoding="utf-8"))
    assert report_payload["checkpoint_available"] is False
    assert report_payload["checkpoint_retention"]["retained"] is False
    assert report_payload["checkpoint_retention"]["deleted_files"] == 0
    assert report_payload["foundation_model"]["model_id"] == "tabicl_v2"
    assert report_payload["foundation_model"]["preprocessing_protocol"] == PREPROCESSING_SHARED
    assert report_payload["foundation_model"]["feature_preprocessing"]["method"] == (
        SHARED_FEATURE_PREPROCESSING
    )
    assert report_payload["foundation_model"]["feature_preprocessing"]["raw_feature_count"] == 1
    assert report_payload["foundation_model"]["feature_preprocessing"]["effective_feature_count"] == 1
    assert report_payload["foundation_model"]["fit_diagnostics"]["tabicl_meta_batch_split"]["fallback_count"] == 2
    assert report_payload["fit_diagnostics"]["tabicl_meta_batch_split"]["dense_label_remap_count"] == 1
    assert report_payload["data_task"]["feature_preprocessing"]["method"] == SHARED_FEATURE_PREPROCESSING
    assert {(row["epoch"], row["split"]) for row in report_payload["epoch_evaluations"]} == {
        (0, "val"),
        (0, "test"),
        (1, "val"),
        (1, "test"),
        (2, "val"),
        (2, "test"),
    }
    assert report_payload["final_checkpoint_metrics"]["test"]["score"] == 0.5
    assert report_payload["selected_checkpoint_metrics"]["test"]["score"] == 1.0
    assert report_payload["final_checkpoint_metrics"]["test"]["score"] == 0.5
    assert report_payload["evaluation"]["selected_epoch"] == 1
    assert report_payload["evaluation"]["final_epoch"] == 2
    assert report_payload["evaluation"]["selected_test_eval_source"] == "periodic_in_memory"
    assert report_payload["evaluation"]["final_test_eval_source"] == "in_memory_final"
    checkpoint_policy = report_payload["checkpoint_policy"]
    assert checkpoint_policy["enabled"] is True
    assert checkpoint_policy["intermediate_checkpoints"] == "suppressed"
    assert checkpoint_policy["export_source"] is None
    assert checkpoint_policy["checkpoint_available"] is False
    assert report_payload["checkpoint_path"] is None
    assert report_payload["selected_checkpoint_path"] is None
    assert report_payload["final_checkpoint_path"] is None
    assert loaded_checkpoint_states == []
    runtime = report_payload["runtime"]
    assert runtime["total_seconds"] == report_payload["runtime_seconds"]
    phase_names = {row["phase"] for row in runtime["phases"]}
    assert {
        "load_data",
        "row_limits",
        "feature_preprocessing",
        "zero_shot_eval",
        "fit_finetuned",
        "checkpoint_or_state_capture",
        "selected_checkpoint_eval",
        "final_checkpoint_eval",
        "periodic_validation_eval",
        "train_history_build",
        "report_write",
    }.issubset(phase_names)
    eval_rows = runtime["eval_splits"]
    assert {row["phase"] for row in eval_rows} == {
        "zero_shot",
        "final_in_memory",
        "periodic_in_memory",
    }
    periodic_eval_rows = [row for row in eval_rows if row["phase"] == "periodic_in_memory"]
    assert len(periodic_eval_rows) == 4
    assert {row["split"] for row in periodic_eval_rows} == {"val", "test"}
    assert {row["epoch"] for row in periodic_eval_rows} == {1, 2}
    assert all(row["eval_source"] == "in_memory" for row in periodic_eval_rows)
    assert all(row["context_rows"] == 2 for row in periodic_eval_rows)
    assert all("predict_seconds" in row for row in periodic_eval_rows)
    fit_events = runtime["fit_events"]
    fit_event_names = {row["event"] for row in fit_events}
    assert {
        "fit_validation",
        "fit_inner_estimator_build",
        "fit_inner_fit",
        "fit_inner_predict_proba",
        "fit_checkpoint_save",
    }.issubset(fit_event_names)
    checkpoint_events = [row for row in fit_events if row["event"] == "fit_checkpoint_save"]
    assert checkpoint_events[0]["epoch"] == 1
    assert checkpoint_events[0]["status"] == "skipped"
    assert checkpoint_events[0]["reason"] == "in_memory_checkpoint_policy"
    metadata = report_payload
    assert metadata["config"]["foundation_model"]["model_id"] == "tabicl_v2"
