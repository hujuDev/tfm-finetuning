from __future__ import annotations

from dataclasses import dataclass
from typing import Never

import numpy as np
import pytest

from tfm_finetuning.finetuning.evaluation.official_epoch_metrics import (
    compute_full_official_epoch_eval_result,
    install_full_official_epoch_eval_metrics,
)


@dataclass
class _EvalResult:
    primary: float
    secondary: dict[str, float]


class _ClassifierEvalEstimator:
    def __init__(self, probabilities: np.ndarray) -> None:
        self._probabilities = probabilities
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> _ClassifierEvalEstimator:
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._probabilities


class _RegressorEvalEstimator:
    def __init__(self, predictions: np.ndarray) -> None:
        self._predictions = predictions

    def fit(self, X: np.ndarray, y: np.ndarray) -> _RegressorEvalEstimator:
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._predictions


class _DummyFinetuner:
    def __init__(self, *, model_type: str, eval_metric: str) -> None:
        self._model_type = model_type
        self.eval_metric = eval_metric
        self.finetuned_estimator_ = object()

    def _evaluate_model(
        self,
        eval_config: dict[str, object],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> _EvalResult:
        return _EvalResult(primary=-1.0, secondary={"fallback": 1.0})


def test_compute_full_official_epoch_eval_result_adds_accuracy_for_classification() -> None:
    """Binary official epoch evals should expose validation accuracy in secondary metrics."""
    finetuner = _DummyFinetuner(model_type="classifier", eval_metric="roc_auc")
    result = compute_full_official_epoch_eval_result(
        finetuner=finetuner,
        eval_config={},
        X_train=np.array([[0.0], [1.0], [2.0]]),
        y_train=np.array([0, 1, 1]),
        X_val=np.array([[0.0], [1.0], [2.0]]),
        y_val=np.array([0, 1, 1]),
        clone_model_for_evaluation_fn=lambda *args, **kwargs: _ClassifierEvalEstimator(
            np.array(
                [
                    [0.90, 0.10],
                    [0.20, 0.80],
                    [0.30, 0.70],
                ]
            )
        ),
        eval_result_type=_EvalResult,
    )

    assert result.primary == pytest.approx(1.0)
    assert result.secondary["accuracy"] == pytest.approx(1.0)
    assert result.secondary["roc_auc"] == pytest.approx(1.0)
    assert result.secondary["log_loss"] == pytest.approx(0.22839300363692283)


def test_compute_full_official_epoch_eval_result_handles_missing_validation_class() -> None:
    """Multiclass epoch eval should stay finite when y_val omits one fitted class."""
    finetuner = _DummyFinetuner(model_type="classifier", eval_metric="log_loss")
    result = compute_full_official_epoch_eval_result(
        finetuner=finetuner,
        eval_config={},
        X_train=np.arange(10.0).reshape(5, 2),
        y_train=np.array(["2", "3", "5", "U", "Y"]),
        X_val=np.arange(8.0).reshape(4, 2),
        y_val=np.array(["2", "3", "5", "U"]),
        clone_model_for_evaluation_fn=lambda *args, **kwargs: _ClassifierEvalEstimator(
            np.array(
                [
                    [0.70, 0.10, 0.10, 0.05, 0.05],
                    [0.10, 0.65, 0.10, 0.10, 0.05],
                    [0.10, 0.10, 0.65, 0.10, 0.05],
                    [0.10, 0.10, 0.10, 0.65, 0.05],
                ]
            )
        ),
        eval_result_type=_EvalResult,
    )

    assert np.isfinite(result.primary)
    assert result.primary == pytest.approx(-result.secondary["log_loss"])
    assert result.secondary["accuracy"] == pytest.approx(1.0)
    assert result.secondary["log_loss"] == pytest.approx(0.4122559230540238)


def test_compute_full_official_epoch_eval_result_adds_regression_core_metrics() -> None:
    """Regression official epoch evals should expose rmse, mae, and r2 together."""
    finetuner = _DummyFinetuner(model_type="regressor", eval_metric="mse")
    result = compute_full_official_epoch_eval_result(
        finetuner=finetuner,
        eval_config={},
        X_train=np.array([[0.0], [1.0], [2.0]]),
        y_train=np.array([1.0, 2.0, 3.0]),
        X_val=np.array([[0.0], [1.0], [2.0]]),
        y_val=np.array([1.0, 2.0, 3.0]),
        clone_model_for_evaluation_fn=lambda *args, **kwargs: _RegressorEvalEstimator(np.array([1.5, 2.5, 2.0])),
        eval_result_type=_EvalResult,
    )

    assert result.primary == pytest.approx(0.5)
    assert result.secondary["rmse"] == pytest.approx(np.sqrt(0.5))
    assert result.secondary["mae"] == pytest.approx(2.0 / 3.0)
    assert result.secondary["r2"] == pytest.approx(0.25)


def test_install_full_official_epoch_eval_metrics_falls_back_to_original(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wrapper installation should preserve upstream eval behavior on helper failure."""
    finetuner = _DummyFinetuner(model_type="classifier", eval_metric="roc_auc")

    def _raise(*args, **kwargs) -> Never:
        raise ValueError("boom")

    monkeypatch.setattr(
        "tfm_finetuning.finetuning.evaluation.official_epoch_metrics.compute_full_official_epoch_eval_result",
        _raise,
    )

    install_full_official_epoch_eval_metrics(finetuner)
    result = finetuner._evaluate_model(
        {},
        np.array([[0.0]]),
        np.array([0]),
        np.array([[0.0]]),
        np.array([0]),
    )

    assert result.primary == pytest.approx(-1.0)
    assert result.secondary == {"fallback": 1.0}
