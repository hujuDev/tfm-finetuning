from __future__ import annotations

import importlib
import types
from typing import Any

import numpy as np
from loguru import logger
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score


def _load_clone_model_for_evaluation() -> Any:
    module = importlib.import_module("tabpfn.finetuning.train_util")
    return module.clone_model_for_evaluation


def _load_eval_result_type() -> type:
    module = importlib.import_module("tabpfn.finetuning.finetuned_base")
    return module.EvalResult


def _normalize_probabilities(probabilities: Any) -> np.ndarray:
    probs = np.asarray(probabilities)
    if probs.ndim == 1:
        positive = probs.astype(float)
        return np.column_stack([1.0 - positive, positive])
    if probs.ndim == 2 and probs.shape[1] == 1:
        positive = probs[:, 0].astype(float)
        return np.column_stack([1.0 - positive, positive])
    return probs.astype(float)


def _classification_labels(eval_estimator: Any, y_train: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    labels = getattr(eval_estimator, "classes_", None)
    if labels is None:
        labels = np.unique(np.asarray(y_train))
    labels = np.asarray(labels).reshape(-1)
    if labels.size != probabilities.shape[1]:
        raise ValueError(
            "Classifier probability column count does not match class labels: "
            f"probability_columns={probabilities.shape[1]} class_label_count={int(labels.size)}."
        )
    return labels


def _classification_predictions_from_probabilities(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if probabilities.ndim != 2:
        raise ValueError(f"Expected 2D probability array, got shape={probabilities.shape!r}.")
    if probabilities.shape[1] == 2:
        indices = (probabilities[:, 1] >= 0.5).astype(int)
    else:
        indices = probabilities.argmax(axis=1).astype(int)
    return labels[indices]


def compute_full_official_epoch_eval_result(
    *,
    finetuner: Any,
    eval_config: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    clone_model_for_evaluation_fn: Any | None = None,
    eval_result_type: type | None = None,
) -> Any:
    """Compute the official epoch validation result with the full core metric set.

    Example:
        >>> class EvalResult:
        ...     def __init__(self, primary, secondary=None):
        ...         self.primary = primary
        ...         self.secondary = secondary or {}
        >>> class DummyEstimator:
        ...     def fit(self, X, y):
        ...         return self
        ...     def predict(self, X):
        ...         return np.array([1.5, 2.5])
        >>> class DummyFinetuner:
        ...     _model_type = "regressor"
        ...     eval_metric = "mse"
        ...     finetuned_estimator_ = object()
        >>> result = compute_full_official_epoch_eval_result(
        ...     finetuner=DummyFinetuner(),
        ...     eval_config={},
        ...     X_train=np.array([[0.0], [1.0]]),
        ...     y_train=np.array([1.0, 3.0]),
        ...     X_val=np.array([[0.0], [1.0]]),
        ...     y_val=np.array([1.0, 3.0]),
        ...     clone_model_for_evaluation_fn=lambda *args, **kwargs: DummyEstimator(),
        ...     eval_result_type=EvalResult,
        ... )
        >>> round(result.secondary["rmse"], 6)
        0.5
    """
    if clone_model_for_evaluation_fn is None:
        clone_model_for_evaluation_fn = _load_clone_model_for_evaluation()
    if eval_result_type is None:
        eval_result_type = _load_eval_result_type()

    eval_estimator = clone_model_for_evaluation_fn(
        finetuner.finetuned_estimator_,
        eval_config,
        finetuner.finetuned_estimator_.__class__,
    )
    eval_estimator.fit(X_train, y_train)

    model_type = str(getattr(finetuner, "_model_type", "")).strip().lower()
    eval_metric = str(getattr(finetuner, "eval_metric", "")).strip().lower()

    if model_type == "classifier":
        probabilities = _normalize_probabilities(eval_estimator.predict_proba(X_val))
        labels = _classification_labels(eval_estimator, y_train, probabilities)
        predictions = _classification_predictions_from_probabilities(probabilities, labels)
        problem_type = "binary" if labels.size <= 2 else "multiclass"
        secondary: dict[str, float] = {
            "accuracy": float(accuracy_score(y_val, predictions)),
            "log_loss": float(log_loss(y_val, probabilities, labels=labels)),
        }
        if problem_type == "binary":
            secondary["roc_auc"] = float(roc_auc_score(np.asarray(y_val) == labels[-1], probabilities[:, -1]))
        else:
            secondary["roc_auc"] = float("nan")

        if eval_metric == "roc_auc":
            primary = secondary["roc_auc"]
        elif eval_metric == "log_loss":
            primary = -secondary["log_loss"]
        else:
            raise ValueError(f"Unsupported classifier eval_metric: {eval_metric!r}")
        return eval_result_type(primary=float(primary), secondary=secondary)

    if model_type == "regressor":
        predictions = np.asarray(eval_estimator.predict(X_val), dtype=float)
        mse = float(mean_squared_error(y_val, predictions))
        secondary = {
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y_val, predictions)),
            "r2": float(r2_score(y_val, predictions)),
        }
        return eval_result_type(primary=mse, secondary=secondary)

    raise ValueError(f"Unsupported official finetuner model_type: {model_type!r}")


def install_full_official_epoch_eval_metrics(finetuner: Any) -> None:
    """Patch a finetuner so per-epoch validation returns the full core metric set.

    Example:
        >>> class DummyFinetuner:
        ...     def __init__(self):
        ...         self._model_type = "classifier"
        ...         self.eval_metric = "roc_auc"
        ...         self.finetuned_estimator_ = object()
        ...     def _evaluate_model(self, *args, **kwargs):
        ...         return None
        >>> finetuner = DummyFinetuner()
        >>> install_full_official_epoch_eval_metrics(finetuner)
        >>> isinstance(finetuner._evaluate_model, types.MethodType)
        True
    """
    original_evaluate_model = getattr(finetuner, "_evaluate_model", None)
    if not callable(original_evaluate_model):
        return

    def _wrapped_evaluate_model(
        this: Any,
        eval_config: dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Any:
        try:
            return compute_full_official_epoch_eval_result(
                finetuner=this,
                eval_config=eval_config,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning(
                "Falling back to upstream official epoch evaluation after full-metric eval failed: {}",
                exc,
            )
            return original_evaluate_model(eval_config, X_train, y_train, X_val, y_val)

    finetuner._evaluate_model = types.MethodType(_wrapped_evaluate_model, finetuner)
