"""Metric helpers for unified foundation-model runs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn import metrics

from tfm_finetuning.finetuning.models.base import TaskKind


def classification_probabilities(prediction: Any) -> np.ndarray:
    """Normalize classifier predictions to a probability array."""
    probs = np.asarray(prediction)
    if probs.ndim == 1:
        probs = np.column_stack([1.0 - probs, probs])
    return probs


def compute_metric_payload(
    y_true: np.ndarray,
    prediction: Any,
    *,
    task_type: TaskKind,
    class_labels: Any = None,
) -> dict[str, Any]:
    """Compute raw metrics and the higher-is-better score used in reports."""
    if task_type == "regression":
        y_pred = np.asarray(prediction).reshape(-1)
        mse = float(metrics.mean_squared_error(y_true, y_pred))
        rmse = float(mse**0.5)
        return {
            "tabarena_metric": "rmse",
            "raw_metric_name": "rmse",
            "raw_metric_value": rmse,
            "score": -rmse,
            "rmse": rmse,
            "mse": mse,
            "mae": float(metrics.mean_absolute_error(y_true, y_pred)),
            "r2": float(metrics.r2_score(y_true, y_pred)),
        }

    probs = classification_probabilities(prediction)
    probs, labels = _align_classification_probabilities(y_true, prediction, probs, class_labels)
    observed_labels = np.unique(y_true)
    pred_labels = labels[probs.argmax(axis=1)]
    log_loss = float(metrics.log_loss(y_true, probs, labels=labels))
    accuracy = float(metrics.accuracy_score(y_true, pred_labels))
    class_payload = _classification_core_payload(
        y_true=np.asarray(y_true),
        pred_labels=pred_labels,
        labels=labels,
        task_type=task_type,
    )
    if task_type == "binary":
        positive_probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
        roc_auc = float(metrics.roc_auc_score(y_true, positive_probs)) if observed_labels.size > 1 else float("nan")
        return {
            "tabarena_metric": "roc_auc",
            "raw_metric_name": "roc_auc",
            "raw_metric_value": roc_auc,
            "score": roc_auc,
            "roc_auc": roc_auc,
            "accuracy": accuracy,
            "log_loss": log_loss,
            **class_payload,
        }
    roc_auc = _multiclass_roc_auc(y_true=np.asarray(y_true), probs=probs, labels=labels)
    return {
        "tabarena_metric": "log_loss",
        "raw_metric_name": "log_loss",
        "raw_metric_value": log_loss,
        "score": -log_loss,
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "log_loss": log_loss,
        **class_payload,
    }


def _align_classification_probabilities(
    y_true: np.ndarray,
    prediction: Any,
    probs: np.ndarray,
    class_labels: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Align probability columns to every fitted or observed query class."""
    labels = None
    if class_labels is not None:
        labels = np.asarray(class_labels).reshape(-1)
    elif isinstance(prediction, pd.DataFrame):
        labels = np.asarray(prediction.columns).reshape(-1)

    observed_labels = np.unique(y_true)
    if labels is not None:
        if labels.size != probs.shape[1]:
            raise ValueError(
                "Classifier probability column count does not match configured class labels: "
                f"probability_columns={probs.shape[1]} observed_label_count={observed_labels.size} "
                f"class_label_count={int(labels.size)}."
            )
        if np.unique(labels).size != labels.size:
            raise ValueError("Configured classifier class labels must be unique.")
        aligned_labels = np.unique(np.concatenate([labels, observed_labels]))
        if np.array_equal(labels, aligned_labels):
            return probs, labels
        aligned_probs = np.zeros((probs.shape[0], aligned_labels.size), dtype=probs.dtype)
        for source_index, label in enumerate(labels):
            target_index = int(np.flatnonzero(aligned_labels == label)[0])
            aligned_probs[:, target_index] = probs[:, source_index]
        return aligned_probs, aligned_labels

    if observed_labels.size == probs.shape[1]:
        return probs, observed_labels

    raise ValueError(
        "Classifier probability column count does not match available labels: "
        f"probability_columns={probs.shape[1]} observed_label_count={observed_labels.size} "
        "class_label_count=None."
    )


def _classification_core_payload(
    *,
    y_true: np.ndarray,
    pred_labels: np.ndarray,
    labels: np.ndarray,
    task_type: TaskKind,
) -> dict[str, Any]:
    binary_kwargs: dict[str, Any]
    if task_type == "binary" and labels.size >= 2:
        binary_kwargs = {"average": "binary", "pos_label": labels[-1]}
    else:
        binary_kwargs = {"average": "weighted"}
    return {
        "balanced_accuracy": float(metrics.balanced_accuracy_score(y_true, pred_labels)),
        "f1": float(metrics.f1_score(y_true, pred_labels, zero_division=0, **binary_kwargs)),
        "f1_macro": float(metrics.f1_score(y_true, pred_labels, average="macro", zero_division=0)),
        "f1_weighted": float(metrics.f1_score(y_true, pred_labels, average="weighted", zero_division=0)),
        "precision": float(metrics.precision_score(y_true, pred_labels, zero_division=0, **binary_kwargs)),
        "precision_macro": float(metrics.precision_score(y_true, pred_labels, average="macro", zero_division=0)),
        "precision_weighted": float(
            metrics.precision_score(y_true, pred_labels, average="weighted", zero_division=0)
        ),
        "recall": float(metrics.recall_score(y_true, pred_labels, zero_division=0, **binary_kwargs)),
        "recall_macro": float(metrics.recall_score(y_true, pred_labels, average="macro", zero_division=0)),
        "recall_weighted": float(metrics.recall_score(y_true, pred_labels, average="weighted", zero_division=0)),
        "classification_report": metrics.classification_report(
            y_true,
            pred_labels,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def _multiclass_roc_auc(*, y_true: np.ndarray, probs: np.ndarray, labels: np.ndarray) -> float:
    if np.unique(y_true).size <= 1:
        return float("nan")
    try:
        return float(metrics.roc_auc_score(y_true, probs, labels=labels, multi_class="ovo", average="macro"))
    except ValueError:
        return float("nan")
