"""Epoch-history row construction for fine-tuning."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from tfm_finetuning.finetuning.datasets.loading import SplitEvaluationData
from tfm_finetuning.finetuning.models.base import InMemoryFinetuneLogger
from tfm_finetuning.finetuning.reporting.wandb_logging import (
    flatten_core_metrics_for_wandb,
    flatten_score_metrics_for_wandb,
)
from tfm_finetuning.finetuning.run import eval_shared as _pipeline_eval_shared

CLASSIFICATION_CORE_METRICS = (
    "roc_auc",
    "accuracy",
    "balanced_accuracy",
    "log_loss",
    "f1",
    "f1_macro",
    "f1_weighted",
    "precision",
    "precision_macro",
    "precision_weighted",
    "recall",
    "recall_macro",
    "recall_weighted",
)
REGRESSION_CORE_METRICS = ("rmse", "mse", "mae", "r2")
EPOCH_CORE_METRICS = (*CLASSIFICATION_CORE_METRICS, *REGRESSION_CORE_METRICS)


def _as_finite_number(value: Any, *, field: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = float("nan")
    if not np.isfinite(numeric):
        raise ValueError(f"Missing finite numeric value for {field}.")
    return float(numeric)


def _primary_score_from_metric_payload(
    metrics: Mapping[str, Any],
    *,
    score_metric: str,
    context: str,
) -> float:
    score = _as_finite_number(metrics.get("score"), field=f"{context}.score")
    metric = score_metric.strip().lower().replace("-", "_")
    if metric == "rmse":
        raw = _as_finite_number(
            metrics.get("raw_metric_value", metrics.get("rmse")), field=f"{context}.raw_metric_value"
        )
        raw_name = str(metrics.get("raw_metric_name") or "rmse").strip().lower()
        if raw_name != "rmse":
            raise ValueError(f"{context} has score_metric=rmse but raw_metric_name={raw_name!r}.")
        if not np.isclose(score, -raw, rtol=1e-6, atol=1e-6):
            raise ValueError(f"{context} has inconsistent RMSE score: score={score} raw_metric_value={raw}.")
    elif metric == "log_loss":
        raw = _as_finite_number(
            metrics.get("raw_metric_value", metrics.get("log_loss")), field=f"{context}.raw_metric_value"
        )
        raw_name = str(metrics.get("raw_metric_name") or "log_loss").strip().lower()
        if raw_name != "log_loss":
            raise ValueError(f"{context} has score_metric=log_loss but raw_metric_name={raw_name!r}.")
        if not np.isclose(score, -raw, rtol=1e-6, atol=1e-6):
            raise ValueError(f"{context} has inconsistent log-loss score: score={score} raw_metric_value={raw}.")
    elif metric == "roc_auc":
        raw = _as_finite_number(
            metrics.get("raw_metric_value", metrics.get("roc_auc")), field=f"{context}.raw_metric_value"
        )
        raw_name = str(metrics.get("raw_metric_name") or "roc_auc").strip().lower()
        if raw_name != "roc_auc":
            raise ValueError(f"{context} has score_metric=roc_auc but raw_metric_name={raw_name!r}.")
        if not np.isclose(score, raw, rtol=1e-6, atol=1e-6):
            raise ValueError(f"{context} has inconsistent ROC-AUC score: score={score} raw_metric_value={raw}.")
    return score


def _epoch_evaluation_rows(
    *,
    periodic_rows: list[dict[str, Any]],
    score_metric: str,
    timing_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build canonical per-epoch validation/test score rows from periodic evals.

    Example:
        >>> _epoch_evaluation_rows(
        ...     periodic_rows=[{"epoch": 1, "eval_source": "in_memory", "metrics": {"val": {"score": -2.0, "raw_metric_name": "rmse", "raw_metric_value": 2.0, "rmse": 2.0}, "test": {"score": -3.0, "raw_metric_name": "rmse", "raw_metric_value": 3.0, "rmse": 3.0}}}],
        ...     score_metric="rmse",
        ...     timing_rows=[],
        ... )[0]["score"]
        -2.0
    """
    timings = {
        (int(row["epoch"]), str(row["split"])): row
        for row in timing_rows
        if row.get("phase") in {"periodic_checkpoint", "periodic_in_memory"}
        and row.get("epoch") is not None
        and row.get("split") is not None
    }
    rows: list[dict[str, Any]] = []
    for periodic in periodic_rows:
        epoch = int(periodic["epoch"])
        metrics_by_split = periodic.get("metrics")
        if not isinstance(metrics_by_split, Mapping):
            raise ValueError(f"Periodic epoch metrics for epoch={epoch} do not contain a metrics mapping.")
        for split in ("val", "test"):
            metrics = metrics_by_split.get(split)
            if not isinstance(metrics, Mapping):
                raise ValueError(f"Missing periodic {split} metrics for epoch={epoch}.")
            context = f"epoch_evaluations epoch={epoch} split={split}"
            score = _primary_score_from_metric_payload(metrics, score_metric=score_metric, context=context)
            timing = timings.get((epoch, split), {})
            row = {
                "epoch": epoch,
                "split": split,
                "score_metric": score_metric,
                "score": score,
                "raw_metric_name": metrics.get("raw_metric_name"),
                "raw_metric_value": metrics.get("raw_metric_value"),
                "eval_source": periodic.get("eval_source"),
                "checkpoint_path": periodic.get("checkpoint_path"),
                "runtime_seconds": periodic.get("runtime_seconds"),
                "n_estimators": timing.get("n_estimators"),
                "context_rows": timing.get("context_rows"),
                "query_rows": timing.get("query_rows"),
            }
            for metric_name in EPOCH_CORE_METRICS:
                if metric_name in metrics:
                    row[metric_name] = metrics[metric_name]
            rows.append({key: value for key, value in row.items() if value is not None})
    return rows


def _baseline_epoch_rows(
    *,
    baseline_metrics: Mapping[str, Any],
    score_metric: str,
    timing_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Represent zero-shot validation/test metrics as epoch-0 evaluation rows.

    Example:
        >>> rows = _baseline_epoch_rows(
        ...     baseline_metrics={"val": {"score": 0.8, "roc_auc": 0.8}},
        ...     score_metric="roc_auc",
        ...     timing_rows=[],
        ... )
        >>> rows[0]["eval_source"]
        'zero_shot_initial'
    """
    return _checkpoint_epoch_rows(
        metrics_by_split=baseline_metrics,
        epoch=0,
        score_metric=score_metric,
        eval_source="zero_shot_initial",
        checkpoint_path=None,
        timing_rows=timing_rows,
        phase="zero_shot",
    )


def _in_memory_validation_epoch_rows(
    *,
    validation_rows: list[dict[str, Any]],
    score_metric: str,
    timing_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build canonical validation rows from runner-owned in-memory evals.

    Example:
        >>> _in_memory_validation_epoch_rows(
        ...     validation_rows=[{"epoch": 1, "metrics": {"val": {"score": 0.8, "roc_auc": 0.8}}}],
        ...     score_metric="roc_auc",
        ...     timing_rows=[],
        ... )[0]["eval_source"]
        'in_memory_validation'
    """
    timings = {
        int(row["epoch"]): row
        for row in timing_rows
        if row.get("phase") == "validation_in_memory" and row.get("split") == "val" and row.get("epoch") is not None
    }
    rows: list[dict[str, Any]] = []
    for validation in validation_rows:
        epoch = int(validation["epoch"])
        metrics_by_split = validation.get("metrics")
        if not isinstance(metrics_by_split, Mapping):
            raise TypeError(f"In-memory validation metrics for epoch={epoch} do not contain a metrics mapping.")
        metrics = metrics_by_split.get("val")
        if not isinstance(metrics, Mapping):
            raise TypeError(f"Missing in-memory validation metrics for epoch={epoch}.")
        rows.append(
            _epoch_row_from_metrics(
                metrics=metrics,
                epoch=epoch,
                split="val",
                score_metric=score_metric,
                eval_source=str(validation.get("eval_source") or "in_memory_validation"),
                checkpoint_path=None,
                timing=timings.get(epoch, {}),
            )
        )
    return rows


def _selectable_periodic_epoch_rows(
    *,
    periodic_rows: list[dict[str, Any]],
    score_metric: str,
    timing_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return periodic epoch rows that are eligible for checkpoint selection.

    Example:
        >>> _selectable_periodic_epoch_rows(
        ...     periodic_rows=[{"epoch": 1, "fallback_source": "final_checkpoint", "metrics": {}}],
        ...     score_metric="roc_auc",
        ...     timing_rows=[],
        ... )
        []
    """
    selection_rows = [row for row in periodic_rows if row.get("fallback_source") is None]
    return _epoch_evaluation_rows(
        periodic_rows=selection_rows,
        score_metric=score_metric,
        timing_rows=timing_rows,
    )


def _periodic_selection(
    *,
    periodic_rows: list[dict[str, Any]],
    score_metric: str,
    timing_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the best validation epoch and paired val/test metrics from periodic rows.

    Example:
        >>> rows = [{"epoch": 1, "eval_source": "checkpoint", "metrics": {
        ...     "val": {"score": 0.8, "roc_auc": 0.8},
        ...     "test": {"score": 0.7, "roc_auc": 0.7},
        ... }}]
        >>> _periodic_selection(periodic_rows=rows, score_metric="roc_auc", timing_rows=[])["epoch"]
        1
    """
    selection_rows = [row for row in periodic_rows if row.get("fallback_source") is None]
    epoch_rows = _selectable_periodic_epoch_rows(
        periodic_rows=selection_rows,
        score_metric=score_metric,
        timing_rows=timing_rows,
    )
    best_epoch = _best_epoch_from_epoch_rows(epoch_rows)
    if best_epoch is None:
        return None
    metrics_by_split: dict[str, dict[str, Any]] = {}
    eval_source: str | None = None
    checkpoint_path: str | None = None
    for row in selection_rows:
        if int(row.get("epoch") or -1) != int(best_epoch):
            continue
        raw_metrics = row.get("metrics")
        if not isinstance(raw_metrics, Mapping):
            continue
        for split in ("val", "test"):
            metrics = raw_metrics.get(split)
            if isinstance(metrics, Mapping):
                metrics_by_split[split] = dict(metrics)
        eval_source = str(row.get("eval_source") or "periodic")
        raw_checkpoint_path = row.get("checkpoint_path")
        checkpoint_path = str(raw_checkpoint_path) if raw_checkpoint_path else None
        break
    if "val" not in metrics_by_split or "test" not in metrics_by_split:
        return None
    return {
        "epoch": int(best_epoch),
        "metrics": metrics_by_split,
        "eval_source": _selected_eval_source_from_periodic(eval_source),
        "checkpoint_path": checkpoint_path,
    }


def _selected_eval_source_from_periodic(eval_source: str | None) -> str:
    text = str(eval_source or "").strip().lower()
    if text == "in_memory":
        return "periodic_in_memory"
    if text == "checkpoint":
        return "periodic_checkpoint"
    return text or "periodic"


def _canonical_epoch_evaluation_rows(
    *,
    train_history: list[dict[str, Any]],
    periodic_rows: list[dict[str, Any]],
    selected_metrics: Mapping[str, Any] | None,
    final_metrics: Mapping[str, Any] | None,
    best_epoch: int | None,
    final_epoch: int | None,
    score_metric: str,
    timing_rows: list[dict[str, Any]],
    selected_checkpoint_path: Path | None,
    final_checkpoint_path: Path | None,
    validation_rows: list[dict[str, Any]] | None = None,
    baseline_metrics: Mapping[str, Any] | None = None,
    selected_eval_source: str = "selected_checkpoint",
    final_eval_source: str = "final_checkpoint",
    selected_timing_phase: str = "selected_checkpoint",
    final_timing_phase: str = "final_checkpoint",
) -> list[dict[str, Any]]:
    runner_validation_rows = _in_memory_validation_epoch_rows(
        validation_rows=validation_rows or [],
        score_metric=score_metric,
        timing_rows=timing_rows,
    )
    rows = runner_validation_rows or _validation_epoch_rows_from_train_history(
        train_history,
        score_metric=score_metric,
    )
    if baseline_metrics is not None:
        rows = [
            *_baseline_epoch_rows(
                baseline_metrics=baseline_metrics,
                score_metric=score_metric,
                timing_rows=timing_rows,
            ),
            *rows,
        ]
    rows.extend(
        _epoch_evaluation_rows(
            periodic_rows=periodic_rows,
            score_metric=score_metric,
            timing_rows=timing_rows,
        )
    )
    if best_epoch is not None and selected_metrics is not None:
        rows.extend(
            _checkpoint_epoch_rows(
                metrics_by_split=selected_metrics,
                epoch=best_epoch,
                score_metric=score_metric,
                eval_source=selected_eval_source,
                checkpoint_path=selected_checkpoint_path,
                timing_rows=timing_rows,
                phase=selected_timing_phase,
            )
        )
    if final_epoch is not None and final_metrics is not None:
        rows.extend(
            _checkpoint_epoch_rows(
                metrics_by_split=final_metrics,
                epoch=final_epoch,
                score_metric=score_metric,
                eval_source=final_eval_source,
                checkpoint_path=final_checkpoint_path,
                timing_rows=timing_rows,
                phase=final_timing_phase,
            )
        )
    return _dedupe_epoch_evaluation_rows(rows)


def _validation_epoch_rows_from_train_history(
    train_history: list[dict[str, Any]],
    *,
    score_metric: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in train_history:
        epoch = record.get("train/epoch")
        if epoch is None:
            continue
        epoch_int = int(float(epoch))
        if epoch_int <= 0:
            continue
        metrics: dict[str, Any] = {"score": record.get("train/val/score")}
        for metric_name in EPOCH_CORE_METRICS:
            value = record.get(f"core/train/val/{metric_name}")
            if value is not None:
                metrics[metric_name] = value
        if metrics.get("score") is None:
            continue
        row = _epoch_row_from_metrics(
            metrics=metrics,
            epoch=epoch_int,
            split="val",
            score_metric=score_metric,
            eval_source="training_log",
            checkpoint_path=None,
            timing={},
        )
        rows.append(row)
    return rows


def _checkpoint_epoch_rows(
    *,
    metrics_by_split: Mapping[str, Any],
    epoch: int,
    score_metric: str,
    eval_source: str,
    checkpoint_path: Path | None,
    timing_rows: list[dict[str, Any]],
    phase: str,
) -> list[dict[str, Any]]:
    timings = {
        str(row["split"]): row for row in timing_rows if row.get("phase") == phase and row.get("split") is not None
    }
    rows: list[dict[str, Any]] = []
    for split in ("val", "test"):
        metrics = metrics_by_split.get(split)
        if not isinstance(metrics, Mapping):
            continue
        rows.append(
            _epoch_row_from_metrics(
                metrics=metrics,
                epoch=int(epoch),
                split=split,
                score_metric=score_metric,
                eval_source=eval_source,
                checkpoint_path=checkpoint_path,
                timing=timings.get(split, {}),
            )
        )
    return rows


def _epoch_row_from_metrics(
    *,
    metrics: Mapping[str, Any],
    epoch: int,
    split: str,
    score_metric: str,
    eval_source: str,
    checkpoint_path: Path | None,
    timing: Mapping[str, Any],
) -> dict[str, Any]:
    context = f"epoch_evaluations epoch={epoch} split={split}"
    score = _primary_score_from_metric_payload(metrics, score_metric=score_metric, context=context)
    raw_metric_name = metrics.get("raw_metric_name") or _raw_metric_name_for_score_metric(score_metric)
    raw_metric_value = metrics.get("raw_metric_value")
    if raw_metric_value is None:
        raw_metric_value = _raw_metric_value_from_score(score=score, score_metric=score_metric)
    row = {
        "epoch": int(epoch),
        "split": split,
        "score_metric": score_metric,
        "score": score,
        "raw_metric_name": raw_metric_name,
        "raw_metric_value": raw_metric_value,
        "eval_source": eval_source,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "runtime_seconds": timing.get("seconds"),
        "n_estimators": timing.get("n_estimators"),
        "context_rows": timing.get("context_rows"),
        "query_rows": timing.get("query_rows"),
    }
    for metric_name in EPOCH_CORE_METRICS:
        if metric_name in metrics:
            row[metric_name] = metrics[metric_name]
    return {key: value for key, value in row.items() if value is not None}


def _raw_metric_name_for_score_metric(score_metric: str) -> str:
    metric = score_metric.strip().lower().replace("-", "_")
    if metric == "rmse":
        return "rmse"
    if metric == "log_loss":
        return "log_loss"
    if metric == "roc_auc":
        return "roc_auc"
    return metric


def _raw_metric_value_from_score(*, score: float, score_metric: str) -> float:
    metric = score_metric.strip().lower().replace("-", "_")
    return -float(score) if metric in {"rmse", "log_loss"} else float(score)


def _dedupe_epoch_evaluation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[int, str, str], dict[str, Any]] = {}
    source_rank = {
        "training_log": 0,
        "checkpoint": 1,
        "in_memory": 1,
        "periodic_checkpoint": 2,
        "periodic_in_memory": 2,
        "selected_checkpoint": 3,
        "in_memory_best": 3,
        "zero_shot_initial": 3,
        "final_checkpoint": 4,
        "in_memory_final": 4,
    }
    for row in rows:
        key = (int(row["epoch"]), str(row["split"]), str(row.get("eval_source") or ""))
        previous = by_key.get(key)
        if previous is None:
            by_key[key] = dict(row)
            continue
        prev_rank = source_rank.get(str(previous.get("eval_source")), 0)
        rank = source_rank.get(str(row.get("eval_source")), 0)
        if rank >= prev_rank:
            by_key[key] = {**previous, **row}
    return [
        by_key[key]
        for key in sorted(
            by_key,
            key=lambda item: (item[0], item[1], source_rank.get(item[2], 0), item[2]),
        )
    ]


def _best_epoch_from_epoch_rows(rows: list[dict[str, Any]]) -> int | None:
    candidates = [
        (int(row["epoch"]), float(row["score"]))
        for row in rows
        if row.get("split") == "val" and row.get("score") is not None and np.isfinite(float(row["score"]))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[1])[0]


def _score_metric_name(data: SplitEvaluationData) -> str:
    return _pipeline_eval_shared.benchmark_metric_for_problem_type(data.problem_type)


def _canonical_metric_key(key: str) -> str:
    return re.sub(r"[^a-z0-9/]+", "_", str(key).strip().lower()).strip("_")


def _record_metric_value(record: Mapping[str, Any], key: str) -> Any:
    if key in record:
        return record[key]
    wanted = _canonical_metric_key(key)
    for raw_key, value in record.items():
        if _canonical_metric_key(str(raw_key)) == wanted:
            return value
    return None


def _training_history_rows(
    *,
    logger_obj: InMemoryFinetuneLogger,
    data: SplitEvaluationData,
    baseline_metrics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in logger_obj.epoch_records:
        payload = dict(record)
        epoch_value = int(payload.get("train/epoch", len(rows))) + 1
        val_payload: dict[str, Any] = {"score": None}
        if data.task_type == "regression":
            rmse = _record_metric_value(payload, "val/rmse")
            mse = _record_metric_value(payload, "val/mse")
            if rmse is not None:
                numeric_rmse = float(rmse)
                val_payload["score"] = -numeric_rmse
                val_payload["rmse"] = numeric_rmse
            elif mse is not None:
                numeric_mse = float(mse)
                rmse = float(np.sqrt(numeric_mse))
                val_payload["score"] = -rmse
                val_payload["rmse"] = rmse
            if mse is not None:
                val_payload["mse"] = float(mse)
        else:
            metric_key = data.metric_name
            value = _record_metric_value(payload, f"val/{metric_key}")
            if value is not None:
                numeric = float(value)
                val_payload["score"] = numeric if data.task_type == "binary" else -numeric
                val_payload[metric_key] = numeric
        for metric_name in EPOCH_CORE_METRICS:
            value = _record_metric_value(payload, f"val/{metric_name}")
            if value is not None:
                val_payload[metric_name] = float(value)
        row = {
            "step": int(payload.get("step", epoch_value) or epoch_value),
            "train/epoch": float(epoch_value),
            "train/loss": payload.get("train/mean_loss"),
            **flatten_score_metrics_for_wandb({"val": val_payload}, prefix="train/"),
            **flatten_core_metrics_for_wandb({"val": val_payload}, prefix="core/train/"),
        }
        test_baseline = baseline_metrics.get("test") if isinstance(baseline_metrics, Mapping) else None
        if isinstance(test_baseline, Mapping) and epoch_value == 0:
            row.update(flatten_score_metrics_for_wandb({"test": test_baseline}, prefix="train/"))
        rows.append({key: value for key, value in row.items() if value is not None})
    return rows
