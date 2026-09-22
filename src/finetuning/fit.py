"""Fit, fallback, and split-evaluation helpers for foundation-model fine-tuning."""

from __future__ import annotations

import contextlib
import gc
import importlib
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from finetuning.data import SplitEvaluationData
from finetuning.distribution_shift import TargetTransformState, inverse_transform_metric_prediction
from finetuning.model_in_memory import InMemoryFitController
from finetuning.models.base import (
    CommonFineTuneConfig,
    FineTuneDiagnosticsAdapter,
    FoundationModelAdapter,
    InMemoryFinetuneLogger,
    TaskKind,
)
from finetuning.models.registry import FoundationModelSpec
from finetuning.resource_policy import (
    _resolve_oom_retry_context_sizes,
    _resolve_resource_fallback_policy,
)
from finetuning.runner import compute_metric_payload
from finetuning.runtime_probe import _FitRuntimeProbe
from utils.config_utils import get_nested


def _configure_periodic_checkpoint_saving(estimator: Any, *, test_eval_interval: int) -> None:
    """Make the test-eval cadence the single source of periodic checkpoint cadence.

    Example:
        >>> obj = type("Obj", (), {"save_checkpoint_interval": 10})()
        >>> _configure_periodic_checkpoint_saving(obj, test_eval_interval=0)
        >>> obj.save_checkpoint_interval is None
        True
    """
    if not hasattr(estimator, "save_checkpoint_interval"):
        return
    interval = int(test_eval_interval)
    estimator.save_checkpoint_interval = interval if interval > 0 else None


def _resolve_device(config: Mapping[str, Any]) -> str | None:
    value = get_nested(config, "foundation_model.device")
    if value is None:
        value = get_nested(config, "evaluation.device")
    if value is None:
        value = get_nested(config, "benchmark.evaluation.device")
    text = str(value or "").strip()
    if not text or text.lower() in {"auto", "none", "null"}:
        return None
    return text


def _attach_epoch_logger(estimator: Any) -> InMemoryFinetuneLogger:
    logger_obj = InMemoryFinetuneLogger()
    if hasattr(estimator, "experiment_logger"):
        estimator.experiment_logger = logger_obj
    elif hasattr(estimator, "_make_experiment_logger"):
        estimator._make_experiment_logger = lambda: logger_obj
    elif hasattr(estimator, "_make_logger"):
        estimator._make_logger = lambda: logger_obj
    else:
        estimator.experiment_logger = logger_obj
    return logger_obj


@dataclass(frozen=True)
class _FitExecutionResult:
    estimator: Any
    fit_controller: InMemoryFitController | None
    epoch_logger: InMemoryFinetuneLogger
    fit_probe: _FitRuntimeProbe
    common_config: CommonFineTuneConfig
    oom_retry: dict[str, Any]
    fit_diagnostics: dict[str, Any]


def _is_cuda_oom_error(exc: BaseException) -> bool:
    visited: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        name = type(current).__name__.lower()
        message = str(current).lower()
        if "outofmemory" in name or "cuda out of memory" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


def _release_cuda_memory() -> None:
    gc.collect()
    with contextlib.suppress(Exception):
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _cleanup_partial_checkpoints(output: Path) -> None:
    if not output.exists():
        return
    for checkpoint in output.glob("*.ckpt"):
        with contextlib.suppress(FileNotFoundError):
            checkpoint.unlink()


def _consume_adapter_finetune_diagnostics(adapter: FoundationModelAdapter) -> dict[str, Any]:
    if not isinstance(adapter, FineTuneDiagnosticsAdapter):
        return {}
    diagnostics = adapter.consume_finetune_diagnostics()
    return dict(diagnostics) if isinstance(diagnostics, Mapping) else {}


def _estimator_class_labels(estimator: Any) -> Any:
    for attr in ("classes_", "classes"):
        value = getattr(estimator, attr, None)
        if value is not None:
            return value
    for attr in ("label_encoder_", "y_encoder_", "encoder_"):
        encoder = getattr(estimator, attr, None)
        value = getattr(encoder, "classes_", None)
        if value is not None:
            return value
    return None


def _as_frame_prediction(prediction: Any, *, task_type: TaskKind) -> Any:
    if task_type == "regression":
        return prediction
    if isinstance(prediction, pd.DataFrame):
        return prediction
    probs = np.asarray(prediction)
    if probs.ndim == 1:
        probs = np.column_stack([1.0 - probs, probs])
    return probs


def _runtime_fields_text(fields: Mapping[str, Any]) -> str:
    parts = [f"{key}={value}" for key, value in fields.items() if value is not None]
    return "" if not parts else " " + " ".join(parts)


def _record_runtime_phase(
    rows: list[dict[str, Any]],
    *,
    phase: str,
    started_at: float,
    **fields: Any,
) -> float:
    seconds = time.monotonic() - started_at
    payload = {"phase": phase, "seconds": seconds, **fields}
    rows.append(payload)
    logger.info(
        "Foundation-model runtime phase: phase={} seconds={:.3f}{}",
        phase,
        seconds,
        _runtime_fields_text(fields),
    )
    return seconds


def _log_eval_timing(row: Mapping[str, Any]) -> None:
    logger.debug(
        (
            "Foundation-model eval timing: phase={phase} split={split} "
            "epoch={epoch} n_estimators={n_estimators} context_rows={context_rows} "
            "query_rows={query_rows} estimator_init={estimator_init_seconds:.3f}s "
            "context_fit={context_fit_seconds:.3f}s predict={predict_seconds:.3f}s "
            "metrics={metric_seconds:.3f}s total={seconds:.3f}s source={eval_source} "
            "checkpoint={checkpoint_path}"
        ),
        phase=row.get("phase"), split=row.get("split"), epoch=row.get("epoch"), n_estimators=row.get("n_estimators"), context_rows=row.get("context_rows"), query_rows=row.get("query_rows"), estimator_init_seconds=float(row.get("estimator_init_seconds") or 0.0), context_fit_seconds=float(row.get("context_fit_seconds") or 0.0), predict_seconds=float(row.get("predict_seconds") or 0.0), metric_seconds=float(row.get("metric_seconds") or 0.0), seconds=float(row.get("seconds") or 0.0), eval_source=row.get("eval_source") or ("checkpoint" if row.get("checkpoint_path") else "direct"), checkpoint_path=row.get("checkpoint_path"),
    )


def _evaluate_estimator_on_split(
    *,
    adapter: FoundationModelAdapter,
    estimator: Any,
    context_X: pd.DataFrame,
    context_y: np.ndarray,
    query_X: pd.DataFrame,
    query_y: np.ndarray,
    task_type: TaskKind,
    timing: dict[str, Any] | None = None,
    target_transform_state: TargetTransformState | None = None,
) -> dict[str, Any]:
    fit_started = time.monotonic()
    estimator = adapter.fit_inference_context(estimator, context_X, context_y)
    if timing is not None:
        timing["context_fit_seconds"] = time.monotonic() - fit_started
    predict_started = time.monotonic()
    prediction = adapter.predict_for_metrics(estimator, query_X, task_type)
    if timing is not None:
        timing["predict_seconds"] = time.monotonic() - predict_started
    metric_started = time.monotonic()
    prediction = inverse_transform_metric_prediction(
        prediction,
        task_type=task_type,
        target_transform_state=target_transform_state,
    )
    class_labels = _estimator_class_labels(estimator) if task_type != "regression" else None
    metrics = compute_metric_payload(
        query_y,
        _as_frame_prediction(prediction, task_type=task_type),
        task_type=task_type,
        class_labels=class_labels,
    )
    if timing is not None:
        timing["metric_seconds"] = time.monotonic() - metric_started
    return metrics


def _fit_with_oom_retry(
    *,
    adapter: FoundationModelAdapter,
    data: SplitEvaluationData,
    spec: FoundationModelSpec,
    config: Mapping[str, Any],
    common_config: CommonFineTuneConfig,
    resolved_seed: int,
    device: str | None,
    output: Path,
    checkpoint_mode: str,
    test_eval_interval: int,
    keep_intermediate_checkpoints: bool,
    eval_timing_rows: list[dict[str, Any]],
) -> _FitExecutionResult:
    context_sizes = _resolve_oom_retry_context_sizes(
        config,
        model_id=spec.model_id,
        common_config=common_config,
    )
    attempts: list[int] = []
    failures: list[dict[str, Any]] = []
    for index, context_size in enumerate(context_sizes):
        attempt_config = replace(common_config, context_size=int(context_size))
        estimator = adapter.resolve_finetune_estimator(data.task_type, attempt_config, int(resolved_seed), device)
        _configure_periodic_checkpoint_saving(estimator, test_eval_interval=test_eval_interval)
        fit_controller = InMemoryFitController.maybe_create(
            adapter=adapter,
            data=data,
            common_config=attempt_config,
            device=device,
            checkpoint_mode=checkpoint_mode,
            test_eval_interval=test_eval_interval,
            keep_intermediate_checkpoints=keep_intermediate_checkpoints,
            evaluate_estimator_on_split=_evaluate_estimator_on_split,
            log_eval_timing=_log_eval_timing,
        )
        if fit_controller is not None:
            fit_controller.bind_timing_rows(eval_timing_rows)
            fit_controller.prepare_estimator(estimator)
        epoch_logger = _attach_epoch_logger(estimator)
        fit_probe = _FitRuntimeProbe(controller=fit_controller).install(estimator)
        attempts.append(int(context_size))
        logger.info(
            (
                "Foundation-model finetune fit start: model_id={} train_rows={} "
                "val_rows={} epochs={} context_size={} configured_context_size={} "
                "attempt={}/{} save_interval={} n_estimators_finetune={} "
                "n_estimators_validation={} n_estimators_inference={} output_dir={}"
            ),
            spec.model_id,
            len(data.train_X),
            len(data.val_X),
            attempt_config.epochs,
            attempt_config.context_size,
            common_config.context_size,
            index + 1,
            len(context_sizes),
            getattr(estimator, "save_interval", None),
            getattr(estimator, "n_estimators_finetune", None),
            getattr(estimator, "n_estimators_validation", None),
            getattr(estimator, "n_estimators_inference", None),
            output,
        )
        try:
            adapter.fit_finetuned(
                estimator,
                data.train_X,
                data.train_y,
                data.val_X,
                data.val_fit_y if data.val_fit_y is not None else data.val_y,
                output,
            )
            if fit_controller is not None:
                fit_controller.after_fit(estimator, validation_device=device)
        except Exception as exc:
            is_oom = _is_cuda_oom_error(exc)
            if is_oom:
                failures.append(
                    {
                        "attempt": index + 1,
                        "context_size": int(context_size),
                        "error_type": type(exc).__name__,
                        "message": str(exc).splitlines()[0][:500],
                    }
                )
            if not is_oom or index + 1 >= len(context_sizes):
                raise
            next_context_size = context_sizes[index + 1]
            logger.warning(
                (
                    "Foundation-model CUDA OOM during fit; retrying with smaller context: "
                    "model_id={} failed_context_size={} next_context_size={} attempt={}/{} error={}"
                ),
                spec.model_id,
                context_size,
                next_context_size,
                index + 1,
                len(context_sizes),
                str(exc).splitlines()[0],
            )
            _cleanup_partial_checkpoints(output)
            _release_cuda_memory()
            continue

        return _FitExecutionResult(
            estimator=estimator,
            fit_controller=fit_controller,
            epoch_logger=epoch_logger,
            fit_probe=fit_probe,
            common_config=attempt_config,
            oom_retry={
                "enabled": len(context_sizes) > 1,
                "resource_fallback_policy": _resolve_resource_fallback_policy(config, model_id=spec.model_id),
                "configured_context_size": int(common_config.context_size),
                "effective_context_size": int(attempt_config.context_size),
                "attempted_context_sizes": attempts,
                "candidate_context_sizes": list(context_sizes),
                "successful_context_size": int(attempt_config.context_size),
                "retry_count": len(failures),
                "oom_retry_triggered": bool(failures),
                "context_size_fraction": float(attempt_config.context_size) / float(common_config.context_size),
                "failures": failures,
            },
            fit_diagnostics=_consume_adapter_finetune_diagnostics(adapter),
        )

    raise RuntimeError("Foundation-model fit did not run any context-size attempts.")
