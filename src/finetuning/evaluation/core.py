"""Evaluation helpers for unified foundation-model fine-tuning."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from finetuning.datasets.loading import (
    TEST_CONTEXT_POLICY_TRAIN_ONLY,
    TEST_CONTEXT_POLICY_TRAIN_PLUS_VALIDATION,
    SplitEvaluationData,
    _test_context_policy_name,
)
from finetuning.models.base import (
    CommonFineTuneConfig,
    FoundationModelAdapter,
    InMemoryStateAdapter,
)
from finetuning.training.fit import _evaluate_estimator_on_split, _log_eval_timing
from utils.gpu_memory import CudaMemoryTracker


def _evaluate_zero_shot(
    *,
    adapter: FoundationModelAdapter,
    data: SplitEvaluationData,
    seed: int,
    device: str | None,
    common_config: CommonFineTuneConfig,
    timing_rows: list[dict[str, Any]] | None = None,
    phase: str = "zero_shot",
    eval_source: str = "zero_shot",
    splits: tuple[str, ...] = ("val", "test"),
) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    split_specs = {
        "val": (
            "val",
            common_config.n_estimators_validation,
            data.val_context_X,
            data.val_context_y,
            data.val_X,
            data.val_y,
        ),
        "test": (
            "test",
            common_config.n_estimators_final_inference,
            data.test_context_X,
            data.test_context_y,
            data.test_X,
            data.test_y,
        ),
    }
    for requested_split in splits:
        split, n_estimators, context_X, context_y, query_X, query_y = split_specs[requested_split]
        split_started = time.monotonic()
        timing = {
            "phase": phase,
            "split": split,
            "epoch": None,
            "checkpoint_path": None,
            "eval_source": eval_source,
            "n_estimators": int(n_estimators),
            "context_rows": len(context_X),
            "query_rows": len(query_X),
        }
        with CudaMemoryTracker(device) as memory_tracker:
            estimator_started = time.monotonic()
            estimator = adapter.resolve_zero_shot_estimator(data.task_type, seed, device, n_estimators)
            timing["estimator_init_seconds"] = time.monotonic() - estimator_started
            metrics[split] = _evaluate_estimator_on_split(
                adapter=adapter,
                estimator=estimator,
                context_X=context_X,
                context_y=context_y,
                query_X=query_X,
                query_y=query_y,
                task_type=data.task_type,
                timing=timing,
                target_transform_state=data.target_transform_state,
            )
        timing.update(memory_tracker.as_report_fields())
        timing["seconds"] = time.monotonic() - split_started
        timing["score"] = metrics[split].get("score")
        if timing_rows is not None:
            timing_rows.append(timing)
        _log_eval_timing(timing)
    return metrics


def _evaluate_checkpoint(
    *,
    adapter: FoundationModelAdapter,
    checkpoint_path: Path,
    data: SplitEvaluationData,
    seed: int,
    device: str | None,
    common_config: CommonFineTuneConfig,
    timing_rows: list[dict[str, Any]] | None = None,
    phase: str = "checkpoint",
    epoch: int | None = None,
    eval_source: str = "checkpoint",
    splits: tuple[str, ...] = ("val", "test"),
) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    split_specs = {
        "val": (
            "val",
            common_config.n_estimators_validation,
            data.val_context_X,
            data.val_context_y,
            data.val_X,
            data.val_y,
        ),
        "test": (
            "test",
            common_config.n_estimators_final_inference,
            data.test_context_X,
            data.test_context_y,
            data.test_X,
            data.test_y,
        ),
    }
    for requested_split in splits:
        split, n_estimators, context_X, context_y, query_X, query_y = split_specs[requested_split]
        split_started = time.monotonic()
        timing = {
            "phase": phase,
            "split": split,
            "epoch": epoch,
            "checkpoint_path": str(checkpoint_path),
            "eval_source": eval_source,
            "n_estimators": int(n_estimators),
            "context_rows": len(context_X),
            "query_rows": len(query_X),
        }
        with CudaMemoryTracker(device) as memory_tracker:
            estimator_started = time.monotonic()
            estimator = adapter.load_checkpoint_for_inference(
                checkpoint_path,
                data.task_type,
                device,
                n_estimators,
                seed=int(seed),
            )
            timing["estimator_init_seconds"] = time.monotonic() - estimator_started
            metrics[split] = _evaluate_estimator_on_split(
                adapter=adapter,
                estimator=estimator,
                context_X=context_X,
                context_y=context_y,
                query_X=query_X,
                query_y=query_y,
                task_type=data.task_type,
                timing=timing,
                target_transform_state=data.target_transform_state,
            )
        timing.update(memory_tracker.as_report_fields())
        timing["seconds"] = time.monotonic() - split_started
        timing["score"] = metrics[split].get("score")
        if timing_rows is not None:
            timing_rows.append(timing)
        _log_eval_timing(timing)
    return metrics


def _evaluate_in_memory_estimator_on_data(
    *,
    adapter: FoundationModelAdapter,
    estimator: Any,
    data: SplitEvaluationData,
    device: str | None,
    common_config: CommonFineTuneConfig,
    timing_rows: list[dict[str, Any]] | None,
    phase: str,
    eval_source: str,
    epoch: int | None,
    state: Any | None = None,
    splits: tuple[str, ...] = ("test",),
) -> dict[str, dict[str, Any]]:
    original_state = None
    if state is not None:
        if not isinstance(adapter, InMemoryStateAdapter):
            raise TypeError("Adapter does not support in-memory state evaluation.")
        original_state = adapter.clone_finetuned_state(estimator)
        adapter.apply_finetuned_state(estimator, state)
    try:
        return _evaluate_current_in_memory_estimator_on_data(
            adapter=adapter,
            estimator=estimator,
            data=data,
            device=device,
            common_config=common_config,
            timing_rows=timing_rows,
            phase=phase,
            eval_source=eval_source,
            epoch=epoch,
            splits=splits,
        )
    finally:
        if state is not None and original_state is not None:
            adapter.apply_finetuned_state(estimator, original_state)


def _evaluate_current_in_memory_estimator_on_data(
    *,
    adapter: FoundationModelAdapter,
    estimator: Any,
    data: SplitEvaluationData,
    device: str | None,
    common_config: CommonFineTuneConfig,
    timing_rows: list[dict[str, Any]] | None,
    phase: str,
    eval_source: str,
    epoch: int | None,
    splits: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    if not isinstance(adapter, InMemoryStateAdapter):
        raise TypeError("Adapter does not support in-memory comparison evaluation.")
    metrics: dict[str, dict[str, Any]] = {}
    split_specs = {
        "val": (
            common_config.n_estimators_validation,
            data.val_context_X,
            data.val_context_y,
            data.val_X,
            data.val_y,
        ),
        "test": (
            common_config.n_estimators_final_inference,
            data.test_context_X,
            data.test_context_y,
            data.test_X,
            data.test_y,
        ),
    }
    was_training = _in_memory_model_was_training(estimator)
    _set_in_memory_eval_mode(estimator)
    try:
        for split in splits:
            n_estimators, context_X, context_y, query_X, query_y = split_specs[split]
            split_started = time.monotonic()
            timing = {
                "phase": phase,
                "split": split,
                "epoch": epoch,
                "checkpoint_path": None,
                "eval_source": eval_source,
                "n_estimators": int(n_estimators),
                "context_rows": len(context_X),
                "query_rows": len(query_X),
            }
            with CudaMemoryTracker(device) as memory_tracker:
                estimator_started = time.monotonic()
                inner = adapter.build_in_memory_inference_estimator(
                    estimator,
                    data.task_type,
                    device,
                    n_estimators,
                )
                timing["estimator_init_seconds"] = time.monotonic() - estimator_started
                metrics[split] = _evaluate_estimator_on_split(
                    adapter=adapter,
                    estimator=inner,
                    context_X=context_X,
                    context_y=context_y,
                    query_X=query_X,
                    query_y=query_y,
                    task_type=data.task_type,
                    timing=timing,
                    target_transform_state=data.target_transform_state,
                )
            timing.update(memory_tracker.as_report_fields())
            timing["seconds"] = time.monotonic() - split_started
            timing["score"] = metrics[split].get("score")
            if timing_rows is not None:
                timing_rows.append(timing)
            _log_eval_timing(timing)
    finally:
        _restore_in_memory_model_mode(estimator, was_training=was_training)
    return metrics


def _set_in_memory_eval_mode(estimator: Any) -> None:
    model = getattr(estimator, "model_", None)
    if hasattr(model, "eval"):
        model.eval()


def _in_memory_model_was_training(estimator: Any) -> bool:
    model = getattr(estimator, "model_", None)
    training = getattr(model, "training", None)
    if isinstance(training, bool):
        return training
    return getattr(model, "mode", None) == "train"


def _restore_in_memory_model_mode(estimator: Any, *, was_training: bool) -> None:
    setter = getattr(estimator, "_set_training_mode", None)
    if was_training and callable(setter):
        setter(True)
        return
    model = getattr(estimator, "model_", None)
    if hasattr(model, "train"):
        model.train(was_training)


def _test_context_policy_for_data(data: SplitEvaluationData) -> str:
    explicit_policy = data.metadata.get("test_context_comparison_policy")
    if isinstance(explicit_policy, str) and explicit_policy.strip():
        return explicit_policy.strip()
    return _test_context_policy_name(
        includes_validation_data=bool(data.metadata.get("test_context_includes_validation_data"))
    )


def _test_context_policy_metadata(data: SplitEvaluationData) -> dict[str, Any]:
    payload = {
        "includes_validation_data": bool(data.metadata.get("test_context_includes_validation_data")),
        "test_context_source": data.metadata.get("test_context_source"),
        "test_context_rows": len(data.test_context_X),
    }
    downsampling = data.metadata.get("test_context_majority_downsampling")
    if isinstance(downsampling, Mapping):
        payload["majority_downsampling"] = dict(downsampling)
    return payload


def _test_only_metrics(metrics: Mapping[str, Any] | None) -> dict[str, dict[str, Any]] | None:
    if not isinstance(metrics, Mapping):
        return None
    test_metrics = metrics.get("test")
    if not isinstance(test_metrics, Mapping):
        return None
    return {"test": dict(test_metrics)}


def _as_finite_metric_delta_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, int | float | np.number):
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _subtract_test_metrics(
    *,
    minuend: Mapping[str, Any] | None,
    subtrahend: Mapping[str, Any] | None,
) -> dict[str, dict[str, float]] | None:
    left = _test_only_metrics(minuend)
    right = _test_only_metrics(subtrahend)
    if left is None or right is None:
        return None
    left_test = left["test"]
    right_test = right["test"]
    deltas: dict[str, float] = {}
    for key in sorted(set(left_test) & set(right_test)):
        left_value = _as_finite_metric_delta_value(left_test.get(key))
        right_value = _as_finite_metric_delta_value(right_test.get(key))
        if left_value is None or right_value is None:
            continue
        deltas[str(key)] = left_value - right_value
    return {"test": deltas} if deltas else None


def _test_context_phase_payload_key(phase_name: str) -> str:
    return f"{phase_name}_metrics" if phase_name == "baseline" else f"{phase_name}_checkpoint_metrics"


def _paired_policy_test_metrics(
    *,
    primary_policy: str,
    alternate_policy: str,
    primary_metrics: Mapping[str, Any] | None,
    alternate_metrics: Mapping[str, Any] | None,
) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    primary_test = _test_only_metrics(primary_metrics)
    alternate_test = _test_only_metrics(alternate_metrics)
    if primary_test is not None:
        out[primary_policy] = primary_test
    if alternate_test is not None:
        out[alternate_policy] = alternate_test
    return out


def _test_context_comparison_phase_pairs(
    *,
    baseline_metrics: Mapping[str, Any] | None,
    alternate_baseline_metrics: Mapping[str, Any] | None,
    selected_checkpoint_metrics: Mapping[str, Any] | None,
    alternate_selected_checkpoint_metrics: Mapping[str, Any] | None,
    final_checkpoint_metrics: Mapping[str, Any] | None,
    alternate_final_checkpoint_metrics: Mapping[str, Any] | None,
    configured_checkpoint_metrics: Mapping[str, Any] | None,
    alternate_configured_checkpoint_metrics: Mapping[str, Any] | None,
) -> dict[str, tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]]:
    return {
        "baseline": (baseline_metrics, alternate_baseline_metrics),
        "selected": (selected_checkpoint_metrics, alternate_selected_checkpoint_metrics),
        "final": (final_checkpoint_metrics, alternate_final_checkpoint_metrics),
        "configured": (configured_checkpoint_metrics, alternate_configured_checkpoint_metrics),
    }


def _test_context_policy_metrics(
    *,
    phase_pairs: Mapping[str, tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]],
    primary_policy: str,
    alternate_policy: str,
    target_policy: str,
) -> dict[str, Mapping[str, Any] | None]:
    out: dict[str, Mapping[str, Any] | None] = {}
    for phase_name, (primary_metrics, alternate_metrics) in phase_pairs.items():
        policy_metrics = {
            primary_policy: primary_metrics,
            alternate_policy: alternate_metrics,
        }
        out[phase_name] = policy_metrics.get(target_policy)
    return out


def _test_context_comparison_deltas(
    *,
    phase_pairs: Mapping[str, tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]],
    primary_policy: str,
    alternate_policy: str,
) -> dict[str, Any]:
    context_policies = {
        TEST_CONTEXT_POLICY_TRAIN_ONLY,
        TEST_CONTEXT_POLICY_TRAIN_PLUS_VALIDATION,
    }
    if {primary_policy, alternate_policy} != context_policies:
        phase_deltas: dict[str, dict[str, dict[str, float]]] = {}
        for phase_name, (primary_metrics, alternate_metrics) in phase_pairs.items():
            delta = _subtract_test_metrics(
                minuend=alternate_metrics,
                subtrahend=primary_metrics,
            )
            if delta is not None:
                phase_deltas[phase_name] = delta
        primary_token = primary_policy.removesuffix("_for_test_icl")
        alternate_token = alternate_policy.removesuffix("_for_test_icl")
        delta_name = f"{alternate_token}_minus_{primary_token}"
        return {delta_name: phase_deltas} if phase_deltas else {}
    plus_validation_metrics = _test_context_policy_metrics(
        phase_pairs=phase_pairs,
        primary_policy=primary_policy,
        alternate_policy=alternate_policy,
        target_policy=TEST_CONTEXT_POLICY_TRAIN_PLUS_VALIDATION,
    )
    train_only_metrics = _test_context_policy_metrics(
        phase_pairs=phase_pairs,
        primary_policy=primary_policy,
        alternate_policy=alternate_policy,
        target_policy=TEST_CONTEXT_POLICY_TRAIN_ONLY,
    )
    phase_deltas: dict[str, dict[str, dict[str, float]]] = {}
    for phase_name in phase_pairs:
        delta = _subtract_test_metrics(
            minuend=plus_validation_metrics.get(phase_name),
            subtrahend=train_only_metrics.get(phase_name),
        )
        if delta is not None:
            phase_deltas[phase_name] = delta
    return {"train_plus_validation_minus_train_only": phase_deltas} if phase_deltas else {}


def _build_test_context_comparison_payload(
    *,
    primary_data: SplitEvaluationData,
    alternate_data: SplitEvaluationData | None,
    baseline_metrics: Mapping[str, Any] | None,
    alternate_baseline_metrics: Mapping[str, Any] | None,
    selected_checkpoint_metrics: Mapping[str, Any] | None = None,
    alternate_selected_checkpoint_metrics: Mapping[str, Any] | None = None,
    final_checkpoint_metrics: Mapping[str, Any] | None = None,
    alternate_final_checkpoint_metrics: Mapping[str, Any] | None = None,
    configured_checkpoint_metrics: Mapping[str, Any] | None = None,
    alternate_configured_checkpoint_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    if alternate_data is None:
        return None
    primary_policy = _test_context_policy_for_data(primary_data)
    alternate_policy = _test_context_policy_for_data(alternate_data)
    if primary_policy == alternate_policy:
        return None
    policies = {
        primary_policy: _test_context_policy_metadata(primary_data),
        alternate_policy: _test_context_policy_metadata(alternate_data),
    }
    phase_pairs = _test_context_comparison_phase_pairs(
        baseline_metrics=baseline_metrics,
        alternate_baseline_metrics=alternate_baseline_metrics,
        selected_checkpoint_metrics=selected_checkpoint_metrics,
        alternate_selected_checkpoint_metrics=alternate_selected_checkpoint_metrics,
        final_checkpoint_metrics=final_checkpoint_metrics,
        alternate_final_checkpoint_metrics=alternate_final_checkpoint_metrics,
        configured_checkpoint_metrics=configured_checkpoint_metrics,
        alternate_configured_checkpoint_metrics=alternate_configured_checkpoint_metrics,
    )
    payload: dict[str, Any] = {
        "enabled": True,
        "primary_policy": primary_policy,
        "alternate_policy": alternate_policy,
        "policies": policies,
    }
    for phase_name, (primary_metrics, alternate_metrics) in phase_pairs.items():
        metrics_by_policy = _paired_policy_test_metrics(
            primary_policy=primary_policy,
            alternate_policy=alternate_policy,
            primary_metrics=primary_metrics,
            alternate_metrics=alternate_metrics,
        )
        if metrics_by_policy:
            payload[_test_context_phase_payload_key(phase_name)] = metrics_by_policy
    deltas = _test_context_comparison_deltas(
        phase_pairs=phase_pairs,
        primary_policy=primary_policy,
        alternate_policy=alternate_policy,
    )
    if deltas:
        payload["deltas"] = deltas
    return payload


def _build_distribution_shift_test_view_payload(
    *,
    primary_data: SplitEvaluationData,
    alternate_data: SplitEvaluationData | None,
    baseline_metrics: Mapping[str, Any] | None,
    alternate_baseline_metrics: Mapping[str, Any] | None,
    selected_checkpoint_metrics: Mapping[str, Any] | None = None,
    alternate_selected_checkpoint_metrics: Mapping[str, Any] | None = None,
    final_checkpoint_metrics: Mapping[str, Any] | None = None,
    alternate_final_checkpoint_metrics: Mapping[str, Any] | None = None,
    configured_checkpoint_metrics: Mapping[str, Any] | None = None,
    alternate_configured_checkpoint_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build paired test-view metrics produced by one fitted model state."""
    if alternate_data is None:
        return None
    metadata = primary_data.metadata.get("evaluation_comparison")
    if not isinstance(metadata, Mapping) or metadata.get("kind") != "distribution_shift_test_view":
        return None
    primary_view = str(metadata.get("primary_view") or "primary")
    alternate_view = str(metadata.get("alternate_view") or "alternate")
    phase_pairs = _test_context_comparison_phase_pairs(
        baseline_metrics=baseline_metrics,
        alternate_baseline_metrics=alternate_baseline_metrics,
        selected_checkpoint_metrics=selected_checkpoint_metrics,
        alternate_selected_checkpoint_metrics=alternate_selected_checkpoint_metrics,
        final_checkpoint_metrics=final_checkpoint_metrics,
        alternate_final_checkpoint_metrics=alternate_final_checkpoint_metrics,
        configured_checkpoint_metrics=configured_checkpoint_metrics,
        alternate_configured_checkpoint_metrics=alternate_configured_checkpoint_metrics,
    )
    payload: dict[str, Any] = {
        "enabled": True,
        "family": metadata.get("family"),
        "fit_state": metadata.get("fit_state"),
        "primary_view": primary_view,
        "alternate_view": alternate_view,
        "same_fitted_state": True,
    }
    for phase_name, (primary_metrics, alternate_metrics) in phase_pairs.items():
        paired = _paired_policy_test_metrics(
            primary_policy=primary_view,
            alternate_policy=alternate_view,
            primary_metrics=primary_metrics,
            alternate_metrics=alternate_metrics,
        )
        if paired:
            payload[_test_context_phase_payload_key(phase_name)] = paired
    return payload
