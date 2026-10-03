from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from tfm_finetuning.finetuning.evaluation.core import (
    _build_distribution_shift_test_view_payload,
    _build_test_context_comparison_payload,
)
from tfm_finetuning.finetuning.reporting.report import finish_run
from tfm_finetuning.finetuning.run.setup import RunSetup
from tfm_finetuning.finetuning.training.checkpoint_retention import checkpoint_retention_payload
from tfm_finetuning.finetuning.training.fit import _record_runtime_phase
from tfm_finetuning.finetuning.training.history import _baseline_epoch_rows, _score_metric_name
from tfm_finetuning.finetuning.training.resource_policy import _runtime_option_payload


def complete_baseline_only_run(
    setup: RunSetup,
    *,
    data: Any,
    comparison_data: Any,
    feature_preprocessing: dict[str, Any],
    common_config: Any,
    configured_common_config: Any,
    estimator_fallback: dict[str, Any],
    baseline_metrics: dict[str, Any],
    comparison_baseline_metrics: dict[str, Any] | None,
    runtime_phases: list[dict[str, Any]],
    eval_timing_rows: list[dict[str, Any]],
    started_at: float,
) -> dict[str, Any]:
    """Build, log, and finish a baseline-only report."""
    elapsed = time.monotonic() - started_at
    score_metric = _score_metric_name(data)
    baseline_epoch_rows = _baseline_epoch_rows(
        baseline_metrics=baseline_metrics,
        score_metric=score_metric,
        timing_rows=eval_timing_rows,
    )
    report = {
        "pipeline_mode": "finetune",
        "finetune": {
            "model_id": setup.spec.model_id,
            "display_name": setup.spec.display_name,
            "version_payload": setup.spec.version_payload,
            "resolved_model_params": setup.adapter.resolved_model_params(common_config),
            "configured_model_params": setup.adapter.resolved_model_params(configured_common_config),
            "preprocessing_protocol": setup.preprocessing_protocol,
            "feature_preprocessing": feature_preprocessing,
            "runtime_options": _runtime_option_payload(common_config),
            "estimator_fallback": estimator_fallback,
            "baseline_only": True,
        },
        "data_task": {
            **data.metadata,
            "problem_type": data.problem_type,
            "task_type": data.task_type,
            "metric_name": data.metric_name,
        },
        "split": {
            "name": setup.split_name,
            "base_seed": setup.base_seed,
            "resolved_seed": setup.resolved_seed,
            "seed_delta": setup.seed_delta,
        },
        "score_metric": score_metric,
        "score_metric_source": "finetune.metric",
        "evaluation": {
            "eval_interval_epochs": int(setup.test_eval_interval),
            "test_evaluation_policy": "baseline_only",
            "selected_checkpoint_policy": "none",
            "selected_epoch": None,
            "final_epoch": 0,
            "selected_test_eval_source": None,
            "validation_row_count": sum(1 for row in baseline_epoch_rows if row.get("split") == "val"),
            "test_row_count": sum(1 for row in baseline_epoch_rows if row.get("split") == "test"),
            "test_context_comparison_enabled": _is_test_context_comparison(comparison_data),
            "distribution_shift_test_views_enabled": _is_distribution_shift_test_view(comparison_data),
        },
        "common_finetune_config": asdict(common_config),
        "configured_common_finetune_config": asdict(configured_common_config),
        "estimator_fallback": estimator_fallback,
        "checkpoint_path": None,
        "selected_checkpoint_path": None,
        "checkpoint_export_mode": setup.checkpoint_mode,
        "checkpoint_policy": {
            "enabled": False,
            "baseline_only": True,
            "periodic_eval_source": "disabled",
            "save_best_val_checkpoint": bool(setup.save_best_val_checkpoint),
        },
        "finetune_baseline_only": True,
        "zero_shot_metrics": baseline_metrics,
        "train_history": [],
        "epoch_evaluations": baseline_epoch_rows,
        "finetune_test_eval_interval_epochs": int(setup.test_eval_interval),
        "final_train_epoch": 0,
        "runtime_seconds": elapsed,
        "runtime": {
            "total_seconds": elapsed,
            "phases": runtime_phases,
            "eval_splits": eval_timing_rows,
            "fit_events": [],
        },
    }
    if _is_distribution_shift_test_view(comparison_data):
        comparison_payload = _build_distribution_shift_test_view_payload(
            primary_data=data,
            alternate_data=comparison_data,
            baseline_metrics=baseline_metrics,
            alternate_baseline_metrics=comparison_baseline_metrics,
        )
        if comparison_payload is not None:
            report["distribution_shift_test_views"] = comparison_payload
    else:
        comparison_payload = _build_test_context_comparison_payload(
            primary_data=data,
            alternate_data=comparison_data,
            baseline_metrics=baseline_metrics,
            alternate_baseline_metrics=comparison_baseline_metrics,
        )
        if comparison_payload is not None:
            report["test_context_comparison"] = comparison_payload
    report["checkpoint_retention"] = checkpoint_retention_payload(setup.output)
    report["checkpoint_available"] = False
    _record_runtime_phase(runtime_phases, phase="report_write", started_at=time.monotonic())
    return finish_run(setup, report)


def _is_distribution_shift_test_view(comparison_data: Any) -> bool:
    metadata = getattr(comparison_data, "metadata", {}) if comparison_data is not None else {}
    comparison = metadata.get("evaluation_comparison") if isinstance(metadata, dict) else None
    return isinstance(comparison, dict) and comparison.get("kind") == "distribution_shift_test_view"


def _is_test_context_comparison(comparison_data: Any) -> bool:
    return comparison_data is not None and not _is_distribution_shift_test_view(comparison_data)
