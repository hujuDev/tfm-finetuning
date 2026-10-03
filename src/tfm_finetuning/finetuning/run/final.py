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
from tfm_finetuning.finetuning.run.training import TrainingOutcome
from tfm_finetuning.finetuning.training.checkpoint_retention import checkpoint_retention_payload
from tfm_finetuning.finetuning.training.checkpoints import _epoch_from_checkpoint, _fit_controller_checkpoint_policy
from tfm_finetuning.finetuning.training.fit import _record_runtime_phase
from tfm_finetuning.finetuning.training.history import (
    _best_epoch_from_epoch_rows,
    _canonical_epoch_evaluation_rows,
    _epoch_evaluation_rows,
)
from tfm_finetuning.finetuning.training.resource_policy import _runtime_option_payload


def complete_finetuned_run(
    setup: RunSetup,
    *,
    data: Any,
    comparison_data: Any,
    feature_preprocessing: dict[str, Any],
    training: TrainingOutcome,
    configured_common_config: Any,
    estimator_fallback: dict[str, Any],
    baseline_metrics: dict[str, Any],
    comparison_baseline_metrics: dict[str, Any] | None,
    eval_timing_rows: list[dict[str, Any]],
    runtime_phases: list[dict[str, Any]],
    started_at: float,
) -> dict[str, Any]:
    """Build, log, and finish the report for a completed fine-tuned run."""
    fit_result = training.fit_result
    fit_controller = training.fit_controller
    fit_probe = training.fit_probe
    effective_common_config = training.effective_common_config
    train_history = training.train_history
    baseline_epoch_rows = training.baseline_epoch_rows
    in_memory_validation_rows = training.in_memory_validation_rows
    fine_tuned_validation_probe_rows = training.fine_tuned_validation_probe_rows
    periodic_rows = training.periodic_rows
    selected_checkpoint_metrics = training.selected_checkpoint_metrics
    selected_checkpoint_for_report = training.selected_checkpoint_for_report
    selected_eval_source = training.selected_eval_source
    selected_epoch = training.selected_epoch
    selected_timing_phase = training.selected_timing_phase
    final_checkpoint = training.final_checkpoint
    final_metrics = training.final_metrics
    final_eval_source = training.final_eval_source
    comparison_final_metrics = training.comparison_final_metrics
    comparison_selected_checkpoint_metrics = training.comparison_selected_checkpoint_metrics
    model_checkpoint = training.model_checkpoint
    configured_metrics = training.configured_metrics
    comparison_configured_metrics = training.comparison_configured_metrics
    best_checkpoint = training.best_checkpoint
    score_metric = training.score_metric

    elapsed = time.monotonic() - started_at
    final_train_epoch_candidates = [
        int(max((row.get("train/epoch", 0) for row in train_history), default=0)),
        fit_controller.final_epoch if fit_controller is not None else None,
        _epoch_from_checkpoint(final_checkpoint) if final_checkpoint is not None else None,
        effective_common_config.epochs,
    ]
    final_train_epoch = int(next(epoch for epoch in final_train_epoch_candidates if epoch is not None and epoch > 0))
    probe_rows = [
        *baseline_epoch_rows,
        *fine_tuned_validation_probe_rows,
        *_epoch_evaluation_rows(periodic_rows=periodic_rows, score_metric=score_metric, timing_rows=eval_timing_rows),
    ]
    selected_epoch = selected_epoch if selected_epoch is not None else _best_epoch_from_epoch_rows(probe_rows)
    epoch_evaluations = _canonical_epoch_evaluation_rows(
        train_history=train_history,
        validation_rows=in_memory_validation_rows,
        periodic_rows=periodic_rows,
        baseline_metrics=baseline_metrics,
        selected_metrics=selected_checkpoint_metrics,
        final_metrics=final_metrics,
        best_epoch=selected_epoch,
        final_epoch=final_train_epoch,
        score_metric=score_metric,
        timing_rows=eval_timing_rows,
        selected_checkpoint_path=selected_checkpoint_for_report,
        final_checkpoint_path=final_checkpoint,
        selected_eval_source=selected_eval_source,
        final_eval_source=final_eval_source,
        selected_timing_phase=selected_timing_phase,
        final_timing_phase="final_in_memory" if fit_controller is not None else "final_checkpoint",
    )
    report = {
        "pipeline_mode": "finetune",
        "finetune": {
            "model_id": setup.spec.model_id,
            "display_name": setup.spec.display_name,
            "version_payload": setup.spec.version_payload,
            "resolved_model_params": setup.adapter.resolved_model_params(effective_common_config),
            "configured_model_params": setup.adapter.resolved_model_params(configured_common_config),
            "preprocessing_protocol": setup.preprocessing_protocol,
            "feature_preprocessing": feature_preprocessing,
            "runtime_options": _runtime_option_payload(effective_common_config),
            "oom_retry": fit_result.oom_retry,
            "estimator_fallback": estimator_fallback,
            "fit_diagnostics": fit_result.fit_diagnostics,
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
            "test_evaluation_policy": "periodic" if setup.test_eval_interval > 0 else "selected_and_final_only",
            "selected_checkpoint_policy": "best_validation",
            "selected_epoch": selected_epoch,
            "final_epoch": final_train_epoch,
            "selected_test_eval_source": selected_eval_source,
            "validation_row_count": sum(1 for row in epoch_evaluations if row.get("split") == "val"),
            "test_row_count": sum(1 for row in epoch_evaluations if row.get("split") == "test"),
            "final_test_eval_source": final_eval_source,
            "test_context_comparison_enabled": _is_test_context_comparison(comparison_data),
            "distribution_shift_test_views_enabled": _is_distribution_shift_test_view(comparison_data),
        },
        "common_finetune_config": asdict(effective_common_config),
        "configured_common_finetune_config": asdict(configured_common_config),
        "oom_retry": fit_result.oom_retry,
        "estimator_fallback": estimator_fallback,
        "fit_diagnostics": fit_result.fit_diagnostics,
        "checkpoint_path": str(model_checkpoint) if model_checkpoint is not None else None,
        "selected_checkpoint_path": str(selected_checkpoint_for_report)
        if selected_checkpoint_for_report is not None
        else None,
        "final_checkpoint_path": str(final_checkpoint) if final_checkpoint is not None else None,
        "checkpoint_export_mode": setup.checkpoint_mode,
        "checkpoint_policy": (
            _fit_controller_checkpoint_policy(
                fit_controller,
                checkpoint_mode=setup.checkpoint_mode,
                save_best_val_checkpoint=setup.save_best_val_checkpoint,
            )
            if fit_controller is not None
            else {
                "enabled": False,
                "intermediate_checkpoints": ("kept" if setup.keep_intermediate_checkpoints else "adapter_default"),
                "periodic_eval_source": "checkpoint" if setup.test_eval_interval > 0 else "disabled",
                "save_best_val_checkpoint": bool(setup.save_best_val_checkpoint),
            }
        ),
        "zero_shot_metrics": baseline_metrics,
        "selected_checkpoint_metrics": selected_checkpoint_metrics,
        "final_checkpoint_metrics": final_metrics,
        "train_history": train_history,
        "epoch_evaluations": epoch_evaluations,
        "finetune_test_eval_interval_epochs": int(setup.test_eval_interval),
        "final_train_epoch": final_train_epoch,
        "runtime_seconds": elapsed,
        "runtime": {
            "total_seconds": elapsed,
            "phases": runtime_phases,
            "eval_splits": eval_timing_rows,
            "fit_events": fit_probe.events,
        },
    }
    comparison_kwargs = {
        "primary_data": data,
        "alternate_data": comparison_data,
        "baseline_metrics": baseline_metrics,
        "alternate_baseline_metrics": comparison_baseline_metrics,
        "selected_checkpoint_metrics": selected_checkpoint_metrics,
        "alternate_selected_checkpoint_metrics": comparison_selected_checkpoint_metrics,
        "final_checkpoint_metrics": final_metrics,
        "alternate_final_checkpoint_metrics": comparison_final_metrics,
        "configured_checkpoint_metrics": configured_metrics,
        "alternate_configured_checkpoint_metrics": comparison_configured_metrics,
    }
    if _is_distribution_shift_test_view(comparison_data):
        comparison_payload = _build_distribution_shift_test_view_payload(**comparison_kwargs)
        if comparison_payload is not None:
            report["distribution_shift_test_views"] = comparison_payload
    else:
        comparison_payload = _build_test_context_comparison_payload(**comparison_kwargs)
        if comparison_payload is not None:
            report["test_context_comparison"] = comparison_payload
    retained_checkpoint_paths = [
        path
        for path in (best_checkpoint, model_checkpoint if selected_eval_source == "zero_shot_initial" else None)
        if setup.save_best_val_checkpoint and path is not None
    ]
    report["checkpoint_retention"] = checkpoint_retention_payload(
        setup.output,
        keep_paths=retained_checkpoint_paths,
    )
    report["checkpoint_available"] = bool(
        (model_checkpoint is not None and model_checkpoint.exists())
        or (best_checkpoint is not None and best_checkpoint.exists())
        or (selected_checkpoint_for_report is not None and selected_checkpoint_for_report.exists())
    )
    if isinstance(report.get("checkpoint_policy"), dict):
        report["checkpoint_policy"]["retained"] = bool(report["checkpoint_retention"]["retained"])
        report["checkpoint_policy"]["checkpoint_available"] = bool(report["checkpoint_available"])
    _record_runtime_phase(runtime_phases, phase="report_write", started_at=time.monotonic())
    return finish_run(setup, report, validation_rows=in_memory_validation_rows, periodic_rows=periodic_rows)


def _is_distribution_shift_test_view(comparison_data: Any) -> bool:
    metadata = getattr(comparison_data, "metadata", {}) if comparison_data is not None else {}
    comparison = metadata.get("evaluation_comparison") if isinstance(metadata, dict) else None
    return isinstance(comparison, dict) and comparison.get("kind") == "distribution_shift_test_view"


def _is_test_context_comparison(comparison_data: Any) -> bool:
    return comparison_data is not None and not _is_distribution_shift_test_view(comparison_data)
