from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tfm_finetuning.finetuning.run.setup import FoundationRunSetup
from tfm_finetuning.finetuning.run.training_eval import (
    _evaluate_final_checkpoint,
    _evaluate_periodic_checkpoints,
    _evaluate_selected_test_context_comparison,
    _select_checkpoint_for_report,
)
from tfm_finetuning.finetuning.training.checkpoints import (
    _materialize_model_checkpoint,
    select_best_checkpoint_path,
    select_checkpoint_path,
)
from tfm_finetuning.finetuning.training.fit import _fit_with_oom_retry, _record_runtime_phase
from tfm_finetuning.finetuning.training.history import (
    _baseline_epoch_rows,
    _in_memory_validation_epoch_rows,
    _score_metric_name,
    _training_history_rows,
    _validation_epoch_rows_from_train_history,
)
from tfm_finetuning.utils.gpu_memory import CudaMemoryTracker


@dataclass
class FoundationTrainingOutcome:
    fit_result: Any
    fit_controller: Any
    fit_probe: Any
    effective_common_config: Any
    train_history: list[dict[str, Any]]
    baseline_epoch_rows: list[dict[str, Any]]
    in_memory_validation_rows: list[dict[str, Any]]
    fine_tuned_validation_probe_rows: list[dict[str, Any]]
    periodic_rows: list[dict[str, Any]]
    selected_checkpoint_metrics: dict[str, Any]
    selected_checkpoint_for_report: Path | None
    selected_eval_source: str
    selected_epoch: int | None
    selected_timing_phase: str
    final_checkpoint: Path | None
    final_metrics: dict[str, Any]
    final_eval_source: str
    comparison_final_metrics: dict[str, Any] | None
    comparison_selected_checkpoint_metrics: dict[str, Any] | None
    model_checkpoint: Path | None
    configured_metrics: dict[str, Any]
    comparison_configured_metrics: dict[str, Any] | None
    best_checkpoint: Path | None
    score_metric: str


def run_finetuned_training_phases(
    setup: FoundationRunSetup,
    *,
    data: Any,
    comparison_data: Any,
    common_config: Any,
    baseline_metrics: dict[str, Any],
    comparison_baseline_metrics: dict[str, Any] | None,
    runtime_phases: list[dict[str, Any]],
    eval_timing_rows: list[dict[str, Any]],
) -> FoundationTrainingOutcome:
    """Fit, evaluate, and select the configured foundation-model checkpoint."""
    phase_started = time.monotonic()
    with CudaMemoryTracker(setup.device) as fit_memory_tracker:
        fit_result = _fit_with_oom_retry(
            adapter=setup.adapter,
            data=data,
            spec=setup.spec,
            config=setup.config,
            common_config=common_config,
            resolved_seed=int(setup.resolved_seed),
            device=setup.device,
            output=setup.output,
            checkpoint_mode=setup.checkpoint_mode,
            test_eval_interval=setup.test_eval_interval,
            keep_intermediate_checkpoints=setup.keep_intermediate_checkpoints,
            eval_timing_rows=eval_timing_rows,
        )
    estimator = fit_result.estimator
    fit_controller = fit_result.fit_controller
    epoch_logger = fit_result.epoch_logger
    fit_probe = fit_result.fit_probe
    effective_common_config = fit_result.common_config
    _record_runtime_phase(
        runtime_phases,
        phase="fit_finetuned",
        started_at=phase_started,
        train_rows=len(data.train_X),
        val_rows=len(data.val_X),
        epochs=effective_common_config.epochs,
        context_size=effective_common_config.context_size,
        configured_context_size=common_config.context_size,
        oom_retries=fit_result.oom_retry.get("retry_count"),
        **fit_memory_tracker.as_report_fields(),
    )

    best_checkpoint, final_checkpoint, model_checkpoint = _capture_checkpoints(
        setup=setup,
        fit_controller=fit_controller,
        runtime_phases=runtime_phases,
    )
    train_history, baseline_epoch_rows, in_memory_validation_rows, fine_tuned_validation_probe_rows, score_metric = (
        _build_training_history_rows(
            data=data,
            epoch_logger=epoch_logger,
            fit_controller=fit_controller,
            baseline_metrics=baseline_metrics,
            eval_timing_rows=eval_timing_rows,
            runtime_phases=runtime_phases,
        )
    )
    final_metrics, final_eval_source, comparison_final_metrics = _evaluate_final_checkpoint(
        setup=setup,
        data=data,
        comparison_data=comparison_data,
        estimator=estimator,
        fit_controller=fit_controller,
        effective_common_config=effective_common_config,
        final_checkpoint=final_checkpoint,
        eval_timing_rows=eval_timing_rows,
        runtime_phases=runtime_phases,
    )
    periodic_rows = _evaluate_periodic_checkpoints(
        setup=setup,
        data=data,
        fit_controller=fit_controller,
        effective_common_config=effective_common_config,
        final_checkpoint=final_checkpoint,
        final_metrics=final_metrics,
        final_eval_source=final_eval_source,
        eval_timing_rows=eval_timing_rows,
        runtime_phases=runtime_phases,
    )
    (
        selected_checkpoint_metrics,
        selected_checkpoint_for_report,
        selected_eval_source,
        selected_epoch,
        selected_timing_phase,
        best_checkpoint,
    ) = _select_checkpoint_for_report(
        setup=setup,
        data=data,
        estimator=estimator,
        fit_controller=fit_controller,
        effective_common_config=effective_common_config,
        baseline_metrics=baseline_metrics,
        baseline_epoch_rows=baseline_epoch_rows,
        fine_tuned_validation_probe_rows=fine_tuned_validation_probe_rows,
        periodic_rows=periodic_rows,
        best_checkpoint=best_checkpoint,
        eval_timing_rows=eval_timing_rows,
        runtime_phases=runtime_phases,
        score_metric=score_metric,
    )
    comparison_selected_checkpoint_metrics = _evaluate_selected_test_context_comparison(
        setup=setup,
        comparison_data=comparison_data,
        comparison_baseline_metrics=comparison_baseline_metrics,
        estimator=estimator,
        fit_controller=fit_controller,
        effective_common_config=effective_common_config,
        selected_checkpoint_for_report=selected_checkpoint_for_report,
        selected_eval_source=selected_eval_source,
        selected_epoch=selected_epoch,
        eval_timing_rows=eval_timing_rows,
        runtime_phases=runtime_phases,
    )
    configured_checkpoint = (
        final_checkpoint if setup.checkpoint_mode == "final" else selected_checkpoint_for_report
    )
    if fit_controller is None and configured_checkpoint is not None:
        model_checkpoint = _materialize_model_checkpoint(source=configured_checkpoint, output_dir=setup.output)
    configured_metrics = final_metrics if setup.checkpoint_mode == "final" else selected_checkpoint_metrics
    comparison_configured_metrics = (
        comparison_final_metrics if setup.checkpoint_mode == "final" else comparison_selected_checkpoint_metrics
    )
    return FoundationTrainingOutcome(
        fit_result=fit_result,
        fit_controller=fit_controller,
        fit_probe=fit_probe,
        effective_common_config=effective_common_config,
        train_history=train_history,
        baseline_epoch_rows=baseline_epoch_rows,
        in_memory_validation_rows=in_memory_validation_rows,
        fine_tuned_validation_probe_rows=fine_tuned_validation_probe_rows,
        periodic_rows=periodic_rows,
        selected_checkpoint_metrics=selected_checkpoint_metrics,
        selected_checkpoint_for_report=selected_checkpoint_for_report,
        selected_eval_source=selected_eval_source,
        selected_epoch=selected_epoch,
        selected_timing_phase=selected_timing_phase,
        final_checkpoint=final_checkpoint,
        final_metrics=final_metrics,
        final_eval_source=final_eval_source,
        comparison_final_metrics=comparison_final_metrics,
        comparison_selected_checkpoint_metrics=comparison_selected_checkpoint_metrics,
        model_checkpoint=model_checkpoint,
        configured_metrics=configured_metrics,
        comparison_configured_metrics=comparison_configured_metrics,
        best_checkpoint=best_checkpoint,
        score_metric=score_metric,
    )


def _capture_checkpoints(
    *,
    setup: FoundationRunSetup,
    fit_controller: Any,
    runtime_phases: list[dict[str, Any]],
) -> tuple[Path | None, Path | None, Path | None]:
    phase_started = time.monotonic()
    if fit_controller is not None:
        best_checkpoint = None
        final_checkpoint = None
        model_checkpoint = None
    else:
        best_checkpoint = select_best_checkpoint_path(setup.output)
        if not best_checkpoint.exists():
            best_checkpoint = None
        final_checkpoint = select_checkpoint_path(setup.output, checkpoint_mode="final")
        if not final_checkpoint.exists():
            final_checkpoint = setup.adapter.export_checkpoint(setup.output, "final")
        model_checkpoint = None
    _record_runtime_phase(
        runtime_phases,
        phase="checkpoint_or_state_capture",
        started_at=phase_started,
        best_checkpoint_path=str(best_checkpoint) if best_checkpoint is not None else None,
        final_checkpoint_path=str(final_checkpoint) if final_checkpoint is not None else None,
        configured_checkpoint_mode=setup.checkpoint_mode,
        model_checkpoint_path=str(model_checkpoint) if model_checkpoint is not None else None,
    )
    return best_checkpoint, final_checkpoint, model_checkpoint


def _build_training_history_rows(
    *,
    data: Any,
    epoch_logger: Any,
    fit_controller: Any,
    baseline_metrics: dict[str, Any],
    eval_timing_rows: list[dict[str, Any]],
    runtime_phases: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    str,
]:
    phase_started = time.monotonic()
    train_history = _training_history_rows(
        logger_obj=epoch_logger,
        data=data,
        baseline_metrics=baseline_metrics,
    )
    score_metric = _score_metric_name(data)
    baseline_epoch_rows = _baseline_epoch_rows(
        baseline_metrics=baseline_metrics,
        score_metric=score_metric,
        timing_rows=eval_timing_rows,
    )
    in_memory_validation_rows = fit_controller.validation_rows if fit_controller is not None else []
    runner_validation_probe_rows = _in_memory_validation_epoch_rows(
        validation_rows=in_memory_validation_rows,
        score_metric=score_metric,
        timing_rows=eval_timing_rows,
    )
    fine_tuned_validation_probe_rows = runner_validation_probe_rows or _validation_epoch_rows_from_train_history(
        train_history,
        score_metric=score_metric,
    )
    _record_runtime_phase(
        runtime_phases,
        phase="train_history_build",
        started_at=phase_started,
        row_count=len(train_history),
    )
    return (
        train_history,
        baseline_epoch_rows,
        in_memory_validation_rows,
        fine_tuned_validation_probe_rows,
        score_metric,
    )
