from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from finetuning.checkpoints import (
    _epoch_from_checkpoint,
    _fallback_periodic_rows_from_final_checkpoint,
    _local_baseline_checkpoint_path,
    _periodic_epoch_metrics,
)
from finetuning.evaluation import _evaluate_checkpoint, _evaluate_in_memory_estimator_on_data
from finetuning.fit import _record_runtime_phase
from finetuning.history import (
    _best_epoch_from_epoch_rows,
    _periodic_selection,
    _selectable_periodic_epoch_rows,
)
from finetuning.run_setup import FoundationRunSetup


def _evaluate_final_checkpoint(
    *,
    setup: FoundationRunSetup,
    data: Any,
    comparison_data: Any,
    estimator: Any,
    fit_controller: Any,
    effective_common_config: Any,
    final_checkpoint: Path | None,
    eval_timing_rows: list[dict[str, Any]],
    runtime_phases: list[dict[str, Any]],
) -> tuple[dict[str, Any], str, dict[str, Any] | None]:
    phase_started = time.monotonic()
    final_eval_source = "final_checkpoint"
    if fit_controller is not None:
        final_eval_epoch = fit_controller.final_epoch or int(effective_common_config.epochs)
        final_eval_source = "in_memory_final"
        final_metrics = fit_controller.evaluate_final(
            estimator,
            epoch=int(final_eval_epoch),
            validation_device=setup.device,
        )
    else:
        final_eval_epoch = None
        final_metrics = _evaluate_checkpoint(
            adapter=setup.adapter,
            checkpoint_path=final_checkpoint,
            data=data,
            seed=int(setup.resolved_seed),
            device=setup.device,
            common_config=effective_common_config,
            timing_rows=eval_timing_rows,
            phase="final_checkpoint",
        )
    _record_runtime_phase(
        runtime_phases,
        phase="final_checkpoint_eval",
        started_at=phase_started,
        checkpoint_path=str(final_checkpoint) if final_checkpoint is not None else None,
        source=final_eval_source,
        splits=len(final_metrics),
    )
    comparison_final_metrics = _evaluate_final_test_context_comparison(
        setup=setup,
        comparison_data=comparison_data,
        estimator=estimator,
        fit_controller=fit_controller,
        effective_common_config=effective_common_config,
        final_checkpoint=final_checkpoint,
        final_eval_source=final_eval_source,
        final_eval_epoch=final_eval_epoch,
        eval_timing_rows=eval_timing_rows,
        runtime_phases=runtime_phases,
    )
    return final_metrics, final_eval_source, comparison_final_metrics


def _evaluate_final_test_context_comparison(
    *,
    setup: FoundationRunSetup,
    comparison_data: Any,
    estimator: Any,
    fit_controller: Any,
    effective_common_config: Any,
    final_checkpoint: Path | None,
    final_eval_source: str,
    final_eval_epoch: int | None,
    eval_timing_rows: list[dict[str, Any]],
    runtime_phases: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if comparison_data is None:
        return None
    phase_started = time.monotonic()
    if fit_controller is not None:
        comparison_final_metrics = _evaluate_in_memory_estimator_on_data(
            adapter=setup.adapter,
            estimator=estimator,
            data=comparison_data,
            device=setup.device,
            common_config=effective_common_config,
            timing_rows=eval_timing_rows,
            phase="final_in_memory_test_context_comparison",
            eval_source="in_memory_final_test_context_comparison",
            epoch=int(final_eval_epoch) if final_eval_epoch is not None else None,
            state=fit_controller.final_state,
            splits=("test",),
        )
    else:
        comparison_final_metrics = _evaluate_checkpoint(
            adapter=setup.adapter,
            checkpoint_path=final_checkpoint,
            data=comparison_data,
            seed=int(setup.resolved_seed),
            device=setup.device,
            common_config=effective_common_config,
            timing_rows=eval_timing_rows,
            phase="final_checkpoint_test_context_comparison",
            eval_source="checkpoint_test_context_comparison",
            epoch=_epoch_from_checkpoint(final_checkpoint) if final_checkpoint is not None else None,
            splits=("test",),
        )
    _record_runtime_phase(
        runtime_phases,
        phase="final_test_context_comparison_eval",
        started_at=phase_started,
        source=f"{final_eval_source}_test_context_comparison",
        splits=len(comparison_final_metrics),
    )
    return comparison_final_metrics


def _evaluate_periodic_checkpoints(
    *,
    setup: FoundationRunSetup,
    data: Any,
    fit_controller: Any,
    effective_common_config: Any,
    final_checkpoint: Path | None,
    final_metrics: dict[str, Any],
    final_eval_source: str,
    eval_timing_rows: list[dict[str, Any]],
    runtime_phases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    phase_started = time.monotonic()
    if fit_controller is not None:
        periodic_rows = fit_controller.periodic_rows
        periodic_source = "in_memory"
        periodic_checkpoint_count = 0
    else:
        periodic_rows = _periodic_epoch_metrics(
            adapter=setup.adapter,
            output_dir=setup.output,
            data=data,
            seed=int(setup.resolved_seed),
            device=setup.device,
            common_config=effective_common_config,
            interval=setup.test_eval_interval,
            timing_rows=eval_timing_rows,
        )
        periodic_source = "checkpoint"
        periodic_checkpoint_count = len(periodic_rows)
    if not periodic_rows:
        fallback_rows = _fallback_periodic_rows_from_final_checkpoint(
            final_metrics=final_metrics,
            final_epoch=_fallback_final_epoch(
                fit_controller=fit_controller,
                final_checkpoint=final_checkpoint,
                effective_common_config=effective_common_config,
            ),
            final_checkpoint_path=final_checkpoint,
            interval=setup.test_eval_interval,
            timing_rows=eval_timing_rows,
            eval_source=final_eval_source,
            timing_phase="final_in_memory" if fit_controller is not None else "final_checkpoint",
        )
        if fallback_rows:
            periodic_rows = fallback_rows
            periodic_source = "final_checkpoint_fallback"
            periodic_checkpoint_count = len(periodic_rows)
    _record_runtime_phase(
        runtime_phases,
        phase="periodic_validation_eval",
        started_at=phase_started,
        interval=setup.test_eval_interval,
        checkpoints=periodic_checkpoint_count,
        row_count=len(periodic_rows),
        source=periodic_source,
        in_fit_validation_seconds=fit_controller.validation_eval_seconds if fit_controller is not None else None,
        in_fit_validation_rows=len(fit_controller.validation_rows) if fit_controller is not None else None,
        in_fit_seconds=fit_controller.periodic_eval_seconds if fit_controller is not None else None,
    )
    return periodic_rows


def _fallback_final_epoch(
    *,
    fit_controller: Any,
    final_checkpoint: Path | None,
    effective_common_config: Any,
) -> int:
    if fit_controller is not None and fit_controller.final_epoch is not None:
        return int(fit_controller.final_epoch)
    if final_checkpoint is not None:
        checkpoint_epoch = _epoch_from_checkpoint(final_checkpoint)
        if checkpoint_epoch is not None:
            return int(checkpoint_epoch)
    return int(effective_common_config.epochs)


def _select_checkpoint_for_report(
    *,
    setup: FoundationRunSetup,
    data: Any,
    estimator: Any,
    fit_controller: Any,
    effective_common_config: Any,
    baseline_metrics: dict[str, Any],
    baseline_epoch_rows: list[dict[str, Any]],
    fine_tuned_validation_probe_rows: list[dict[str, Any]],
    periodic_rows: list[dict[str, Any]],
    best_checkpoint: Path | None,
    eval_timing_rows: list[dict[str, Any]],
    runtime_phases: list[dict[str, Any]],
    score_metric: str,
) -> tuple[dict[str, Any], Path | None, str, int | None, str, Path | None]:
    periodic_selection = _periodic_selection(
        periodic_rows=periodic_rows,
        score_metric=score_metric,
        timing_rows=eval_timing_rows,
    )
    periodic_probe_rows = _selectable_periodic_epoch_rows(
        periodic_rows=periodic_rows,
        score_metric=score_metric,
        timing_rows=eval_timing_rows,
    )
    selected_epoch = _best_epoch_from_epoch_rows(
        [
            *baseline_epoch_rows,
            *(periodic_probe_rows if periodic_selection is not None else fine_tuned_validation_probe_rows),
        ]
    )
    phase_started = time.monotonic()
    selected_checkpoint_for_report = best_checkpoint
    selected_timing_phase = "selected_checkpoint"
    if selected_epoch == 0:
        selected_eval_source = "zero_shot_initial"
        selected_checkpoint_metrics = baseline_metrics
        selected_checkpoint_for_report = _local_baseline_checkpoint_path(setup.adapter, data.task_type)
        selected_timing_phase = "zero_shot"
    elif periodic_selection is not None:
        selected_eval_source = str(periodic_selection["eval_source"])
        selected_checkpoint_metrics = periodic_selection["metrics"]
        selected_checkpoint_for_report = _checkpoint_from_periodic_selection(
            best_checkpoint=best_checkpoint,
            periodic_selection=periodic_selection,
            save_best_val_checkpoint=setup.save_best_val_checkpoint,
        )
        selected_timing_phase = selected_eval_source
    elif fit_controller is not None and fit_controller.best_metrics is not None:
        selected_eval_source = fit_controller.best_metrics_source or "in_memory_best"
        selected_checkpoint_metrics = fit_controller.best_metrics
        selected_epoch = int(fit_controller.best_epoch) if fit_controller.best_epoch is not None else selected_epoch
        selected_timing_phase = "selected_checkpoint"
    elif best_checkpoint is None:
        raise RuntimeError(
            "Best-validation metrics require periodic eval rows, in-memory best metrics, "
            "or an exported best checkpoint."
        )
    else:
        selected_eval_source = "selected_checkpoint"
        selected_checkpoint_metrics = _evaluate_checkpoint(
            adapter=setup.adapter,
            checkpoint_path=best_checkpoint,
            data=data,
            seed=int(setup.resolved_seed),
            device=setup.device,
            common_config=effective_common_config,
            timing_rows=eval_timing_rows,
            phase="selected_checkpoint",
        )
    if fit_controller is not None and setup.save_best_val_checkpoint and selected_eval_source != "zero_shot_initial":
        best_checkpoint = fit_controller.export_checkpoint(estimator, setup.output, checkpoint_mode="best")
        selected_checkpoint_for_report = best_checkpoint
    if fit_controller is None and setup.save_best_val_checkpoint and selected_eval_source != "zero_shot_initial":
        if best_checkpoint is None:
            raise RuntimeError(
                "foundation_model.save_best_val_checkpoint=true requires a real best-validation checkpoint, "
                f"but none was found in {setup.output}."
            )
        selected_checkpoint_for_report = best_checkpoint
    _record_runtime_phase(
        runtime_phases,
        phase="selected_checkpoint_eval",
        started_at=phase_started,
        checkpoint_path=str(selected_checkpoint_for_report) if selected_checkpoint_for_report is not None else None,
        source=selected_eval_source,
        splits=len(selected_checkpoint_metrics),
    )
    return (
        selected_checkpoint_metrics,
        selected_checkpoint_for_report,
        selected_eval_source,
        selected_epoch,
        selected_timing_phase,
        best_checkpoint,
    )


def _checkpoint_from_periodic_selection(
    *,
    best_checkpoint: Path | None,
    periodic_selection: dict[str, Any],
    save_best_val_checkpoint: bool,
) -> Path | None:
    periodic_checkpoint_path = periodic_selection.get("checkpoint_path")
    if save_best_val_checkpoint and best_checkpoint is not None:
        return best_checkpoint
    if periodic_checkpoint_path:
        return Path(periodic_checkpoint_path)
    return best_checkpoint


def _evaluate_selected_test_context_comparison(
    *,
    setup: FoundationRunSetup,
    comparison_data: Any,
    comparison_baseline_metrics: dict[str, Any] | None,
    estimator: Any,
    fit_controller: Any,
    effective_common_config: Any,
    selected_checkpoint_for_report: Path | None,
    selected_eval_source: str,
    selected_epoch: int | None,
    eval_timing_rows: list[dict[str, Any]],
    runtime_phases: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if comparison_data is None:
        return None
    phase_started = time.monotonic()
    if selected_epoch == 0:
        comparison_selected_checkpoint_metrics = comparison_baseline_metrics
    elif fit_controller is not None:
        comparison_selected_checkpoint_metrics = _evaluate_in_memory_estimator_on_data(
            adapter=setup.adapter,
            estimator=estimator,
            data=comparison_data,
            device=setup.device,
            common_config=effective_common_config,
            timing_rows=eval_timing_rows,
            phase="selected_checkpoint_test_context_comparison",
            eval_source=f"{selected_eval_source}_test_context_comparison",
            epoch=int(selected_epoch) if selected_epoch is not None else None,
            state=fit_controller.best_state,
            splits=("test",),
        )
    elif selected_checkpoint_for_report is not None:
        comparison_selected_checkpoint_metrics = _evaluate_checkpoint(
            adapter=setup.adapter,
            checkpoint_path=selected_checkpoint_for_report,
            data=comparison_data,
            seed=int(setup.resolved_seed),
            device=setup.device,
            common_config=effective_common_config,
            timing_rows=eval_timing_rows,
            phase="selected_checkpoint_test_context_comparison",
            eval_source="checkpoint_test_context_comparison",
            epoch=int(selected_epoch) if selected_epoch is not None else None,
            splits=("test",),
        )
    else:
        comparison_selected_checkpoint_metrics = None
    if comparison_selected_checkpoint_metrics is not None:
        _record_runtime_phase(
            runtime_phases,
            phase="selected_test_context_comparison_eval",
            started_at=phase_started,
            source=f"{selected_eval_source}_test_context_comparison",
            splits=len(comparison_selected_checkpoint_metrics),
        )
    return comparison_selected_checkpoint_metrics
