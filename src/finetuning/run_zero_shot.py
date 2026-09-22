from __future__ import annotations

import time
from typing import Any

from finetuning.data import SplitEvaluationData
from finetuning.evaluation import _evaluate_zero_shot
from finetuning.fit import _record_runtime_phase
from finetuning.models.base import CommonFineTuneConfig, FoundationModelAdapter


def evaluate_zero_shot_baselines(
    *,
    adapter: FoundationModelAdapter,
    data: SplitEvaluationData,
    comparison_data: SplitEvaluationData | None,
    resolved_seed: int,
    device: str | None,
    common_config: CommonFineTuneConfig,
    runtime_phases: list[dict[str, Any]],
    eval_timing_rows: list[dict[str, Any]],
    splits: tuple[str, ...] = ("val", "test"),
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Evaluate primary and optional alternate-test-context zero-shot baselines."""
    phase_started = time.monotonic()
    baseline_metrics = _evaluate_zero_shot(
        adapter=adapter,
        data=data,
        seed=int(resolved_seed),
        device=device,
        common_config=common_config,
        timing_rows=eval_timing_rows,
        splits=splits,
    )
    _record_runtime_phase(
        runtime_phases,
        phase="zero_shot_eval",
        started_at=phase_started,
        splits=len(baseline_metrics),
    )
    comparison_baseline_metrics = None
    if comparison_data is not None:
        phase_started = time.monotonic()
        comparison_baseline_metrics = _evaluate_zero_shot(
            adapter=adapter,
            data=comparison_data,
            seed=int(resolved_seed),
            device=device,
            common_config=common_config,
            timing_rows=eval_timing_rows,
            phase="zero_shot_test_context_comparison",
            eval_source="zero_shot_test_context_comparison",
            splits=("test",),
        )
        _record_runtime_phase(
            runtime_phases,
            phase="zero_shot_test_context_comparison_eval",
            started_at=phase_started,
            splits=len(comparison_baseline_metrics),
        )
    return baseline_metrics, comparison_baseline_metrics
