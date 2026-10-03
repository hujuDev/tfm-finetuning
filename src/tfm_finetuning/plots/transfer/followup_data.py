"""Covertype target-budget evidence reused by the final six-target comparison.

Example: ``budgets = build_budget_contrasts(config, schedule_tables)``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tfm_finetuning.plots.metrics import fine_tuning_gain, split_raw_metric_value
from tfm_finetuning.plots.results import Run, read_runs
from tfm_finetuning.plots.transfer.transfer_data import MODEL_ORDER, TransferTables

BUDGET_ROWS = (64, 256, 1024, 4096, 16384, 50000)


def build_budget_contrasts(input_config: Mapping[str, Any], schedules: TransferTables) -> tuple[BudgetContrast, ...]:
    """Load accepted Covertype budgets and use schedule runs for the full-row endpoint."""
    lookup = {(run.dataset, run.model, run.seed, run.route): run for run in schedules.runs}
    result = _budget_contrasts(read_runs(Path(input_config["results_root"]), ["dataset_transfer_data_budget"]), lookup)
    if len(result) != 162:
        raise ValueError(f"Expected 162 Covertype budget contrasts, got {len(result)}.")
    return result


@dataclass(frozen=True)
class BudgetContrast:
    """One seed-paired fine-tuning-setting contrast against pure ICL."""

    model: str
    seed: int
    target_rows: int
    setting: str
    target_steps: int
    metric: str
    icl_value: float
    setting_value: float
    gain: float


def _budget_contrasts(
    records: Iterable[Run],
    e1_lookup: Mapping[tuple[str, str, int, str], Any],
) -> tuple[BudgetContrast, ...]:
    values: dict[tuple[str, int, int, str], tuple[str, float]] = {}
    baselines: dict[tuple[str, int, int], tuple[str, float]] = {}
    for run in records:
        report = run.report
        model, seed, metric = run.model, run.seed, report["score_metric"]
        for view in report["evaluations"]:
            axes = view["axes"]
            rows = int(axes["targetTrainRows"])
            condition = str(axes["targetBudgetCondition"])
            values[(model, seed, rows, condition)] = (
                metric,
                _metric(view["selected_checkpoint_metrics"]["test"], metric),
            )
            if condition == "satimage_initialized_100_target_steps":
                baselines[(model, seed, rows)] = (metric, _metric(view["zero_shot_metrics"]["test"], metric))
    for model in MODEL_ORDER:
        for seed in (0, 1, 2):
            target = e1_lookup[("covertype", model, seed, "target_compute_matched")]
            source = e1_lookup[("covertype", model, seed, "sequential_satimage_then_target")]
            values[(model, seed, 50000, "target_only_100_steps")] = (
                target.metric,
                float(target.target_100),
            )
            values[(model, seed, 50000, "target_only_200_steps")] = (
                target.metric,
                float(target.target_200),
            )
            values[(model, seed, 50000, "satimage_initialized_100_target_steps")] = (
                source.metric,
                source.test_value,
            )
            baselines[(model, seed, 50000)] = (source.metric, source.icl_value)
    contrasts = []
    for model in MODEL_ORDER:
        for seed in (0, 1, 2):
            for rows in BUDGET_ROWS:
                metric, icl = baselines[(model, seed, rows)]
                for setting, condition, target_steps in (
                    ("source_initialized", "satimage_initialized_100_target_steps", 100),
                    ("target_matched", "target_only_100_steps", 100),
                    ("total_matched", "target_only_200_steps", 200),
                ):
                    setting_metric, setting_value = values[(model, seed, rows, condition)]
                    if setting_metric != metric:
                        raise ValueError(f"Mixed primary metrics in E2 {model}/{seed}/{rows}.")
                    contrasts.append(
                        BudgetContrast(
                            model,
                            seed,
                            rows,
                            setting,
                            target_steps,
                            metric,
                            icl,
                            setting_value,
                            _gain(icl, setting_value, metric),
                        )
                    )
    return tuple(contrasts)


def _metric(metrics: Mapping[str, Any], metric: str) -> float:
    value = split_raw_metric_value(metrics, metric, score_metric=metric)
    if value is None:
        raise ValueError(f"Missing primary metric {metric!r}.")
    return float(value)


def _gain(control: float, method: float, metric: str) -> float:
    value = fine_tuning_gain(control, method, metric)
    if value is None:
        raise ValueError("A transfer follow-up contrast produced a non-finite gain.")
    return float(value)
