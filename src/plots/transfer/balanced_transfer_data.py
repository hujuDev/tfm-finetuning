"""Load the balanced six-target transfer extension from exact run reports.

The main estimand compares source-initialized and target-only fits that use the
same target rows and 100 target updates.  The longer target-only trajectory is
retained in the audit table but is not part of that primary comparison.

Example:
    ``tables = build_balanced_transfer_tables(config, followups.budgets)``
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from plots.metrics import (
    fine_tuning_gain,
    split_raw_metric_value,
)
from plots.results import read_runs
from plots.transfer.followup_data import BUDGET_ROWS, BudgetContrast
from plots.transfer.transfer_data import MODEL_ORDER

TASK_DATASETS = {
    "multiclass": ("covertype", "sdss17"),
    "binary": ("amazon_employee_access", "diabetes130us"),
    "regression": ("physiochemical_protein", "food_delivery_time"),
}
DATASET_ORDER = tuple(dataset for datasets in TASK_DATASETS.values() for dataset in datasets)
DATASET_TASK = {dataset: task_family for task_family, datasets in TASK_DATASETS.items() for dataset in datasets}
DATASET_SOURCE = {
    "covertype": "satimage",
    "sdss17": "satimage",
    "amazon_employee_access": "bank_heloc",
    "diabetes130us": "bank_heloc",
    "physiochemical_protein": "houses",
    "food_delivery_time": "houses",
}
SOURCE_LABELS = {
    "satimage": "Satimage",
    "bank_heloc": "Bank + HELOC",
    "houses": "Houses",
}
SOURCE_IDENTITIES = {
    ("covertype", "sequential_satimage_then_target"): "satimage",
    ("credit-g", "pooled_sources_then_target"): "bank_heloc",
    ("houses", "source_only_checkpoint"): "houses",
}
SETTING_ALIASES = {
    "target_only_100_steps": "target_100",
    "target_only_200_steps": "target_200",
    "source_initialized_100_target_steps": "source_initialized",
    "satimage_initialized_100_target_steps": "source_initialized",
}


@dataclass(frozen=True)
class BalancedBudgetSetting:
    """One seed-level target-budget setting against its identical-context ICL run."""

    dataset: str
    task_family: str
    model: str
    seed: int
    target_rows: int
    source: str
    metric: str
    setting: str
    icl_value: float
    value: float
    gain_over_icl: float


@dataclass(frozen=True)
class SourceInitializationAdvantage:
    """Direct source-initialization effect with target rows and updates fixed."""

    dataset: str
    task_family: str
    model: str
    seed: int
    target_rows: int
    source: str
    metric: str
    target_only_value: float
    source_initialized_value: float
    advantage: float


@dataclass(frozen=True)
class BalancedSourceOnlyResult:
    """One source-only checkpoint evaluated with labeled target ICL context."""

    dataset: str
    task_family: str
    model: str
    seed: int
    source: str
    metric: str
    icl_value: float
    source_value: float
    source_gain: float
    target_only_value: float
    target_only_gain: float


@dataclass(frozen=True)
class BalancedTransferTables:
    """Analysis-ready balanced target-budget and source-only evidence."""

    budget_settings: tuple[BalancedBudgetSetting, ...]
    source_advantages: tuple[SourceInitializationAdvantage, ...]
    source_only: tuple[BalancedSourceOnlyResult, ...]


def build_balanced_transfer_tables(
    input_config: Mapping[str, Any],
    covertype_budgets: Iterable[BudgetContrast],
) -> BalancedTransferTables:
    """Build and validate the exact six-target balanced transfer grid."""
    settings = list(_covertype_settings(covertype_budgets))
    settings.extend(
        _load_budget_settings(
            input_config,
            prefix="balanced_classification",
            expected_producers=324,
        )
    )
    settings.extend(
        _load_budget_settings(
            input_config,
            prefix="balanced_regression",
            expected_producers=216,
        )
    )
    budget_settings = tuple(settings)
    source_advantages = pair_source_advantages(budget_settings)
    source_only = _load_source_only_results(
        input_config,
        budget_settings,
    )
    _validate_balanced_counts(budget_settings, source_advantages, source_only)
    return BalancedTransferTables(budget_settings, source_advantages, source_only)


def pair_source_advantages(
    settings: Iterable[BalancedBudgetSetting],
) -> tuple[SourceInitializationAdvantage, ...]:
    """Pair source-initialized and target-only 100-update settings exactly."""
    grouped: dict[tuple[str, str, int, int], dict[str, BalancedBudgetSetting]] = {}
    for row in settings:
        key = (row.dataset, row.model, row.seed, row.target_rows)
        if row.setting in grouped.setdefault(key, {}):
            raise ValueError(f"Duplicate balanced budget setting: {key}/{row.setting}.")
        grouped[key][row.setting] = row

    advantages = []
    for key, rows in sorted(grouped.items()):
        if set(rows) != {"target_100", "target_200", "source_initialized"}:
            raise ValueError(f"Incomplete balanced budget settings for {key}: {sorted(rows)}.")
        target = rows["target_100"]
        source = rows["source_initialized"]
        _validate_pair(target, source)
        advantages.append(
            SourceInitializationAdvantage(
                dataset=target.dataset,
                task_family=target.task_family,
                model=target.model,
                seed=target.seed,
                target_rows=target.target_rows,
                source=target.source,
                metric=target.metric,
                target_only_value=target.value,
                source_initialized_value=source.value,
                advantage=_gain(target.value, source.value, target.metric),
            )
        )
    return tuple(advantages)


def write_balanced_transfer_tables(
    tables: BalancedTransferTables,
    output_dir: Path,
) -> tuple[Path, ...]:
    """Write the exact seed-level evidence behind the balanced figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return (
        _write_csv(output_dir / "balanced_budget_settings.csv", tables.budget_settings),
        _write_csv(output_dir / "source_initialization_advantages.csv", tables.source_advantages),
        _write_csv(output_dir / "balanced_source_only_results.csv", tables.source_only),
    )


def _covertype_settings(
    rows: Iterable[BudgetContrast],
) -> tuple[BalancedBudgetSetting, ...]:
    setting_names = {
        "source_initialized": "source_initialized",
        "target_matched": "target_100",
        "total_matched": "target_200",
    }
    converted = []
    for row in rows:
        converted.append(
            BalancedBudgetSetting(
                dataset="covertype",
                task_family="multiclass",
                model=row.model,
                seed=row.seed,
                target_rows=row.target_rows,
                source="satimage",
                metric=row.metric,
                setting=setting_names[row.setting],
                icl_value=row.icl_value,
                value=row.setting_value,
                gain_over_icl=row.gain,
            )
        )
    if len(converted) != 162:
        raise ValueError(f"Expected 162 completed Covertype budget settings, found {len(converted)}.")
    return tuple(converted)


def _load_budget_settings(
    input_config: Mapping[str, Any], *, prefix: str, expected_producers: int
) -> tuple[BalancedBudgetSetting, ...]:
    """Read direct target-budget evaluations emitted by current training runs."""
    study = {
        "balanced_classification": "dataset_transfer_balanced_budget_classification",
        "balanced_regression": "dataset_transfer_balanced_budget_regression",
    }[prefix]
    records = list(read_runs(Path(input_config["results_root"]), [study]))
    if len(records) != expected_producers:
        raise ValueError(f"Expected {expected_producers} {study} runs, found {len(records)}")
    rows = []
    for run in records:
        for view in run.report["evaluations"]:
            axes = view["axes"]
            dataset = view["target"]
            metric, baseline, value = _view_metric(view, run.report)
            rows.append(
                BalancedBudgetSetting(
                    dataset,
                    DATASET_TASK[dataset],
                    run.model,
                    run.seed,
                    int(axes["targetTrainRows"]),
                    DATASET_SOURCE[dataset],
                    metric,
                    SETTING_ALIASES[axes["targetBudgetCondition"]],
                    baseline,
                    value,
                    _gain(baseline, value, metric),
                )
            )
    return tuple(rows)


def _load_source_only_results(
    input_config: Mapping[str, Any], budget_settings: tuple[BalancedBudgetSetting, ...]
) -> tuple[BalancedSourceOnlyResult, ...]:
    """Pair source checkpoint evaluations with equal-seed target-only controls."""
    targets = {
        (row.dataset, row.model, row.seed): row
        for row in budget_settings
        if row.target_rows == 50000 and row.setting == "target_100"
    }
    rows = []
    studies = ["dataset_transfer_source_only_eval_classification", "dataset_transfer_source_only_eval_regression"]
    for run in read_runs(Path(input_config["results_root"]), studies):
        transfer = run.report["dataset_transfer"]
        source = SOURCE_IDENTITIES[(transfer["checkpoint_origin"], transfer["checkpoint_route"])]
        for view in run.report["evaluations"]:
            dataset = view["target"]
            metric, baseline, value = _view_metric(view, run.report)
            target = targets[(dataset, run.model, run.seed)]
            if metric != target.metric:
                raise ValueError("Source-only and target-only metrics differ")
            rows.append(
                BalancedSourceOnlyResult(
                    dataset,
                    DATASET_TASK[dataset],
                    run.model,
                    run.seed,
                    source,
                    metric,
                    baseline,
                    value,
                    _gain(baseline, value, metric),
                    target.value,
                    target.gain_over_icl,
                )
            )
    return tuple(rows)


def _view_metric(view: Mapping[str, Any], report: Mapping[str, Any]) -> tuple[str, float, float]:
    metric = view["selected_checkpoint_metrics"]["test"].get("tabarena_metric", report["score_metric"])
    baseline = split_raw_metric_value(view["zero_shot_metrics"]["test"], metric, score_metric=metric)
    value = split_raw_metric_value(view["selected_checkpoint_metrics"]["test"], metric, score_metric=metric)
    if baseline is None or value is None:
        raise ValueError(f"Missing finite {metric} source or baseline value")
    return metric, float(baseline), float(value)


def _validate_pair(left: BalancedBudgetSetting, right: BalancedBudgetSetting) -> None:
    if (
        left.dataset,
        left.task_family,
        left.model,
        left.seed,
        left.target_rows,
        left.source,
        left.metric,
    ) != (
        right.dataset,
        right.task_family,
        right.model,
        right.seed,
        right.target_rows,
        right.source,
        right.metric,
    ) or not math.isclose(left.icl_value, right.icl_value, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("Balanced source and target settings are not exactly comparable.")


def _validate_balanced_counts(
    settings: tuple[BalancedBudgetSetting, ...],
    advantages: tuple[SourceInitializationAdvantage, ...],
    source_only: tuple[BalancedSourceOnlyResult, ...],
) -> None:
    expected_advantages = {
        (dataset, model, seed, rows)
        for dataset in DATASET_ORDER
        for model in MODEL_ORDER
        for seed in (0, 1, 2)
        for rows in BUDGET_ROWS
    }
    observed_advantages = {(row.dataset, row.model, row.seed, row.target_rows) for row in advantages}
    if len(settings) != 972 or observed_advantages != expected_advantages:
        raise ValueError("Balanced target-budget evidence is not the complete six-target grid.")
    expected_source_only = {
        (dataset, model, seed, source)
        for dataset in DATASET_ORDER
        for model in MODEL_ORDER
        for seed in (0, 1, 2)
        for source in (("satimage", "bank_heloc") if DATASET_TASK[dataset] != "regression" else ("houses",))
    }
    observed_source_only = {(row.dataset, row.model, row.seed, row.source) for row in source_only}
    if len(source_only) != 90 or observed_source_only != expected_source_only:
        raise ValueError("Balanced source-only evidence is not the complete 90-cell grid.")


def _gain(baseline: float, adapted: float, metric: str) -> float:
    value = fine_tuning_gain(baseline, adapted, metric)
    if value is None:
        raise ValueError("Balanced transfer contrast produced a non-finite gain.")
    return float(value)


def _write_csv(path: Path, rows: tuple[Any, ...]) -> Path:
    records = [asdict(row) for row in rows]
    if not records:
        raise ValueError(f"Refusing to write empty balanced-transfer table {path}.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return path


__all__ = [
    "BUDGET_ROWS",
    "DATASET_ORDER",
    "SOURCE_LABELS",
    "TASK_DATASETS",
    "BalancedBudgetSetting",
    "BalancedSourceOnlyResult",
    "BalancedTransferTables",
    "SourceInitializationAdvantage",
    "build_balanced_transfer_tables",
    "pair_source_advantages",
    "write_balanced_transfer_tables",
]
