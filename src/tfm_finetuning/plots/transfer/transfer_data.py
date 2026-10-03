"""Load and pair the completed transfer E1 transfer schedule evidence.

Run reports preserve seed pairing between every transfer route and its
target-only checkpoint views.

Example:
    ``tables = build_transfer_tables(Path("results"))``
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from tfm_finetuning.plots.metrics import (
    fine_tuning_gain,
    split_raw_metric_value,
)
from tfm_finetuning.plots.results import Run, read_runs

DATASET_ORDER = ("credit-g", "covertype", "satimage")
MODEL_ORDER = ("tabpfnv25", "tabiclv2", "mitra")
MODEL_ALIASES = {
    "tabpfn_v2_5": "tabpfnv25",
    "tabicl_v2": "tabiclv2",
    "mitra": "mitra",
}
TARGET_ROUTE = "target_compute_matched"

ROUTE_ORDER = {
    "credit-g": (
        "pooled_target_sources",
        "pooled_sources_then_target",
        "sequential_bank_then_heloc_then_target",
        "sequential_heloc_then_bank_then_target",
    ),
    "covertype": (
        "pooled_target_satimage",
        "sequential_satimage_then_target",
        "pooled_target_satimage_segment_ablation",
        "sequential_satimage_then_segment_then_target",
        "sequential_segment_then_satimage_then_target",
    ),
    "satimage": (
        "pooled_target_covertype",
        "sequential_covertype_then_target",
        "pooled_target_covertype_segment_ablation",
        "sequential_covertype_then_segment_then_target",
        "sequential_segment_then_covertype_then_target",
    ),
}
DISPLAY_ROUTE_ORDER = {
    "credit-g": (
        "pooled_target_sources",
        "sequential_bank_then_heloc_then_target",
        "sequential_heloc_then_bank_then_target",
    ),
    "covertype": ROUTE_ORDER["covertype"][2:],
    "satimage": ROUTE_ORDER["satimage"][2:],
}
CONTROL_SETTING_ORDER = {
    "credit-g": ("icl_only", "target_100", "target_300"),
    "covertype": ("icl_only", "target_100", "target_200", "target_300"),
    "satimage": ("icl_only", "target_100", "target_200", "target_300"),
}
ROUTE_LABELS = {
    "pooled_target_sources": "(T + Sᴬ + Sᴮ)₃₀₀",
    "pooled_sources_then_target": "(Sᴬ + Sᴮ)₂₀₀ → T₁₀₀",
    "sequential_bank_then_heloc_then_target": "Sᴬ₁₀₀ → Sᴮ₁₀₀ → T₁₀₀",
    "sequential_heloc_then_bank_then_target": "Sᴮ₁₀₀ → Sᴬ₁₀₀ → T₁₀₀",
    "pooled_target_satimage": "(T + Sᴬ)₂₀₀",
    "sequential_satimage_then_target": "Sᴬ₁₀₀ → T₁₀₀",
    "pooled_target_satimage_segment_ablation": "(T + Sᴬ + Sᴮ)₃₀₀",
    "sequential_satimage_then_segment_then_target": "Sᴬ₁₀₀ → Sᴮ₁₀₀ → T₁₀₀",
    "sequential_segment_then_satimage_then_target": "Sᴮ₁₀₀ → Sᴬ₁₀₀ → T₁₀₀",
    "pooled_target_covertype": "(T + Sᴬ)₂₀₀",
    "sequential_covertype_then_target": "Sᴬ₁₀₀ → T₁₀₀",
    "pooled_target_covertype_segment_ablation": "(T + Sᴬ + Sᴮ)₃₀₀",
    "sequential_covertype_then_segment_then_target": "Sᴬ₁₀₀ → Sᴮ₁₀₀ → T₁₀₀",
    "sequential_segment_then_covertype_then_target": "Sᴮ₁₀₀ → Sᴬ₁₀₀ → T₁₀₀",
}
ROUTE_COMPUTE_STEPS = {
    "credit-g": defaultdict(lambda: 300),
    "covertype": defaultdict(
        lambda: 300,
        {
            "pooled_target_satimage": 200,
            "sequential_satimage_then_target": 200,
        },
    ),
    "satimage": defaultdict(
        lambda: 300,
        {
            "pooled_target_covertype": 200,
            "sequential_covertype_then_target": 200,
        },
    ),
}
PAIR_SPECS = {
    "credit-g": (
        (
            "pooled_target_sources",
            "pooled_sources_then_target",
            "Joint pool vs.\nsource pool → target",
        ),
        (
            "sequential_bank_then_heloc_then_target",
            "sequential_heloc_then_bank_then_target",
            "Bank-last vs.\nHELOC-last source order",
        ),
    ),
    "covertype": (
        (
            "pooled_target_satimage",
            "sequential_satimage_then_target",
            "Joint pool vs.\nSatimage → target",
        ),
        (
            "pooled_target_satimage_segment_ablation",
            "pooled_target_satimage",
            "Add Segment\nto joint pool",
        ),
        (
            "sequential_satimage_then_segment_then_target",
            "sequential_segment_then_satimage_then_target",
            "Segment-last vs.\nSatimage-last source order",
        ),
    ),
    "satimage": (
        (
            "pooled_target_covertype",
            "sequential_covertype_then_target",
            "Joint pool vs.\nCovertype → target",
        ),
        (
            "pooled_target_covertype_segment_ablation",
            "pooled_target_covertype",
            "Add Segment\nto joint pool",
        ),
        (
            "sequential_covertype_then_segment_then_target",
            "sequential_segment_then_covertype_then_target",
            "Segment-last vs.\nCovertype-last source order",
        ),
    ),
}


@dataclass(frozen=True)
class RunMetric:
    """One held-out result with target-only checkpoint views."""

    dataset: str
    model: str
    seed: int
    route: str
    metric: str
    test_value: float
    icl_value: float
    target_100: float | None = None
    target_200: float | None = None
    target_300: float | None = None


@dataclass(frozen=True)
class Contrast:
    """One seed-paired, orientation-correct performance contrast."""

    dataset: str
    model: str
    seed: int
    route: str
    label: str
    comparison: str
    control_steps: int
    metric: str
    control_value: float
    route_value: float
    gain: float


@dataclass(frozen=True)
class PairContrast:
    """One direct seed-paired route comparison."""

    dataset: str
    model: str
    seed: int
    label: str
    left_route: str
    right_route: str
    metric: str
    left_value: float
    right_value: float
    gain: float


@dataclass(frozen=True)
class TransferTables:
    """Analysis-ready E1 tables."""

    runs: tuple[RunMetric, ...]
    contrasts: tuple[Contrast, ...]
    contrast_summary: tuple[dict[str, Any], ...]
    pairs: tuple[PairContrast, ...]
    pair_summary: tuple[dict[str, Any], ...]
    coverage: tuple[dict[str, Any], ...]


def build_transfer_tables(results_root: Path, *, study: str = "dataset_transfer") -> TransferTables:
    """Load fresh schedule runs and construct exactly paired comparisons."""
    runs = tuple(_load_run(run) for run in read_runs(results_root, [study]))
    _validate_grid(runs)
    controls = {(run.dataset, run.model, run.seed): run for run in runs if run.route == TARGET_ROUTE}
    contrasts: list[Contrast] = []
    for run in runs:
        if run.route == TARGET_ROUTE:
            continue
        control = controls[(run.dataset, run.model, run.seed)]
        for comparison, steps in (
            ("icl_only", 0),
            ("target_exposure", 100),
            ("compute_matched", int(ROUTE_COMPUTE_STEPS[run.dataset][run.route])),
        ):
            control_value = run.icl_value if comparison == "icl_only" else _control_value(control, steps)
            gain = fine_tuning_gain(control_value, run.test_value, run.metric)
            if gain is None:
                raise ValueError("A paired E1 contrast produced a non-finite gain.")
            contrasts.append(
                Contrast(
                    dataset=run.dataset,
                    model=run.model,
                    seed=run.seed,
                    route=run.route,
                    label=ROUTE_LABELS[run.route],
                    comparison=comparison,
                    control_steps=steps,
                    metric=run.metric,
                    control_value=control_value,
                    route_value=run.test_value,
                    gain=gain,
                )
            )

    lookup = {(run.dataset, run.model, run.seed, run.route): run for run in runs}
    pairs: list[PairContrast] = []
    for dataset, specs in PAIR_SPECS.items():
        for left, right, label in specs:
            for model in MODEL_ORDER:
                for seed in (0, 1, 2):
                    left_run = lookup[(dataset, model, seed, left)]
                    right_run = lookup[(dataset, model, seed, right)]
                    gain = fine_tuning_gain(
                        right_run.test_value,
                        left_run.test_value,
                        left_run.metric,
                    )
                    if gain is None:
                        raise ValueError("A direct route comparison produced a non-finite gain.")
                    pairs.append(
                        PairContrast(
                            dataset=dataset,
                            model=model,
                            seed=seed,
                            label=label,
                            left_route=left,
                            right_route=right,
                            metric=left_run.metric,
                            left_value=left_run.test_value,
                            right_value=right_run.test_value,
                            gain=gain,
                        )
                    )

    return TransferTables(
        runs=runs,
        contrasts=tuple(contrasts),
        contrast_summary=_summarize(
            contrasts, ("dataset", "model", "route", "label", "comparison", "control_steps", "metric")
        ),
        pairs=tuple(pairs),
        pair_summary=_summarize(pairs, ("dataset", "model", "label", "left_route", "right_route", "metric")),
        coverage=_coverage(runs),
    )


def write_transfer_tables(tables: TransferTables, output_dir: Path) -> tuple[Path, ...]:
    """Write auditable run-level and aggregate CSV evidence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = (
        _write_csv(output_dir / "run_level_primary_metrics.csv", tables.runs),
        _write_csv(output_dir / "paired_target_control_contrasts.csv", tables.contrasts),
        _write_csv(output_dir / "route_contrast_summary.csv", tables.contrast_summary),
        _write_csv(output_dir / "paired_route_contrasts.csv", tables.pairs),
        _write_csv(output_dir / "route_pair_summary.csv", tables.pair_summary),
        _write_csv(output_dir / "coverage.csv", tables.coverage),
    )
    return outputs


def training_order_icl_gains(
    tables: TransferTables,
    dataset: str,
    *,
    routes: tuple[str, ...] | None = None,
) -> tuple[list[tuple[str, int, str, float]], tuple[str, ...]]:
    """Calculate every displayed setting's primary-metric gain over its own ICL result."""
    all_routes = ROUTE_ORDER[dataset]
    selected_routes = all_routes if routes is None else routes
    unknown_routes = set(selected_routes).difference(all_routes)
    if unknown_routes:
        raise ValueError(f"Unknown routes requested for {dataset}: {sorted(unknown_routes)}")
    controls = CONTROL_SETTING_ORDER[dataset]
    dataset_runs = [row for row in tables.runs if row.dataset == dataset]
    gains: list[tuple[str, int, str, float]] = []
    for seed in (0, 1, 2):
        for model in MODEL_ORDER:
            model_runs = [row for row in dataset_runs if row.model == model and row.seed == seed]
            target_runs = [row for row in model_runs if row.route == TARGET_ROUTE]
            if len(target_runs) != 1:
                raise ValueError(
                    f"Expected one target-only run for {dataset}/{model}/{seed}, found {len(target_runs)}."
                )
            target = target_runs[0]
            available_control_values = {
                "icl_only": target.icl_value,
                "target_100": target.target_100,
                "target_200": target.target_200,
                "target_300": target.target_300,
            }
            for setting in controls:
                value = available_control_values[setting]
                if value is None:
                    raise ValueError(f"Missing {setting} control for {dataset}/{model}/{seed}.")
                gains.append((model, seed, setting, fine_tuning_gain(target.icl_value, value, target.metric)))
            route_runs = {row.route: row for row in model_runs if row.route != TARGET_ROUTE}
            if set(route_runs) != set(all_routes):
                raise ValueError(f"Incomplete transfer routes for {dataset}/{model}/{seed}.")
            for route in selected_routes:
                row = route_runs[route]
                gains.append((model, seed, route, fine_tuning_gain(target.icl_value, row.test_value, target.metric)))
    return gains, controls


def _load_run(run: Run) -> RunMetric:
    report = run.report
    route = report["dataset_transfer"]["route"]
    metric = report["score_metric"]
    value = _metric(report["selected_checkpoint_metrics"]["test"], metric)
    icl = _metric(report["zero_shot_metrics"]["test"], metric)
    views = report.get("target_update_metrics", {})
    boundaries = {
        f"target_{epoch}": _metric(view["selected_checkpoint_metrics"]["test"], metric)
        for epoch, view in views.items()
        if epoch in {"100", "200", "300"}
    }
    if route == TARGET_ROUTE:
        boundaries["target_300"] = value
    return RunMetric(run.dataset, run.model, run.seed, route, metric, value, icl, **boundaries)


def _metric(metrics: dict[str, Any], metric: str) -> float:
    value = split_raw_metric_value(metrics, metric, score_metric=metric)
    if value is None:
        raise ValueError(f"Missing primary metric {metric!r}.")
    return value


def _control_value(control: RunMetric, steps: int) -> float:
    value = getattr(control, f"target_{steps}", None)
    if value is None:
        raise ValueError(f"Target-only control has no {steps}-step checkpoint view.")
    return float(value)


def _validate_grid(runs: tuple[RunMetric, ...]) -> None:
    keys = {(run.dataset, run.model, run.seed, run.route) for run in runs}
    if len(keys) != len(runs):
        raise ValueError("E1 aggregate contains duplicate scientific run units.")
    expected = {
        (dataset, model, seed, route)
        for dataset in DATASET_ORDER
        for model in MODEL_ORDER
        for seed in (0, 1, 2)
        for route in (TARGET_ROUTE, *ROUTE_ORDER[dataset])
    }
    if keys != expected:
        raise ValueError(f"E1 grid mismatch: missing={expected - keys}, extra={keys - expected}")


def _summarize(rows: list[Any], keys: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        groups[tuple(getattr(row, key) for key in keys)].append(float(row.gain))
    summary: list[dict[str, Any]] = []
    for identity, values in groups.items():
        item = dict(zip(keys, identity, strict=True))
        item.update(
            mean_gain=fmean(values),
            sem_gain=stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0,
            seed_count=len(values),
        )
        summary.append(item)
    return tuple(summary)


def _coverage(runs: tuple[RunMetric, ...]) -> tuple[dict[str, Any], ...]:
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for run in runs:
        counts[(run.dataset, run.model, run.route)] += 1
    return tuple(
        {
            "dataset": dataset,
            "model": model,
            "route": route,
            "seed_count": count,
        }
        for (dataset, model, route), count in sorted(counts.items())
    )


def _write_csv(path: Path, rows: tuple[Any, ...]) -> Path:
    records = [asdict(row) if hasattr(row, "__dataclass_fields__") else row for row in rows]
    if not records:
        raise ValueError(f"Refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return path


__all__ = [
    "CONTROL_SETTING_ORDER",
    "DATASET_ORDER",
    "DISPLAY_ROUTE_ORDER",
    "MODEL_ORDER",
    "PAIR_SPECS",
    "ROUTE_LABELS",
    "ROUTE_ORDER",
    "TransferTables",
    "build_transfer_tables",
    "training_order_icl_gains",
    "write_transfer_tables",
]
