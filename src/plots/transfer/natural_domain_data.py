"""Read and aggregate completed natural-domain transfer runs.

Example:
    ``rows = load_natural_domain_rows(Path("results"))``
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from plots.metrics import (
    best_relative_normalized_score,
    fine_tuning_gain,
    metric_is_lower_better,
)
from plots.results import read_runs

PANEL_ORDER = ("heart", "wids", "covertype")
PANEL_LABELS = {
    "heart": "Heart-disease hospitals",
    "wids": "WiDS hospitals",
    "covertype": "Covertype areas",
}
PANEL_METRICS = {"heart": "roc_auc", "wids": "roc_auc", "covertype": "log_loss"}
PANEL_DATASETS = {
    "heart": (
        "heart_disease_cleveland",
        "heart_disease_hungary",
        "heart_disease_va_long_beach",
    ),
    "wids": (
        "wids_diabetes_hospital_118",
        "wids_diabetes_hospital_19",
        "wids_diabetes_hospital_188",
    ),
    "covertype": ("covertype_comanche_peak", "covertype_neota", "covertype_rawah"),
}
MODEL_ORDER = ("tabpfnv25", "tabiclv2", "mitra")
TARGET_100_ROUTE = "target_only_100_steps"
TARGET_PRODUCER_ROUTE = "target_control_300_steps"
FIXED_TARGET_CONTEXT_POLICY = "target_train_plus_validation"
ROUTE_MATCHED_CONTEXT_POLICY = "route_matched_train_plus_validation"
ROUTE_ORDER = (TARGET_100_ROUTE, "source_only_pooled", "target_inclusive_pooled")
ROUTE_LABELS = {
    TARGET_100_ROUTE: "Target\n$T_{100}$",
    "source_only_pooled": "Sources\n$(S^{A} + S^{B})_{200}$",
    "target_inclusive_pooled": "Mixed\n$(T + S^{A} + S^{B})_{300}$",
}
NORMALIZED_METHOD_ORDER = tuple(f"{route}_{phase}" for route in ROUTE_ORDER for phase in ("icl", "sft"))


@dataclass(frozen=True)
class NaturalDomainRow:
    """One target-domain result averaged over the three fixed seeds."""

    panel: str
    dataset: str
    model: str
    route: str
    metric: str
    baseline: float
    final: float
    gain: float
    gain_se: float
    seed_count: int


@dataclass(frozen=True)
class NaturalDomainNormalizedScore:
    """One target-wise best-relative score for an ICL or SFT route."""

    panel: str
    dataset: str
    model: str
    route: str
    phase: str
    score: float


def load_natural_domain_rows(
    results_root: Path, *, fixed_target_context: bool = False
) -> tuple[NaturalDomainRow, ...]:
    """Aggregate fresh natural-domain measurements with identical-context baselines."""
    context = FIXED_TARGET_CONTEXT_POLICY if fixed_target_context else ROUTE_MATCHED_CONTEXT_POLICY
    studies = ["dataset_transfer_natural_domain_training"]
    if not fixed_target_context:
        studies.append("dataset_transfer_natural_domain")
    values = defaultdict(dict)
    for run in read_runs(results_root, studies):
        report = run.report
        transfer = report["dataset_transfer"]
        route = transfer["route"]
        metric = report["score_metric"]
        if transfer.get("execution_mode") == "checkpoint_eval":
            route = transfer["checkpoint_route"]
            if route == TARGET_PRODUCER_ROUTE:
                continue  # The figure uses the producer's 100-update control, not this 300-update probe.
        if route == TARGET_PRODUCER_ROUTE:
            view = report["target_update_metrics"]["100"]
            route = TARGET_100_ROUTE
        elif route in ROUTE_ORDER and transfer["context_policy"] == context:
            view = report
        else:
            continue
        baseline = float(view["zero_shot_metrics"]["test"][metric])
        final = float(view["selected_checkpoint_metrics"]["test"][metric])
        gain = fine_tuning_gain(baseline, final, metric)
        if gain is None:
            raise ValueError(f"Invalid natural-domain gain: {run.path}")
        key = (run.dataset, run.model, route)
        if run.seed in values[key]:
            raise ValueError(f"Duplicate natural-domain seed {key}/{run.seed}")
        values[key][run.seed] = (baseline, final, gain)
    rows = []
    for panel, datasets in PANEL_DATASETS.items():
        for dataset in datasets:
            for model in MODEL_ORDER:
                for route in ROUTE_ORDER:
                    seeds = values[(dataset, model, route)]
                    if set(seeds) != {0, 1, 2}:
                        raise ValueError(f"Missing natural-domain seeds for {dataset}/{model}/{route}")
                    baseline, final, gain = zip(*seeds.values(), strict=True)
                    rows.append(
                        NaturalDomainRow(
                            panel,
                            dataset,
                            model,
                            route,
                            PANEL_METRICS[panel],
                            fmean(baseline),
                            fmean(final),
                            fmean(gain),
                            stdev(gain) / len(gain) ** 0.5,
                            len(gain),
                        )
                    )
    return tuple(rows)


def summarize_natural_domain_rows(rows: tuple[NaturalDomainRow, ...]) -> tuple[dict[str, Any], ...]:
    """Average measured gains across the three rotated targets in each panel."""
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row.panel, row.model, row.route, row.metric)].append(row.gain)
    return tuple(
        {
            "panel": panel,
            "model": model,
            "route": route,
            "metric": metric,
            "target_count": len(values),
            "mean_gain": fmean(values),
            "positive_target_count": sum(value > 0 for value in values),
        }
        for (panel, model, route, metric), values in sorted(grouped.items())
    )


def summarize_mixed_contrasts(rows: tuple[NaturalDomainRow, ...]) -> tuple[dict[str, Any], ...]:
    """Compare the equal-budget mixed route directly with target-only fine-tuning."""
    lookup = {(row.panel, row.dataset, row.model, row.route): row for row in rows}
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for panel in PANEL_ORDER:
        for dataset in PANEL_DATASETS[panel]:
            for model in MODEL_ORDER:
                target = lookup[(panel, dataset, model, TARGET_100_ROUTE)]
                mixed = lookup[(panel, dataset, model, "target_inclusive_pooled")]
                gain = fine_tuning_gain(target.final, mixed.final, target.metric)
                if gain is None:
                    raise ValueError(f"Non-finite mixed-route contrast for {dataset}/{model}.")
                grouped[(panel, model, target.metric)].append(gain)
    return tuple(
        {
            "panel": panel,
            "model": model,
            "metric": metric,
            "target_count": len(values),
            "mean_mixed_vs_target_gain": fmean(values),
            "mixed_wins": sum(value > 0 for value in values),
        }
        for (panel, model, metric), values in sorted(grouped.items())
    )


def summarize_normalized_scores(rows: tuple[NaturalDomainRow, ...]) -> tuple[dict[str, Any], ...]:
    """Average target-wise best-relative ICL and SFT scores within each panel."""
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for score in normalize_natural_domain_scores(rows):
        grouped[(score.panel, score.model, f"{score.route}_{score.phase}")].append(score.score)
    return tuple(
        {
            "panel": panel,
            "model": model,
            "method": method,
            "target_count": len(grouped[(panel, model, method)]),
            "mean_normalized_score": fmean(grouped[(panel, model, method)]),
        }
        for panel in PANEL_ORDER
        for model in MODEL_ORDER
        for method in NORMALIZED_METHOD_ORDER
    )


def normalize_natural_domain_scores(
    rows: tuple[NaturalDomainRow, ...],
) -> tuple[NaturalDomainNormalizedScore, ...]:
    """Normalize every route phase against the best result on the same target."""
    normalized_scores: list[NaturalDomainNormalizedScore] = []
    target_rows: dict[tuple[str, str], list[NaturalDomainRow]] = defaultdict(list)
    for row in rows:
        target_rows[(row.panel, row.dataset)].append(row)
    for (panel, dataset), candidates in target_rows.items():
        metric = candidates[0].metric
        raw_values = [value for row in candidates for value in (row.baseline, row.final)]
        best = min(raw_values) if metric_is_lower_better(metric) else max(raw_values)
        for model in MODEL_ORDER:
            model_rows = [row for row in candidates if row.model == model]
            method_values = {
                f"{row.route}_{phase}": value
                for row in model_rows
                for phase, value in (("icl", row.baseline), ("sft", row.final))
            }
            if set(method_values) != set(NORMALIZED_METHOD_ORDER):
                raise ValueError(f"Incomplete normalized-score candidates for {dataset}/{model}.")
            for method, value in method_values.items():
                normalized = best_relative_normalized_score(value, best, metric)
                if normalized is None:
                    raise ValueError(f"Invalid normalized score for {dataset}/{model}/{method}.")
                route, phase = method.rsplit("_", maxsplit=1)
                normalized_scores.append(
                    NaturalDomainNormalizedScore(
                        panel=panel,
                        dataset=dataset,
                        model=model,
                        route=route,
                        phase=phase,
                        score=normalized,
                    )
                )
    return tuple(normalized_scores)


def write_natural_domain_tables(rows: tuple[NaturalDomainRow, ...], output_dir: Path) -> tuple[Path, ...]:
    """Write measured target rows and panel means as reproducible CSV tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = (
        output_dir / "target_results.csv",
        output_dir / "panel_summary.csv",
        output_dir / "mixed_vs_target_control.csv",
        output_dir / "normalized_score_summary.csv",
    )
    _write_csv(paths[0], [asdict(row) for row in rows])
    _write_csv(paths[1], list(summarize_natural_domain_rows(rows)))
    _write_csv(paths[2], list(summarize_mixed_contrasts(rows)))
    _write_csv(paths[3], list(summarize_normalized_scores(rows)))
    return paths


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "FIXED_TARGET_CONTEXT_POLICY",
    "MODEL_ORDER",
    "NORMALIZED_METHOD_ORDER",
    "PANEL_DATASETS",
    "PANEL_LABELS",
    "PANEL_METRICS",
    "PANEL_ORDER",
    "ROUTE_LABELS",
    "ROUTE_MATCHED_CONTEXT_POLICY",
    "ROUTE_ORDER",
    "TARGET_100_ROUTE",
    "TARGET_PRODUCER_ROUTE",
    "NaturalDomainNormalizedScore",
    "NaturalDomainRow",
    "load_natural_domain_rows",
    "normalize_natural_domain_scores",
    "summarize_mixed_contrasts",
    "summarize_natural_domain_rows",
    "summarize_normalized_scores",
    "write_natural_domain_tables",
]
