"""Read Bank/Covertype retention and matched single-dataset references.

Example: ``references, evaluations = load_retention({"results_root": "results"})``.
The expected three-model, three-seed grid is checked before making the final table.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from plots.metrics import fine_tuning_gain, split_raw_metric_value
from plots.results import read_runs
from plots.transfer.transfer_data import MODEL_ALIASES, MODEL_ORDER

BANK = "Bank_Customer_Churn"
COVERTYPE = "covertype"
RETENTION_TARGETS = (BANK, COVERTYPE)
RETENTION_ROUTES = ("sequential_bank_then_covertype", "sequential_covertype_then_bank", "pooled_bank_covertype")


@dataclass(frozen=True)
class SingleDatasetReference:
    """One 100-update single-dataset checkpoint evaluated on its own dataset."""

    dataset: str
    model: str
    seed: int
    metric: str
    icl_value: float
    checkpoint_value: float
    gain: float


@dataclass(frozen=True)
class LearnedRetentionResult:
    """A terminal Bank/Covertype checkpoint evaluated on one of both datasets."""

    evaluation_target: str
    model: str
    seed: int
    route: str
    metric: str
    icl_value: float
    checkpoint_value: float
    gain: float


def load_retention(
    config: Mapping[str, Any],
) -> tuple[tuple[SingleDatasetReference, ...], tuple[LearnedRetentionResult, ...]]:
    """Load completed current reports and reject missing or duplicate evidence."""
    references, evaluations = [], []
    studies = ["dataset_transfer", "dataset_transfer_target_controls", "dataset_transfer_learned_retention_eval"]
    for run in read_runs(Path(config["results_root"]), studies):
        report = run.report
        transfer = report.get("dataset_transfer", {})
        model_id = report.get("foundation_model", {}).get("model_id")
        if model_id not in MODEL_ALIASES:
            continue
        model = MODEL_ALIASES[model_id]
        route = transfer.get("route")
        if (run.dataset == COVERTYPE and route == "target_compute_matched") or (
            run.dataset == BANK and route == "target_control_300_steps"
        ):
            metric = report["score_metric"]
            payload = report["target_update_metrics"]["100"]
            baseline, adapted, gain = _comparison(payload, metric)
            references.append(SingleDatasetReference(run.dataset, model, run.seed, metric, baseline, adapted, gain))
        if route == "learned_retention_probe":
            for view in report["evaluations"]:
                metric = view["score_metric"]
                baseline, adapted, gain = _comparison(view, metric)
                evaluations.append(
                    LearnedRetentionResult(
                        view["target"], model, run.seed, view["route"], metric, baseline, adapted, gain
                    )
                )
    expected = {(dataset, model, seed) for dataset in RETENTION_TARGETS for model in MODEL_ORDER for seed in (0, 1, 2)}
    observed = {(row.dataset, row.model, row.seed) for row in references}
    if observed != expected or len(references) != len(expected):
        raise ValueError(
            f"Single-dataset retention references incomplete or duplicated: missing={expected - observed}."
        )
    expected_evaluations = {(*key, route) for key in expected for route in RETENTION_ROUTES}
    observed_evaluations = {(row.evaluation_target, row.model, row.seed, row.route) for row in evaluations}
    if observed_evaluations != expected_evaluations or len(evaluations) != len(expected_evaluations):
        raise ValueError(
            f"Retention evaluations incomplete or duplicated: missing={expected_evaluations - observed_evaluations}."
        )
    by_reference = {(row.dataset, row.model, row.seed): row for row in references}
    for row in evaluations:
        if by_reference[(row.evaluation_target, row.model, row.seed)].metric != row.metric:
            raise ValueError("Retention and its matched single-dataset reference use different metrics.")
    return (
        tuple(sorted(references, key=lambda row: (row.dataset, row.model, row.seed))),
        tuple(sorted(evaluations, key=lambda row: (row.evaluation_target, row.route, row.model, row.seed))),
    )


def _comparison(payload: Mapping[str, Any], metric: str) -> tuple[float, float, float]:
    baseline = split_raw_metric_value(payload["zero_shot_metrics"]["test"], metric, score_metric=metric)
    adapted = split_raw_metric_value(payload["selected_checkpoint_metrics"]["test"], metric, score_metric=metric)
    gain = fine_tuning_gain(baseline, adapted, metric)
    if baseline is None or adapted is None or gain is None:
        raise ValueError(f"Retention comparison lacks a finite {metric} value.")
    return float(baseline), float(adapted), float(gain)


def _write_csv(path: Path, rows: Sequence[Any]) -> Path:
    records = [asdict(row) for row in rows]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return path
