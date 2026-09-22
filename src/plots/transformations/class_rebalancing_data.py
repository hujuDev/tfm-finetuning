"""Class-rebalancing analysis for matched full-data and balanced ICL contexts.

Example: ``summarize(shared_root, context_root, shared_suffix="d4d875", context_suffix="ee3c37")``.
Folds and seeds are averaged within datasets before means and standard errors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from plots.results import read_runs

MODEL_KEYS = {"tabpfn_v2_5": "tabpfnv25", "tabicl_v2": "tabiclv2", "mitra": "mitra"}


def _load_reports(root: Path, study: str) -> list[dict[str, Any]]:
    return [{"report": run.report, "metadata": run.metadata} for run in read_runs(root, [study])]


def _identity(payload: dict[str, Any]) -> dict[str, object]:
    return {key: payload["metadata"][key] for key in ("dataset", "model", "fold", "seed")}


def _class_accuracy(metrics: dict[str, Any], label: object) -> float:
    class_report = metrics["test"]["classification_report"]
    return float(class_report[str(label)]["recall"])


def _shared_differences_from_icl(reports: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for payload in reports:
        report = payload["report"]
        shift = report["data_task"]["distribution_shift"]["pre_finite"]["rebalance_train"]
        condition = "original" if shift["requested_target_focal_rate"] == "original" else "balanced"
        identity = _identity(payload)
        metric_sets = {
            "icl": report["zero_shot_metrics"],
            "sft": report["selected_checkpoint_metrics"],
        }
        for method, metrics in metric_sets.items():
            for role, label in (("majority", shift["focal_label"]), ("minority", shift["other_label"])):
                rows.append(
                    {
                        **identity,
                        "metric": f"{method}_{condition}",
                        "role": role,
                        "accuracy": _class_accuracy(metrics, label),
                    }
                )
    frame = pd.DataFrame(rows)
    keys = ["dataset", "model", "fold", "seed", "role"]
    if frame.duplicated([*keys, "metric"]).any():
        raise ValueError("Duplicate class-rebalancing observation in accepted reports")
    wide = frame.pivot_table(
        index=keys,
        columns="metric",
        values="accuracy",
        aggfunc="first",
    ).reset_index()
    required = ["icl_original", "sft_balanced"]
    if wide[required].isna().any().any():
        raise RuntimeError("Shared-downsampling reports do not form complete original/balanced pairs")
    wide["baseline"] = wide["icl_original"]
    wide["route_accuracy"] = wide["sft_balanced"]
    wide["difference"] = wide["route_accuracy"] - wide["baseline"]
    wide["route"] = "sft_shared"
    return wide


def _context_differences_from_icl(reports: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    full = "full_original_for_test_icl"
    balanced = "balanced_original_majority_downsampled_for_test_icl"
    for payload in reports:
        report = payload["report"]
        comparison = report["test_context_comparison"]
        shift = comparison["policies"][balanced]["majority_downsampling"]
        identity = _identity(payload)
        route_metrics = {
            "icl_balanced": comparison["baseline_metrics"][balanced],
            "sft_full": comparison["selected_checkpoint_metrics"][full],
            "sft_context": comparison["selected_checkpoint_metrics"][balanced],
        }
        for route, metrics in route_metrics.items():
            for role, label in (("majority", shift["focal_label"]), ("minority", shift["other_label"])):
                baseline = _class_accuracy(comparison["baseline_metrics"][full], label)
                route_accuracy = _class_accuracy(metrics, label)
                rows.append(
                    {
                        **identity,
                        "route": route,
                        "role": role,
                        "baseline": baseline,
                        "route_accuracy": route_accuracy,
                        "difference": route_accuracy - baseline,
                    }
                )
    return pd.DataFrame(rows)


def _validate_matching_identities(shared: pd.DataFrame, context: pd.DataFrame) -> None:
    keys = ["dataset", "model", "fold", "seed", "role"]
    shared_identities = shared[keys].drop_duplicates()
    context_identities = context[keys].drop_duplicates()
    matched = shared_identities.merge(
        context_identities,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not (matched["_merge"] == "both").all():
        raise RuntimeError("Shared and context-only studies do not contain the same run identities")


def summarize(results_root: Path, *, shared_study: str, context_study: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return class-accuracy differences from the common full-data ICL baseline."""
    shared_reports = _load_reports(results_root, shared_study)
    context_reports = _load_reports(results_root, context_study)
    if len(shared_reports) != 162 or len(context_reports) != 81:
        raise RuntimeError(
            "Expected 162 shared-downsampling and 81 context-only reports, "
            f"found {len(shared_reports)} and {len(context_reports)}"
        )
    shared = _shared_differences_from_icl(shared_reports)
    context = _context_differences_from_icl(context_reports)
    _validate_matching_identities(shared, context)
    observations = pd.concat(
        [
            shared[
                [
                    "dataset",
                    "model",
                    "fold",
                    "seed",
                    "route",
                    "role",
                    "baseline",
                    "route_accuracy",
                    "difference",
                ]
            ],
            context,
        ],
        ignore_index=True,
    )
    dataset = (
        observations.groupby(["dataset", "model", "route", "role"], as_index=False)[
            ["baseline", "route_accuracy", "difference"]
        ]
        .mean()
        .rename(
            columns={
                "baseline": "baseline_mean",
                "route_accuracy": "route_accuracy_mean",
                "difference": "dataset_mean",
            }
        )
    )
    summary = (
        dataset.groupby(["model", "route", "role"])["dataset_mean"]
        .agg(mean="mean", std="std", datasets="count")
        .reset_index()
    )
    summary["se"] = summary["std"] / np.sqrt(summary["datasets"])
    return dataset, summary
