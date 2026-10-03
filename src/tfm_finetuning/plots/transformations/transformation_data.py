"""Build canonical run-level tables for transformation transformation findings.

Completed runs supply conventional metrics, selected-checkpoint results, and
same-checkpoint test views from their per-run fine-tuning reports.

Example:
    ``tables = build_transformation_tables(Path("configs/plots/transformations/transformation_analysis.toml"))``
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from tfm_finetuning.plots.metrics import (
    best_relative_normalized_score,
    fine_tuning_gain,
    metric_is_lower_better,
    normalize_metric_key,
    split_raw_metric_value,
)
from tfm_finetuning.plots.results import read_runs

REPO_ROOT = Path(__file__).resolve().parents[4]
METRICS = ("roc_auc", "accuracy", "log_loss", "rmse", "mse", "mae", "r2")
MODEL_KEYS = {
    "tabpfn_v2": "tabpfnv2",
    "tabpfnv2": "tabpfnv2",
    "tabpfn_v2_5": "tabpfnv25",
    "tabpfnv2_5": "tabpfnv25",
    "tabpfnv25": "tabpfnv25",
    "tabicl_v2": "tabiclv2",
    "tabiclv2": "tabiclv2",
    "mitra": "mitra",
}

BASE_COLUMNS = [
    "family",
    "dataset",
    "task_family",
    "score_metric",
    "metric",
    "model",
    "fold",
    "repeat",
    "sample",
    "seed",
    "run_id",
    "study",
    "report_path",
    "fit_condition",
    "test_condition",
    "condition",
    "transformation_setting",
    "dose",
    "prior_displacement",
    "method",
    "checkpoint",
    "value",
]
FIT_KEY = ["family", "dataset", "model", "fold", "repeat", "sample", "seed", "fit_condition"]
PAIR_KEY = [
    "family",
    "dataset",
    "task_family",
    "score_metric",
    "metric",
    "model",
    "fold",
    "repeat",
    "sample",
    "seed",
]
SUMMARY_KEY = [
    "family",
    "dataset",
    "task_family",
    "score_metric",
    "metric",
    "model",
    "fit_condition",
    "test_condition",
    "condition",
    "transformation_setting",
    "method",
    "checkpoint",
]


@dataclass(frozen=True)
class TransformationTables:
    """Canonical run-level, contrast, summary, trajectory, and coverage tables."""

    run_metrics: pd.DataFrame
    run_gains: pd.DataFrame
    normalized_run_scores: pd.DataFrame
    control_contrasts: pd.DataFrame
    method_contrasts: pd.DataFrame
    dataset_summary: pd.DataFrame
    method_summary: pd.DataFrame
    score_summary: pd.DataFrame
    normalized_score_summary: pd.DataFrame
    epoch_metrics: pd.DataFrame
    coverage: pd.DataFrame
    missing_run_units: pd.DataFrame


@dataclass(frozen=True)
class _Condition:
    fit: str
    test: str
    combined: str
    transformation_setting: str
    dose: float | None = None
    prior_displacement: float | None = None


def build_transformation_tables(
    config: Mapping[str, Any] | Path | str,
    *,
    repo_root: Path = REPO_ROOT,
) -> TransformationTables:
    """Build paired transformation measurements from fresh completed runs."""
    resolved = _read_config(config)
    expected = _expected_fit_units(resolved)
    metrics, epochs, observed = [], [], []
    studies = resolved["studies"]
    for family, study in studies.items():
        for run in read_runs(repo_root / resolved["input"]["results_root"], [study]):
            report = run.report
            score_metric = normalize_metric_key(report["score_metric"])
            base = {
                **{
                    key: run.metadata[key]
                    for key in ("dataset", "model", "fold", "repeat", "sample", "seed", "run_id")
                },
                "dataset": run.dataset.lower().replace("-", "_").replace(" ", "_"),
                "family": family,
                "task_family": _task_family(None, score_metric),
                "score_metric": score_metric,
                "study": study,
                "report_path": str(run.path.with_name("finetune_report.json")),
            }
            condition = _condition(family, report)
            observed.append({**{column: base[column] for column in FIT_KEY[:-1]}, "fit_condition": condition.fit})
            metrics.extend(_report_metric_rows(base, condition, report))
            if family == "diamonds_mitigation":
                epochs.extend(_validation_epoch_rows(base, condition, report))

    observed_frame = pd.DataFrame(observed, columns=FIT_KEY)
    _validate_logical_fits(observed_frame)
    run_metrics = pd.DataFrame(metrics, columns=BASE_COLUMNS)
    epoch_columns = [*BASE_COLUMNS, "epoch", "split", "eval_source"]
    epoch_metrics = pd.DataFrame(epochs, columns=epoch_columns)
    run_gains = _run_gain_table(run_metrics)
    controls = {str(key): str(value) for key, value in resolved.get("controls", {}).items()}
    control_contrasts = _control_gain_contrasts(run_gains, controls)
    method_contrasts = _method_performance_contrasts(run_metrics, controls)
    dataset_summary = _dataset_summary(run_gains, control_contrasts)
    method_summary = _summary(method_contrasts, prefix="performance_change")
    selected_scores = run_metrics[
        ((run_metrics["method"] == "icl") & (run_metrics["checkpoint"] == "baseline"))
        | ((run_metrics["method"] == "sft") & (run_metrics["checkpoint"] == "selected"))
    ]
    score_summary = _summary(selected_scores, prefix="score")
    normalized_run_scores = _normalized_scores(selected_scores)
    normalized_score_summary = _summary(normalized_run_scores, prefix="normalized_score")
    coverage, missing = _coverage_tables(expected, observed_frame)
    return TransformationTables(
        run_metrics=run_metrics,
        run_gains=run_gains,
        normalized_run_scores=normalized_run_scores,
        control_contrasts=control_contrasts,
        method_contrasts=method_contrasts,
        dataset_summary=dataset_summary,
        method_summary=method_summary,
        score_summary=score_summary,
        normalized_score_summary=normalized_score_summary,
        epoch_metrics=epoch_metrics,
        coverage=coverage,
        missing_run_units=missing,
    )


def _read_config(config: Mapping[str, Any] | Path | str) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    path = Path(config)
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _expected_fit_units(config: Mapping[str, Any]) -> pd.DataFrame:
    datasets = config.get("datasets", {})
    conditions = config.get("fit_conditions", {})
    model_config = config.get("models", {})
    splits = config.get("splits", {})
    seeds = [int(value) for value in splits.get("seeds", [0])]
    small = {str(value) for value in splits.get("small_datasets", [])}
    small_folds = [int(value) for value in splits.get("small_dataset_folds", [0])]
    standard_folds = [int(value) for value in splits.get("standard_dataset_folds", [0])]
    records: list[dict[str, Any]] = []
    for family, family_datasets in datasets.items():
        family_models = model_config.get(family, model_config.get("default", []))
        family_conditions = conditions.get(family, [])
        for dataset in family_datasets:
            folds = small_folds if str(dataset) in small else standard_folds
            for model, fold, seed, condition in product(family_models, folds, seeds, family_conditions):
                records.append(
                    {
                        "family": str(family),
                        "dataset": str(dataset),
                        "model": str(model),
                        "fold": int(fold),
                        "repeat": 0,
                        "sample": 0,
                        "seed": int(seed),
                        "fit_condition": str(condition),
                    }
                )
    return pd.DataFrame(records, columns=FIT_KEY).drop_duplicates(ignore_index=True)


def _task_family(value: object, score_metric: str) -> str:
    token = str(value or "").lower().replace("_classification", "")
    if token in {"binary", "multiclass", "regression"}:
        return token
    return {"roc_auc": "binary", "log_loss": "multiclass", "rmse": "regression"}.get(score_metric, "unknown")


def _condition(family: str, report: Mapping[str, Any]) -> _Condition:
    shift = (report.get("data_task") or {}).get("distribution_shift") or {}
    if family == "subsample":
        item = _shift_item(shift, "subsample_train")
        fit = str(item.get("requested", "1"))
        return _Condition(fit, "original", fit, _setting(fit == "1", "fit"), _fraction_value(fit))
    if family == "class_prior":
        item = _shift_item(shift, "rebalance_train")
        fit = "original" if item.get("family_reference") else _number_token(item.get("requested_target_pos_rate"))
        resolved = _finite(item.get("resolved_target_pos_rate"))
        original = _finite(item.get("pos_rate_before"))
        test_rate = _test_positive_rate(report, item.get("pos_label"))
        if resolved is not None and original is not None and test_rate is not None:
            displacement = abs(resolved - test_rate) - abs(original - test_rate)
        else:
            displacement = abs(resolved - original) if resolved is not None and original is not None else None
        return _Condition(fit, "original", fit, _setting(fit == "original", "fit"), resolved, displacement)
    if family == "label_noise":
        item = _shift_item(shift, "label_noise")
        fit = _number_token(item.get("rate", 0))
        dose = _finite(item.get("rate"))
        return _Condition(fit, "original", fit, _setting(dose == 0, "fit"), dose)
    if family == "feature_ablation":
        item = _shift_item(shift, "feature_mask")
        fit = _number_token(item.get("fraction", 0))
        dose = _finite(item.get("fraction"))
        return _Condition(fit, fit, fit, _setting(dose == 0, "matched"), dose)
    if family == "feature_order":
        views = report.get("distribution_shift_test_views") or {}
        item = _shift_item(shift, "shuffle_feature_order")
        fit = str(views.get("fit_state") or item.get("fit_state") or item.get("update_state") or "")
        if fit == "fixed_permutation":
            fit = "permuted"
        return _Condition(fit, "", "", "")
    if family == "numeric_scale":
        views = report.get("distribution_shift_test_views") or {}
        item = _shift_item(shift, "numeric_scale")
        fit = str(views.get("fit_state") or item.get("fit_state") or item.get("update_state") or "")
        return _Condition(fit, "", "", "")
    if family == "spurious_correlation":
        item = _shift_item(shift, "spurious_feature")
        fit = _number_token(item.get("strength", 0))
        dose = _finite(item.get("strength"))
        return _Condition(fit, "neutral", fit, _setting(dose == 0, "fit"), dose)
    if family == "diamonds_mitigation":
        item = _shift_item(shift, "target_transform")
        fit = str(item.get("mode") or "none")
        return _Condition(fit, "original", fit, _setting(fit == "none", "fit"))
    if family == "contextual_scale":
        item = _shift_item(shift, "contextual_scale")
        condition = str(item.get("condition") or "")
        setting = str(item.get("transformation_setting") or "")
        spread = _finite(item.get("scale_spread"))
        return _Condition(condition, condition, condition, setting, spread)
    if family == "heavy_tail":
        contamination = _shift_item(shift, "heavy_tail_contamination")
        target = _shift_item(shift, "target_transform")
        rate = _finite(contamination.get("rate"))
        contamination_label = "original" if rate == 0 else "fit_side_contaminated"
        target_mode = str(target.get("mode") or "none")
        condition = f"{contamination_label}/{target_mode}"
        is_control = contamination_label == "original" and target_mode == "none"
        return _Condition(condition, "original", condition, _setting(is_control, "fit"), rate)
    raise ValueError(f"Unsupported transformation family: {family}")


def _shift_item(shift: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    for stage in ("pre_finite", "post_finite"):
        stage_payload = shift.get(stage)
        if isinstance(stage_payload, Mapping) and isinstance(stage_payload.get(name), Mapping):
            return stage_payload[name]
    return {}


def _test_positive_rate(report: Mapping[str, Any], positive_label: object) -> float | None:
    test_metrics = (report.get("zero_shot_metrics") or {}).get("test") or {}
    class_report = test_metrics.get("classification_report") or {}
    positive = class_report.get(str(positive_label))
    total = class_report.get("macro avg") or class_report.get("macro_avg")
    if not isinstance(positive, Mapping) or not isinstance(total, Mapping):
        return None
    positive_support = _finite(positive.get("support"))
    total_support = _finite(total.get("support"))
    if positive_support is None or total_support is None or total_support <= 0:
        return None
    return positive_support / total_support


def _setting(is_control: bool, transformed: str) -> str:
    if is_control:
        return "no_transformation"
    return {
        "fit": "fit_side_transformation",
        "test": "test_side_transformation",
        "matched": "matched_transformation",
    }[transformed]


def _view_condition(family: str, fit: str, test: str) -> _Condition:
    original = "canonical" if family == "feature_order" else "raw"
    if fit == original and test == original:
        setting = "no_transformation"
    elif fit != original and test == original:
        setting = "fit_side_transformation"
    elif fit == original and test != original:
        setting = "test_side_transformation"
    elif fit == test:
        setting = "matched_transformation"
    else:
        raise ValueError(f"Unsupported {family} fit/test view: {fit}/{test}")
    return _Condition(fit, test, f"{fit}/{test}", setting)


def _report_metric_rows(
    base: Mapping[str, Any],
    fit_condition: _Condition,
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if base["family"] in {"feature_order", "numeric_scale"}:
        views = report.get("distribution_shift_test_views") or {}
        if not views.get("enabled"):
            original = "canonical" if base["family"] == "feature_order" else "raw"
            condition = _view_condition(str(base["family"]), fit_condition.fit, original)
            return _metric_rows_for_condition(
                base,
                condition,
                {
                    "baseline": ("icl", (report.get("zero_shot_metrics") or {}).get("test")),
                    "selected": ("sft", (report.get("selected_checkpoint_metrics") or {}).get("test")),
                    "final": ("sft", (report.get("final_checkpoint_metrics") or {}).get("test")),
                },
            )
        if not views.get("same_fitted_state"):
            raise ValueError(f"{base['family']} run {base['run_id']} lacks same-checkpoint test views.")
        test_views = [str(views.get("primary_view") or ""), str(views.get("alternate_view") or "")]
        rows: list[dict[str, Any]] = []
        for test_view in dict.fromkeys(test_views):
            condition = _view_condition(str(base["family"]), fit_condition.fit, test_view)
            rows.extend(
                _metric_rows_for_condition(
                    base,
                    condition,
                    {
                        "baseline": ("icl", (views.get("baseline_metrics") or {}).get(test_view, {}).get("test")),
                        "selected": (
                            "sft",
                            (views.get("selected_checkpoint_metrics") or {}).get(test_view, {}).get("test"),
                        ),
                        "final": ("sft", (views.get("final_checkpoint_metrics") or {}).get(test_view, {}).get("test")),
                    },
                )
            )
        return rows
    return _metric_rows_for_condition(
        base,
        fit_condition,
        {
            "baseline": ("icl", (report.get("zero_shot_metrics") or {}).get("test")),
            "selected": ("sft", (report.get("selected_checkpoint_metrics") or {}).get("test")),
            "final": ("sft", (report.get("final_checkpoint_metrics") or {}).get("test")),
        },
    )


def _metric_rows_for_condition(
    base: Mapping[str, Any],
    condition: _Condition,
    checkpoints: Mapping[str, tuple[str, object]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for checkpoint, (method, raw_metrics) in checkpoints.items():
        if not isinstance(raw_metrics, Mapping):
            continue
        for metric in _available_metrics(raw_metrics, str(base["score_metric"])):
            value = split_raw_metric_value(raw_metrics, metric, score_metric=base["score_metric"])
            if value is None:
                continue
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "fit_condition": condition.fit,
                    "test_condition": condition.test,
                    "condition": condition.combined,
                    "transformation_setting": condition.transformation_setting,
                    "dose": condition.dose,
                    "prior_displacement": condition.prior_displacement,
                    "method": method,
                    "checkpoint": checkpoint,
                    "value": value,
                }
            )
    return rows


def _available_metrics(metrics: Mapping[str, Any], score_metric: str) -> list[str]:
    explicit = {normalize_metric_key(key) for key in metrics}
    ordered = [score_metric, *METRICS]
    return list(dict.fromkeys(metric for metric in ordered if metric in explicit or metric == score_metric))


def _validation_epoch_rows(
    base: Mapping[str, Any],
    condition: _Condition,
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for evaluation in report.get("epoch_evaluations") or []:
        if evaluation.get("split") != "val":
            continue
        epoch = int(evaluation.get("epoch", 0))
        source = str(evaluation.get("eval_source") or "")
        method = "icl" if source == "zero_shot_initial" else "sft"
        for metric in _available_metrics(evaluation, str(base["score_metric"])):
            value = split_raw_metric_value(evaluation, metric, score_metric=base["score_metric"])
            if value is None:
                continue
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "fit_condition": condition.fit,
                    "test_condition": condition.test,
                    "condition": condition.combined,
                    "transformation_setting": condition.transformation_setting,
                    "dose": condition.dose,
                    "prior_displacement": condition.prior_displacement,
                    "method": method,
                    "checkpoint": f"epoch_{epoch}",
                    "value": value,
                    "epoch": epoch,
                    "split": "val",
                    "eval_source": source,
                }
            )
    return rows


def _run_gain_table(run_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [*BASE_COLUMNS, "baseline", "adapted"]
    if run_metrics.empty:
        return pd.DataFrame(columns=columns)
    identity = [column for column in BASE_COLUMNS if column not in {"method", "checkpoint", "value"}]
    baseline = run_metrics[(run_metrics["method"] == "icl") & (run_metrics["checkpoint"] == "baseline")]
    selected = run_metrics[(run_metrics["method"] == "sft") & (run_metrics["checkpoint"] == "selected")]
    pairs = (
        baseline[[*identity, "value"]]
        .rename(columns={"value": "baseline"})
        .merge(
            selected[[*identity, "value"]].rename(columns={"value": "adapted"}),
            on=identity,
            how="inner",
            validate="one_to_one",
        )
    )
    pairs["value"] = [
        fine_tuning_gain(baseline_value, adapted_value, metric)
        for baseline_value, adapted_value, metric in pairs[["baseline", "adapted", "metric"]].itertuples(
            index=False, name=None
        )
    ]
    pairs = pairs.dropna(subset=["value"])
    pairs["method"] = "sft_minus_icl"
    pairs["checkpoint"] = "selected"
    return pairs[columns]


def _control_gain_contrasts(run_gains: pd.DataFrame, controls: Mapping[str, str]) -> pd.DataFrame:
    columns = [*BASE_COLUMNS, "gain", "control_gain"]
    if run_gains.empty:
        return pd.DataFrame(columns=columns)
    frames: list[pd.DataFrame] = []
    for family, rows in run_gains.groupby("family", sort=False):
        control = controls.get(str(family))
        if control is None:
            continue
        control_rows = rows[rows["condition"] == control][[*PAIR_KEY, "value"]].rename(
            columns={"value": "control_gain"}
        )
        current = rows.rename(columns={"value": "gain"}).merge(
            control_rows,
            on=PAIR_KEY,
            how="inner",
            validate="many_to_one",
        )
        current["value"] = current["gain"] - current["control_gain"]
        frames.append(current[columns])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def _method_performance_contrasts(run_metrics: pd.DataFrame, controls: Mapping[str, str]) -> pd.DataFrame:
    columns = [*BASE_COLUMNS, "condition_value", "control_value"]
    if run_metrics.empty:
        return pd.DataFrame(columns=columns)
    comparison_key = [*PAIR_KEY, "method", "checkpoint"]
    frames: list[pd.DataFrame] = []
    for family, rows in run_metrics.groupby("family", sort=False):
        control = controls.get(str(family))
        if control is None:
            continue
        control_rows = rows[rows["condition"] == control][[*comparison_key, "value"]].rename(
            columns={"value": "control_value"}
        )
        current = rows.rename(columns={"value": "condition_value"}).merge(
            control_rows,
            on=comparison_key,
            how="inner",
            validate="many_to_one",
        )
        current["value"] = [
            fine_tuning_gain(control_value, condition_value, metric)
            for control_value, condition_value, metric in current[
                ["control_value", "condition_value", "metric"]
            ].itertuples(index=False, name=None)
        ]
        frames.append(current.dropna(subset=["value"])[columns])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def _dataset_summary(run_gains: pd.DataFrame, contrasts: pd.DataFrame) -> pd.DataFrame:
    gains = _summary(run_gains, prefix="gain")
    changes = _summary(contrasts, prefix="control_gain_change")
    if gains.empty:
        return gains
    change_columns = [
        *SUMMARY_KEY,
        "control_gain_change_mean",
        "control_gain_change_std",
        "control_gain_change_sem",
        "control_gain_change_n",
    ]
    return gains.merge(changes[change_columns], on=SUMMARY_KEY, how="left", validate="one_to_one")


def _summary(frame: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    columns = [
        *SUMMARY_KEY,
        "dose",
        "prior_displacement",
        f"{prefix}_mean",
        f"{prefix}_std",
        f"{prefix}_sem",
        f"{prefix}_n",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    grouped = frame.groupby(SUMMARY_KEY, dropna=False, sort=True)
    summary = grouped.agg(
        dose=("dose", "mean"),
        prior_displacement=("prior_displacement", "mean"),
        **{
            f"{prefix}_mean": ("value", "mean"),
            f"{prefix}_std": ("value", "std"),
            f"{prefix}_n": ("value", "count"),
        },
    ).reset_index()
    summary[f"{prefix}_sem"] = summary[f"{prefix}_std"] / summary[f"{prefix}_n"].pow(0.5)
    return summary[columns]


def _normalized_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Normalize every dataset/metric run unit across all candidate settings."""
    if scores.empty:
        return scores.copy()
    candidate_key = ["family", "dataset", "metric", "fold", "repeat", "sample", "seed"]
    frames: list[pd.DataFrame] = []
    for metric, metric_rows in scores.groupby("metric", sort=False):
        rows = metric_rows.copy()
        grouped = rows.groupby(candidate_key, dropna=False)["value"]
        best = grouped.transform("min" if metric_is_lower_better(metric) else "max")
        rows["value"] = [
            best_relative_normalized_score(value, best_value, metric)
            for value, best_value in zip(rows["value"], best, strict=True)
        ]
        frames.append(rows.dropna(subset=["value"]))
    return pd.concat(frames, ignore_index=True)


def _validate_logical_fits(observed: pd.DataFrame) -> None:
    if observed.empty:
        return
    duplicates = observed[observed.duplicated(FIT_KEY, keep=False)]
    if not duplicates.empty:
        sample = duplicates[FIT_KEY].drop_duplicates().head(5).to_dict(orient="records")
        raise ValueError(f"Multiple accepted reports map to the same logical fit cell: {sample}")


def _coverage_tables(expected: pd.DataFrame, observed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed_unique = observed.drop_duplicates(FIT_KEY) if not observed.empty else observed
    merged = expected.merge(observed_unique.assign(_observed=True), on=FIT_KEY, how="left")
    missing = merged[merged["_observed"].isna()][FIT_KEY].reset_index(drop=True)
    records: list[dict[str, Any]] = []
    for (family, dataset), rows in merged.groupby(["family", "dataset"], sort=True):
        observed_count = int(rows["_observed"].notna().sum())
        expected_count = len(rows)
        records.append(
            {
                "family": family,
                "dataset": dataset,
                "expected_run_units": expected_count,
                "observed_run_units": observed_count,
                "missing_run_units": expected_count - observed_count,
                "coverage_fraction": observed_count / expected_count if expected_count else math.nan,
                "complete": observed_count == expected_count,
            }
        )
    return pd.DataFrame(records), missing


def _number_token(value: object) -> str:
    finite = _finite(value)
    return format(finite, ".12g") if finite is not None else str(value)


def _fraction_value(value: object) -> float | None:
    try:
        return float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError):
        return _finite(value)


def _finite(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


__all__ = ["TransformationTables", "build_transformation_tables"]
