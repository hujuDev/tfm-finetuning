"""Focused tests for canonical transformation transformation result tables."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from tfm_finetuning.plots.transformations import transformation_analysis as analysis
from tfm_finetuning.plots.transformations import transformation_data as data


def _metric(value: float, metric: str) -> dict[str, float | str]:
    score = -value if metric in {"log_loss", "rmse", "mse", "mae"} else value
    return {
        "raw_metric_name": metric,
        "raw_metric_value": value,
        "score": score,
        metric: value,
    }


def _config(
    *,
    attempts: dict[str, list[str]],
    datasets: dict[str, list[str]],
    conditions: dict[str, list[str]],
    controls: dict[str, str],
    seeds: list[int] | None = None,
) -> dict[str, object]:
    return {
        "input": {"results_root": "results"},
        "studies": {family: values[0] for family, values in attempts.items()},
        "datasets": datasets,
        "fit_conditions": conditions,
        "controls": controls,
        "models": {"default": ["tabpfnv25"], "diamonds_mitigation": ["tabpfnv25"]},
        "splits": {
            "seeds": seeds or [0],
            "small_dataset_folds": [0, 1, 2],
            "standard_dataset_folds": [0],
            "small_datasets": [],
        },
    }


def _base_report(
    *,
    dataset: str,
    metric: str,
    shift: dict[str, object],
    seed: int = 0,
    baseline: float = 0.6,
    selected: float = 0.5,
    final: float = 0.55,
) -> dict[str, object]:
    return {
        "data_task": {
            "dataset_id": dataset,
            "fold": 0,
            "repeat": 0,
            "sample": 0,
            "distribution_shift": shift,
        },
        "split": {"base_seed": seed, "resolved_seed": seed},
        "finetune": {"model_id": "tabpfn_v2_5"},
        "score_metric": metric,
        "zero_shot_metrics": {"test": _metric(baseline, metric)},
        "selected_checkpoint_metrics": {"test": _metric(selected, metric)},
        "final_checkpoint_metrics": {"test": _metric(final, metric)},
    }


def _write_run(
    root: Path,
    *,
    experiment_id: str,
    run_id: str,
    dataset: str,
    task_type: str,
    report: dict[str, object],
) -> dict[str, object]:
    path = root / "results" / experiment_id / run_id
    path.mkdir(parents=True, exist_ok=True)
    (path / "finetune_report.json").write_text(json.dumps(report))
    (path / "run.json").write_text(json.dumps({
        "status": "complete", "study": experiment_id, "run_id": run_id, "dataset": dataset,
        "model": "tabpfnv25", "preprocessing": "shared", "seed": report["split"]["resolved_seed"],
        "fold": 0, "repeat": 0, "sample": 0, "parameters": report["data_task"]["distribution_shift"], "config": {},
    }))
    return {}


def _write_manifest(root: Path, experiment_id: str, summaries: list[dict[str, object]]) -> None:
    # Runs are now self-contained; no aggregate manifest is needed.
    pass


@pytest.mark.parametrize(
    ("family", "shift", "fit", "test", "combined", "setting", "dose"),
    [
        (
            "subsample",
            {"pre_finite": {"subsample_train": {"requested": "1/4"}}},
            "1/4",
            "original",
            "1/4",
            "fit_side_transformation",
            0.25,
        ),
        (
            "label_noise",
            {"pre_finite": {"label_noise": {"rate": 0.15}}},
            "0.15",
            "original",
            "0.15",
            "fit_side_transformation",
            0.15,
        ),
        (
            "feature_ablation",
            {"pre_finite": {"feature_mask": {"fraction": 0.2}}},
            "0.2",
            "0.2",
            "0.2",
            "matched_transformation",
            0.2,
        ),
        (
            "spurious_correlation",
            {"pre_finite": {"spurious_feature": {"strength": 0.8}}},
            "0.8",
            "neutral",
            "0.8",
            "fit_side_transformation",
            0.8,
        ),
        (
            "diamonds_mitigation",
            {"pre_finite": {"target_transform": {"mode": "robust_log1p"}}},
            "robust_log1p",
            "original",
            "robust_log1p",
            "fit_side_transformation",
            None,
        ),
        (
            "contextual_scale",
            {
                "pre_finite": {
                    "contextual_scale": {
                        "condition": "scale_sx_with_unit_column",
                        "transformation_setting": "matched_transformation",
                        "scale_spread": 3,
                    }
                }
            },
            "scale_sx_with_unit_column",
            "scale_sx_with_unit_column",
            "scale_sx_with_unit_column",
            "matched_transformation",
            3.0,
        ),
        (
            "heavy_tail",
            {
                "pre_finite": {
                    "heavy_tail_contamination": {"rate": 0.05},
                    "target_transform": {"mode": "robust_log1p"},
                }
            },
            "fit_side_contaminated/robust_log1p",
            "original",
            "fit_side_contaminated/robust_log1p",
            "fit_side_transformation",
            0.05,
        ),
    ],
)
def test_dose_family_conditions_are_reader_facing(
    family: str,
    shift: dict[str, object],
    fit: str,
    test: str,
    combined: str,
    setting: str,
    dose: float | None,
) -> None:
    condition = data._condition(family, {"data_task": {"distribution_shift": shift}})

    assert (condition.fit, condition.test, condition.combined) == (fit, test, combined)
    assert condition.transformation_setting == setting
    assert condition.dose == pytest.approx(dose) if dose is not None else condition.dose is None


def test_loader_pairs_selected_metrics_and_reports_missing_cells_without_zero_fill(tmp_path: Path) -> None:
    experiment_id = "accepted-label-noise"
    control_report = _base_report(
        dataset="dataset",
        metric="log_loss",
        shift={"pre_finite": {"label_noise": {"rate": 0}}},
        baseline=0.6,
        selected=0.5,
    )
    noisy_report = _base_report(
        dataset="dataset",
        metric="log_loss",
        shift={"pre_finite": {"label_noise": {"rate": 0.3}}},
        baseline=0.8,
        selected=0.9,
    )
    control = _write_run(
        tmp_path,
        experiment_id=experiment_id,
        run_id="control",
        dataset="dataset",
        task_type="multiclass",
        report=control_report,
    )
    noisy = _write_run(
        tmp_path,
        experiment_id=experiment_id,
        run_id="noisy",
        dataset="dataset",
        task_type="multiclass",
        report=noisy_report,
    )
    _write_manifest(tmp_path, experiment_id, [control, control, noisy])
    config = _config(
        attempts={"label_noise": [experiment_id]},
        datasets={"label_noise": ["dataset"]},
        conditions={"label_noise": ["0", "0.3"]},
        controls={"label_noise": "0"},
        seeds=[0, 1],
    )

    tables = data.build_transformation_tables(config, repo_root=tmp_path)

    gains = tables.run_gains[tables.run_gains["metric"] == "log_loss"].set_index("condition")
    assert gains.loc["0", "value"] == pytest.approx(math.log(0.6 / 0.5))
    assert gains.loc["0.3", "value"] == pytest.approx(math.log(0.8 / 0.9))
    changes = tables.control_contrasts[tables.control_contrasts["metric"] == "log_loss"].set_index("condition")
    assert changes.loc["0.3", "value"] == pytest.approx(math.log(0.8 / 0.9) - math.log(0.6 / 0.5))
    method_changes = tables.method_contrasts[
        (tables.method_contrasts["metric"] == "log_loss")
        & (tables.method_contrasts["condition"] == "0.3")
        & (tables.method_contrasts["checkpoint"].isin(["baseline", "selected"]))
    ].set_index("method")
    assert method_changes.loc["icl", "value"] == pytest.approx(math.log(0.6 / 0.8))
    assert method_changes.loc["sft", "value"] == pytest.approx(math.log(0.5 / 0.9))
    assert tables.coverage[["expected_run_units", "observed_run_units", "missing_run_units"]].sum().to_dict() == {
        "expected_run_units": 4,
        "observed_run_units": 2,
        "missing_run_units": 2,
    }
    assert len(tables.missing_run_units) == 2
    assert set(tables.missing_run_units["seed"]) == {1}
    log_loss_summary = tables.dataset_summary[tables.dataset_summary["metric"] == "log_loss"]
    assert set(log_loss_summary["gain_n"]) == {1}
    assert log_loss_summary["gain_sem"].isna().all()
    score_summary = tables.score_summary[tables.score_summary["metric"] == "log_loss"].set_index(
        ["condition", "method"]
    )
    assert score_summary.loc[("0", "sft"), "score_mean"] == pytest.approx(0.5)
    assert score_summary.loc[("0.3", "icl"), "score_mean"] == pytest.approx(0.8)
    normalized = tables.normalized_score_summary[
        tables.normalized_score_summary["metric"] == "log_loss"
    ].set_index(["condition", "method"])
    assert normalized.loc[("0", "sft"), "normalized_score_mean"] == pytest.approx(1.0)
    assert normalized.loc[("0.3", "sft"), "normalized_score_mean"] == pytest.approx(0.5 / 0.9)
    normalized_runs = tables.normalized_run_scores[
        tables.normalized_run_scores["metric"] == "log_loss"
    ].set_index(["condition", "method"])
    assert normalized_runs.loc[("0", "sft"), "value"] == pytest.approx(1.0)
    assert normalized_runs.loc[("0.3", "sft"), "value"] == pytest.approx(0.5 / 0.9)




def test_class_prior_displacement_is_the_change_in_fit_test_prior_gap() -> None:
    condition = data._condition(
        "class_prior",
        {
            "data_task": {
                "distribution_shift": {
                    "pre_finite": {
                        "rebalance_train": {
                            "requested_target_pos_rate": 0.5,
                            "resolved_target_pos_rate": 0.49,
                            "pos_rate_before": 0.2,
                            "pos_label": True,
                            "family_reference": False,
                        }
                    }
                }
            },
            "zero_shot_metrics": {
                "test": {
                    "classification_report": {
                        "False": {"support": 75},
                        "True": {"support": 25},
                        "macro avg": {"support": 100},
                    }
                }
            },
        },
    )

    assert condition.fit == "0.5"
    assert condition.prior_displacement == pytest.approx(0.19)


def test_same_checkpoint_feature_order_reports_all_four_fit_test_cells(tmp_path: Path) -> None:
    experiment_id = "accepted-feature-order"
    summaries = []
    for fit, offset in (("canonical", 0.0), ("permuted", 0.1)):
        report = _base_report(
            dataset="dataset",
            metric="roc_auc",
            shift={"post_finite": {"shuffle_feature_order": {"fit_state": fit}}},
        )
        report["distribution_shift_test_views"] = {
            "enabled": True,
            "same_fitted_state": True,
            "fit_state": fit,
            "primary_view": "canonical",
            "alternate_view": "permuted",
            "baseline_metrics": {
                "canonical": {"test": _metric(0.60 + offset, "roc_auc")},
                "permuted": {"test": _metric(0.50 + offset, "roc_auc")},
            },
            "selected_checkpoint_metrics": {
                "canonical": {"test": _metric(0.65 + offset, "roc_auc")},
                "permuted": {"test": _metric(0.55 + offset, "roc_auc")},
            },
            "final_checkpoint_metrics": {
                "canonical": {"test": _metric(0.64 + offset, "roc_auc")},
                "permuted": {"test": _metric(0.54 + offset, "roc_auc")},
            },
        }
        summaries.append(
            _write_run(
                tmp_path,
                experiment_id=experiment_id,
                run_id=f"run-{fit}",
                dataset="dataset",
                task_type="binary",
                report=report,
            )
        )
    _write_manifest(tmp_path, experiment_id, summaries)
    config = _config(
        attempts={"feature_order": [experiment_id]},
        datasets={"feature_order": ["dataset"]},
        conditions={"feature_order": ["canonical", "permuted"]},
        controls={"feature_order": "canonical/canonical"},
    )

    tables = data.build_transformation_tables(config, repo_root=tmp_path)

    gains = tables.run_gains[tables.run_gains["metric"] == "roc_auc"]
    assert set(gains["condition"]) == {
        "canonical/canonical",
        "canonical/permuted",
        "permuted/canonical",
        "permuted/permuted",
    }
    settings = gains.set_index("condition")["transformation_setting"].to_dict()
    assert settings["canonical/permuted"] == "test_side_transformation"
    assert settings["permuted/canonical"] == "fit_side_transformation"
    assert settings["permuted/permuted"] == "matched_transformation"
    assert tables.coverage.iloc[0]["observed_run_units"] == 2


@pytest.mark.parametrize(
    ("family", "shift_name", "control", "transformed_input", "transformed_condition"),
    [
        ("feature_order", "shuffle_feature_order", "canonical", "fixed_permutation", "permuted"),
        ("numeric_scale", "numeric_scale", "raw", "standardized", "standardized"),
    ],
)
def test_update_only_design_keeps_evaluation_representation_canonical(
    tmp_path: Path,
    family: str,
    shift_name: str,
    control: str,
    transformed_input: str,
    transformed_condition: str,
) -> None:
    experiment_id = f"accepted-{family}"
    summaries = []
    for update_state in (control, transformed_input):
        report = _base_report(
            dataset="dataset",
            metric="roc_auc",
            shift={"post_finite": {shift_name: {"update_state": update_state}}},
        )
        summaries.append(
            _write_run(
                tmp_path,
                experiment_id=experiment_id,
                run_id=f"run-{update_state}",
                dataset="dataset",
                task_type="binary",
                report=report,
            )
        )
    _write_manifest(tmp_path, experiment_id, summaries)
    config = _config(
        attempts={family: [experiment_id]},
        datasets={family: ["dataset"]},
        conditions={family: [control, transformed_condition]},
        controls={family: f"{control}/{control}"},
    )

    tables = data.build_transformation_tables(config, repo_root=tmp_path)

    gains = tables.run_gains[tables.run_gains["metric"] == "roc_auc"]
    assert set(gains["condition"]) == {
        f"{control}/{control}",
        f"{transformed_condition}/{control}",
    }
    settings = gains.set_index("condition")["transformation_setting"].to_dict()
    assert settings[f"{control}/{control}"] == "no_transformation"
    assert settings[f"{transformed_condition}/{control}"] == "fit_side_transformation"
    assert tables.coverage.iloc[0]["observed_run_units"] == 2


def test_diamonds_keeps_selected_final_and_validation_epoch_metrics(tmp_path: Path) -> None:
    experiment_id = "accepted-diamonds"
    report = _base_report(
        dataset="diamonds",
        metric="rmse",
        shift={"pre_finite": {"target_transform": {"mode": "none"}}},
        baseline=500,
        selected=300,
        final=900,
    )
    report["epoch_evaluations"] = [
        {"epoch": 0, "split": "val", "eval_source": "zero_shot_initial", **_metric(520, "rmse")},
        {"epoch": 1, "split": "val", "eval_source": "periodic", **_metric(310, "rmse")},
        {"epoch": 1, "split": "test", "eval_source": "periodic", **_metric(305, "rmse")},
    ]
    summary = _write_run(
        tmp_path,
        experiment_id=experiment_id,
        run_id="diamonds-none",
        dataset="diamonds",
        task_type="regression",
        report=report,
    )
    _write_manifest(tmp_path, experiment_id, [summary])
    config = _config(
        attempts={"diamonds_mitigation": [experiment_id]},
        datasets={"diamonds_mitigation": ["diamonds"]},
        conditions={"diamonds_mitigation": ["none"]},
        controls={"diamonds_mitigation": "none"},
    )

    tables = data.build_transformation_tables(config, repo_root=tmp_path)

    heldout = tables.run_metrics[tables.run_metrics["metric"] == "rmse"].set_index("checkpoint")
    assert heldout.loc["baseline", "value"] == pytest.approx(500)
    assert heldout.loc["selected", "value"] == pytest.approx(300)
    assert heldout.loc["final", "value"] == pytest.approx(900)
    trajectory = tables.epoch_metrics[tables.epoch_metrics["metric"] == "rmse"]
    assert set(trajectory["epoch"]) == {0, 1}
    assert set(trajectory["split"]) == {"val"}
    assert set(trajectory["method"]) == {"icl", "sft"}


def test_heavy_tail_stability_is_selected_to_final_log_rmse_ratio() -> None:
    rows = []
    for seed, selected, final in ((0, 10.0, 20.0), (1, 12.0, 24.0)):
        for checkpoint, value in (("selected", selected), ("final", final)):
            rows.append(
                {
                    "family": "heavy_tail",
                    "dataset": "dataset",
                    "model": "tabpfnv25",
                    "fold": 0,
                    "repeat": 0,
                    "sample": 0,
                    "seed": seed,
                    "condition": "fit_side_contaminated/robust_log1p",
                    "metric": "rmse",
                    "score_metric": "rmse",
                    "method": "sft",
                    "checkpoint": checkpoint,
                    "value": value,
                }
            )

    summary = analysis._heavy_tail_stability_summary(pd.DataFrame(rows)).iloc[0]

    assert summary["stability_mean"] == pytest.approx(math.log(0.5))
    assert summary["stability_n"] == 2
    assert summary["contamination"] == "fit_side_contaminated"
    assert summary["target_mode"] == "robust_log1p"
