"""Drift check pairing: same dataset and split, old vs new library."""

from __future__ import annotations

import math

import pytest

from tests.plots.preprint.synthetic_results import write_run
from tfm_finetuning.plots.preprint.drift import drift_summary, drift_table


def _write(root, study, dataset, seed, metric, baseline, selected):
    write_run(
        root,
        study=study,
        dataset=dataset,
        model="tabpfnv25",
        seed=seed,
        metric=metric,
        baseline=baseline,
        selected=selected,
    )


def test_drift_pairs_runs_per_split_and_signs_changes_by_metric_direction(tmp_path) -> None:
    for seed in (0, 1):
        _write(tmp_path, "overview_baseline", "binary_a", seed, "roc_auc", 0.80, 0.82)
        _write(tmp_path, "preprint_tabpfn_drift_check", "binary_a", seed, "roc_auc", 0.81, 0.82)
        _write(tmp_path, "overview_baseline", "reg_a", seed, "rmse", 4.0, 2.0)
        _write(tmp_path, "preprint_tabpfn_drift_check", "reg_a", seed, "rmse", 2.0, 2.0)
    # A baseline run without a counterpart is not paired.
    _write(tmp_path, "overview_baseline", "only_old", 0, "roc_auc", 0.9, 0.9)

    table = drift_table(tmp_path)

    assert len(table) == 4 and set(table["dataset"]) == {"binary_a", "reg_a"}
    binary = table.loc[table["dataset"] == "binary_a"].iloc[0]
    assert binary["icl_change"] == pytest.approx(0.01) and binary["sft_change"] == pytest.approx(0.0)
    regression = table.loc[table["dataset"] == "reg_a"].iloc[0]
    assert regression["icl_change"] == pytest.approx(math.log(2.0))
    assert regression["sft_change"] == pytest.approx(0.0)
    assert table["speedup"].tolist() == pytest.approx([1.0] * 4)


def test_drift_summary_averages_splits_per_dataset(tmp_path) -> None:
    for seed, new in ((0, 0.81), (1, 0.79)):
        _write(tmp_path, "overview_baseline", "binary_a", seed, "roc_auc", 0.80, 0.80)
        _write(tmp_path, "preprint_tabpfn_drift_check", "binary_a", seed, "roc_auc", new, new)

    summary = drift_summary(drift_table(tmp_path)).iloc[0]

    assert summary["splits"] == 2
    assert summary["icl_change"] == pytest.approx(0.0)
    assert summary["abs_icl_change"] == pytest.approx(0.01)


def test_drift_ignores_other_preprocessing_and_normalizes_dataset_names(tmp_path) -> None:
    for study, seconds_scale in (("overview_baseline", 1), ("preprint_tabpfn_drift_check", 1)):
        write_run(
            tmp_path,
            study=study,
            dataset="credit-g",
            model="tabpfnv25",
            seed=0,
            metric="roc_auc",
            baseline=0.8,
            selected=0.8,
            preprocessing="shared",
        )
        # The baseline study also holds the other preprocessing protocol, which must not be paired.
        write_run(
            tmp_path,
            study=study,
            dataset="credit-g",
            model="tabpfnv25",
            seed=0,
            metric="roc_auc",
            baseline=0.7,
            selected=0.7,
            preprocessing="native",
        )

    table = drift_table(tmp_path)

    assert len(table) == 1 and table.loc[0, "dataset"] == "credit_g"
    assert table.loc[0, "seconds_old"] == 600 and table.loc[0, "seconds_new"] == 600
