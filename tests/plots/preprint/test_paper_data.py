"""Panel construction, headroom closure, the per-model table, and run cost extraction used by the paper assets."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tests.plots.preprint.synthetic_results import STUDIES, write_results, write_run
from tfm_finetuning.plots.common.model_display import MODEL_COLORS, MODEL_LABELS, PAPER_MODELS, PERMISSIVE_MODELS
from tfm_finetuning.plots.preprint.data import (
    bootstrap_mean,
    cost_frame,
    cross_model_comparison,
    generation_summary,
    model_table,
    models_present,
    preprint_frame,
)


def _panel() -> pd.DataFrame:
    rows = []
    for dataset, task, icl, sft, effect in [
        ("a", "binary", 0.80, 0.90, 0.10),
        ("b", "binary", 0.95, 0.94, -0.01),
        ("c", "regression", 0.99, 0.99, 0.0),
        ("d", "regression", 0.90, 0.95, 0.05),
    ]:
        rows.append(
            {
                "dataset": dataset,
                "task_family": task,
                "model": "tabpfnv2",
                "normalized_baseline": icl,
                "normalized_selected": sft,
                "normalized_gain": sft - icl,
                "effect": effect,
            }
        )
    return pd.DataFrame(rows)


def test_model_table_counts_within_model_wins_from_the_gain() -> None:
    panel = _panel()
    summary = generation_summary(panel, seed=0, resamples=100)

    table = model_table(panel, summary).iloc[0]

    assert table["model"] == "tabpfnv2" and table["datasets"] == 4
    # A gain of exactly zero is neither an SFT nor an ICL win.
    assert (table["sft_wins"], table["icl_wins"]) == (2, 1)
    assert (table["sft_wins_binary"], table["datasets_binary"]) == (1, 2)
    assert (table["sft_wins_regression"], table["datasets_regression"]) == (1, 2)
    assert (table["sft_wins_multiclass"], table["datasets_multiclass"]) == (0, 0)
    assert table["gain"] == pytest.approx(np.mean([0.10, -0.01, 0.0, 0.05]))
    assert table["gain_low"] <= table["gain"] <= table["gain_high"]


def _write_cost_run(root, model, dataset, seed, *, total, in_fit_eval, allocated_mib, reserved_mib) -> None:
    write_run(root, study="s", dataset=dataset, model=model, seed=seed, metric="roc_auc", baseline=0.7, selected=0.8)
    path = next(root.glob(f"s/{dataset}-{model}-shared-{seed}/finetune_report.json"))
    report = json.loads(path.read_text())
    report["runtime"] = {
        "total_seconds": total,
        "phases": [
            {
                "phase": "fit_finetuned",
                "seconds": total - 10,
                "gpu_memory_peak_allocated_mib": allocated_mib,
                "gpu_memory_peak_reserved_mib": reserved_mib,
            },
            {"phase": "periodic_validation_eval", "seconds": 0, "in_fit_seconds": in_fit_eval},
        ],
    }
    path.write_text(json.dumps(report))


def test_cost_frame_removes_test_evaluation_time_and_keeps_the_largest_memory_peak(tmp_path) -> None:
    _write_cost_run(
        tmp_path, "tabpfnv25", "MiniData-x", 0, total=7200, in_fit_eval=1800, allocated_mib=2048, reserved_mib=4096
    )
    _write_cost_run(
        tmp_path, "tabpfnv25", "MiniData-x", 1, total=3600, in_fit_eval=0, allocated_mib=3072, reserved_mib=3072
    )
    _write_cost_run(tmp_path, "tabpfnv3", "airlines", 0, total=60, in_fit_eval=0, allocated_mib=1, reserved_mib=1)

    frame = cost_frame(tmp_path, studies=["s"]).set_index(["dataset", "model"])

    assert list(frame.index) == [("minidata_x", "tabpfnv25")]
    row = frame.loc[("minidata_x", "tabpfnv25")]
    assert row["runs"] == 2
    assert row["hours"] == pytest.approx((2.0 + 1.0) / 2)
    assert row["hours_without_test_eval"] == pytest.approx((1.5 + 1.0) / 2)
    assert row["peak_allocated_gib"] == pytest.approx(3.0)
    assert row["peak_reserved_gib"] == pytest.approx(4.0)


def test_cross_model_comparison_counts_wins_ties_and_losses_per_task() -> None:
    rows = []
    for dataset, task, perm_sft, newer_icl in [
        ("a", "binary", 0.95, 0.90),
        ("b", "binary", 0.90, 0.903),
        ("c", "regression", 0.80, 0.90),
    ]:
        rows.append(
            {
                "dataset": dataset,
                "task_family": task,
                "model": "tabiclv2",
                "normalized_baseline": 0.5,
                "normalized_selected": perm_sft,
            }
        )
        rows.append(
            {
                "dataset": dataset,
                "task_family": task,
                "model": "tabpfnv3",
                "normalized_baseline": newer_icl,
                "normalized_selected": 1.0,
            }
        )
    table = cross_model_comparison(pd.DataFrame(rows), fine_tuned=["tabiclv2"], without_fine_tuning=["tabpfnv3"])
    by = table.set_index("task_family")

    assert (by.loc["all", "wins"], by.loc["all", "ties"], by.loc["all", "losses"]) == (1, 1, 1)
    assert by.loc["all", "mean_difference"] == pytest.approx((0.05 - 0.003 - 0.10) / 3)
    assert (by.loc["binary", "datasets"], by.loc["regression", "losses"]) == (2, 1)
    assert by.loc["multiclass", "datasets"] == 0


def test_frame_keeps_only_the_default_preprocessing_and_normalizes_over_all_models(tmp_path) -> None:
    write_results(tmp_path)

    frame = preprint_frame(tmp_path, studies=STUDIES)

    assert set(frame["preprocessing"]) == {"shared"}
    assert models_present(frame) == PAPER_MODELS
    assert frame["dataset"].nunique() == 12
    assert len(frame) == 12 * len(PAPER_MODELS)
    # The best value of each dataset, over both ICL and SFT of every model, normalizes to one.
    best = frame.groupby("dataset")[["normalized_baseline", "normalized_selected"]].max().max(axis=1)
    assert np.allclose(best, 1.0)
    assert (frame["normalized_gain"] > 0).all()


def test_excluded_datasets_are_dropped_case_insensitively(tmp_path) -> None:
    write_results(tmp_path)
    for model in PAPER_MODELS:
        write_run(
            tmp_path,
            study="preprint_tabpfn_v3" if model.startswith("tabpfnv3") else "overview_baseline",
            dataset="MiniBooNE",
            model=model,
            seed=0,
            metric="roc_auc",
            baseline=0.9,
            selected=0.91,
        )

    frame = preprint_frame(tmp_path, studies=STUDIES)

    assert "miniboone" not in set(frame["dataset"])


def test_datasets_missing_for_one_model_are_dropped_for_all_models(tmp_path) -> None:
    write_results(tmp_path, skip={("binary_0", "tabpfnv35"), ("regression_3", "mitra")})

    frame = preprint_frame(tmp_path, studies=STUDIES)

    assert frame.attrs["dropped_datasets"] == ["binary_0", "regression_3"]
    assert frame["dataset"].nunique() == 10
    assert frame.groupby("model")["dataset"].nunique().nunique() == 1


def test_model_without_any_result_is_not_required_for_other_models(tmp_path) -> None:
    write_results(tmp_path)
    frame = preprint_frame(tmp_path, studies=STUDIES, models=[m for m in PAPER_MODELS if m != "tabpfnv35fast"])

    assert "tabpfnv35fast" not in models_present(frame)
    assert frame["dataset"].nunique() == 12


def test_bootstrap_is_deterministic_and_brackets_the_mean() -> None:
    values = [0.1, 0.2, 0.4, 0.3, 0.0, 0.25]

    first = bootstrap_mean(values, seed=1, resamples=500)

    assert first == bootstrap_mean(values, seed=1, resamples=500)
    assert first[0] == pytest.approx(np.mean(values)) and first[1] <= first[0] <= first[2]
    assert all(np.isnan(bootstrap_mean([float("nan")], seed=1, resamples=10)))


def test_every_model_has_a_label_and_color_and_permissive_models_are_known() -> None:
    assert set(PAPER_MODELS) <= set(MODEL_LABELS) == set(MODEL_COLORS)
    assert set(PERMISSIVE_MODELS) <= set(MODEL_LABELS)
    assert not {"tabpfnv25", "tabpfnv3", "tabpfnv35", "tabpfnv35fast"} & set(PERMISSIVE_MODELS)
