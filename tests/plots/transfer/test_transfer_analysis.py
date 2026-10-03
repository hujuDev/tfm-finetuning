from __future__ import annotations

import pytest

from tfm_finetuning.plots.metrics import fine_tuning_gain
from tfm_finetuning.plots.transfer.transfer_data import (
    DATASET_ORDER,
    DISPLAY_ROUTE_ORDER,
    MODEL_ORDER,
    ROUTE_COMPUTE_STEPS,
    ROUTE_ORDER,
    TARGET_ROUTE,
    Contrast,
    RunMetric,
    TransferTables,
    training_order_icl_gains,
)
from tfm_finetuning.plots.transfer.transfer_figures import TRAINING_ORDER_ROUTE_LABELS


def _complete_training_order_tables() -> TransferTables:
    runs = []
    for dataset in DATASET_ORDER:
        metric = "roc_auc" if dataset == "credit-g" else "log_loss"
        for model_index, model in enumerate(MODEL_ORDER):
            for seed in range(3):
                if metric == "roc_auc":
                    icl = 0.72 + model_index * 0.02 + seed * 0.001
                    target_100 = icl + 0.01
                    target_200 = target_100 + 0.005
                    target_300 = target_100 + 0.008
                else:
                    icl = 0.34 - model_index * 0.02 + seed * 0.001
                    target_100 = icl - 0.03
                    target_200 = target_100 - 0.01
                    target_300 = target_100 - 0.015
                runs.append(
                    RunMetric(
                        dataset=dataset,
                        model=model,
                        seed=seed,
                        route=TARGET_ROUTE,
                        metric=metric,
                        test_value=target_300,
                        icl_value=icl,
                        target_100=target_100,
                        target_200=None if dataset == "credit-g" else target_200,
                        target_300=target_300,
                    )
                )
                for route_index, route in enumerate(ROUTE_ORDER[dataset]):
                    route_value = (
                        target_100 + route_index * 0.002 if metric == "roc_auc" else target_100 - route_index * 0.002
                    )
                    runs.append(
                        RunMetric(
                            dataset=dataset,
                            model=model,
                            seed=seed,
                            route=route,
                            metric=metric,
                            test_value=route_value,
                            icl_value=icl,
                        )
                    )
    return TransferTables(
        runs=tuple(runs),
        contrasts=tuple(
            Contrast(
                dataset=dataset,
                model=model,
                seed=seed,
                route=route,
                label=TRAINING_ORDER_ROUTE_LABELS[route],
                comparison=comparison,
                control_steps=0,
                metric="roc_auc" if dataset == "credit-g" else "log_loss",
                control_value=0.5,
                route_value=0.5,
                gain=(route_index - 1) * 0.005 + seed * 0.001,
            )
            for dataset in DATASET_ORDER
            for route_index, route in enumerate(ROUTE_ORDER[dataset])
            for model in MODEL_ORDER
            for comparison in ("icl_only",)
            for seed in range(3)
        ),
        contrast_summary=(),
        pairs=(),
        pair_summary=(),
        coverage=(),
    )


def test_training_order_gains_use_each_model_seed_icl_as_zero() -> None:
    records, controls = training_order_icl_gains(
        _complete_training_order_tables(),
        "covertype",
        routes=DISPLAY_ROUTE_ORDER["covertype"],
    )

    assert controls == ("icl_only", "target_100", "target_200", "target_300")
    icl_gains = [gain for _model, _seed, setting, gain in records if setting == "icl_only"]
    assert len(icl_gains) == len(MODEL_ORDER) * 3
    assert all(gain == pytest.approx(0.0) for gain in icl_gains)
    assert all(gain >= 0 for _model, _seed, setting, gain in records if setting != "icl_only")


def test_section3_route_grid_matches_153_scientific_units() -> None:
    route_count = sum(1 + len(ROUTE_ORDER[dataset]) for dataset in DATASET_ORDER)
    assert route_count * len(MODEL_ORDER) * 3 == 153


def test_remote_primary_routes_use_200_step_compute_control() -> None:
    assert ROUTE_COMPUTE_STEPS["covertype"]["pooled_target_satimage"] == 200
    assert ROUTE_COMPUTE_STEPS["satimage"]["sequential_covertype_then_target"] == 200
    assert ROUTE_COMPUTE_STEPS["covertype"]["pooled_target_satimage_segment_ablation"] == 300


def test_gain_orientation_is_consistent_across_metrics() -> None:
    assert fine_tuning_gain(0.7, 0.8, "roc_auc") == pytest.approx(0.1)
    assert fine_tuning_gain(0.8, 0.4, "log_loss") > 0


def test_source_boundary_budget_excludes_the_unrun_target_stage() -> None:
    assert ROUTE_COMPUTE_STEPS["credit-g"]["pooled_sources_then_target"] - 100 == 200
    assert ROUTE_COMPUTE_STEPS["covertype"]["sequential_satimage_then_target"] - 100 == 100
