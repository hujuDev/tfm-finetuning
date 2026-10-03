from __future__ import annotations

import pytest

from plots.transfer.natural_domain_data import (
    MODEL_ORDER,
    PANEL_DATASETS,
    PANEL_METRICS,
    PANEL_ORDER,
    ROUTE_LABELS,
    ROUTE_ORDER,
    NaturalDomainRow,
    normalize_natural_domain_scores,
    summarize_normalized_scores,
)


def _complete_rows() -> tuple[NaturalDomainRow, ...]:
    return tuple(
        NaturalDomainRow(
            panel=panel,
            dataset=dataset,
            model=model,
            route=route,
            metric=PANEL_METRICS[panel],
            baseline=0.5,
            final=0.51,
            gain=0.01 * (target_index + 1),
            gain_se=0.001,
            seed_count=3,
        )
        for panel in PANEL_ORDER
        for target_index, dataset in enumerate(PANEL_DATASETS[panel])
        for model in MODEL_ORDER
        for route in ROUTE_ORDER
    )


def test_natural_domain_grid_has_243_seed_results() -> None:
    target_count = sum(len(PANEL_DATASETS[panel]) for panel in PANEL_ORDER)
    assert target_count * len(MODEL_ORDER) * len(ROUTE_ORDER) * 3 == 243




def test_natural_domain_panels_use_task_primary_metrics() -> None:
    assert PANEL_METRICS == {"heart": "roc_auc", "wids": "roc_auc", "covertype": "log_loss"}
    assert ROUTE_ORDER == (
        "target_only_100_steps",
        "source_only_pooled",
        "target_inclusive_pooled",
    )
    assert all("₁" not in label for label in ROUTE_LABELS.values())
    assert ROUTE_LABELS == {
        "target_only_100_steps": "Target\n$T_{100}$",
        "source_only_pooled": "Sources\n$(S^{A} + S^{B})_{200}$",
        "target_inclusive_pooled": "Mixed\n$(T + S^{A} + S^{B})_{300}$",
    }




def test_target_wise_normalized_scores_preserve_route_specific_baselines() -> None:
    scores = normalize_natural_domain_scores(_complete_rows())

    assert len(scores) == len(_complete_rows()) * 2
    heart_icl = next(
        score
        for score in scores
        if score.panel == "heart"
        and score.dataset == PANEL_DATASETS["heart"][0]
        and score.model == "tabpfnv25"
        and score.route == "target_only_100_steps"
        and score.phase == "icl"
    )
    heart_sft = next(
        score
        for score in scores
        if score.panel == "heart"
        and score.dataset == PANEL_DATASETS["heart"][0]
        and score.model == "tabpfnv25"
        and score.route == "target_only_100_steps"
        and score.phase == "sft"
    )
    assert heart_icl.score == pytest.approx(0.5 / 0.51)
    assert heart_sft.score == pytest.approx(1.0)


def test_normalized_score_summary_respects_metric_orientation() -> None:
    summary = summarize_normalized_scores(_complete_rows())
    lookup = {(str(row["panel"]), str(row["model"]), str(row["method"])): row for row in summary}

    assert lookup[("heart", "tabpfnv25", "target_only_100_steps_icl")]["mean_normalized_score"] == pytest.approx(
        0.5 / 0.51
    )
    assert lookup[("heart", "tabpfnv25", "source_only_pooled_sft")]["mean_normalized_score"] == pytest.approx(1.0)
    assert lookup[("covertype", "tabpfnv25", "target_only_100_steps_icl")][
        "mean_normalized_score"
    ] == pytest.approx(1.0)
    assert lookup[("covertype", "tabpfnv25", "source_only_pooled_sft")]["mean_normalized_score"] == pytest.approx(
        0.5 / 0.51
    )
    assert all(row["target_count"] == 3 for row in summary)
