"""Regression tests for the expanded Section 2 reporting suite."""

import tomllib
from itertools import pairwise
from pathlib import Path

import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.container import ErrorbarContainer

from tfm_finetuning.plots.transformations import transformation_analysis as analysis
from tfm_finetuning.plots.transformations import transformation_overviews as overviews


def test_subsampling_pooling_uses_unweighted_dataset_means_and_cross_dataset_sem() -> None:
    rows = pd.DataFrame(
        [
            {"dataset": "a", "model": "mitra", "condition": "1/2", "value": 0.2},
            {"dataset": "a", "model": "mitra", "condition": "1/2", "value": 0.4},
            {"dataset": "b", "model": "mitra", "condition": "1/2", "value": 0.8},
        ]
    )

    pooled = overviews.pooled_subsampling_summary(
        rows,
        value_column="value",
        group_columns=("model", "condition"),
    ).iloc[0]

    assert pooled["mean"] == pytest.approx(0.55)
    assert pooled["count"] == 2
    assert pooled["sem"] == pytest.approx(0.25)


def test_subsampling_composite_has_one_asset_clean_terms_and_fixed_score_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, matplotlib.figure.Figure] = {}

    def capture(
        figure: matplotlib.figure.Figure,
        path: Path,
        **_kwargs: object,
    ) -> None:
        captured["figure"] = figure
        figure.savefig(path)

    monkeypatch.setattr(overviews, "save_section1_figure", capture)
    normalized_rows = []
    gain_rows = []
    conditions = ("1/16", "1/8", "1/4", "1/2", "1")
    condition_ranks = {condition: index for index, condition in enumerate(conditions)}
    for dataset_index, dataset in enumerate(("a", "b", "c")):
        for model_index, model in enumerate(("tabpfnv25", "tabiclv2", "mitra")):
            for condition in reversed(conditions):
                condition_index = condition_ranks[condition]
                for method in ("icl", "sft"):
                    normalized_rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "method": method,
                            "condition": condition,
                            "metric": "roc_auc",
                            "score_metric": "roc_auc",
                            "normalized_score_mean": 0.66
                            + 0.04 * condition_index
                            + 0.01 * model_index
                            + 0.005 * dataset_index
                            + (0.015 if method == "sft" else 0),
                        }
                    )
                gain_rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "condition": condition,
                        "metric": "roc_auc",
                        "score_metric": "roc_auc",
                        "gain_mean": 0.005 * condition_index + 0.002 * dataset_index,
                    }
                )

    output = overviews.plot_subsampling_pooled_composite(
        pd.DataFrame(normalized_rows),
        pd.DataFrame(gain_rows),
        output_path=tmp_path / "subsample_pooled_scores_and_gains.pdf",
        condition_order=conditions,
        condition_labels=("1/16", "1/8", "1/4", "1/2", "Full"),
    )

    assert output.name == "subsample_pooled_scores_and_gains.pdf"
    figure = captured["figure"]
    score_axis, gain_axis = figure.axes
    assert score_axis.get_ylim()[0] <= 0.6
    assert score_axis.get_ylim()[0] > 0.59
    assert score_axis.get_ylim()[1] >= 1.0
    assert 0.6 in score_axis.get_yticks()
    assert 1.0 in score_axis.get_yticks()
    latex_inclusion_width = overviews.latex_fraction_width(overviews.POOLED_COMPOSITE_LATEX_WIDTH_FRACTION)
    assert figure.get_figwidth() == pytest.approx(latex_inclusion_width)
    assert figure.get_figheight() == pytest.approx(overviews.POOLED_COMPOSITE_HEIGHT_INCHES)
    inclusion_scale = latex_inclusion_width / figure.get_figwidth()
    assert inclusion_scale == pytest.approx(1.0)
    assert score_axis.get_position().x1 < gain_axis.get_position().x0
    assert score_axis.get_position().y0 == pytest.approx(gain_axis.get_position().y0)
    assert score_axis.get_position().height == pytest.approx(gain_axis.get_position().height)
    assert score_axis.get_title(loc="left") == "A"
    assert gain_axis.get_title(loc="left") == "B"
    assert score_axis._left_title.get_fontsize() == pytest.approx(10.0)
    assert gain_axis._left_title.get_fontsize() == pytest.approx(10.0)
    assert all(label.get_visible() for label in score_axis.get_xticklabels())
    assert all(label.get_visible() for label in gain_axis.get_xticklabels())
    assert tuple(label.get_text() for label in score_axis.get_xticklabels()) == (
        "1/16",
        "1/8",
        "1/4",
        "1/2",
        "Full",
    )
    assert tuple(label.get_text() for label in gain_axis.get_xticklabels()) == (
        "1/16",
        "1/8",
        "1/4",
        "1/2",
        "Full",
    )
    assert len(figure.legends) == 1
    assert {text.get_text() for text in figure.legends[0].get_texts()} == {
        "TabPFN-2.5",
        "TabICLv2",
        "Mitra",
        "ICL",
        "SFT",
    }
    for axis in (score_axis, gain_axis):
        assert all(label.get_fontsize() * inclusion_scale >= 9.0 for label in axis.get_yticklabels())
        assert axis.yaxis.label.get_fontsize() * inclusion_scale >= 9.0
        series = [container.lines[0] for container in axis.containers if isinstance(container, ErrorbarContainer)]
        assert series
        assert all(tuple(line.get_xdata()) == tuple(range(len(conditions))) for line in series)
    assert all(label.get_fontsize() * inclusion_scale >= 9.0 for label in gain_axis.get_xticklabels())
    rendered_text = " ".join(text.get_text() for text in [*figure.texts, *score_axis.texts, *gain_axis.texts]).lower()
    assert "labeled budget" not in rendered_text
    assert "matched" not in rendered_text
    plt.close(figure)


def test_horizontal_alignment_labels_wrap_without_overlap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, matplotlib.figure.Figure] = {}

    def capture(figure, path, **_kwargs):
        captured["figure"] = figure
        figure.savefig(path)

    monkeypatch.setattr(overviews, "save_section1_figure", capture)
    conditions = ("canonical/canonical", "permuted/canonical", "canonical/permuted", "permuted/permuted")
    labels = ("Original", "Fit only", "Test only", "Both")
    normalized_rows = []
    gain_rows = []
    for dataset in ("a", "b"):
        for condition_index, condition in enumerate(conditions):
            for method in ("icl", "sft"):
                normalized_rows.append(
                    {
                        "dataset": dataset,
                        "model": "mitra",
                        "method": method,
                        "condition": condition,
                        "metric": "roc_auc",
                        "score_metric": "roc_auc",
                        "normalized_score_mean": 0.9 - 0.1 * condition_index,
                    }
                )
            gain_rows.append(
                {
                    "dataset": dataset,
                    "model": "mitra",
                    "condition": condition,
                    "metric": "roc_auc",
                    "score_metric": "roc_auc",
                    "gain_mean": 0.01 * condition_index,
                }
            )

    overviews.plot_pooled_primary_score_gain_composite(
        pd.DataFrame(normalized_rows),
        pd.DataFrame(gain_rows),
        output_path=tmp_path / "alignment.pdf",
        condition_order=conditions,
        condition_labels=labels,
        x_axis_label="Feature-order setting (fit/test)",
    )

    figure = captured["figure"]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for axis in figure.axes:
        tick_labels = axis.get_xticklabels()
        assert [label.get_text() for label in tick_labels] == [
            "Original",
            "Fit only",
            "Test only",
            "Both",
        ]
        bounds = [label.get_window_extent(renderer) for label in tick_labels]
        assert all(left.x1 + 1.0 <= right.x0 for left, right in pairwise(bounds))
    plt.close(figure)


def test_alignment_studies_only_supply_representation_comparisons() -> None:
    config = tomllib.loads(analysis.DEFAULT_CONFIG.read_text(encoding="utf-8"))

    assert set(config["alignment_studies"]) == {"feature_order", "numeric_scale"}
    assert {
        "spurious_correlation",
        "diamonds_mitigation",
        "contextual_scale",
        "heavy_tail",
    }.issubset(config["studies"])
