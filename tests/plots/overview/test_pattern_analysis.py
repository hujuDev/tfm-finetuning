from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import pytest

import tfm_finetuning.plots.overview.pattern_diagnostic_plots as diagnostic_plots
from tfm_finetuning.plots.common import plot_style
from tfm_finetuning.plots.common.plot_style import FULL_WIDTH_INCHES, apply_plot_style
from tfm_finetuning.plots.figure_text_audit import inventory_visible_text
from tfm_finetuning.plots.metrics import fine_tuning_gain
from tfm_finetuning.plots.overview.pattern_data import _analysis_subset
from tfm_finetuning.plots.overview.pattern_diagnostic_plots import (
    _checkpoint_collapse_summary,
    _checkpoint_plot_frame,
    _plot_checkpoint_collapses_compact,
    _plot_checkpoint_gap,
)
from tfm_finetuning.plots.overview.pattern_metric_plots import (
    _draw_metric_gain_panel,
    _metric_display_name,
    _plot_metric_gains,
    _plot_preprocessing_difference,
)
from tfm_finetuning.plots.overview.pattern_model_comparison import (
    DEFAULT_DECISIVE_WINNER_MARGIN,
    _critical_difference_table,
    _draw_critical_difference_panel,
    _global_winner_table,
)
from tfm_finetuning.plots.overview.pattern_statistics import (
    _lodo_spearman,
    association_table,
    fine_tuning_outcome_table,
    structural_hypothesis_association_table,
)
from tfm_finetuning.plots.overview.result_table import dataset_result_table_latex, write_dataset_result_table


def test_effect_uses_auc_difference_and_log_error_ratio() -> None:
    """Task-family effects should be positive when fine-tuning improves."""
    assert fine_tuning_gain(0.7, 0.72, "roc_auc") == pytest.approx(0.02)
    assert fine_tuning_gain(2.0, 1.0, "rmse") == pytest.approx(math.log(2.0))
    assert fine_tuning_gain(0.4, 0.2, "log_loss") == pytest.approx(math.log(2.0))
    assert fine_tuning_gain(0.0, 0.0, "rmse") is None








def test_r2_metric_title_uses_full_size_unicode_superscript() -> None:
    """The exported R² title must not shrink its exponent through MathText."""
    frame = pd.DataFrame(
        [{"model": model, "effect": 0.01} for model in ["tabpfnv2", "tabpfnv25", "tabiclv2", "mitra"]]
    )
    figure, axis = plt.subplots()

    _draw_metric_gain_panel(axis, frame, "r2")

    assert _metric_display_name("r2") == "R²"
    assert axis.get_title() == "R²"
    assert "$" not in axis.get_title()
    assert axis.title.get_fontsize() >= 9.0
    plt.close(figure)


def test_metric_gain_y_labels_fit_the_fixed_canvas(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Publication layout must keep the leftmost metric label inside the PDF canvas."""
    frame = pd.DataFrame(
        [
            {"task_family": "multiclass", "metric": metric, "model": model, "effect": 0.01}
            for metric in ["accuracy", "log_loss"]
            for model in ["tabpfnv2", "tabpfnv25", "tabiclv2", "mitra"]
        ]
    )
    label_bounds: list[tuple[float, float]] = []

    def capture_bounds(figure: object, _path: Path, **_kwargs: object) -> None:
        figure.canvas.draw()
        bounds = figure.axes[0].yaxis.label.get_window_extent(figure.canvas.get_renderer())
        label_bounds.append((bounds.x0, figure.get_window_extent().width - bounds.x1))

    monkeypatch.setattr("tfm_finetuning.plots.common.plot_style.save_figure", capture_bounds)

    _plot_metric_gains(frame, tmp_path)

    assert label_bounds
    assert label_bounds[0][0] >= 0
    assert label_bounds[0][1] >= 0


def _preprocessing_comparison_frame() -> pd.DataFrame:
    rows = []
    for task_index, task in enumerate(("binary", "multiclass", "regression")):
        for model_index, model in enumerate(("tabpfnv2", "tabpfnv25", "tabiclv2", "mitra")):
            for dataset_index in range(3):
                native_gain = 0.01 * (task_index + model_index + dataset_index)
                for preprocessing, offset in (
                    ("native", 0.0),
                    ("shared", 0.005 * (dataset_index - 1)),
                ):
                    rows.append(
                        {
                            "dataset": f"{task}-{dataset_index}",
                            "model": model,
                            "task_family": task,
                            "preprocessing": preprocessing,
                            "effect": native_gain + offset,
                        }
                    )
    return pd.DataFrame(rows)


def test_preprocessing_difference_uses_exact_full_width_horizontal_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    original_save = plot_style.save_figure

    def capture_figure(figure: object, path: Path, **kwargs: object) -> None:
        captured["width"] = figure.get_figwidth()
        captured["axes"] = len(figure.axes)
        captured["titles"] = [axis.get_title() for axis in figure.axes]
        captured["inclusion_fraction"] = kwargs.get("inclusion_fraction", 1.0)
        original_save(figure, path, **kwargs)

    monkeypatch.setattr(plot_style, "save_figure", capture_figure)
    with mpl.rc_context():
        apply_plot_style()
        _plot_preprocessing_difference(_preprocessing_comparison_frame(), tmp_path)

    assert captured["width"] == pytest.approx(FULL_WIDTH_INCHES)
    assert captured["axes"] == 3
    assert captured["titles"] == ["Binary", "Multiclass", "Regression"]
    assert captured["inclusion_fraction"] == 1.0


def test_preprocessing_difference_visible_text_meets_final_print_floor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    effective_sizes: list[float] = []
    original_save = plot_style.save_figure

    def capture_text(figure: object, path: Path, **kwargs: object) -> None:
        effective_sizes.extend(
            element.effective_font_pt for element in inventory_visible_text(figure, exact_width=True)
        )
        original_save(figure, path, **kwargs)

    monkeypatch.setattr(plot_style, "save_figure", capture_text)
    with mpl.rc_context():
        apply_plot_style()
        _plot_preprocessing_difference(_preprocessing_comparison_frame(), tmp_path)

    assert effective_sizes
    assert min(effective_sizes) >= 8.0




def test_default_analysis_subset_uses_only_configured_preprocessing() -> None:
    """Main thesis analyses should not count both preprocessing paths as dataset evidence."""
    frame = pd.DataFrame(
        {
            "dataset": ["dataset", "dataset"],
            "preprocessing": ["native", "shared"],
            "effect": [0.1, 0.2],
        }
    )

    result = _analysis_subset(frame, preprocessing="shared")

    assert result[["dataset", "preprocessing", "effect"]].to_dict(orient="records") == [
        {"dataset": "dataset", "preprocessing": "shared", "effect": 0.2}
    ]






def test_dataset_result_table_uses_current_models_metrics_and_missing_cells() -> None:
    frame = pd.DataFrame(
        [
            {
                "dataset": "example_binary",
                "model": "tabpfnv2",
                "task_family": "binary",
                "metric": "roc_auc",
                "baseline": 0.7,
                "selected": 0.72,
                "effect": 0.02,
            },
            {
                "dataset": "example_binary",
                "model": "tabpfnv25",
                "task_family": "binary",
                "metric": "roc_auc",
                "baseline": 0.71,
                "selected": 0.73,
                "effect": -0.00001,
            },
            {
                "dataset": "example_regression",
                "model": "tabpfnv2",
                "task_family": "regression",
                "metric": "rmse",
                "baseline": 2.0,
                "selected": 1.5,
                "effect": 0.2877,
            },
        ]
    )

    result = dataset_result_table_latex(
        frame,
        model_order=["tabpfnv2", "tabpfnv25", "tabiclv2", "mitra"],
        model_labels={
            "tabpfnv2": "TabPFNv2",
            "tabpfnv25": "TabPFN-2.5",
            "tabiclv2": "TabICLv2",
            "mitra": "Mitra",
        },
    )

    assert r"\multicolumn{2}{c}{TabPFNv2}" in result
    assert r"\resultmetric{ROC-AUC}" in result
    assert r"\resultmetric{RMSE}" in result
    assert r"\resultft{0.72}{+0.0200}" in result
    assert r"\resultft{\bestresult{0.73}}{0.0000}" in result
    assert r"\resultft{\bestresult{1.5}}{+0.2877}" in result
    assert result.count("-- & --") == 5
    assert result.count(r"\resultdataset{") == 2
    assert r"\begin{landscape}" not in result
    assert "Fine-tuning results for individual datasets" in result
    # Auxiliary metrics cannot replace the task's primary metric in this table.
    auxiliary = frame.iloc[[0]].assign(metric="accuracy", baseline=0.999, selected=0.999)
    assert dataset_result_table_latex(
        pd.concat([frame, auxiliary]),
        model_order=["tabpfnv2", "tabpfnv25", "tabiclv2", "mitra"],
        model_labels={"tabpfnv2": "TabPFNv2", "tabpfnv25": "TabPFN-2.5", "tabiclv2": "TabICLv2", "mitra": "Mitra"},
    ) == result


def test_dataset_result_table_uses_overview_filename(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "dataset": "example_binary",
                "model": "tabpfnv2",
                "task_family": "binary",
                "metric": "roc_auc",
                "baseline": 0.7,
                "selected": 0.72,
                "effect": 0.02,
            }
        ]
    )

    path = write_dataset_result_table(
        frame,
        output_dir=tmp_path,
        model_order=["tabpfnv2"],
        model_labels={"tabpfnv2": "TabPFNv2"},
    )

    assert path.name == "default_overview_dataset_table.tex"
    assert path.exists()


def test_statistics_use_dataset_rows_and_report_lodo_sensitivity() -> None:
    """Association output should expose dataset count and leave-one-out stability."""
    frame = pd.DataFrame(
        {
            "dataset": [f"d{index}" for index in range(8)],
            "model": ["model"] * 8,
            "preprocessing": ["native"] * 8,
            "task_family": ["binary"] * 8,
            "effect": [float(index) for index in range(8)],
            "rows": [float(index + 1) for index in range(8)],
        }
    )

    result = association_table(frame, feature_columns=["rows"])

    assert result.iloc[0]["n_datasets"] == 8
    assert result.iloc[0]["spearman_rho"] == pytest.approx(1.0)
    assert result.iloc[0]["bootstrap_ci_low"] == pytest.approx(1.0)
    assert result.iloc[0]["bootstrap_ci_high"] == pytest.approx(1.0)
    assert result.iloc[0]["lodo_sign_stability"] == pytest.approx(1.0)
    assert result.iloc[0]["p_value_holm"] == pytest.approx(result.iloc[0]["p_value"])


def test_statistics_accept_five_datasets_by_default() -> None:
    frame = pd.DataFrame(
        {
            "dataset": [f"d{index}" for index in range(5)],
            "model": ["model"] * 5,
            "preprocessing": ["native"] * 5,
            "task_family": ["multiclass"] * 5,
            "effect": [float(index) for index in range(5)],
            "feature": [float(index + 1) for index in range(5)],
        }
    )

    result = association_table(frame, feature_columns=["feature"])

    assert result.iloc[0]["n_datasets"] == 5


def test_structural_associations_keep_only_declared_tasks_and_direction() -> None:
    frame = pd.DataFrame(
        {
            "dataset": [f"d{index}" for index in range(8)] * 2,
            "model": ["model"] * 16,
            "preprocessing": ["shared"] * 16,
            "task_family": ["binary"] * 8 + ["regression"] * 8,
            "effect": [float(index) for index in range(8)] * 2,
            "structure": [float(index + 1) for index in range(8)] * 2,
        }
    )

    result = structural_hypothesis_association_table(
        frame,
        hypotheses=[
            {
                "feature": "structure",
                "family": "signal_geometry",
                "expected_direction": "positive",
                "tasks": ["binary"],
                "hypothesis": "Distributed signal should increase SFT gain.",
            }
        ],
    )

    assert result["task_family"].tolist() == ["binary"]
    assert result.iloc[0]["direction_matches"]
    assert result.iloc[0]["family"] == "signal_geometry"


def test_lodo_spearman_recomputes_each_reduced_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each LODO estimate must omit one pair and rerank the remaining observations."""
    x = pd.Series([1.0, 2.0, 2.0, 4.0]).to_numpy()
    y = pd.Series([1.0, 4.0, 2.0, 3.0]).to_numpy()
    calls: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
    real_spearmanr = __import__("scipy").stats.spearmanr

    def recording_spearmanr(left: object, right: object) -> object:
        calls.append((tuple(left), tuple(right)))
        return real_spearmanr(left, right)

    monkeypatch.setattr("tfm_finetuning.plots.overview.pattern_statistics.stats.spearmanr", recording_spearmanr)

    result = _lodo_spearman(x, y)

    assert [len(left) for left, _ in calls] == [3, 3, 3, 3, 4]
    assert calls == [
        ((2.0, 2.0, 4.0), (4.0, 2.0, 3.0)),
        ((1.0, 2.0, 4.0), (1.0, 2.0, 3.0)),
        ((1.0, 2.0, 4.0), (1.0, 4.0, 3.0)),
        ((1.0, 2.0, 2.0), (1.0, 4.0, 2.0)),
        ((1.0, 2.0, 2.0, 4.0), (1.0, 4.0, 2.0, 3.0)),
    ]
    assert result["lodo_rho_min"] < result["lodo_rho_max"]


def test_fine_tuning_outcomes_use_bh_detectability_and_preserve_untestable_rows() -> None:
    """Nonsignificant and single-run gains must not be labelled neutral effects."""
    frame = pd.DataFrame(
        {
            "dataset": ["help", "harm", "not_detected", "single_run"],
            "effect": [0.1, -0.1, 0.0, 0.1],
            "effect_se": [0.001, 0.001, 0.001, None],
            "run_count": [3, 3, 3, 1],
        }
    )

    outcomes = fine_tuning_outcome_table(frame).set_index("dataset")

    assert outcomes.loc["help", "outcome"] == "help"
    assert outcomes.loc["harm", "outcome"] == "harm"
    assert outcomes.loc["not_detected", "outcome"] == "not_detected"
    assert outcomes.loc["single_run", "outcome"] == "insufficient_replication"
    assert math.isnan(outcomes.loc["single_run", "q_value_bh"])


def test_checkpoint_plot_excludes_catastrophic_error_collapse() -> None:
    """An epoch-100 error above ten times ICL should not set the plot scale."""
    frame = pd.DataFrame(
        {
            "dataset": ["stable", "collapsed"],
            "score_metric": ["rmse", "rmse"],
            "final_effect": [-0.1, -math.log(11.0)],
            "checkpoint_gap": [0.1, 3.0],
        }
    )

    plotted = _checkpoint_plot_frame(frame)

    assert plotted["dataset"].tolist() == ["stable"]


def test_checkpoint_plot_keeps_collapses_visible_without_flattening_central_panel(tmp_path, monkeypatch) -> None:
    """The inclusive view spans collapses, while the central view retains visible boxes."""
    frame = pd.DataFrame(
        {
            "model": ["tabpfnv2", "tabpfnv2", "tabpfnv2", "tabpfnv2"],
            "score_metric": ["rmse"] * 4,
            "final_effect": [-0.1, -0.1, -0.1, -math.log(11.0)],
            "checkpoint_gap": [-0.03, 0.0, 0.05, 3.0],
        }
    )
    captured = {}

    def capture(fig, path, **kwargs):
        captured["figure"] = fig
        return path

    monkeypatch.setattr(diagnostic_plots, "save_and_close_figure", capture)
    _plot_checkpoint_gap(frame, tmp_path)
    figure = captured["figure"]
    try:
        inclusive, central = figure.axes
        assert inclusive.get_ylim()[1] >= 3.0
        assert 0.05 <= central.get_ylim()[1] < 0.5
    finally:
        plt.close(figure)


def test_compact_checkpoint_plot_isolates_four_tabpfn_regression_collapses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "task_family": "regression",
                "model": model,
                "score_metric": "rmse",
                "final_effect": -math.log(ratio),
                "checkpoint_gap": math.log(ratio),
            }
            for dataset, model, ratio in (
                ("diamonds", "tabpfnv2", 40.0),
                ("diamonds", "tabpfnv25", 73.0),
                ("miami_housing", "tabpfnv2", 16.0),
                ("miami_housing", "tabpfnv25", 29.0),
                ("stable", "tabpfnv2", 1.02),
            )
        ]
    )
    collapses, stable = _checkpoint_collapse_summary(frame)

    assert len(collapses) == 4
    assert stable["dataset"].tolist() == ["stable"]
    assert sorted(collapses["epoch100_error_ratio"].round().astype(int)) == [16, 29, 40, 73]

    captured: dict[str, object] = {}

    def capture(fig, path, **kwargs):
        captured.update(figure=fig, path=path, kwargs=kwargs)
        return path

    monkeypatch.setattr(diagnostic_plots, "save_and_close_figure", capture)
    path = _plot_checkpoint_collapses_compact(frame, tmp_path)

    assert path.name == "checkpoint_collapses_compact.pdf"
    assert captured["kwargs"]["inclusion_fraction"] == pytest.approx(0.48)
    axis = captured["figure"].axes[0]
    assert axis.get_xscale() == "log"
    assert [label.get_text() for label in axis.get_yticklabels()] == [
        "Miami Housing",
        "Diamonds",
        "Other regression\ndatasets",
    ]
    plt.close(captured["figure"])


def test_critical_difference_ranks_icl_and_sft_jointly_on_complete_datasets() -> None:
    """CD inputs should compare every model-method combination on one paired panel."""
    frame = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "model": model,
                "task_family": "binary",
                "score_metric": "roc_auc",
                "baseline": baseline,
                "selected": selected,
            }
            for dataset, values in {
                "d1": {"tabpfnv2": (0.50, 0.70), "tabiclv2": (0.80, 0.90)},
                "d2": {"tabpfnv2": (0.60, 0.65), "tabiclv2": (0.75, 0.85)},
            }.items()
            for model, (baseline, selected) in values.items()
        ]
        + [
            {
                "dataset": "incomplete",
                "model": "tabpfnv2",
                "task_family": "binary",
                "score_metric": "roc_auc",
                "baseline": 0.95,
                "selected": 0.96,
            }
        ]
    )

    comparisons = _critical_difference_table(frame).set_index(["model", "method"])

    assert set(comparisons["dataset_count"]) == {2}
    assert set(comparisons["comparison_count"]) == {4}
    assert comparisons.loc[("tabiclv2", "sft"), "mean_rank"] == pytest.approx(1.0)
    assert comparisons.loc[("tabiclv2", "icl"), "mean_rank"] == pytest.approx(2.0)
    assert comparisons.loc[("tabpfnv2", "sft"), "mean_rank"] == pytest.approx(3.0)
    assert comparisons.loc[("tabpfnv2", "icl"), "mean_rank"] == pytest.approx(4.0)
    assert comparisons["critical_difference"].iloc[0] > 0


def test_cd_panel_omits_numeric_test_annotations_but_keeps_visual_cd() -> None:
    """The compact panel should retain the visual CD without printed statistics."""
    variants = [
        ("tabiclv2", "sft", 3.00),
        ("tabpfnv25", "sft", 3.18),
        ("tabiclv2", "icl", 3.36),
        ("tabpfnv2", "sft", 3.36),
        ("tabpfnv25", "icl", 4.55),
        ("tabpfnv2", "icl", 5.73),
        ("mitra", "sft", 5.82),
        ("mitra", "icl", 7.00),
    ]
    rows = pd.DataFrame(
        [
            {
                "task_family": "multiclass",
                "model": model,
                "method": method,
                "mean_rank": rank,
                "dataset_count": 11,
                "comparison_count": 8,
                "friedman_p_value": 0.0001,
                "critical_difference": 3.17,
            }
            for model, method, rank in variants
        ]
    )
    figure, axis = plt.subplots(figsize=(7.1, 2.1))
    figure.subplots_adjust(left=0.19, right=0.81, top=0.96, bottom=0.04)

    _draw_critical_difference_panel(axis, rows)
    figure.canvas.draw()
    labels = [text.get_text() for text in axis.texts]
    assert "CD" in labels
    assert not any(label.startswith(("Friedman", "Nemenyi")) for label in labels)
    assert any(line.get_linewidth() == pytest.approx(2.0) for line in axis.lines)
    plt.close(figure)


def test_global_winners_distinguish_observed_wins_from_effective_ties() -> None:
    frame = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "task_family": "regression",
                "score_metric": "rmse",
                "model": model,
                "baseline": baseline,
                "baseline_se": 0.001,
                "selected": selected,
                "selected_se": 0.001,
                "normalized_baseline": normalized_baseline,
                "normalized_selected": normalized_selected,
            }
            for dataset, values in {
                "decisive": {
                    "tabpfnv2": (1.03, 1.00, 1.00 / 1.03, 1.00),
                    "mitra": (1.04, 1.02, 1.00 / 1.04, 1.00 / 1.02),
                },
                "near": {
                    "tabpfnv2": (1.02, 1.00, 1.00 / 1.02, 1.00),
                    "mitra": (1.03, 1.003, 1.00 / 1.03, 1.00 / 1.003),
                },
                "exact": {
                    "tabpfnv2": (1.00, 1.02, 1.00, 1.00 / 1.02),
                    "mitra": (1.00, 1.03, 1.00, 1.00 / 1.03),
                },
            }.items()
            for model, (baseline, selected, normalized_baseline, normalized_selected) in values.items()
        ]
    )

    winners = _global_winner_table(frame).sort_values(["dataset", "model"])
    decisive = winners[winners["dataset"] == "decisive"].iloc[0]
    near = winners[winners["dataset"] == "near"].iloc[0]
    exact = winners[winners["dataset"] == "exact"]

    assert decisive[["model", "method"]].tolist() == ["tabpfnv2", "sft"]
    assert decisive["normalized_margin"] > DEFAULT_DECISIVE_WINNER_MARGIN
    assert decisive["decisive_winner"]
    assert decisive["uncertainty_95_separated"]
    assert near["normalized_margin"] < DEFAULT_DECISIVE_WINNER_MARGIN
    assert not near["decisive_winner"]
    assert set(exact["model"]) == {"tabpfnv2", "mitra"}
    assert set(exact["tie_size"]) == {2}
    assert not exact["decisive_winner"].any()
