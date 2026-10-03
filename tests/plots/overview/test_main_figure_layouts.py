"""Focused layout contracts for the main-text figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.colors import to_hex

import tfm_finetuning.plots.overview.pattern_diagnostic_plots as diagnostic_plots
import tfm_finetuning.plots.overview.pattern_metric_plots as metric_plots
import tfm_finetuning.plots.overview.pattern_model_comparison as model_comparison
from tfm_finetuning.plots.common.model_display import MODEL_COLORS, MODEL_LABELS, MODEL_ORDER, TASK_LABELS, TASK_ORDER
from tfm_finetuning.plots.common.plot_style import apply_plot_style, latex_fraction_width
from tfm_finetuning.plots.figure_text_audit import inventory_visible_text


def _minimum_visible_text_size(figure: plt.Figure) -> float:
    return min(element.effective_font_pt for element in inventory_visible_text(figure, exact_width=True))


def _assert_visible_text_inside_canvas(figure: plt.Figure) -> None:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    canvas = figure.get_window_extent(renderer)
    tick_labels = {id(label) for axis in figure.axes for label in (*axis.get_xticklabels(), *axis.get_yticklabels())}
    for text in figure.findobj(match=mpl.text.Text):
        if not text.get_visible() or not text.get_text().strip() or id(text) in tick_labels:
            continue
        bounds = text.get_window_extent(renderer)
        assert bounds.x0 >= canvas.x0 - 1.0, text.get_text()
        assert bounds.x1 <= canvas.x1 + 1.0, text.get_text()
        assert bounds.y0 >= canvas.y0 - 1.0, text.get_text()
        assert bounds.y1 <= canvas.y1 + 1.0, text.get_text()


def _metric_frame() -> pd.DataFrame:
    task_metrics = {
        "binary": ("roc_auc", "accuracy", "log_loss"),
        "multiclass": ("accuracy", "log_loss"),
        "regression": ("rmse", "mae", "r2"),
    }
    return pd.DataFrame(
        [
            {
                "dataset": f"{task}-{dataset_index}",
                "task_family": task,
                "metric": metric,
                "model": model,
                "preprocessing": "shared",
                "effect": 0.01 * (dataset_index + model_index + metric_index + 1),
            }
            for task, metrics in task_metrics.items()
            for metric_index, metric in enumerate(metrics)
            for model_index, model in enumerate(MODEL_ORDER)
            for dataset_index in range(4)
        ]
    )


def _effect_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": f"{task}-{dataset_index}",
                "task_family": task,
                "model": model,
                "effect": 0.02 * (dataset_index - 1) + 0.01 * model_index,
            }
            for task in TASK_ORDER
            for model_index, model in enumerate(MODEL_ORDER)
            for dataset_index in range(4)
        ]
    )


def _preprocessing_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": f"{task}-{dataset_index}",
                "task_family": task,
                "model": model,
                "preprocessing": preprocessing,
                "effect": 0.01 * (dataset_index + model_index) + offset,
            }
            for task in TASK_ORDER
            for model_index, model in enumerate(MODEL_ORDER)
            for dataset_index in range(3)
            for preprocessing, offset in (
                ("native", 0.0),
                ("shared", 0.005 * (dataset_index - 1)),
            )
        ]
    )


def test_metric_gain_figures_use_shallow_exact_width_metric_rows(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, tuple[plt.Figure, float]] = {}

    def capture(figure: plt.Figure, path: Path, **kwargs: object) -> Path:
        captured[path.name] = (figure, float(kwargs["inclusion_fraction"]))
        return path

    monkeypatch.setattr(metric_plots, "save_and_close_figure", capture)
    with mpl.rc_context():
        apply_plot_style()
        metric_plots._plot_metric_gains(_metric_frame(), tmp_path)

    expected_layouts = {
        "binary": (3, 1.0),
        "multiclass": (2, 0.78),
        "regression": (3, 1.0),
    }
    for task, (panel_count, width_fraction) in expected_layouts.items():
        figure, inclusion_fraction = captured[f"metric_gains_{task}.pdf"]
        assert figure.get_figwidth() == pytest.approx(latex_fraction_width(width_fraction))
        assert inclusion_fraction == pytest.approx(width_fraction)
        assert figure.get_figheight() == pytest.approx(2.65)
        assert len(figure.axes) == panel_count
        assert len({round(axis.get_position().y0, 6) for axis in figure.axes}) == 1
        assert [axis.get_position().x0 for axis in figure.axes] == sorted(
            axis.get_position().x0 for axis in figure.axes
        )
        assert not any(figure.axes[0].get_shared_y_axes().joined(figure.axes[0], axis) for axis in figure.axes[1:])
        assert figure._suptitle is None
        assert not figure.legends
        assert max(axis.get_position().y1 for axis in figure.axes) >= 0.87
        assert _minimum_visible_text_size(figure) >= 9.0
        assert all(label.get_fontsize() >= 9.0 for axis in figure.axes for label in axis.get_xticklabels())
        _assert_visible_text_inside_canvas(figure)
        plt.close(figure)


def test_metric_agreement_is_full_width_horizontal_row(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, plt.Figure] = {}

    def capture(figure: plt.Figure, path: Path, **_kwargs: object) -> Path:
        captured["figure"] = figure
        return path

    monkeypatch.setattr(metric_plots, "save_and_close_figure", capture)
    with mpl.rc_context():
        apply_plot_style()
        metric_plots._plot_metric_gain_agreement(_metric_frame(), tmp_path)

    figure = captured["figure"]
    assert figure.get_figwidth() == pytest.approx(latex_fraction_width(1.0))
    assert figure.get_figheight() <= 2.25
    assert len({axis.get_position().y0 for axis in figure.axes}) == 1
    assert len(figure.axes) == 4
    assert _minimum_visible_text_size(figure) >= 8.0
    _assert_visible_text_inside_canvas(figure)
    plt.close(figure)


def _critical_difference_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "task_family": task,
                "model": model,
                "method": method,
                "mean_rank": float(index + 1),
                "dataset_count": 11,
                "comparison_count": 8,
                "friedman_p_value": 0.01,
                "critical_difference": 2.5,
            }
            for task in TASK_ORDER
            for index, (model, method) in enumerate(
                [(model, method) for model in MODEL_ORDER for method in ("icl", "sft")]
            )
        ]
    )


def test_sft_dataset_win_plot_uses_five_equal_length_method_bars(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rows = []
    for dataset_index in range(3):
        for model_index, model in enumerate(MODEL_ORDER):
            baseline = 0.70 + 0.01 * model_index
            selected = baseline + (0.02 if dataset_index != 2 else -0.01)
            if dataset_index == 0 and model == "tabiclv2":
                selected = 0.95
            rows.append(
                {
                    "dataset": f"dataset-{dataset_index}",
                    "task_family": "binary",
                    "model": model,
                    "effect": selected - baseline,
                    "normalized_baseline": baseline,
                    "normalized_selected": selected,
                }
            )
    frame = pd.DataFrame(rows)
    summary = model_comparison._sft_dataset_win_table(frame).set_index("model")
    assert summary.loc["tabiclv2", "own_icl_improvement_count"] == 2
    assert summary.loc["tabiclv2", "own_icl_win_count"] == 1
    assert summary.loc["tabiclv2", "own_tie_count"] == 0
    assert summary.loc["tabiclv2", "global_best_count"] == 1

    captured: dict[str, object] = {}

    def capture(figure: plt.Figure, path: Path, **kwargs: object) -> Path:
        captured["figure"] = figure
        captured["path"] = path
        captured["fraction"] = kwargs.get("inclusion_fraction")
        return path

    monkeypatch.setattr(model_comparison, "save_and_close_figure", capture)
    model_comparison._plot_sft_dataset_wins(frame, tmp_path)

    figure = captured["figure"]
    assert isinstance(figure, plt.Figure)
    assert captured["path"] == tmp_path / "sft_dataset_wins.pdf"
    assert captured["fraction"] == pytest.approx(0.48)
    assert figure.get_figwidth() == pytest.approx(latex_fraction_width(0.48))
    assert figure.get_figheight() == pytest.approx(2.75)

    axis = figure.axes[0]
    assert [text.get_text() for text in axis.get_yticklabels()] == [
        *[MODEL_LABELS[model] for model in MODEL_ORDER],
        "Best overall",
    ]
    bar_widths_by_y: dict[float, float] = {}
    for patch in axis.patches:
        y = round(float(patch.get_y() + patch.get_height() / 2), 2)
        bar_widths_by_y[y] = bar_widths_by_y.get(y, 0.0) + float(patch.get_width())
    assert len(bar_widths_by_y) == 5
    assert all(width == pytest.approx(3.0) for width in bar_widths_by_y.values())
    separators = [
        line
        for line in axis.lines
        if len(line.get_xdata()) == 2
        and line.get_xdata()[0] == pytest.approx(line.get_xdata()[1])
        and to_hex(line.get_color()) == "#111827"
    ]
    assert len(separators) == 5

    legend = figure.axes[0].get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["ICL", "SFT"]
    all_text = [text.get_text() for text in figure.findobj(match=mpl.text.Text)]
    assert "ICL 1" in all_text
    assert "SFT 2" in all_text
    assert "Best of all 8 variants" not in all_text
    assert "Positive gain over own ICL" not in all_text
    figure.canvas.draw()
    assert legend.get_window_extent().y1 <= figure.bbox.y1
    assert _minimum_visible_text_size(figure) >= 8.0
    _assert_visible_text_inside_canvas(figure)
    plt.close(figure)


def test_final_gain_variance_has_three_task_panels(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    def capture(figure, path, **kwargs):
        captured.update(figure=figure, path=path)
        return path

    monkeypatch.setattr(metric_plots, "save_and_close_figure", capture)
    with mpl.rc_context():
        apply_plot_style()
        metric_plots._plot_effect_variance(_effect_frame(), tmp_path)
    figure = captured["figure"]
    assert captured["path"] == tmp_path / "fine_tuning_gain_variance.pdf"
    assert len(figure.axes) == 3
    assert [ax.get_title() for ax in figure.axes] == [TASK_LABELS[task] for task in TASK_ORDER]
    assert all(len(ax.collections) == 2 * len(MODEL_ORDER) for ax in figure.axes)
    assert _minimum_visible_text_size(figure) >= 8.0
    _assert_visible_text_inside_canvas(figure)
    plt.close(figure)


def _association_frame() -> pd.DataFrame:
    features = [
        "baseTrainRows",
        "maxCategoricalCardinality",
        "rowToFeatureRatio",
        "univariateSignalTop5Share",
    ]
    model_pairs = [
        (MODEL_ORDER[0], MODEL_ORDER[1]),
        (MODEL_ORDER[2], MODEL_ORDER[3]),
        (MODEL_ORDER[0], MODEL_ORDER[2]),
        (MODEL_ORDER[1], MODEL_ORDER[3]),
    ]
    return pd.DataFrame(
        [
            {
                "task_family": task,
                "preprocessing": "shared",
                "model": model,
                "feature": feature,
                "spearman_rho": -0.7 + 0.18 * (2 * feature_index + model_index),
                "lodo_rho_min": -0.78 + 0.18 * (2 * feature_index + model_index),
                "lodo_rho_max": -0.62 + 0.18 * (2 * feature_index + model_index),
                "bootstrap_ci_low": -0.86 + 0.18 * (2 * feature_index + model_index),
                "bootstrap_ci_high": -0.54 + 0.18 * (2 * feature_index + model_index),
                "n_datasets": 11,
            }
            for task in TASK_ORDER
            for feature_index, (feature, models) in enumerate(zip(features, model_pairs, strict=True))
            for model_index, model in enumerate(models)
        ]
    )


def test_combined_association_heatmap_groups_tasks_and_descriptor_families(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def capture(figure: plt.Figure, path: Path, **_kwargs: object) -> Path:
        captured["figure"] = figure
        captured["path"] = path
        return path

    monkeypatch.setattr(diagnostic_plots, "save_and_close_figure", capture)
    general = pd.DataFrame(
        [
            {
                "task_family": task,
                "preprocessing": "shared",
                "model": model,
                "feature": feature,
                "spearman_rho": 0.4 if feature == "baseTrainRows" else -0.3,
                "bootstrap_ci_low": 0.1 if feature == "baseTrainRows" else -0.6,
                "bootstrap_ci_high": 0.7 if feature == "baseTrainRows" else 0.1,
                "q_value_bh": 0.01 if model == MODEL_ORDER[0] else 0.2,
            }
            for task in TASK_ORDER
            for model in MODEL_ORDER
            for feature in ("baseTrainRows", "rowToFeatureRatio")
        ]
    )
    structural = pd.DataFrame(
        [
            {
                "task_family": task,
                "preprocessing": "shared",
                "model": model,
                "feature": ("localTargetVariationIqr" if task == "regression" else "localClassDisagreementRate"),
                "spearman_rho": -0.2,
                "bootstrap_ci_low": -0.5,
                "bootstrap_ci_high": 0.2,
                "q_value_bh": 0.4,
            }
            for task in TASK_ORDER
            for model in MODEL_ORDER
        ]
    )
    with mpl.rc_context():
        apply_plot_style()
        diagnostic_plots._plot_combined_association_heatmap(general, structural, tmp_path)

    figure = captured["figure"]
    assert isinstance(figure, plt.Figure)
    assert captured["path"] == tmp_path / "association_heatmap_combined.pdf"
    assert figure.get_figwidth() == pytest.approx(latex_fraction_width(1.0))
    assert figure.get_figheight() == pytest.approx(6.45)
    assert figure._suptitle is None
    assert [label.get_text() for label in figure.axes[0].get_yticklabels()] == [
        "Training rows",
        "Total features",
        "Training table cells",
        "Row-to-feature ratio",
    ]
    assert [label.get_text() for label in figure.axes[2].get_yticklabels()] == [
        "Normalized class entropy",
        "Robust target-tail index",
        "Top-five univariate\nsignal share",
        "Nearest-neighbour\ntarget variation",
    ]
    all_text = [text.get_text() for text in figure.findobj(match=mpl.text.Text)]
    assert "1  Dataset scale" in all_text
    assert "2  Feature structure and distribution" in all_text
    assert "3  Target structure" in all_text
    assert "4  Predictive structure" not in all_text
    assert "Numeric-feature share" in all_text
    assert "Target P99 / median" not in all_text
    assert ".40" in all_text
    assert "×" in all_text
    assert all_text.count("--") == 12
    assert figure.axes[0].collections
    assert any(text.get_visible() and text.get_text() == "Binary" for text in figure.findobj(match=mpl.text.Text))
    header_axis = figure.axes[3]
    assert [text.get_text() for text in header_axis.texts[:3]] == [TASK_LABELS[task] for task in TASK_ORDER]
    assert len(header_axis.texts) == len(TASK_ORDER)
    assert all(axis.lines for axis in figure.axes[:3])
    assert [to_hex(label.get_color()) for label in figure.axes[2].get_xticklabels()] == [
        MODEL_COLORS[model] for _, model in [(task, model) for task in TASK_ORDER for model in MODEL_ORDER]
    ]
    assert all(tick.tick1line.get_markersize() > 0 for tick in figure.axes[2].xaxis.get_major_ticks())
    assert all(
        label._linespacing == pytest.approx(0.85) for axis in figure.axes[:3] for label in axis.get_yticklabels()
    )
    assert _minimum_visible_text_size(figure) >= 7.5
    _assert_visible_text_inside_canvas(figure)
    plt.close(figure)


def test_combined_association_heatmap_can_label_epoch100_estimand(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def capture(figure: plt.Figure, path: Path, **_kwargs: object) -> Path:
        captured["figure"] = figure
        captured["path"] = path
        return path

    monkeypatch.setattr(diagnostic_plots, "save_and_close_figure", capture)
    empty = pd.DataFrame(
        columns=[
            "task_family",
            "preprocessing",
            "model",
            "feature",
            "spearman_rho",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "q_value_bh",
        ]
    )
    with mpl.rc_context():
        apply_plot_style()
        diagnostic_plots._plot_combined_association_heatmap(
            empty,
            empty,
            tmp_path,
            filename="association_heatmap_epoch100_combined.pdf",
            gain_label="epoch-100 SFT gain over ICL",
        )

    figure = captured["figure"]
    assert isinstance(figure, plt.Figure)
    assert captured["path"] == tmp_path / "association_heatmap_epoch100_combined.pdf"
    assert figure.axes[-1].get_ylabel() == ("Spearman correlation with epoch-100 SFT gain over ICL")
    plt.close(figure)


def _grouped_metadata_frame(group_column: str) -> pd.DataFrame:
    groups = (
        ("science_engineering", "medical_healthcare", "generic_benchmark")
        if group_column == "domain"
        else ("tabarena", "openml_addendum")
    )
    return pd.DataFrame(
        [
            {
                "dataset": f"{task}-{group}-{model}",
                "task_family": task,
                group_column: group,
                "model": model,
                "effect": 0.01 * (group_index + model_index + 1),
            }
            for task in TASK_ORDER
            for group_index, group in enumerate(groups)
            for model_index, model in enumerate(MODEL_ORDER)
        ]
    )
