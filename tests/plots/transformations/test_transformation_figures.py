"""Regression tests for transformation publication figure primitives."""

import shutil
from pathlib import Path

import matplotlib.figure
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.container import ErrorbarContainer

from tests.support.pdf_text_audit import assert_pdf_minimum_text_size
from tfm_finetuning.plots.figure_text_audit import inventory_visible_text
from tfm_finetuning.plots.transformations import transformation_figures as figures

MODELS = ("tabpfnv25", "tabiclv2", "mitra")
DATASETS = tuple(f"dataset_{index}" for index in range(5))
SUPPLEMENTAL_SCORE_GAIN_CASES = (
    (
        "contextual_scale_accuracy_dataset_scores_gains.pdf",
        "accuracy",
        ("1×\nno unit", "1×\nunit", "3×\nno unit", "3×\nunit"),
        "SDSS17",
    ),
    (
        "contextual_scale_mae_dataset_scores_gains.pdf",
        "mae",
        ("1×\nno unit", "1×\nunit", "3×\nno unit", "3×\nunit"),
        "SDSS17",
    ),
    (
        "contextual_scale_roc_auc_dataset_scores_gains.pdf",
        "roc_auc",
        ("1×\nno unit", "1×\nunit", "3×\nno unit", "3×\nunit"),
        "SDSS17",
    ),
    ("diamonds_mitigation_mae_dataset_scores_gains.pdf", "mae", ("Raw", "Winsorized", "Robust log"), "Diamonds"),
    ("diamonds_mitigation_rmse_dataset_scores_gains.pdf", "rmse", ("Raw", "Winsorized", "Robust log"), "Diamonds"),
    ("feature_ablation_accuracy_dataset_scores_gains.pdf", "accuracy", ("0%", "10%", "20%", "30%"), "Wine quality"),
    ("feature_ablation_mae_dataset_scores_gains.pdf", "mae", ("0%", "10%", "20%", "30%"), "Wine quality"),
    (
        "heavy_tail_mae_dataset_scores_gains.pdf",
        "mae",
        ("O/raw", "O/wins.", "O/log", "C/raw", "C/wins.", "C/log"),
        "Healthcare expenses",
    ),
    (
        "label_noise_accuracy_dataset_scores_gains.pdf",
        "accuracy",
        ("0%", "5%", "15%", "30%"),
        "Amazon employee access",
    ),
)


def _capture_figure(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    captured: dict[str, object] = {}
    real_save = figures.save_figure

    def capture(
        figure: matplotlib.figure.Figure,
        path: Path,
        *,
        inclusion_fraction: float = 1.0,
    ) -> None:
        captured["figure"] = figure
        captured["inclusion_fraction"] = inclusion_fraction
        real_save(figure, path, inclusion_fraction=inclusion_fraction)
        FigureCanvasAgg(figure)
        figure.canvas.draw()

    monkeypatch.setattr(figures, "save_figure", capture)
    return captured


def test_diamonds_plot_uses_log_scales_without_selected_to_final_connectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _capture_figure(monkeypatch)
    modes = ("raw", "signed_log", "quantile")
    trajectory_rows = []
    heldout_rows = []
    for mode_index, mode in enumerate(modes):
        for seed in range(3):
            for epoch in (1, 2, 3):
                trajectory_rows.append(
                    {
                        "target_mode": mode,
                        "seed": seed,
                        "epoch": epoch,
                        "validation_rmse": 10000 / (epoch + mode_index + 1) + seed,
                    }
                )
            heldout_rows.append(
                {
                    "target_mode": mode,
                    "seed": seed,
                    "icl_rmse": 9000 + seed,
                    "selected_rmse": 6000 - 500 * mode_index + seed,
                    "epoch50_rmse": 6500 - 500 * mode_index + seed,
                }
            )

    figures.plot_diamonds_mitigation(
        pd.DataFrame(trajectory_rows),
        pd.DataFrame(heldout_rows),
        output_path=tmp_path / "diamonds.pdf",
        target_mode_order=modes,
        target_mode_labels={"raw": "Raw", "signed_log": "Signed log", "quantile": "Quantile"},
    )

    figure = captured["figure"]
    assert figure.get_figwidth() == pytest.approx(
        figures.latex_fraction_width(figures.HORIZONTAL_AB_LATEX_WIDTH_FRACTION)
    )
    assert captured["inclusion_fraction"] == pytest.approx(figures.HORIZONTAL_AB_LATEX_WIDTH_FRACTION)
    assert [axis.get_title() for axis in figure.axes] == ["A", "B"]
    assert figure._suptitle is None
    assert figure.axes[0].get_position().x1 < figure.axes[1].get_position().x0
    assert figure.axes[0].get_position().y0 == pytest.approx(figure.axes[1].get_position().y0)
    assert [axis.get_yscale() for axis in figure.axes] == ["log", "log"]
    assert all("$" not in axis.yaxis.get_major_formatter()(1000, 0) for axis in figure.axes)
    assert all(axis.yaxis.get_major_formatter()(1000, 0) == "1000" for axis in figure.axes)
    assert all("$" not in axis.yaxis.get_minor_formatter()(3000, 0) for axis in figure.axes)
    assert min(element.effective_font_pt for element in inventory_visible_text(figure, exact_width=True)) >= 9.0
    if shutil.which("mutool") is not None:
        assert_pdf_minimum_text_size(tmp_path / "diamonds.pdf", minimum_pt=9.0)
    assert len(figure.axes[0].collections) >= len(modes)
    assert sum(isinstance(container, ErrorbarContainer) for container in figure.axes[1].containers) == len(modes) * 3
    assert all(axis.get_legend() is None for axis in figure.axes)
    assert len(figure.legends) == 1
    assert len(figure.legends[0].get_texts()) == 6
    renderer = figure.canvas.get_renderer()
    assert figure.legends[0].get_window_extent(renderer).y0 > max(axis.bbox.y1 for axis in figure.axes)
    connectors = [line for line in figure.axes[1].lines if len(line.get_xdata()) == 2]
    assert not connectors
    assert not figure.axes[0].texts
    assert not figure.axes[1].texts


def test_heavy_tail_plot_uses_shallow_horizontal_stability_panels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _capture_figure(monkeypatch)
    datasets = ("healthcare", "houses", "miami")
    modes = ("none", "winsor", "robust_log")
    rows = []
    for dataset in datasets:
        for model_index, model in enumerate(MODELS):
            for contamination in ("original", "fit_side_contaminated"):
                for mode_index, mode in enumerate(modes):
                    rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "contamination": contamination,
                            "target_mode": mode,
                            "stability_mean": -0.1 * model_index + 0.05 * mode_index,
                            "stability_sem": 0.02,
                        }
                    )
    frame = pd.DataFrame(rows)

    figures.plot_heavy_tail_mitigation(
        frame,
        output_path=tmp_path / "heavy-tail.pdf",
        dataset_order=datasets,
        target_mode_order=modes,
        target_mode_labels={"none": "None", "winsor": "Winsorized", "robust_log": "Robust log"},
    )

    figure = captured["figure"]
    assert figure.get_figwidth() == pytest.approx(figures.latex_fraction_width(0.94))
    assert figure.get_figheight() < 3.2
    assert captured["inclusion_fraction"] == pytest.approx(0.94)
    assert len(figure.axes) == 3
    assert [axis.get_title() for axis in figure.axes] == ["A", "B", "C"]
    assert figure._suptitle is None
    assert 1.0 - max(axis.get_position().y1 for axis in figure.axes) < 0.22
    assert len(figure.legends) == 2
    assert [legend._ncols for legend in figure.legends] == [3, 2]
    renderer = figure.canvas.get_renderer()
    legend_boxes = [legend.get_window_extent(renderer) for legend in figure.legends]
    title_boxes = [axis.title.get_window_extent(renderer) for axis in figure.axes]
    assert min(legend_box.y0 for legend_box in legend_boxes) > max(title_box.y1 for title_box in title_boxes)
    assert [label.get_text() for label in figure.axes[0].get_xticklabels()] == [
        "None",
        "Winsorized",
        "Robust log",
    ]
    assert all(label.get_rotation() == pytest.approx(30) for label in figure.axes[0].get_xticklabels())
    assert all(label.get_ha() == "right" for label in figure.axes[0].get_xticklabels())
    assert all(label.get_va() == "top" for label in figure.axes[0].get_xticklabels())
    assert all(axis.get_xlim() == pytest.approx((-0.32, 2.32)) for axis in figure.axes)
    assert min(element.effective_font_pt for element in inventory_visible_text(figure, exact_width=True)) >= 9.0
    if shutil.which("mutool") is not None:
        assert_pdf_minimum_text_size(tmp_path / "heavy-tail.pdf", minimum_pt=9.0)
    for axis in figure.axes:
        model_lines = [
            line for line in axis.lines if len(line.get_xdata()) == len(modes) and line.get_linestyle() in {"-", "--"}
        ]
        assert len(model_lines) == 2 * len(MODELS)
