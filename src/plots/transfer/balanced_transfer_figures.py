"""Publication figures for the balanced six-target transfer extension.

Example:
    ``render_balanced_transfer_figures(tables, analysis_dir, thesis_dir)``
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from plots.common.model_display import MODEL_COLORS, MODEL_LABELS
from plots.common.plot_style import (
    FULL_WIDTH_INCHES,
    apply_section1_plot_style,
    save_and_close_section1_figure,
    shared_y_tick_interval,
    style_section1_y_axes,
)
from plots.transfer.balanced_transfer_data import (
    BUDGET_ROWS,
    TASK_DATASETS,
    BalancedTransferTables,
    SourceInitializationAdvantage,
)
from plots.transfer.transfer_data import MODEL_ORDER

BALANCED_FIGURE_NAMES = ("source_initialization_by_target_data.pdf",)
TASK_ORDER = ("multiclass", "binary", "regression")
TASK_HEADERS = {
    "multiclass": "Multiclass",
    "binary": "Binary",
    "regression": "Regression",
}
TASK_SOURCE_HEADERS = {
    "multiclass": "Source: Satimage",
    "binary": "Source: Bank + HELOC",
    "regression": "Source: Houses",
}
BUDGET_TICK_LABELS = ("64", "256", "1k", "4k", "16k", "Full\n≤50k")
PANEL_LABELS = {
    "covertype": "Covertype",
    "sdss17": "SDSS17",
    "amazon_employee_access": "Amazon Employee Access",
    "diabetes130us": "Diabetes130US",
    "physiochemical_protein": "Physiochemical Protein",
    "food_delivery_time": "Food Delivery Time",
}


def render_balanced_transfer_figures(
    tables: BalancedTransferTables,
    output_dir: Path,
    thesis_asset_dir: Path,
) -> tuple[Path, ...]:
    """Render and mirror the two balanced six-target findings figures."""
    apply_section1_plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    thesis_asset_dir.mkdir(parents=True, exist_ok=True)
    figures = (
        _render_source_initialization_advantage(
            tables.source_advantages,
            output_dir / BALANCED_FIGURE_NAMES[0],
        ),
    )
    for figure in figures:
        (thesis_asset_dir / figure.name).write_bytes(figure.read_bytes())
    return figures


def _render_source_initialization_advantage(
    rows: Iterable[SourceInitializationAdvantage],
    output: Path,
) -> Path:
    figure, axes = _six_target_canvas(height=5.55)
    records = tuple(rows)
    x_positions = np.arange(len(BUDGET_ROWS), dtype=float)
    for row_index in range(2):
        for column_index, task_family in enumerate(TASK_ORDER):
            axis = axes[row_index, column_index]
            dataset = TASK_DATASETS[task_family][row_index]
            for model in MODEL_ORDER:
                means, errors = _budget_summary(records, dataset, model)
                axis.errorbar(
                    x_positions,
                    means,
                    yerr=errors,
                    color=MODEL_COLORS[model],
                    marker="o",
                    linewidth=1.2,
                    markersize=4.8,
                    markeredgecolor="white",
                    markeredgewidth=0.5,
                    capsize=2.0,
                    elinewidth=0.8,
                    capthick=0.8,
                    zorder=3,
                )
            _style_panel(axis, dataset, emphasize_zero=True)
            axis.set_xticks(x_positions)
            if row_index == 1:
                axis.set_xticklabels(BUDGET_TICK_LABELS)
                axis.tick_params(axis="x", labelsize=8.0)
                axis.get_xticklabels()[-1].set_ha("left")
            else:
                axis.tick_params(axis="x", labelbottom=False)
            axis.set_xlim(-0.35, len(BUDGET_ROWS) - 0.45)
    _style_task_columns(axes)
    _add_column_headers(figure, axes, show_sources=True)
    _add_model_legend(figure, anchor=(0.5, 0.99))
    figure.supylabel(
        "Source-initialization advantage over Target-only",
        x=0.012,
        fontsize=9.5,
    )
    figure.supxlabel("Available target training rows", y=0.03, fontsize=9.5)
    figure.subplots_adjust(left=0.115, right=0.975, bottom=0.16, top=0.76, hspace=0.39, wspace=0.27)
    return save_and_close_section1_figure(figure, output, tight=False)


def _six_target_canvas(*, height: float) -> tuple[plt.Figure, np.ndarray]:
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(FULL_WIDTH_INCHES, height),
        sharey="col",
        constrained_layout=False,
        squeeze=False,
    )
    return figure, axes


def _budget_summary(
    rows: tuple[SourceInitializationAdvantage, ...],
    dataset: str,
    model: str,
) -> tuple[list[float], list[float]]:
    means, errors = [], []
    for target_rows in BUDGET_ROWS:
        values = [
            row.advantage
            for row in rows
            if row.dataset == dataset and row.model == model and row.target_rows == target_rows
        ]
        _require_count(values, 3, f"budget {dataset}/{model}/{target_rows}")
        means.append(float(np.mean(values)))
        errors.append(_sem(values))
    return means, errors


def _style_panel(axis: Axes, dataset: str, *, emphasize_zero: bool = False) -> None:
    axis.set_title(
        PANEL_LABELS[dataset],
        loc="left",
        x=-0.18,
        fontsize=8.7,
        pad=7,
    )
    axis.axhline(
        0.0,
        color="#303030" if emphasize_zero else "#555555",
        linewidth=1.25 if emphasize_zero else 0.75,
        zorder=2 if emphasize_zero else 0,
        label="_zero_reference",
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)


def _style_task_columns(axes: np.ndarray) -> None:
    for column_index in range(len(TASK_ORDER)):
        column_axes = axes[:, column_index]
        style_section1_y_axes(
            column_axes,
            tick_interval=shared_y_tick_interval(column_axes, max_guides=12),
        )


def _add_column_headers(
    figure: plt.Figure,
    axes: np.ndarray,
    *,
    show_sources: bool = False,
) -> None:
    for column_index, task_family in enumerate(TASK_ORDER):
        position = axes[0, column_index].get_position()
        center = (position.x0 + position.x1) / 2
        figure.text(
            center,
            0.84,
            TASK_HEADERS[task_family],
            ha="center",
            va="bottom",
            fontsize=8.7,
            weight="semibold",
        )
        if show_sources:
            figure.text(
                center,
                0.81,
                TASK_SOURCE_HEADERS[task_family],
                ha="center",
                va="bottom",
                fontsize=8.0,
                color="#555555",
            )


def _add_model_legend(figure: plt.Figure, *, anchor: tuple[float, float]) -> None:
    figure.legend(
        handles=[
            Line2D(
                [], [], color=MODEL_COLORS[model], marker="o", linewidth=1.5, markersize=4.5, label=MODEL_LABELS[model]
            )
            for model in MODEL_ORDER
        ],
        ncols=len(MODEL_ORDER),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=anchor,
        fontsize=9,
        title="Model",
        title_fontproperties={"size": 9.0, "weight": "semibold"},
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=1.0,
    )


def _sem(values: list[float]) -> float:
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _require_count(values: list[float], expected: int, context: str) -> None:
    if len(values) != expected:
        raise ValueError(f"Expected {expected} values for {context}, found {len(values)}.")


__all__ = [
    "BALANCED_FIGURE_NAMES",
    "BUDGET_TICK_LABELS",
    "PANEL_LABELS",
    "TASK_HEADERS",
    "TASK_SOURCE_HEADERS",
    "render_balanced_transfer_figures",
]
