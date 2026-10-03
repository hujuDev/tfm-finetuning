"""Publication figures for the transfer E1 transfer schedule study."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator

from tfm_finetuning.plots.common.model_display import MODEL_COLORS, MODEL_LABELS
from tfm_finetuning.plots.common.plot_style import (
    FULL_WIDTH_INCHES,
    apply_plot_style,
    save_and_close_figure,
)
from tfm_finetuning.plots.transfer.transfer_data import (
    DISPLAY_ROUTE_ORDER,
    MODEL_ORDER,
    TransferTables,
    training_order_icl_gains,
)

_LEGEND_TITLE_FONT = {"size": 9.0, "weight": "bold"}
TRAINING_ORDER_PANEL_TITLES = {
    "credit-g": "Credit-G target $T$\n$S^{A}$: Bank · $S^{B}$: HELOC",
    "covertype": "Covertype target $T$\n$S^{A}$: Satimage · $S^{B}$: Segment",
    "satimage": "Satimage target $T$\n$S^{A}$: Covertype · $S^{B}$: Segment",
}
TRAINING_ORDER_ROUTE_LABELS = {
    "pooled_target_sources": r"$(T + S^{A} + S^{B})_{300}$",
    "pooled_sources_then_target": "$(S^{A} + S^{B})_{200}$\n$\\rightarrow T_{100}$",
    "sequential_bank_then_heloc_then_target": "$S^{A}_{100} \\rightarrow S^{B}_{100}$\n$\\rightarrow T_{100}$",
    "sequential_heloc_then_bank_then_target": "$S^{B}_{100} \\rightarrow S^{A}_{100}$\n$\\rightarrow T_{100}$",
    "pooled_target_satimage": r"$(T + S^{A})_{200}$",
    "sequential_satimage_then_target": r"$S^{A}_{100} \rightarrow T_{100}$",
    "pooled_target_satimage_segment_ablation": r"$(T + S^{A} + S^{B})_{300}$",
    "sequential_satimage_then_segment_then_target": "$S^{A}_{100} \\rightarrow S^{B}_{100}$\n$\\rightarrow T_{100}$",
    "sequential_segment_then_satimage_then_target": "$S^{B}_{100} \\rightarrow S^{A}_{100}$\n$\\rightarrow T_{100}$",
    "pooled_target_covertype": r"$(T + S^{A})_{200}$",
    "sequential_covertype_then_target": r"$S^{A}_{100} \rightarrow T_{100}$",
    "pooled_target_covertype_segment_ablation": r"$(T + S^{A} + S^{B})_{300}$",
    "sequential_covertype_then_segment_then_target": "$S^{A}_{100} \\rightarrow S^{B}_{100}$\n$\\rightarrow T_{100}$",
    "sequential_segment_then_covertype_then_target": "$S^{B}_{100} \\rightarrow S^{A}_{100}$\n$\\rightarrow T_{100}$",
}
SUPERSEDED_FIGURES = (
    "schedule_target_exposure_contrasts.pdf",
    "schedule_compute_matched_contrasts.pdf",
    "schedule_route_pair_contrasts.pdf",
)


def render_transfer_figures(
    tables: TransferTables,
    output_dir: Path,
    thesis_asset_dir: Path,
) -> tuple[Path, ...]:
    """Render the consolidated E1 findings figure and mirror it into the thesis."""
    apply_plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    thesis_asset_dir.mkdir(parents=True, exist_ok=True)
    for directory in (output_dir, thesis_asset_dir):
        for filename in SUPERSEDED_FIGURES:
            (directory / filename).unlink(missing_ok=True)
    figure = _render_training_order_comparison(
        tables,
        output=output_dir / "transfer_training_order_comparison.pdf",
    )
    (thesis_asset_dir / figure.name).write_bytes(figure.read_bytes())
    return (figure,)


def _render_training_order_comparison(tables: TransferTables, *, output: Path) -> Path:
    figure = plt.figure(figsize=(FULL_WIDTH_INCHES, 6.2), constrained_layout=False)
    grid = figure.add_gridspec(2, 2)
    covertype_axis = figure.add_subplot(grid[0, 0])
    axes_by_dataset = {
        "covertype": covertype_axis,
        "satimage": figure.add_subplot(grid[0, 1], sharey=covertype_axis),
        "credit-g": figure.add_subplot(grid[1, :]),
    }
    for dataset, gain_axis in axes_by_dataset.items():
        routes = DISPLAY_ROUTE_ORDER[dataset]
        gain_records, _controls = training_order_icl_gains(tables, dataset, routes=routes)
        _plot_setting_means(gain_axis, gain_records, routes)
        _plot_target_100_reference_lines(gain_axis, gain_records)
        gain_axis.set_title(TRAINING_ORDER_PANEL_TITLES[dataset], fontsize=10, pad=4)
        gain_axis.set_xticks(range(len(routes)))
        gain_axis.set_xticklabels(
            [TRAINING_ORDER_ROUTE_LABELS.get(route, route) for route in routes],
            fontsize=8,
            rotation=25,
            ha="right",
            rotation_mode="anchor",
        )
        gain_axis.set_axisbelow(True)
        gain_axis.spines[["top", "right"]].set_visible(False)
        if dataset != "satimage":
            gain_axis.set_ylabel(f"Gain over ICL\n({_gain_metric_label(dataset)})", fontsize=9)
        _configure_training_order_gain_axis(
            gain_axis,
            dataset=dataset,
            show_grid=dataset == "credit-g",
        )
    axes_by_dataset["satimage"].tick_params(
        axis="y",
        which="both",
        left=False,
        labelleft=False,
    )
    add_model_legend(figure, anchor=(0.36, 0.995))
    figure.legend(
        handles=[
            Line2D(
                [],
                [],
                color="#555555",
                linestyle=(0, (4, 2)),
                linewidth=1.1,
                label=r"Target-only $T_{100}$ reference",
            )
        ],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.77, 0.995),
        fontsize=9,
        title="Reference",
        title_fontproperties=_LEGEND_TITLE_FONT,
        handlelength=2.4,
        handletextpad=0.5,
    )
    figure.supxlabel("Fine-tuning order", y=0.025, fontsize=9.5)
    figure.subplots_adjust(
        left=0.145,
        right=0.995,
        bottom=0.15,
        top=0.825,
        hspace=0.88,
        wspace=0.16,
    )
    return save_and_close_figure(figure, output, tight=False)


def add_model_legend(figure, *, anchor: tuple[float, float]) -> None:
    """Add the compact model-color legend shared by transfer figures."""
    figure.legend(
        handles=[
            Line2D([], [], color=MODEL_COLORS[model], linewidth=4, label=MODEL_LABELS[model]) for model in MODEL_ORDER
        ],
        ncols=len(MODEL_ORDER),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=anchor,
        fontsize=9,
        title="Model",
        title_fontproperties=_LEGEND_TITLE_FONT,
        handlelength=1.6,
        handletextpad=0.45,
        columnspacing=1.1,
    )


def _plot_setting_means(
    axis: Axes,
    records: list[tuple[str, int, str, float]],
    settings: tuple[str, ...],
) -> None:
    model_offsets = dict(zip(MODEL_ORDER, np.linspace(-0.22, 0.22, len(MODEL_ORDER)), strict=True))
    for model in MODEL_ORDER:
        for index, setting in enumerate(settings):
            values = [
                score
                for record_model, _seed, record_setting, score in records
                if record_model == model and record_setting == setting
            ]
            if len(values) != 3:
                raise ValueError(f"Expected three seeds for {model}/{setting}, found {len(values)}.")
            x = index + model_offsets[model]
            mean = float(np.mean(values))
            sem = float(np.std(values, ddof=1) / np.sqrt(len(values)))
            axis.errorbar(
                x,
                mean,
                yerr=sem,
                color=MODEL_COLORS[model],
                fmt="none",
                capsize=2.2,
                elinewidth=0.8,
                capthick=0.8,
                zorder=3,
            )
            axis.scatter(
                x,
                mean,
                color=MODEL_COLORS[model],
                edgecolors="white",
                linewidths=0.55,
                marker="o",
                s=34,
                zorder=4,
            )


def _plot_target_100_reference_lines(
    axis: Axes,
    records: list[tuple[str, int, str, float]],
) -> None:
    """Plot each model's mean 100-update target-only gain across the panel."""
    for model in MODEL_ORDER:
        values = [
            gain for record_model, _seed, setting, gain in records if record_model == model and setting == "target_100"
        ]
        if len(values) != 3:
            raise ValueError(f"Expected three target-only seeds for {model}, found {len(values)}.")
        axis.axhline(
            float(np.mean(values)),
            color=MODEL_COLORS[model],
            linestyle=(0, (4, 2)),
            linewidth=1.0,
            alpha=0.75,
            zorder=1,
            label=f"_target_100_{model}",
        )


def _gain_metric_label(dataset: str) -> str:
    return "Δ ROC-AUC" if dataset == "credit-g" else "log-loss ratio"


def _configure_training_order_gain_axis(
    axis: Axes,
    *,
    dataset: str,
    show_grid: bool = True,
) -> None:
    axis.axhline(0.0, color="#555555", linewidth=0.8, zorder=0)
    label_interval = 0.005 if dataset == "credit-g" else 0.10
    axis.yaxis.set_major_locator(MultipleLocator(label_interval))
    if dataset == "credit-g":
        axis.yaxis.set_minor_locator(MultipleLocator(0.0025))
    else:
        axis.yaxis.set_minor_locator(MultipleLocator(0.01))
    precision = 3 if dataset == "credit-g" else 2
    axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{0.0 if np.isclose(value, 0.0) else value:.{precision}f}")
    )
    axis.tick_params(axis="y", which="major", length=3)
    axis.tick_params(axis="y", which="minor", length=2)
    if dataset == "credit-g":
        lower, upper = axis.get_ylim()
        axis.set_ylim(
            np.floor(lower / label_interval) * label_interval,
            np.ceil(upper / label_interval) * label_interval,
        )
    if show_grid:
        axis.grid(axis="y", which="major", color="#dddddd", linewidth=0.55, zorder=-1)
        axis.grid(axis="y", which="minor", color="#f2f2f2", linewidth=0.3, zorder=-1)


__all__ = ["TRAINING_ORDER_PANEL_TITLES", "TRAINING_ORDER_ROUTE_LABELS", "add_model_legend", "render_transfer_figures"]
