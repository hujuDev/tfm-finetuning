"""Paired source-only and target-only transfer display for the submitted findings.

Example: ``plot_transfer_pairs(seed_results, Path("transfer.pdf"))``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from tfm_finetuning.plots.common.model_display import MODEL_COLORS, MODEL_LABELS
from tfm_finetuning.plots.common.plot_style import FULL_WIDTH_INCHES, apply_plot_style, save_figure
from tfm_finetuning.plots.transfer.balanced_transfer_data import SOURCE_LABELS, TASK_DATASETS
from tfm_finetuning.plots.transfer.balanced_transfer_figures import PANEL_LABELS

MODELS = ("tabpfnv25", "tabiclv2", "mitra")
TASK_LABELS = {
    "multiclass": "Log-loss gain: log(ICL / SFT)",
    "binary": "ROC-AUC gain: SFT − ICL",
    "regression": "RMSE gain: log(ICL / SFT)",
}


def plot_transfer_pairs(frame: pd.DataFrame, output: Path) -> Path:
    """Retain per-seed uncertainty and a separate metric scale for each task."""
    apply_plot_style()
    keys = ["dataset", "model", "source", "seed"]
    if frame.duplicated(keys).any():
        raise ValueError("Repeated seed results in transfer input")
    if not frame.groupby(keys[:-1]).size().eq(3).all():
        raise ValueError("Transfer comparisons require three seeds per route")
    references = frame.groupby(["dataset", "model", "seed"])["target_only_gain"]
    if references.nunique().gt(1).any():
        raise ValueError("Target-only reference differs between sources for the same seed")
    fig, axes = plt.subplots(
        3, 3, figsize=(FULL_WIDTH_INCHES, 6.1), sharex="row", sharey="row", gridspec_kw={"height_ratios": [4, 4, 2]}
    )
    for y, (task, datasets) in enumerate(TASK_DATASETS.items()):
        task_rows = frame[frame.task_family == task]
        routes = [
            (dataset, source)
            for dataset in datasets
            for source in SOURCE_LABELS
            if ((task_rows.dataset == dataset) & (task_rows.source == source)).any()
        ]
        limits = [0.0]
        for x, model in enumerate(MODELS):
            axis = axes[y, x]
            for position, (dataset, source) in enumerate(routes):
                rows = task_rows[
                    (task_rows.dataset == dataset) & (task_rows.model == model) & (task_rows.source == source)
                ]
                if len(rows) != 3:
                    raise ValueError(f"Incomplete transfer route: {dataset}/{source}/{model}")
                means = rows[["source_gain", "target_only_gain"]].mean()
                errors = rows[["source_gain", "target_only_gain"]].sem()
                for column, marker in [("source_gain", "o"), ("target_only_gain", "D")]:
                    axis.errorbar(
                        means[column],
                        position,
                        xerr=errors[column],
                        fmt=marker,
                        color=MODEL_COLORS[model],
                        markersize=4.5,
                        markerfacecolor="white" if column == "target_only_gain" else MODEL_COLORS[model],
                        capsize=2,
                        elinewidth=0.7,
                        zorder=3,
                    )
                    limits.extend([means[column] - errors[column], means[column] + errors[column]])
            axis.axvline(0, color="#475569", linewidth=0.9)
            axis.set_yticks(
                range(len(routes)), [f"T: {PANEL_LABELS[d]}\nS: {SOURCE_LABELS[s]}" for d, s in routes], fontsize=8
            )
            axis.set_ylim(len(routes) - 0.5, -0.5)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=3))
            axis.tick_params(axis="x", labelsize=8)
            axis.grid(axis="x", alpha=0.17)
            axis.spines[["top", "right", "left"]].set_visible(False)
            axis.tick_params(axis="y", length=0)
            if y == 0:
                axis.set_title(MODEL_LABELS[model], fontsize=9, color=MODEL_COLORS[model], fontweight="bold")
        span = max(0.025, np.ptp(limits))
        axes[y, 0].set_xlim(min(limits) - span * 0.10, max(limits) + span * 0.10)
        axes[y, 1].set_xlabel(TASK_LABELS[task], fontsize=8, labelpad=8)
    fig.legend(
        handles=[
            Line2D([], [], marker="o", color="#475569", linestyle="none", label="Source-only SFT"),
            Line2D(
                [],
                [],
                marker="D",
                color="#475569",
                markerfacecolor="white",
                linestyle="none",
                label="Target-only SFT",
            ),
        ],
        loc="upper center",
        frameon=False,
        fontsize=8,
        ncol=2,
    )
    fig.subplots_adjust(left=0.285, right=0.985, top=0.88, bottom=0.095, hspace=0.57, wspace=0.20)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output)
    plt.close(fig)
    return output
