"""Final representation, class-rebalancing and heavy-tail figures.

These renderers consume the existing dataset-level analysis tables. Example:
``plot_representation_heatmaps(summaries, Path("alignment.pdf"))``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from tfm_finetuning.plots.common.model_display import MODEL_COLORS, MODEL_LABELS
from tfm_finetuning.plots.common.plot_style import (
    FULL_WIDTH_INCHES,
    apply_plot_style,
    latex_fraction_width,
    save_figure,
)
from tfm_finetuning.plots.transformations.transformation_overviews import (
    REPRESENTATION_SCORE_CONDITION_ORDER,
    pooled_subsampling_summary,
)

MODELS = ("tabpfnv25", "tabiclv2", "mitra")
METHODS = ("icl", "sft")
MITIGATIONS = ("none", "winsor_p01_p99", "robust_log1p")
MODEL_HANDLES = [Line2D([], [], color=MODEL_COLORS[m], marker="o", label=MODEL_LABELS[m]) for m in MODELS]


def _save(fig: plt.Figure, output: Path, fraction: float = 0.94) -> Path:
    """Save at the manuscript's physical inclusion width and close the canvas."""
    output.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output, inclusion_fraction=fraction)
    plt.close(fig)
    return output


def _clean(axis: plt.Axes) -> None:
    """Use the shared light grid and open top/right axes."""
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.15)
    axis.set_axisbelow(True)


def plot_representation_heatmaps(summaries: dict[str, pd.DataFrame], output: Path) -> Path:
    """Show the five matched representation conditions as annotated heatmaps."""
    apply_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(latex_fraction_width(0.94), 5.0), layout="constrained")
    cmap = LinearSegmentedColormap.from_list("white_gray_green", [(0, "#ffffff"), (0.4, "#dedede"), (1, "#006837")])
    cmap.set_bad("#eeeeee")
    columns = [(model, method) for model in MODELS for method in METHODS]
    labels = [
        "Original control",
        "Consistent transformation",
        "Fine-tuning train",
        "Fine-tuning + ICL train",
        "Test data",
    ]
    for axis, (title, frame), letter in zip(axes, summaries.items(), "AB", strict=True):
        rows = frame[frame.metric == frame.score_metric]
        pooled = pooled_subsampling_summary(
            rows, value_column="normalized_score_mean", group_columns=("model", "method", "condition")
        ).set_index(["condition", "model", "method"])
        values = np.full((5, 6), np.nan)
        for y, condition in enumerate(REPRESENTATION_SCORE_CONDITION_ORDER):
            for x, (model, method) in enumerate(columns):
                if condition == "fine_tuning_train" and method == "icl":
                    axis.plot([x - 0.30, x + 0.30], [y + 0.30, y - 0.30], color="#64748b", linewidth=1.2)
                    continue
                key = (condition, model, method)
                if key not in pooled.index:
                    continue
                row = pooled.loc[key]
                values[y, x] = row["mean"]
                axis.text(
                    x,
                    y,
                    f"{row['mean']:.2f}\n±{row['sem']:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if np.dot(cmap(row["mean"])[:3], [0.299, 0.587, 0.114]) < 0.52 else "#17212b",
                )
        mesh = axis.imshow(np.ma.masked_invalid(values), vmin=0, vmax=1, cmap=cmap, aspect="auto")
        axis.set_yticks([])
        for y, label in enumerate(labels):
            axis.text(
                -0.025,
                y + (0.16 if y in (0, 2, 3) else 0),
                label,
                transform=axis.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=8,
            )
        for y, heading in (
            (0, "Controls"),
            (2, "Fine-tuning–test mismatch"),
            (3, "ICL-train–test mismatch"),
        ):
            axis.text(
                -0.025,
                y - 0.29,
                heading,
                transform=axis.get_yaxis_transform(),
                fontsize=8,
                fontweight="bold",
                ha="right",
                va="center",
            )
        axis.set_yticks(np.arange(-0.5, 5, 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.5)
        axis.set_xticks(range(6), [method.upper() for _, method in columns], fontsize=8)
        axis.xaxis.tick_top()
        for index, model in enumerate(MODELS):
            axis.text(
                (2 * index + 1) / 6,
                1.16,
                MODEL_LABELS[model],
                transform=axis.transAxes,
                ha="center",
                fontsize=9,
                fontweight="bold",
                color=MODEL_COLORS[model],
            )
        for boundary in (1.5, 3.5):
            axis.axvline(boundary, color="#334155", linewidth=1.6, zorder=2)
        axis.set_title(f"{letter}  {title}", loc="left", pad=33)
        axis.set_xticks(np.arange(-0.5, 6, 1), minor=True)

        axis.tick_params(which="both", length=0)
        axis.spines[:].set_visible(False)
    bar = fig.colorbar(mesh, ax=axes, location="right", shrink=0.86, fraction=0.045, pad=0.025)
    bar.set_label("Mean normalized score (1 = best)")
    return _save(fig, output)


def plot_class_tradeoff(summary: pd.DataFrame, output: Path) -> Path:
    """Pair class-accuracy changes without treating classes as independent runs."""
    apply_plot_style()
    routes = {
        "icl_balanced": ("o", "Balanced ICL\n(no SFT)"),
        "sft_shared": ("s", "Balanced SFT +\nbalanced ICL"),
        "sft_context": ("^", "Full-data SFT +\nbalanced ICL"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH_INCHES, 2.8), sharex=True, sharey=True)
    for axis, model in zip(axes, MODELS, strict=True):
        data = summary[summary.model == model].set_index(["route", "role"])
        for route, (marker, _) in routes.items():
            major, minor = data.loc[(route, "majority")], data.loc[(route, "minority")]
            axis.errorbar(
                major["mean"],
                minor["mean"],
                xerr=major["se"],
                yerr=minor["se"],
                fmt=marker,
                color=MODEL_COLORS[model],
                markersize=6,
                markerfacecolor="white" if route == "icl_balanced" else MODEL_COLORS[model],
                markeredgecolor="#374151",
                markeredgewidth=0.6,
                elinewidth=0.8,
                capsize=2,
                zorder=3,
            )
        axis.set_title(MODEL_LABELS[model], color=MODEL_COLORS[model], fontweight="bold")
        axis.set_xlim(-0.30, -0.16)
        axis.set_ylim(0.33, 0.57)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=3))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.tick_params(labelsize=8)
        _clean(axis)
    axes[0].set_ylabel("Minority-class accuracy change")
    fig.supxlabel("Majority-class accuracy change", y=0.025, fontsize=9)
    handles = [
        Line2D(
            [],
            [],
            marker=marker,
            linestyle="none",
            color="#374151",
            markerfacecolor="white" if route == "icl_balanced" else "#64748b",
            label=label,
        )
        for route, (marker, label) in routes.items()
    ]
    fig.legend(
        handles=handles, loc="upper center", ncol=3, fontsize=8, frameon=False, columnspacing=1.4, handletextpad=0.5
    )
    fig.subplots_adjust(left=0.12, right=0.985, top=0.76, bottom=0.23, wspace=0.16)
    return _save(fig, output, 1.0)


def paired_score(
    axis: plt.Axes,
    rows: list,
    position: float,
    model: str,
    method: str,
    mean: str = "normalized_score_mean",
    sem: str = "normalized_score_sem",
    marker_size: float = 4.5,
) -> None:
    """Connect two states with open/filled marks; circle = ICL and square = SFT."""
    color = MODEL_COLORS[model]
    axis.plot([row[mean] for row in rows], [position, position], color=color, alpha=0.55, linewidth=1)
    for state, row in enumerate(rows):
        axis.errorbar(
            row[mean],
            position,
            xerr=row[sem],
            fmt="o" if method == "icl" else "s",
            color=color,
            markersize=marker_size,
            markerfacecolor="white" if state == 0 else color,
            elinewidth=0.7,
            capsize=1.5,
            zorder=3,
        )


def paired_handles(first: str, second: str) -> list[Line2D]:
    """Order two-column legend entries with ICL above SFT and states left to right."""
    return [
        Line2D(
            [],
            [],
            color="#475569",
            marker="o" if method == "icl" else "s",
            markerfacecolor="white" if state == 0 else "#475569",
            linestyle="none",
            label=f"{method.upper()} · {label}",
        )
        for state, label in enumerate((first, second))
        for method in METHODS
    ]


def plot_heavy_tail_pairs(summary: pd.DataFrame, output: Path) -> Path:
    """Render the final horizontal three-dataset heavy-tail comparison."""
    apply_plot_style()
    datasets = {
        "healthcare_insurance_expenses": "Healthcare expenses",
        "houses": "Houses",
        "miami_housing": "Miami housing",
    }
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH_INCHES, 3.6), sharex=True, sharey=True)
    for axis, (dataset, label) in zip(axes, datasets.items(), strict=True):
        rows = summary[summary.dataset == dataset].set_index(["model", "method", "condition"])
        for index, mitigation in enumerate(MITIGATIONS):
            for model_index, model in enumerate(MODELS):
                for method_index, method in enumerate(METHODS):
                    pair = [
                        rows.loc[(model, method, f"{state}/{mitigation}")]
                        for state in ("original", "fit_side_contaminated")
                    ]
                    position = index * 7.3 + model_index * 2.4 + method_index
                    paired_score(axis, pair, position, model, method, marker_size=4.0)
        title = label.replace("Healthcare expenses", "Healthcare\nexpenses")
        axis.set_title(title, fontsize=9, loc="left", fontweight="bold", pad=5)
        axis.set_yticks([2.9, 10.2, 17.5], ["Raw\ntarget", "Winsorized", "Log\ntransformation"], fontsize=8)
        axis.set_xlim(0, 1.04)
        axis.set_ylim(21.1, -0.7)
        axis.set_xticks([0, 0.5, 1])
        axis.tick_params(labelsize=8, axis="y", length=0)
        _clean(axis)
        axis.grid(axis="y", visible=False)
        for boundary in [6.55, 13.85]:
            axis.axhline(boundary, color="#e2e8f0", linewidth=0.6)
    fig.legend(
        handles=MODEL_HANDLES,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=8,
        handlelength=1,
        handletextpad=0.4,
        columnspacing=0.8,
    )
    state_handles = paired_handles("Uncontaminated", "Contaminated")
    fig.legend(
        handles=state_handles[2:] + state_handles[:2],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=False,
        fontsize=8,
        handlelength=0.8,
        handletextpad=0.4,
        columnspacing=0.8,
    )
    fig.supxlabel("Normalized score (1 = best)", fontsize=8, y=0.015)
    fig.subplots_adjust(left=0.19, right=0.98, top=0.68, bottom=0.14, wspace=0.20)
    return _save(fig, output, 1.0)
