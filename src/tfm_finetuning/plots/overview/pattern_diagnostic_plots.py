"""Resource, robustness, metadata, and per-dataset Section 1 plots."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from tfm_finetuning.plots.common.distribution_style import draw_distribution
from tfm_finetuning.plots.common.model_display import (
    CATASTROPHIC_FINAL_ERROR_RATIO,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_ORDER,
    TASK_LABELS,
    TASK_ORDER,
)
from tfm_finetuning.plots.common.plot_style import (
    FULL_WIDTH_INCHES,
    latex_fraction_width,
    save_and_close_section1_figure,
    style_section1_y_axes,
)
from tfm_finetuning.plots.metrics import metric_is_lower_better
from tfm_finetuning.plots.overview.metadata_plots import AXIS_FEATURE_LABELS

_ASSOCIATION_TAXONOMY = (
    (
        "1  Dataset scale",
        ("baseTrainRows", "nFeaturesTotal", "trainingTableCells", "rowToFeatureRatio"),
    ),
    (
        "2  Feature structure and distribution",
        (
            "numericFeatureShare",
            "schemaTypeEntropyNormalized",
            "missingValueShare",
            "numericEffectiveRankRatio",
            "marginalShapeHeterogeneity",
            "sparseRegionOutlierMass",
            "maxCategoricalCardinality",
            "categoricalRareValueShare",
            "categoricalCooccurrenceSparsity",
        ),
    ),
    (
        "3  Target structure",
        (
            "labelEntropyNormalized",
            "targetRobustTailIndex",
            "univariateSignalTop5Share",
            "localTargetVariationByTask",
        ),
    ),
)

_ASSOCIATION_TASK_APPLICABILITY = {
    "labelEntropyNormalized": frozenset({"binary", "multiclass"}),
    "targetRobustTailIndex": frozenset({"regression"}),
}


def _plot_checkpoint_gap(frame: pd.DataFrame, output_dir: Path) -> Path:
    all_results = frame.dropna(subset=["checkpoint_gap"]).copy()
    central_results = _checkpoint_plot_frame(frame)
    width_fraction = 0.72
    fig, axes = plt.subplots(1, 2, figsize=(latex_fraction_width(width_fraction), 2.7))
    panels = [(all_results, "All results"), (central_results, "Excluding collapses")]
    rng = np.random.default_rng(42)
    for ax, (plotted, title) in zip(axes, panels, strict=True):
        for position, model in enumerate(MODEL_ORDER):
            values = plotted.loc[plotted["model"] == model, "checkpoint_gap"].dropna()
            draw_distribution(ax, values, position, MODEL_COLORS[model], rng=rng)
        ax.set_xticks(range(len(MODEL_ORDER)), [MODEL_LABELS[m] for m in MODEL_ORDER])
        ax.set_xlim(-0.4, len(MODEL_ORDER) - 0.5)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_title(title, fontsize=9.2)
        ax.tick_params(axis="x", labelrotation=35, labelsize=9.0)
        for tick, model in zip(ax.get_xticklabels(), MODEL_ORDER, strict=True):
            tick.set_color(MODEL_COLORS[model])
            tick.set_horizontalalignment("right")
        ax.tick_params(axis="y", labelsize=9.0)
    axes[0].set_ylabel("Selected − epoch-100 gain", fontsize=9)
    # The inclusive panel deliberately spans the catastrophic collapses. Style
    # each panel independently so that its coarse tick interval cannot flatten
    # the robust, collapse-excluded central view.
    style_section1_y_axes([axes[0]])
    style_section1_y_axes([axes[1]])
    fig.subplots_adjust(left=0.13, right=0.985, top=0.88, bottom=0.25, wspace=0.40)
    return save_and_close_section1_figure(
        fig,
        output_dir / "checkpoint_selection_gap.pdf",
        tight=False,
        inclusion_fraction=width_fraction,
    )


def _checkpoint_collapse_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return catastrophic TabPFN regression cells and their stable peers."""
    usable = frame[frame["task_family"].eq("regression") & frame["model"].isin(["tabpfnv2", "tabpfnv25"])].dropna(
        subset=["checkpoint_gap", "final_effect"]
    )
    collapse = usable["score_metric"].map(metric_is_lower_better) & (
        usable["final_effect"] < -math.log(CATASTROPHIC_FINAL_ERROR_RATIO)
    )
    summarized = usable.assign(epoch100_error_ratio=np.exp(usable["checkpoint_gap"]))
    return summarized[collapse].copy(), summarized[~collapse].copy()


def _plot_checkpoint_collapses_compact(frame: pd.DataFrame, output_dir: Path) -> Path:
    """Highlight the four late-epoch TabPFN regression collapses at half width."""
    collapses, stable = _checkpoint_collapse_summary(frame)
    width_fraction = 0.48
    fig, ax = plt.subplots(figsize=(latex_fraction_width(width_fraction), 2.65))
    dataset_positions = {"miami_housing": 0.0, "diamonds": 1.0}
    model_offsets = {"tabpfnv2": 0.10, "tabpfnv25": -0.10}

    stable_ratios = stable["epoch100_error_ratio"].to_numpy()
    if stable_ratios.size:
        stable_y = 2.0
        ax.hlines(
            stable_y,
            stable_ratios.min(),
            stable_ratios.max(),
            color="#8b95a5",
            linewidth=1.2,
            zorder=2,
        )
        ax.scatter(
            np.median(stable_ratios),
            stable_y,
            marker="D",
            s=20,
            color="#667085",
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        ax.annotate(
            f"median {np.median(stable_ratios):.2f}×",
            (max(stable_ratios.max(), 1.0), stable_y),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.0,
            color="#4b5563",
        )

    for row in collapses.itertuples():
        if row.dataset not in dataset_positions:
            continue
        y_position = dataset_positions[row.dataset] + model_offsets[row.model]
        ratio = float(row.epoch100_error_ratio)
        color = MODEL_COLORS[row.model]
        ax.hlines(y_position, 1.0, ratio, color=color, linewidth=1.15, alpha=0.85, zorder=2)
        ax.scatter(
            ratio,
            y_position,
            s=26,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        ax.annotate(
            f"{ratio:.0f}×",
            (ratio, y_position),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.2,
            color="#111827",
        )

    ax.set_xscale("log")
    ax.set_xlim(0.8, 120)
    ax.set_xticks([1, 10, 100], ["1×", "10×", "100×"])
    ax.set_yticks([0, 1, 2], ["Miami Housing", "Diamonds", "Other regression\ndatasets"])
    ax.set_ylim(-0.25, 2.45)
    ax.tick_params(axis="both", labelsize=8.4)
    ax.axvline(1.0, color="#374151", linewidth=0.8, zorder=1)
    ax.grid(axis="x", which="major", color="#d7dce3", linewidth=0.65)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Epoch-100 RMSE /\nselected-checkpoint RMSE", fontsize=8.4)
    ax.set_title("Late-epoch TabPFN collapse", fontsize=9.4, pad=24)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=MODEL_COLORS[model],
            markeredgecolor="none",
            markersize=5,
            label=MODEL_LABELS[model],
        )
        for model in ("tabpfnv2", "tabpfnv25")
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=2,
        frameon=False,
        fontsize=8.0,
        handlelength=0.8,
        handletextpad=0.35,
        columnspacing=0.8,
    )
    fig.subplots_adjust(left=0.36, right=0.91, top=0.72, bottom=0.30)
    return save_and_close_section1_figure(
        fig,
        output_dir / "checkpoint_collapses_compact.pdf",
        tight=False,
        inclusion_fraction=width_fraction,
    )


def _checkpoint_plot_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Exclude catastrophic error collapses from checkpoint-sensitivity summaries."""
    usable = frame.dropna(subset=["checkpoint_gap"]).copy()
    collapse = usable["score_metric"].map(metric_is_lower_better) & (
        usable["final_effect"] < -math.log(CATASTROPHIC_FINAL_ERROR_RATIO)
    )
    return usable[~collapse]


def _plot_combined_association_heatmap(
    associations: pd.DataFrame,
    structural_associations: pd.DataFrame,
    output_dir: Path,
    *,
    filename: str = "association_heatmap_combined.pdf",
    gain_label: str = "SFT gain over ICL",
) -> Path:
    """Compare associations using a compact three-part dataset-characteristic taxonomy."""
    preprocessing = "shared"
    general = associations[associations["preprocessing"] == preprocessing].copy()
    structural = structural_associations.reindex(columns=associations.columns)
    structural = structural[structural["preprocessing"] == preprocessing].copy()
    structural = structural[structural["feature"] != "univariateSignalTop5Share"]
    combined = pd.concat([general, structural], ignore_index=True, sort=False)
    combined["feature"] = combined["feature"].replace(
        {
            "localClassDisagreementRate": "localTargetVariationByTask",
            "localTargetVariationIqr": "localTargetVariationByTask",
        }
    )
    columns = [(task, model) for task in TASK_ORDER for model in MODEL_ORDER]
    figure, axes = plt.subplots(
        len(_ASSOCIATION_TAXONOMY),
        1,
        figsize=(FULL_WIDTH_INCHES, 6.45),
        sharex=True,
        gridspec_kw={
            "height_ratios": [len(features) for _, features in _ASSOCIATION_TAXONOMY],
            "hspace": 0.20,
        },
    )
    image = None
    for panel_index, (panel_title, features) in enumerate(_ASSOCIATION_TAXONOMY):
        axis = axes[panel_index]
        matrix = np.full((len(features), len(columns)), np.nan)
        cell_rows: dict[tuple[int, int], pd.Series] = {}
        for feature_index, feature in enumerate(features):
            feature_rows = combined[combined["feature"] == feature]
            for column_index, (task, model) in enumerate(columns):
                match = feature_rows[(feature_rows["task_family"] == task) & (feature_rows["model"] == model)]
                if match.empty:
                    applicable_tasks = _ASSOCIATION_TASK_APPLICABILITY.get(feature)
                    empty_symbol = "--" if applicable_tasks is not None and task not in applicable_tasks else "×"
                    axis.text(
                        column_index,
                        feature_index,
                        empty_symbol,
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color="#6b7280",
                    )
                    continue
                row = match.iloc[0]
                rho = float(row["spearman_rho"])
                if not np.isfinite(rho):
                    continue
                matrix[feature_index, column_index] = rho
                cell_rows[(feature_index, column_index)] = row
        image = axis.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        axis.set_yticks(
            range(len(features)),
            [AXIS_FEATURE_LABELS.get(feature, feature) for feature in features],
            fontsize=8.0,
        )
        axis.tick_params(axis="y", length=0, pad=4)
        for tick_label in axis.get_yticklabels():
            tick_label.set_linespacing(0.85)
        axis.set_title(panel_title, loc="left", fontsize=8.5, pad=2, fontweight="semibold")
        axis.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(features), 1), minor=True)
        axis.grid(which="minor", color="#d1d5db", linewidth=0.55)
        axis.tick_params(which="minor", bottom=False, left=False)
        for row_boundary in np.arange(0.5, len(features) - 0.5, 1):
            axis.plot(
                [-0.55, 1.0],
                [row_boundary, row_boundary],
                transform=axis.get_yaxis_transform(),
                color="#c9ced6",
                linewidth=0.55,
                clip_on=False,
                zorder=0,
            )
        if panel_title == "2  Feature structure and distribution":
            categorical_start = features.index("maxCategoricalCardinality")
            axis.axhline(categorical_start - 0.5, color="#6b7280", linewidth=1.0)
        for boundary in (3.5, 7.5):
            axis.axvline(boundary, color="#4b5563", linewidth=1.2)
        for (feature_index, column_index), row in cell_rows.items():
            rho = float(row["spearman_rho"])
            interval_excludes_zero = bool(float(row["bootstrap_ci_low"]) > 0 or float(row["bootstrap_ci_high"]) < 0)
            q_value = pd.to_numeric(pd.Series([row.get("q_value_bh")]), errors="coerce").iloc[0]
            adjusted = bool(np.isfinite(q_value) and q_value < 0.05)
            magnitude = f"{abs(rho):.2f}".removeprefix("0")
            value_label = ("−" if rho < 0 else "") + magnitude
            text_color = "white" if abs(rho) >= 0.52 else "#111827"
            axis.text(
                column_index,
                feature_index,
                value_label,
                ha="center",
                va="center",
                fontsize=7.5,
                fontweight="bold" if interval_excludes_zero else "normal",
                color=text_color,
            )
            if adjusted:
                axis.scatter(
                    column_index + 0.34,
                    feature_index - 0.29,
                    marker="*",
                    s=17,
                    linewidth=0,
                    color=text_color,
                    zorder=4,
                )
    axes[0].tick_params(axis="x", top=False, labeltop=False, bottom=False, labelbottom=False)
    axes[-1].set_xticks(
        range(len(columns)),
        [MODEL_LABELS[model] for _, model in columns],
        rotation=48,
        ha="right",
        rotation_mode="anchor",
        fontsize=8.0,
    )
    axes[-1].tick_params(axis="x", bottom=True, length=3.5, width=0.8, pad=3)
    for axis in axes[:-1]:
        axis.tick_params(axis="x", bottom=False, labelbottom=False)
    for label, (_, model) in zip(axes[-1].get_xticklabels(), columns, strict=True):
        label.set_color(MODEL_COLORS[model])
        label.set_fontweight("semibold")
    for tick, (_, model) in zip(axes[-1].xaxis.get_major_ticks(), columns, strict=True):
        tick.tick1line.set_color(MODEL_COLORS[model])
    assert image is not None
    figure.subplots_adjust(left=0.31, right=0.84, top=0.865, bottom=0.115, hspace=0.20)
    heatmap_bounds = axes[0].get_position()
    header_axis = figure.add_axes([heatmap_bounds.x0, 0.89, heatmap_bounds.width, 0.035])
    header_axis.set_xlim(-0.5, len(columns) - 0.5)
    header_axis.set_ylim(0, 1)
    header_axis.axis("off")
    task_centres = [index * len(MODEL_ORDER) + (len(MODEL_ORDER) - 1) / 2 for index in range(len(TASK_ORDER))]
    for centre, task in zip(task_centres, TASK_ORDER, strict=True):
        header_axis.text(
            centre,
            0.5,
            TASK_LABELS[task],
            ha="center",
            va="center",
            fontsize=9.0,
            fontweight="semibold",
        )
    for boundary in (3.5, 7.5):
        header_axis.axvline(boundary, color="#4b5563", linewidth=1.0)
    colorbar_axis = figure.add_axes([0.865, 0.19, 0.022, 0.53])
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label(f"Spearman correlation with {gain_label}", fontsize=8.5)
    colorbar.ax.tick_params(labelsize=8.0)
    return save_and_close_section1_figure(
        figure,
        output_dir / filename,
        tight=False,
        inclusion_fraction=1.0,
        max_inclusion_height_fraction=0.82,
        minimum_text_size=7.5,
    )
