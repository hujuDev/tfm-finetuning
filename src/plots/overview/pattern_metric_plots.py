"""Metric-oriented Section 1 effect plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.common.distribution_style import draw_distribution
from plots.common.model_display import (
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_ORDER,
    TASK_LABELS,
    TASK_ORDER,
)
from plots.common.plot_style import (
    FULL_WIDTH_INCHES,
    latex_fraction_width,
    save_and_close_section1_figure,
    style_section1_y_axes,
)

_GAIN_AXIS_LABELS = {
    "binary": "ROC-AUC gain",
    "multiclass": "Log-loss gain",
    "regression": "RMSE gain",
}


def _draw_effect_variance_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    task: str,
    *,
    show_model_labels: bool = True,
) -> None:
    """Draw one task-family box summary of cross-dataset SFT gains."""
    family = frame[frame["task_family"] == task]
    for model_index, model in enumerate(MODEL_ORDER):
        values = family.loc[family["model"] == model, "effect"].dropna()
        if values.empty:
            continue
        color = MODEL_COLORS[model]
        ax.boxplot(
            [values],
            positions=[model_index],
            widths=0.46,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            boxprops={"facecolor": color, "alpha": 0.4, "edgecolor": color},
            whiskerprops={"color": color},
            capprops={"color": color},
            medianprops={"linewidth": 1.5, "color": color},
        )
    ax.axhline(0, color="#374151", linewidth=0.8)
    ax.set_xticks(
        range(len(MODEL_ORDER)),
        [MODEL_LABELS[model] for model in MODEL_ORDER],
        rotation=35,
        ha="right",
    )
    ax.tick_params(axis="x", labelsize=9.0, pad=1)
    if not show_model_labels:
        ax.tick_params(axis="x", labelbottom=False)
    ax.set_ylabel(_GAIN_AXIS_LABELS[task], fontsize=9.0)


def _preprocessing_difference_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return paired shared-minus-native SFT-gain differences."""
    pivot = frame.pivot_table(
        index=["dataset", "model", "task_family"], columns="preprocessing", values="effect", aggfunc="first"
    )
    required = {"native", "shared"}
    if not required.issubset(pivot.columns):
        raise RuntimeError("Both preprocessing paths are required for the preprocessing comparison.")
    paired = pivot.dropna(subset=list(required)).reset_index()
    paired["difference"] = paired["shared"] - paired["native"]
    return paired


def _draw_preprocessing_difference_panel(
    ax: plt.Axes,
    paired: pd.DataFrame,
    task: str,
    *,
    show_model_labels: bool = True,
) -> None:
    """Draw one task-family box summary of preprocessing-path differences."""
    family = paired[paired["task_family"] == task]
    rng = np.random.default_rng(42)
    for position, model in enumerate(MODEL_ORDER):
        values = family.loc[family["model"] == model, "difference"]
        draw_distribution(ax, values, position, MODEL_COLORS[model], rng=rng)
    ax.axhline(0, color="#374151", linewidth=0.8)
    ax.set_xticks(range(len(MODEL_ORDER)), [MODEL_LABELS[model] for model in MODEL_ORDER])
    ax.tick_params(axis="x", labelrotation=35, labelsize=9.0, pad=1)
    for tick, model in zip(ax.get_xticklabels(), MODEL_ORDER, strict=True):
        tick.set_color(MODEL_COLORS[model])
        tick.set_horizontalalignment("right")
    if not show_model_labels:
        ax.tick_params(axis="x", labelbottom=False)


def _plot_preprocessing_difference(frame: pd.DataFrame, output_dir: Path) -> Path:
    """Compare shared and native gain on the same dataset/model cell."""
    paired = _preprocessing_difference_frame(frame)
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH_INCHES, 2.9))
    metric_labels = {
        "binary": "ROC-AUC gain difference",
        "multiclass": "Log-loss gain difference",
        "regression": "RMSE gain difference",
    }
    for ax, task in zip(axes, TASK_ORDER, strict=True):
        _draw_preprocessing_difference_panel(ax, paired, task)
        ax.set_title(TASK_LABELS[task], fontsize=10.0, pad=2)
        ax.set_ylabel(metric_labels[task], fontsize=9.0)
    style_section1_y_axes(axes)
    fig.subplots_adjust(left=0.11, right=0.99, top=0.88, bottom=0.25, wspace=0.47)
    return save_and_close_section1_figure(
        fig,
        output_dir / "preprocessing_gain_difference.pdf",
        tight=False,
    )


def _plot_effect_variance(frame: pd.DataFrame, output_dir: Path) -> Path:
    """Render cross-dataset box summaries with room to distinguish small gains."""
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH_INCHES, 2.6))
    rng = np.random.default_rng(42)
    for ax, task in zip(axes, TASK_ORDER, strict=True):
        _draw_effect_variance_panel(ax, frame, task)
        for position, model in enumerate(MODEL_ORDER):
            values = frame.loc[(frame["task_family"] == task) & (frame["model"] == model), "effect"].dropna()
            if values.empty:
                continue
            ax.scatter(
                position + 0.32 + rng.uniform(-0.055, 0.055, len(values)),
                values,
                s=10,
                color=MODEL_COLORS[model],
                alpha=0.25,
                edgecolors="none",
                zorder=3,
            )
            ax.scatter(
                position + 0.32,
                values.mean(),
                marker="D",
                s=25,
                color=MODEL_COLORS[model],
                edgecolors="white",
                linewidths=0.5,
                zorder=4,
            )
        ax.set_title(TASK_LABELS[task], fontsize=10.0, pad=3)
    style_section1_y_axes(axes, tick_interval=0.05)
    fig.subplots_adjust(left=0.11, right=0.99, top=0.90, bottom=0.25, wspace=0.47)
    return save_and_close_section1_figure(
        fig,
        output_dir / "fine_tuning_gain_variance.pdf",
        tight=False,
    )


def _plot_metric_gains(metric_frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Render compact horizontal metric rows without mixing their scales.

    Binary classification and regression use shallow three-panel rows at the
    full thesis width.  The two-panel multiclass row uses a narrower centered
    canvas so its panels retain comparable physical dimensions.
    """
    metric_order = {
        "binary": ["roc_auc", "accuracy", "log_loss"],
        "multiclass": ["accuracy", "log_loss"],
        "regression": ["rmse", "mae", "r2"],
    }
    paths: list[Path] = []
    for task in TASK_ORDER:
        available = [
            metric
            for metric in metric_order[task]
            if not metric_frame[(metric_frame["task_family"] == task) & (metric_frame["metric"] == metric)].empty
        ]
        if not available:
            continue
        width_fraction = 0.78 if task == "multiclass" else 1.0
        fig, raw_axes = plt.subplots(
            1,
            len(available),
            figsize=(latex_fraction_width(width_fraction), 2.65),
            squeeze=False,
        )
        axes = raw_axes.ravel()
        for ax, metric in zip(axes, available, strict=True):
            rows = metric_frame[(metric_frame["task_family"] == task) & (metric_frame["metric"] == metric)]
            _draw_metric_gain_panel(ax, rows, metric)
        style_section1_y_axes(axes)
        left_margin = 0.15 if task == "multiclass" else 0.10
        fig.subplots_adjust(
            left=left_margin,
            right=0.99,
            top=0.89,
            bottom=0.25,
            wspace=0.48,
        )
        paths.append(
            save_and_close_section1_figure(
                fig,
                output_dir / f"metric_gains_{task}.pdf",
                tight=False,
                inclusion_fraction=width_fraction,
            )
        )
    return paths


def _draw_metric_gain_panel(ax: plt.Axes, rows: pd.DataFrame, metric: str) -> None:
    """Draw one metric panel while preserving its independent y scale."""
    positions = np.arange(len(MODEL_ORDER), dtype=float)
    rng = np.random.default_rng(42)
    for index, model in enumerate(MODEL_ORDER):
        values = rows.loc[rows["model"] == model, "effect"].dropna()
        draw_distribution(ax, values, positions[index], MODEL_COLORS[model], rng=rng)
    ax.axhline(0, color="#374151", linewidth=0.8)
    ax.set_xticks(positions, [MODEL_LABELS[model] for model in MODEL_ORDER], rotation=35, ha="right")
    ax.set_xlim(positions[0] - 0.4, positions[-1] + 0.5)
    ax.tick_params(axis="x", labelsize=9.0, pad=1)
    ax.set_ylabel(_metric_gain_label(metric))
    ax.set_title(_metric_display_name(metric))


def _metric_gain_label(metric: str) -> str:
    return "SFT − ICL" if metric in {"roc_auc", "accuracy", "r2"} else "log(ICL / SFT)"


def _plot_metric_gain_agreement(metric_frame: pd.DataFrame, output_dir: Path) -> Path:
    """Show whether SFT gains over ICL agree across complementary metrics."""
    comparisons = [
        ("binary", "roc_auc", "accuracy"),
        ("binary", "roc_auc", "log_loss"),
        ("multiclass", "accuracy", "log_loss"),
        ("regression", "rmse", "mae"),
    ]
    # Match the 55-dataset population used in the main overview.
    shared = metric_frame.loc[
        metric_frame["preprocessing"].eq("shared") & ~metric_frame["dataset"].isin(["airlines", "miniboone"])
    ]
    width_fraction = 1.0
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(FULL_WIDTH_INCHES, 2.25),
        squeeze=False,
    )
    for ax, (task, left_metric, right_metric) in zip(axes.flat, comparisons, strict=True):
        rows = shared[shared["task_family"] == task]
        pivot = rows.pivot_table(index=["dataset", "model"], columns="metric", values="effect", aggfunc="first")
        if left_metric not in pivot or right_metric not in pivot:
            ax.set_visible(False)
            continue
        common = pivot[[left_metric, right_metric]].dropna().reset_index()
        for model in MODEL_ORDER:
            model_rows = common[common["model"] == model]
            ax.scatter(
                model_rows[left_metric],
                model_rows[right_metric],
                s=18,
                alpha=0.72,
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
            )
        rho = common[left_metric].corr(common[right_metric], method="spearman") if len(common) >= 3 else np.nan
        ax.axhline(0, color="#c0c5ce", linewidth=0.6)
        ax.axvline(0, color="#c0c5ce", linewidth=0.6)
        ax.set_xlabel(f"{_metric_display_name(left_metric)} gain")
        ax.set_ylabel(f"{_metric_display_name(right_metric)} gain")
        ax.set_title(
            f"{TASK_LABELS[task]}\nρ={rho:.2f} · n={len(common)}",
            fontsize=8,
        )
    style_section1_y_axes(axes.flat)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=4,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        columnspacing=0.8,
        handletextpad=0.2,
    )
    fig.subplots_adjust(left=0.10, right=0.99, top=0.68, bottom=0.22, wspace=0.85)
    return save_and_close_section1_figure(
        fig,
        output_dir / "metric_gain_agreement.pdf",
        tight=False,
        inclusion_fraction=width_fraction,
    )


def _metric_display_name(metric: str) -> str:
    names = {
        "roc_auc": "ROC-AUC",
        "accuracy": "Accuracy",
        "log_loss": "Log-loss",
        "rmse": "RMSE",
        "mae": "MAE",
        # Keep the superscript at the title's full point size.  MathText would
        # render the exponent as a smaller child glyph in the exported PDF.
        "r2": "R²",
    }
    return names.get(metric, metric.replace("_", " ").title())
