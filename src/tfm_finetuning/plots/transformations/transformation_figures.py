"""Publication figures for the completed Section 2 transformation studies.

The functions consume already summarized data frames so that experiment-result
selection and figure rendering remain separate.  Every export uses the fixed
Section 1 thesis canvas and typography.

Example:
    ``plot_dataset_dose_response(summary, output_path=Path("noise.pdf"), ...)``
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatter

from tfm_finetuning.plots.common.model_display import MODEL_COLORS, MODEL_LABELS, MODEL_ORDER
from tfm_finetuning.plots.common.plot_style import (
    apply_section1_plot_style,
    latex_fraction_width,
    save_section1_figure,
)

GROUP2_MODEL_ORDER = ("tabpfnv25", "tabiclv2", "mitra")
DEFAULT_TARGET_MODE_COLORS = ("#6b7280", "#4f78cf", "#2f8f83")
HORIZONTAL_AB_LATEX_WIDTH_FRACTION = 0.94
METHOD_STYLES = {
    "icl": {"linestyle": "--", "marker": "o", "markerfacecolor": "white"},
    "sft": {"linestyle": "-", "marker": "^", "markerfacecolor": None},
}


def _use_plain_log_tick_labels(axis: plt.Axes) -> None:
    """Keep adaptive log ticks while avoiding MathText script-sized glyphs."""
    axis.yaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=True))
    axis.yaxis.set_minor_formatter(LogFormatter(base=10, labelOnlyBase=False))


def plot_heavy_tail_mitigation(
    stability_summary: pd.DataFrame,
    *,
    output_path: Path,
    dataset_order: Sequence[str],
    target_mode_order: Sequence[str],
    target_mode_labels: Mapping[str, str],
    inclusion_fraction: float = 0.94,
) -> Path:
    """Show selected-to-epoch-50 stability in one horizontal dataset row."""
    common = {"dataset", "model", "contamination", "target_mode"}
    _require_columns(stability_summary, common | {"stability_mean", "stability_sem"})
    datasets = list(dataset_order)
    if not 1 <= len(datasets) <= 4:
        raise ValueError("A heavy-tail mitigation figure requires one to four datasets.")
    models = _ordered_models(stability_summary)
    positions = {mode: index for index, mode in enumerate(target_mode_order)}
    target_labels = [target_mode_labels.get(mode, mode) for mode in target_mode_order]
    contaminated_dash = (0, (2.5, 1.0))
    contamination_styles = {
        "original": ("-", "o"),
        "fit_side_contaminated": (contaminated_dash, "s"),
    }

    apply_section1_plot_style()
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(latex_fraction_width(inclusion_fraction), 2.70),
        squeeze=False,
    )
    for column_index, dataset in enumerate(datasets):
        ax = axes[0, column_index]
        rows = stability_summary[stability_summary["dataset"] == dataset]
        for contamination, (linestyle, marker) in contamination_styles.items():
            for model in models:
                line_rows = rows[(rows["contamination"] == contamination) & (rows["model"] == model)].copy()
                line_rows["_position"] = line_rows["target_mode"].map(positions)
                line_rows = line_rows.dropna(subset=["_position", "stability_mean"]).sort_values("_position")
                if line_rows.empty:
                    continue
                ax.errorbar(
                    line_rows["_position"],
                    line_rows["stability_mean"],
                    yerr=line_rows["stability_sem"].fillna(0),
                    color=MODEL_COLORS[model],
                    linestyle=linestyle,
                    marker=marker,
                    markersize=3.5,
                    linewidth=1.15,
                    capsize=2,
                )
        ax.axhline(0, color="#374151", linewidth=0.8, zorder=0)
        ax.set_xticks(
            range(len(target_mode_order)),
            target_labels,
            fontsize=9,
            rotation=30,
            ha="right",
            va="top",
            rotation_mode="anchor",
        )
        ax.set_xlim(-0.32, len(target_mode_order) - 1 + 0.32)
        ax.grid(axis="y", alpha=0.2)
        ax.set_title(
            chr(ord("A") + column_index),
            fontsize=10,
            fontweight="bold",
            pad=2,
        )
    axes[0, 0].set_ylabel("log(selected / epoch-50 RMSE)", fontsize=9)

    model_handles = _model_handles(models)
    contamination_handles = [
        Line2D([], [], color="#4b5563", linestyle="-", marker="o", label="Uncontaminated training targets"),
        Line2D(
            [],
            [],
            color="#4b5563",
            linestyle=contaminated_dash,
            marker="s",
            label="Contaminated training targets",
        ),
    ]
    fig.legend(
        handles=model_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        frameon=False,
        ncol=len(model_handles),
        fontsize=9,
        handlelength=1.1,
        handletextpad=0.3,
        columnspacing=0.8,
        borderpad=0.0,
    )
    fig.legend(
        handles=contamination_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        frameon=False,
        ncol=len(contamination_handles),
        fontsize=9,
        handlelength=2.8,
        handletextpad=0.5,
        columnspacing=1.4,
        borderpad=0.0,
    )
    fig.subplots_adjust(left=0.12, right=0.97, top=0.79, bottom=0.34, wspace=0.30)
    return _save(fig, output_path, inclusion_fraction=inclusion_fraction)


def plot_diamonds_mitigation(
    trajectories: pd.DataFrame,
    heldout: pd.DataFrame,
    *,
    output_path: Path,
    target_mode_order: Sequence[str],
    target_mode_labels: Mapping[str, str],
    inclusion_fraction: float = HORIZONTAL_AB_LATEX_WIDTH_FRACTION,
    target_mode_colors: Mapping[str, str] | None = None,
    mode_column: str = "target_mode",
    seed_column: str = "seed",
    epoch_column: str = "epoch",
    validation_column: str = "validation_rmse",
    icl_column: str = "icl_rmse",
    selected_column: str = "selected_rmse",
    final_column: str = "epoch50_rmse",
) -> Path:
    """Plot validation trajectories and held-out RMSE for Diamonds mitigation.

    ``trajectories`` is long-form with ``target_mode``, ``seed``, ``epoch``,
    and ``validation_rmse``. ``heldout`` has one row per representation/seed
    with held-out ICL, selected-checkpoint, and epoch-50 RMSE. Individual seed
    trajectories are shown faintly behind the mean and standard-error band.
    """
    _require_columns(
        trajectories,
        {mode_column, seed_column, epoch_column, validation_column},
    )
    _require_columns(
        heldout,
        {
            mode_column,
            seed_column,
            icl_column,
            selected_column,
            final_column,
        },
    )
    colors = target_mode_colors or {
        mode: DEFAULT_TARGET_MODE_COLORS[index % len(DEFAULT_TARGET_MODE_COLORS)]
        for index, mode in enumerate(target_mode_order)
    }
    _require_positive(trajectories[validation_column], validation_column)
    for column in (icl_column, selected_column, final_column):
        _require_positive(heldout[column], column)

    apply_section1_plot_style()
    fig, (trajectory_ax, heldout_ax) = plt.subplots(
        1,
        2,
        figsize=(latex_fraction_width(inclusion_fraction), 3.55),
        gridspec_kw={"width_ratios": (1.05, 1)},
    )
    for mode in target_mode_order:
        mode_rows = trajectories[trajectories[mode_column] == mode]
        for _, seed_rows in mode_rows.groupby(seed_column, sort=True):
            seed_rows = seed_rows.sort_values(epoch_column)
            trajectory_ax.plot(
                seed_rows[epoch_column],
                seed_rows[validation_column],
                color=colors[mode],
                alpha=0.2,
                linewidth=0.85,
            )
        summary = (
            mode_rows.groupby(epoch_column, as_index=False)[validation_column]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        summary["sem"] = summary["std"] / summary["count"].pow(0.5)
        summary.loc[summary["count"] < 2, "sem"] = 0.0
        lower = np.maximum(summary["mean"] - summary["sem"].fillna(0.0), np.finfo(float).tiny)
        upper = summary["mean"] + summary["sem"].fillna(0.0)
        trajectory_ax.fill_between(
            summary[epoch_column],
            lower,
            upper,
            color=colors[mode],
            alpha=0.14,
            linewidth=0,
        )
        trajectory_ax.plot(
            summary[epoch_column],
            summary["mean"],
            color=colors[mode],
            linewidth=2.0,
            label=target_mode_labels.get(mode, mode),
        )
    trajectory_ax.set_yscale("log")
    _use_plain_log_tick_labels(trajectory_ax)
    trajectory_ax.set_title("A")
    trajectory_ax.set_xlabel("Fine-tuning epoch")
    trajectory_ax.set_ylabel("Validation RMSE\n(log scale; lower is better)")
    trajectory_ax.grid(alpha=0.2)
    offsets = {icl_column: -0.23, selected_column: 0.0, final_column: 0.23}
    markers = {icl_column: "o", selected_column: "s", final_column: "D"}
    for mode_index, mode in enumerate(target_mode_order):
        rows = heldout[heldout[mode_column] == mode]
        for column in (icl_column, selected_column, final_column):
            x = np.full(len(rows), mode_index + offsets[column])
            heldout_ax.scatter(
                x,
                rows[column],
                color=colors[mode],
                marker=markers[column],
                s=18,
                alpha=0.45,
                edgecolors="none",
                zorder=2,
            )
            if not rows.empty:
                standard_error = float(rows[column].std(ddof=1) / np.sqrt(len(rows))) if len(rows) > 1 else 0.0
                heldout_ax.errorbar(
                    mode_index + offsets[column],
                    rows[column].mean(),
                    yerr=standard_error,
                    color=colors[mode],
                    marker=markers[column],
                    markeredgecolor="#111827",
                    markeredgewidth=0.65,
                    markersize=6.5,
                    linestyle="none",
                    elinewidth=1.0,
                    capsize=2.5,
                    zorder=3,
                )
    heldout_ax.set_yscale("log")
    _use_plain_log_tick_labels(heldout_ax)
    heldout_ax.set_title("B")
    heldout_ax.set_xticks(
        range(len(target_mode_order)),
        [target_mode_labels.get(mode, mode) for mode in target_mode_order],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    heldout_ax.set_xlabel("Target representation")
    heldout_ax.set_ylabel("Test RMSE\n(log scale; lower is better)")
    heldout_ax.grid(axis="y", alpha=0.2)
    legend_handles = [
        *[
            Line2D(
                [],
                [],
                color=colors[mode],
                linewidth=1.8,
                label=target_mode_labels.get(mode, mode),
            )
            for mode in target_mode_order
        ],
        Line2D([], [], color="#374151", marker="o", linestyle="None", label="ICL"),
        Line2D([], [], color="#374151", marker="s", linestyle="None", label="Selected checkpoint"),
        Line2D([], [], color="#374151", marker="D", linestyle="None", label="Epoch 50"),
    ]
    # Matplotlib fills legend columns first; interleave the two semantic rows.
    fig.legend(
        handles=[handle for column in range(3) for handle in legend_handles[column::3]],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        frameon=False,
        fontsize=9,
        handlelength=1.4,
        handletextpad=0.45,
        columnspacing=1.1,
    )

    fig.tight_layout(rect=(0.025, 0.02, 0.995, 0.82), w_pad=1.5)
    return _save(fig, output_path, inclusion_fraction=inclusion_fraction)


def _ordered_models(frame: pd.DataFrame) -> list[str]:
    present = set(frame["model"].dropna().astype(str))
    unknown = present.difference(MODEL_ORDER)
    if unknown:
        raise ValueError(f"Unknown model keys cannot use the shared thesis style: {sorted(unknown)}")
    preferred = [model for model in GROUP2_MODEL_ORDER if model in present]
    return preferred + [model for model in MODEL_ORDER if model in present and model not in preferred]


def _model_handles(models: Sequence[str]) -> list[Line2D]:
    return [
        Line2D(
            [],
            [],
            color=MODEL_COLORS[model],
            marker="o",
            linewidth=1.4,
            markersize=4,
            label=MODEL_LABELS[model],
        )
        for model in models
    ]


def _require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required plotting columns: {sorted(missing)}")


def _require_positive(values: pd.Series, column: str) -> None:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric).all() or (numeric <= 0).any():
        raise ValueError(f"{column} must contain only positive finite RMSE values for log scaling.")


def _save(
    fig: plt.Figure,
    path: Path,
    *,
    inclusion_fraction: float = 1.0,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_section1_figure(fig, path, inclusion_fraction=inclusion_fraction)
    plt.close(fig)
    return path


__all__ = ["plot_diamonds_mitigation"]
