"""Cross-dataset overview figures and raw-score tables for Section 2.

These helpers complement the dataset-faceted figures in
``transformation_figures``.  They deliberately pool only
best-relative normalized scores or dimensionless gain quantities; raw metric
values remain dataset-specific and are exported as LaTeX tables.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from tfm_finetuning.plots.common.model_display import MODEL_COLORS, MODEL_LABELS, MODEL_ORDER
from tfm_finetuning.plots.common.plot_style import (
    apply_section1_plot_style,
    latex_fraction_width,
    save_section1_figure,
)
from tfm_finetuning.plots.transformations.transformation_figures import (
    GROUP2_MODEL_ORDER,
    HORIZONTAL_AB_LATEX_WIDTH_FRACTION,
    METHOD_STYLES,
)

# These horizontal composites are included nearly full width. Matching that
# physical width in the source PDF preserves Matplotlib point sizes in print.
POOLED_COMPOSITE_LATEX_WIDTH_FRACTION = HORIZONTAL_AB_LATEX_WIDTH_FRACTION
POOLED_COMPOSITE_HEIGHT_INCHES = 3.35
REPRESENTATION_COMPOSITE_HEIGHT_INCHES = 5.65


# The findings comparison puts the matched transformation control directly
# beside the original control. The appendix tables retain the experimental
# order above.
REPRESENTATION_SCORE_CONDITION_ORDER = (
    "original",
    "all_data",
    "fine_tuning_train",
    "both_train_datasets",
    "test_data",
)


def combine_representation_alignment_summaries(
    update_summary: pd.DataFrame,
    alignment_summary: pd.DataFrame,
    *,
    value_column: str,
    control_condition: str,
    fine_tuning_only_condition: str,
    alignment_condition_order: Sequence[str],
) -> pd.DataFrame:
    """Combine the two run batches into one five-condition alignment design.

    The control is available in both batches and is averaged within each
    dataset--model cell. The fine-tuning-only condition comes from the newer
    run batch. The remaining representation allocations come from the earlier
    batch in which the fine-tuning and ICL train data use the same
    representation.
    """
    if len(alignment_condition_order) != 4:
        raise ValueError("Alignment condition order must contain four settings.")
    _require_columns(update_summary, {"dataset", "model", "condition", value_column})
    _require_columns(alignment_summary, {"dataset", "model", "condition", value_column})

    update_map = {
        control_condition: "original",
        fine_tuning_only_condition: "fine_tuning_train",
    }
    alignment_map = dict(
        zip(
            alignment_condition_order,
            ("original", "both_train_datasets", "test_data", "all_data"),
            strict=True,
        )
    )
    update_rows = update_summary[update_summary["condition"].isin(update_map)].copy()
    alignment_rows = alignment_summary[alignment_summary["condition"].isin(alignment_map)].copy()
    missing_update = set(update_map).difference(update_rows["condition"])
    missing_alignment = set(alignment_map).difference(alignment_rows["condition"])
    if missing_update or missing_alignment:
        raise ValueError(
            "Missing representation conditions: "
            f"update={sorted(missing_update)}, alignment={sorted(missing_alignment)}"
        )
    update_rows["condition"] = update_rows["condition"].map(update_map)
    alignment_rows["condition"] = alignment_rows["condition"].map(alignment_map)
    combined = pd.concat([update_rows, alignment_rows], ignore_index=True)

    identity_candidates = (
        "family",
        "dataset",
        "task_family",
        "score_metric",
        "metric",
        "model",
        "method",
        "checkpoint",
        "condition",
    )
    identity = [column for column in identity_candidates if column in combined.columns]
    return combined.groupby(identity, as_index=False, dropna=False)[value_column].mean()


def pooled_subsampling_summary(
    frame: pd.DataFrame,
    *,
    value_column: str,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    """Pool dataset-level subsampling summaries with cross-dataset uncertainty."""
    required = {"dataset", *group_columns, value_column}
    _require_columns(frame, required)
    dataset_level = frame.groupby(["dataset", *group_columns], as_index=False, dropna=False)[value_column].mean()
    pooled = (
        dataset_level.groupby(list(group_columns), as_index=False, dropna=False)[value_column]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    pooled["sem"] = pooled["std"] / pooled["count"].pow(0.5)
    pooled.loc[pooled["count"] < 2, "sem"] = 0.0
    return pooled


def plot_pooled_primary_score_gain_composite(
    normalized_summary: pd.DataFrame,
    gain_summary: pd.DataFrame,
    *,
    output_path: Path,
    condition_order: Sequence[str],
    condition_labels: Sequence[str],
    x_axis_label: str,
    score_ylim: tuple[float, float] | None = None,
    score_ticks: Sequence[float] | None = None,
    stacked: bool = False,
) -> Path:
    """Plot pooled primary scores and SFT gains for one ordered experiment."""
    if len(condition_order) != len(condition_labels):
        raise ValueError("Condition labels must match the condition order.")
    score_rows = normalized_summary[normalized_summary["metric"] == normalized_summary["score_metric"]].copy()
    gain_rows = gain_summary[gain_summary["metric"] == gain_summary["score_metric"]].copy()
    score_pooled = pooled_subsampling_summary(
        score_rows,
        value_column="normalized_score_mean",
        group_columns=("model", "method", "condition"),
    )
    gain_pooled = pooled_subsampling_summary(
        gain_rows,
        value_column="gain_mean",
        group_columns=("model", "condition"),
    )
    models = _ordered_models(pd.concat([score_rows[["model"]], gain_rows[["model"]]]))
    positions = {condition: index for index, condition in enumerate(condition_order)}

    apply_section1_plot_style()
    fig, axes = plt.subplots(
        2 if stacked else 1,
        1 if stacked else 2,
        figsize=(
            latex_fraction_width(POOLED_COMPOSITE_LATEX_WIDTH_FRACTION),
            REPRESENTATION_COMPOSITE_HEIGHT_INCHES if stacked else POOLED_COMPOSITE_HEIGHT_INCHES,
        ),
        sharex=True,
    )
    for model in models:
        for method in ("icl", "sft"):
            rows = score_pooled[
                (score_pooled["model"] == model)
                & (score_pooled["method"] == method)
                & score_pooled["condition"].isin(condition_order)
            ].copy()
            rows["position"] = rows["condition"].map(positions)
            rows = rows.sort_values("position")
            if rows.empty:
                continue
            style = METHOD_STYLES[method]
            axes[0].errorbar(
                rows["position"],
                rows["mean"],
                yerr=rows["sem"].fillna(0),
                color=MODEL_COLORS[model],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markerfacecolor=style["markerfacecolor"] or MODEL_COLORS[model],
                markersize=4,
                linewidth=1.35,
                capsize=2,
            )
        rows = gain_pooled[(gain_pooled["model"] == model) & gain_pooled["condition"].isin(condition_order)].copy()
        rows["position"] = rows["condition"].map(positions)
        rows = rows.sort_values("position")
        if rows.empty:
            continue
        axes[1].errorbar(
            rows["position"],
            rows["mean"],
            yerr=rows["sem"].fillna(0),
            color=MODEL_COLORS[model],
            marker="o",
            markersize=4,
            linewidth=1.35,
            capsize=2,
        )

    score_low = float((score_pooled["mean"] - score_pooled["sem"].fillna(0)).min())
    score_high = float((score_pooled["mean"] + score_pooled["sem"].fillna(0)).max())
    if score_ylim is None:
        score_span = max(0.2, score_high - score_low)
        score_ylim = (max(0.0, score_low - 0.08 * score_span), min(1.02, max(1.005, score_high + 0.04 * score_span)))
    axes[0].set_ylim(*score_ylim)
    if score_ticks is not None:
        axes[0].set_yticks(score_ticks)
    axes[0].set_title("A", loc="left", fontsize=10)
    axes[0].set_ylabel("Normalized score", fontsize=9)
    axes[1].axhline(0, color="#374151", linewidth=0.8, zorder=0)
    axes[1].set_title("B", loc="left", fontsize=10)
    axes[1].set_ylabel("SFT gain over ICL", fontsize=9)
    for ax in axes:
        ax.set_xlim(-0.2, len(condition_order) - 0.8)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes:
        tick_labels = _horizontal_condition_labels(condition_labels)
        ax.set_xticks(range(len(condition_order)), tick_labels, fontsize=9)
        if len(condition_order) > 5:
            ax.tick_params(axis="x", labelrotation=25)
            for label in ax.get_xticklabels():
                label.set_horizontalalignment("right")
    axes[1].margins(y=0.14)
    model_handles = _model_handles(models)
    method_handles = _method_handles(["icl", "sft"])
    method_handles[-1].set_label("SFT")
    fig.legend(
        handles=[*model_handles, *method_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        frameon=False,
        ncol=len(model_handles) + len(method_handles),
        fontsize=9,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=0.8,
    )
    if stacked:
        fig.tight_layout(rect=(0.0, 0.08, 1, 0.91), h_pad=1.1)
    else:
        fig.tight_layout(rect=(0.0, 0.10, 1, 0.87), w_pad=1.4)
    fig.supxlabel(x_axis_label, y=0.015, fontsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_section1_figure(
        fig,
        output_path,
        inclusion_fraction=POOLED_COMPOSITE_LATEX_WIDTH_FRACTION,
    )
    plt.close(fig)
    return output_path


def _horizontal_condition_labels(condition_labels: Sequence[str]) -> list[str]:
    """Wrap paired fit/test abbreviations without splitting numeric fractions."""
    labels = [str(label) for label in condition_labels]
    should_wrap = (
        bool(labels) and all("/" in label and "\n" not in label for label in labels) and max(map(len, labels)) > 7
    )
    return [label.replace("/", "/\n", 1) for label in labels] if should_wrap else labels


def plot_subsampling_pooled_composite(
    normalized_summary: pd.DataFrame,
    gain_summary: pd.DataFrame,
    *,
    output_path: Path,
    condition_order: Sequence[str],
    condition_labels: Sequence[str],
) -> Path:
    """Plot the train-data-subsampling composite on its fixed score range."""
    return plot_pooled_primary_score_gain_composite(
        normalized_summary,
        gain_summary,
        output_path=output_path,
        condition_order=condition_order,
        condition_labels=condition_labels,
        x_axis_label="Train-data fraction",
        score_ylim=(0.595, 1.005),
        score_ticks=(0.6, 0.8, 1.0),
    )


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


def _method_handles(methods: Sequence[str]) -> list[Line2D]:
    labels = {"icl": "ICL", "sft": "SFT (selected)"}
    return [
        Line2D(
            [],
            [],
            color="#374151",
            linestyle=METHOD_STYLES[method]["linestyle"],
            marker=METHOD_STYLES[method]["marker"],
            markerfacecolor=METHOD_STYLES[method]["markerfacecolor"] or "#374151",
            linewidth=1.2,
            markersize=4,
            label=labels[method],
        )
        for method in methods
    ]


def _require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required reporting columns: {sorted(missing)}")


__all__ = []
