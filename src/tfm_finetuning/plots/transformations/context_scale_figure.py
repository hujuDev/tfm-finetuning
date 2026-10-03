"""Final contextual feature-scale comparison.

Example: ``plot_context_interactions(summary, Path("context.pdf"))``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from tfm_finetuning.plots.common.plot_style import apply_section1_plot_style, latex_fraction_width
from tfm_finetuning.plots.transformations.transformation_comparisons import (
    METHODS,
    MODEL_COLORS,
    MODEL_LABELS,
    MODELS,
    _clean,
    _save,
)
from tfm_finetuning.plots.transformations.transformation_overviews import pooled_subsampling_summary


def plot_context_interactions(summary: pd.DataFrame, output: Path) -> Path:
    """Compare methods on aligned axes, retaining the pooled mean and dataset SE."""
    apply_section1_plot_style()
    pooled = pooled_subsampling_summary(
        summary, value_column="normalized_score_mean", group_columns=("model", "method", "condition")
    ).set_index(["model", "method", "condition"])
    fig, axes = plt.subplots(3, 2, figsize=(latex_fraction_width(0.94), 5.7), sharex=True, sharey=True)
    for y, model in enumerate(MODELS):
        for x, method in enumerate(METHODS):
            axis = axes[y, x]
            for scale, style, marker in [("1x", "--", "o"), ("sx", "-", "s")]:
                rows = [pooled.loc[(model, method, f"scale_{scale}_{unit}_unit_column")] for unit in ("no", "with")]
                axis.errorbar(
                    [0, 1],
                    [r["mean"] for r in rows],
                    yerr=[r["sem"] for r in rows],
                    color=MODEL_COLORS[model],
                    linestyle=style,
                    marker=marker,
                    markerfacecolor="white" if scale == "1x" else MODEL_COLORS[model],
                    linewidth=1.3,
                    markersize=4.5,
                    capsize=2,
                )
            axis.set_title(
                f"{MODEL_LABELS[model]} · {method.upper()}", fontsize=9, fontweight="bold", color=MODEL_COLORS[model]
            )
            axis.set_xticks([0, 1], ["No unit", "Matching unit"], fontsize=8)
            axis.tick_params(labelbottom=True)
            axis.set_xlim(-0.2, 1.2)
            axis.set_ylim(0.50, 1.02)
            _clean(axis)
    fig.supylabel("Mean normalized score (1 = best)", fontsize=9, x=0.015)
    fig.legend(
        handles=[
            Line2D(
                [], [], color="#475569", linestyle="--", marker="o", markerfacecolor="white", label="Unscaled (1×)"
            ),
            Line2D([], [], color="#475569", linestyle="-", marker="s", label="Scaled (3×)"),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    fig.subplots_adjust(left=0.13, right=0.98, top=0.89, bottom=0.06, hspace=0.65, wspace=0.18)
    return _save(fig, output)
