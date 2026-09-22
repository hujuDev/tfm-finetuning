"""Render paired ICL/SFT normalized scores for the thesis's 55 datasets.

Example: ``plot_normalized_distributions(canonical_frame, output_dir)``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from plots.common.distribution_style import draw_distribution
from plots.common.model_display import MODEL_COLORS, MODEL_LABELS, MODEL_ORDER, TASK_LABELS, TASK_ORDER
from plots.common.plot_style import FULL_WIDTH_INCHES, save_and_close_section1_figure, style_section1_y_axes


def plot_normalized_distributions(frame, output_dir: Path) -> Path:
    """Plot existing best-relative scores with hollow ICL and filled SFT marks."""
    frame = frame.loc[frame["preprocessing"].eq("shared") & ~frame["dataset"].isin(["airlines", "miniboone"])]
    fig, axes = plt.subplots(3, 1, figsize=(FULL_WIDTH_INCHES, 5.3))
    rng = np.random.default_rng(42)
    metrics = {"binary": "ROC-AUC", "multiclass": "Log-loss", "regression": "RMSE"}
    for ax, task in zip(axes, TASK_ORDER, strict=True):
        family = frame.loc[frame["task_family"].eq(task)]
        for index, model in enumerate(MODEL_ORDER):
            rows = family.loc[family["model"].eq(model)]
            for delta, column, filled in [(-0.52, "normalized_baseline", False), (0.52, "normalized_selected", True)]:
                draw_distribution(ax, rows[column], index * 2.6 + delta, MODEL_COLORS[model], rng=rng, filled=filled)
        ax.set_xticks(np.arange(len(MODEL_ORDER)) * 2.6, [MODEL_LABELS[m] for m in MODEL_ORDER])
        ax.set_xlim(-1.0, 2.6 * (len(MODEL_ORDER) - 1) + 1.1)
        ax.axhline(1, color="#8b96a8", linewidth=0.8, linestyle="--", zorder=0)
        ax.set_title(f"{TASK_LABELS[task]} · n={family.dataset.nunique()} · {metrics[task]}", fontsize=10)
    style_section1_y_axes(axes)
    for ax, limits in zip(axes, [(0.735, 1.02), (0.1, 1.02), (0.55, 1.02)], strict=True):
        ax.set_ylim(*limits)
    fig.supylabel("Best-relative normalized primary score", fontsize=9, x=0.015)
    fig.legend(
        handles=[
            Patch(facecolor="none", edgecolor="#64748b", label="ICL (left)"),
            Patch(facecolor="#64748b", alpha=0.4, edgecolor="#64748b", label="SFT (right)"),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.88, bottom=0.06, hspace=0.65)
    return save_and_close_section1_figure(fig, output_dir / "normalized_score_distributions.pdf", tight=False)
