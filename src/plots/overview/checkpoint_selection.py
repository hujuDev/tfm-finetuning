"""Plot dataset-level validation-selected epochs from the canonical evidence.

Example: ``plot_checkpoint_selection(frame, output_dir)`` in the Section 1
generator, where ``frame`` contains the default preprocessing route only.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.common.model_display import (
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_ORDER,
    TASK_LABELS,
    TASK_ORDER,
)
from plots.common.plot_style import (
    FULL_WIDTH_INCHES,
    apply_section1_plot_style,
    save_and_close_section1_figure,
    style_section1_y_axes,
)


def checkpoint_selection_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep one observed mean epoch per dataset/model from 100-epoch runs.

    The selected epoch already averages runs. Each dataset receives equal
    weight, and epoch zero is a valid selected state.
    """
    columns = ["dataset", "model", "task_family", "selected_epoch"]
    selected = frame.loc[
        frame["training_epochs"].eq(100) & frame["selected_epoch"].between(0, 100),
        columns,
    ].copy()
    if selected.duplicated(["dataset", "model"]).any():
        raise ValueError("Checkpoint plot requires one preprocessing route and row per dataset/model.")
    return selected.sort_values(["task_family", "model", "dataset"]).reset_index(drop=True)


def plot_checkpoint_selection(frame: pd.DataFrame, output_dir: Path) -> Path | None:
    """Save box plots and their source table using the shared thesis style."""
    data = checkpoint_selection_data(frame)
    if data.empty:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "checkpoint_selection_epochs.csv", index=False)
    apply_section1_plot_style()
    figure, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH_INCHES, 2.4), sharey=True, layout="constrained")
    rng = np.random.default_rng(42)
    for axis, task in zip(axes, TASK_ORDER, strict=True):
        for position, model in enumerate(MODEL_ORDER):
            values = data.loc[data["task_family"].eq(task) & data["model"].eq(model), "selected_epoch"].to_numpy()
            if not len(values):
                continue
            color = MODEL_COLORS[model]
            axis.boxplot(
                [values],
                positions=[position],
                widths=0.46,
                showfliers=False,
                patch_artist=True,
                boxprops={"facecolor": color, "alpha": 0.4, "edgecolor": color},
                medianprops={"color": color, "linewidth": 1.5},
                whiskerprops={"color": color},
                capprops={"color": color},
            )
            axis.scatter(
                position + 0.32 + rng.uniform(-0.055, 0.055, len(values)),
                values,
                s=10,
                color=color,
                alpha=0.25,
                edgecolors="none",
                zorder=3,
            )
            axis.scatter(
                position + 0.32,
                values.mean(),
                marker="D",
                s=25,
                color=color,
                edgecolors="white",
                linewidths=0.5,
                zorder=4,
            )
        axis.set_title(TASK_LABELS[task])
        axis.set_xticks(range(len(MODEL_ORDER)), [MODEL_LABELS[m] for m in MODEL_ORDER], rotation=45, ha="right")
        axis.set_xlim(-0.45, len(MODEL_ORDER) - 0.4)
        axis.set_ylim(0, 100)
    axes[0].set_ylabel("Selected epoch\n(mean per dataset)")
    style_section1_y_axes(axes, tick_interval=20)
    return save_and_close_section1_figure(figure, output_dir / "checkpoint_selection_epochs.pdf", tight=False)
