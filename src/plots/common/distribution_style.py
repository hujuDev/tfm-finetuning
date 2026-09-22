"""Shared box and adjacent-point styling used by the findings figures.

Example: ``draw_distribution(ax, values, 0, color, rng=rng, filled=True)``.
"""

import numpy as np


def draw_distribution(ax, values, position, color, *, rng, filled=True, width=0.46, offset=0.32) -> None:
    """Draw a model-colored distribution, using hollow marks for ICL."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return
    ax.boxplot(
        [values],
        positions=[position],
        widths=width,
        showfliers=False,
        patch_artist=True,
        manage_ticks=False,
        boxprops={"facecolor": color if filled else "none", "alpha": 0.4 if filled else 1.0, "edgecolor": color},
        medianprops={"color": color, "linewidth": 1.5},
        whiskerprops={"color": color},
        capprops={"color": color},
    )
    ax.scatter(
        position + offset + rng.uniform(-0.055, 0.055, len(values)),
        values,
        s=10,
        color=color,
        alpha=0.25,
        edgecolors="none",
        zorder=3,
    )
    ax.scatter(
        position + offset,
        values.mean(),
        marker="D",
        s=25,
        facecolors=color if filled else "white",
        edgecolors="white" if filled else color,
        linewidths=0.5 if filled else 1.0,
        zorder=4,
    )
