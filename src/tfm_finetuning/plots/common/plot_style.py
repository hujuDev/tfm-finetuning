"""Shared publication typography for Section 1 thesis figures.

Example:
    ``apply_section1_plot_style()`` before constructing any Matplotlib figures.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator

from tfm_finetuning.plots.figure_text_audit import assert_minimum_text_size

# The thesis text block is 438.17227 TeX pt wide.  TeX uses 72.27 pt/in.
FULL_WIDTH_INCHES = 438.17227 / 72.27
# The thesis text block is 241 mm high.  Result figures are normally capped at
# 82% of that height by LaTeX, so callers can include the cap in the final-size
# audit when a tall source canvas may be post-scaled.
TEXT_HEIGHT_INCHES = 241.0 / 25.4
FIGURE_TITLE_SIZE = 11.0
MINIMUM_FINAL_TEXT_SIZE = 8.0
Y_TICK_INTERVALS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
_SUPERSCRIPT_TRANSLATION = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def apply_section1_plot_style() -> None:
    """Apply stable typography shared by every Section 1 plot generator."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.labelsize": 9.0,
            "axes.titlesize": 10.0,
            "axes.titleweight": "bold",
            "figure.titlesize": FIGURE_TITLE_SIZE,
            "figure.titleweight": "bold",
            "legend.fontsize": 9.0,
            "legend.title_fontsize": 9.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
        }
    )


def latex_fraction_width(fraction: float) -> float:
    """Return a source-canvas width matching a fraction of thesis text width."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("LaTeX figure width fraction must be in (0, 1].")
    return FULL_WIDTH_INCHES * fraction


def plain_log_power_label(value: float, _position: float | None = None) -> str:
    """Format powers of ten without MathText's undersized exponent glyphs."""
    if not np.isfinite(value) or value <= 0:
        return ""
    exponent = int(round(float(np.log10(value))))
    if not np.isclose(value, 10.0**exponent, rtol=1e-10, atol=0.0):
        return ""
    return f"10{str(exponent).translate(_SUPERSCRIPT_TRANSLATION)}"


def save_section1_figure(
    figure: Figure,
    path: Path,
    *,
    inclusion_fraction: float = 1.0,
    max_inclusion_height_fraction: float | None = None,
    minimum_text_size: float = MINIMUM_FINAL_TEXT_SIZE,
) -> None:
    r"""Validate and save a canvas generated at its final physical width.

    ``inclusion_fraction`` is the matching LaTeX ``\\textwidth`` fraction.  A
    mismatch is an error because post-scaling would also scale every text
    element and invalidate the final-print typography audit.
    """
    expected_width = latex_fraction_width(inclusion_fraction)
    actual_width = figure.get_figwidth()
    if not math.isclose(actual_width, expected_width, rel_tol=0.0, abs_tol=1e-3):
        raise AssertionError(
            "Section 1 figure width does not match its LaTeX inclusion width: "
            f"source={actual_width:.5f} in, expected={expected_width:.5f} in "
            f"({inclusion_fraction:.0%} of text width)."
        )
    inclusion_scale = 1.0
    if max_inclusion_height_fraction is not None:
        if not 0.0 < max_inclusion_height_fraction <= 1.0:
            raise ValueError("LaTeX figure-height fraction must be in (0, 1].")
        maximum_height = TEXT_HEIGHT_INCHES * max_inclusion_height_fraction
        inclusion_scale = min(1.0, maximum_height / figure.get_figheight())
    assert_minimum_text_size(
        figure,
        minimum_pt=minimum_text_size,
        exact_width=inclusion_scale == 1.0,
        inclusion_scale=None if inclusion_scale == 1.0 else inclusion_scale,
    )
    figure.savefig(path)


def save_and_close_section1_figure(
    figure: Figure,
    path: Path,
    *,
    tight: bool = True,
    inclusion_fraction: float = 1.0,
    max_inclusion_height_fraction: float | None = None,
    minimum_text_size: float = MINIMUM_FINAL_TEXT_SIZE,
) -> Path:
    """Finalize, save, and release one Section 1 publication figure."""
    if tight:
        figure.tight_layout()
    save_section1_figure(
        figure,
        path,
        inclusion_fraction=inclusion_fraction,
        max_inclusion_height_fraction=max_inclusion_height_fraction,
        minimum_text_size=minimum_text_size,
    )
    plt.close(figure)
    return path


def shared_y_tick_interval(axes: Iterable[Axes], *, max_guides: int = 18) -> float:
    """Choose the smallest common publication interval that avoids grid overload."""
    spans = [abs(axis.get_ylim()[1] - axis.get_ylim()[0]) for axis in axes if axis.get_visible()]
    widest_span = max(spans, default=0.0)
    for interval in Y_TICK_INTERVALS:
        if widest_span / interval <= max_guides:
            return interval
    return Y_TICK_INTERVALS[-1]


def style_section1_y_axes(
    axes: Iterable[Axes],
    *,
    tick_interval: float | None = None,
) -> float:
    """Apply shared faint ticks and sparse labelled reference lines.

    Every base interval remains visible as a minor tick/grid guide. Labels and
    stronger grid lines use a coarser multiple when the plotted span is wide.
    """
    visible_axes = [axis for axis in axes if axis.get_visible()]
    interval = tick_interval or shared_y_tick_interval(visible_axes)
    decimals = max(0, int(np.ceil(-np.log10(interval)))) if interval < 1 else 0

    def format_tick(value: float, _position: float) -> str:
        normalized = 0.0 if np.isclose(value, 0.0) else value
        return f"{normalized:.{decimals}f}"

    for axis in visible_axes:
        lower, upper = axis.get_ylim()
        snapped_lower = np.floor(lower / interval) * interval
        snapped_upper = np.ceil(upper / interval) * interval
        if snapped_lower < snapped_upper:
            axis.set_ylim(snapped_lower, snapped_upper)
        axis_span = abs(snapped_upper - snapped_lower)
        label_interval = interval
        for multiplier in (1, 2, 4, 5, 10):
            candidate = interval * multiplier
            if axis_span / candidate <= 8:
                label_interval = candidate
                break
        axis.yaxis.set_major_locator(MultipleLocator(label_interval))
        axis.yaxis.set_major_formatter(FuncFormatter(format_tick))
        if not np.isclose(label_interval, interval):
            axis.yaxis.set_minor_locator(MultipleLocator(interval))
            axis.grid(axis="y", which="minor", color="#f0f2f5", linewidth=0.3)
            axis.tick_params(axis="y", which="minor", length=2)
        axis.grid(axis="y", which="major", color="#d7dce3", linewidth=0.65)
        axis.tick_params(axis="y", which="major", length=3)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    return interval


def add_section1_model_legend(
    figure: Figure,
    *,
    model_order: Sequence[str],
    model_labels: Mapping[str, str],
    model_colors: Mapping[str, str],
    anchor: tuple[float, float],
) -> None:
    """Render the titled, frameless model legend used by Section 3 figures."""
    handles = [
        Line2D(
            [], [], color=model_colors[model], marker="o", linestyle="none", markersize=5, label=model_labels[model]
        )
        for model in model_order
    ]
    figure.legend(
        handles=handles,
        ncols=len(handles),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=anchor,
        fontsize=9.0,
        title="Model",
        title_fontproperties={"size": 9.0, "weight": "semibold"},
        handlelength=0.9,
        handletextpad=0.35,
        columnspacing=1.2,
    )


__all__ = [
    "FIGURE_TITLE_SIZE",
    "FULL_WIDTH_INCHES",
    "MINIMUM_FINAL_TEXT_SIZE",
    "Y_TICK_INTERVALS",
    "add_section1_model_legend",
    "apply_section1_plot_style",
    "latex_fraction_width",
    "plain_log_power_label",
    "save_and_close_section1_figure",
    "save_section1_figure",
    "shared_y_tick_interval",
    "style_section1_y_axes",
]
