"""Inventory and validate visible Matplotlib text at its final print scale.

Example:
    ``assert_minimum_text_size(figure, minimum_pt=8, exact_width=True)``
    validates a figure generated at its final LaTeX inclusion width.  Pass an
    explicit ``inclusion_scale`` instead when LaTeX will resize the source PDF.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.text import Annotation, Text


@dataclass(frozen=True, slots=True)
class VisibleTextElement:
    """One rendered text artist and its effective final-print font size."""

    location: str
    text: str
    source_font_pt: float
    inclusion_scale: float
    effective_font_pt: float


@dataclass(frozen=True, slots=True)
class _TextLocation:
    name: str
    parent_visible: bool


def _resolve_inclusion_scale(*, inclusion_scale: float | None, exact_width: bool) -> float:
    if exact_width == (inclusion_scale is not None):
        raise ValueError("Specify exactly one of inclusion_scale or exact_width=True.")
    scale = 1.0 if exact_width else inclusion_scale
    if isinstance(scale, bool) or scale is None or not math.isfinite(scale) or scale <= 0:
        raise ValueError("inclusion_scale must be a finite positive number.")
    return float(scale)


def _text_locations(figure: Figure) -> dict[int, _TextLocation]:
    locations: dict[int, _TextLocation] = {}
    figure_visible = figure.get_visible()

    for index, artist in enumerate(figure.texts):
        locations[id(artist)] = _TextLocation(f"figure.text[{index}]", figure_visible)

    for axes_index, axis in enumerate(figure.axes):
        axis_visible = figure_visible and axis.get_visible()
        for index, artist in enumerate(axis.texts):
            kind = "annotation" if isinstance(artist, Annotation) else "text"
            locations[id(artist)] = _TextLocation(f"axes[{axes_index}].{kind}[{index}]", axis_visible)

        for position, artist in (
            ("title", axis.title),
            ("title.left", axis._left_title),
            ("title.right", axis._right_title),
        ):
            locations[id(artist)] = _TextLocation(f"axes[{axes_index}].{position}", axis_visible)

        for axis_name, matplotlib_axis in (("xaxis", axis.xaxis), ("yaxis", axis.yaxis)):
            component_visible = axis_visible and matplotlib_axis.get_visible()
            locations[id(matplotlib_axis.label)] = _TextLocation(
                f"axes[{axes_index}].{axis_name}.label",
                component_visible,
            )
            locations[id(matplotlib_axis.get_offset_text())] = _TextLocation(
                f"axes[{axes_index}].{axis_name}.offset",
                component_visible,
            )
            for level, ticks in (
                ("major", matplotlib_axis.get_major_ticks()),
                ("minor", matplotlib_axis.get_minor_ticks()),
            ):
                for tick_index, tick in enumerate(ticks):
                    for label_name, label in (("label1", tick.label1), ("label2", tick.label2)):
                        locations[id(label)] = _TextLocation(
                            f"axes[{axes_index}].{axis_name}.{level}[{tick_index}].{label_name}",
                            component_visible,
                        )

    legends = figure.findobj(match=Legend)
    for legend_index, legend in enumerate(legends):
        parent_visible = figure_visible and legend.get_visible()
        if legend.axes is not None:
            parent_visible = parent_visible and legend.axes.get_visible()
        locations[id(legend.get_title())] = _TextLocation(
            f"legend[{legend_index}].title",
            parent_visible,
        )
        for text_index, artist in enumerate(legend.get_texts()):
            locations[id(artist)] = _TextLocation(
                f"legend[{legend_index}].text[{text_index}]",
                parent_visible,
            )

    for name, artist in (
        ("figure.suptitle", figure._suptitle),
        ("figure.supxlabel", figure._supxlabel),
        ("figure.supylabel", figure._supylabel),
    ):
        if artist is not None:
            locations[id(artist)] = _TextLocation(name, figure_visible)
    return locations


def inventory_visible_text(
    figure: Figure,
    *,
    inclusion_scale: float | None = None,
    exact_width: bool = False,
) -> tuple[VisibleTextElement, ...]:
    """Return every non-empty, visible text artist with its final font size.

    ``exact_width=True`` declares that the source canvas is generated at its
    final physical inclusion width.  Otherwise, ``inclusion_scale`` must state
    the LaTeX source-to-final width ratio explicitly.
    """
    scale = _resolve_inclusion_scale(inclusion_scale=inclusion_scale, exact_width=exact_width)
    figure.canvas.draw()
    locations = _text_locations(figure)
    elements: list[VisibleTextElement] = []
    fallback_index = 0

    for artist in figure.findobj(match=Text):
        text = artist.get_text()
        location = locations.get(id(artist))
        parent_visible = True if location is None else location.parent_visible
        if not artist.get_visible() or not parent_visible or not text.strip():
            continue
        alpha = artist.get_alpha()
        if alpha is not None and alpha <= 0:
            continue
        if location is None:
            location = _TextLocation(f"text[{fallback_index}]", figure.get_visible())
            fallback_index += 1
        source_font_pt = float(artist.get_fontsize())
        elements.append(
            VisibleTextElement(
                location=location.name,
                text=text,
                source_font_pt=source_font_pt,
                inclusion_scale=scale,
                effective_font_pt=source_font_pt * scale,
            )
        )
    return tuple(elements)


def assert_minimum_text_size(
    figure: Figure,
    *,
    minimum_pt: float = 8.0,
    inclusion_scale: float | None = None,
    exact_width: bool = False,
) -> tuple[VisibleTextElement, ...]:
    """Raise ``AssertionError`` if visible text falls below ``minimum_pt``."""
    if not math.isfinite(minimum_pt) or minimum_pt <= 0:
        raise ValueError("minimum_pt must be a finite positive number.")
    elements = inventory_visible_text(
        figure,
        inclusion_scale=inclusion_scale,
        exact_width=exact_width,
    )
    offenders = [element for element in elements if element.effective_font_pt + 1e-9 < minimum_pt]
    if offenders:
        details = "\n".join(_format_offender(element) for element in offenders)
        raise AssertionError(f"Visible plot text is below the {minimum_pt:.2f} pt final-print minimum:\n{details}")
    return elements


def _format_offender(element: VisibleTextElement) -> str:
    text = element.text.replace("\n", "\\n")
    return (
        f"- {element.location} {text!r}: "
        f"source={element.source_font_pt:.2f} pt, "
        f"scale={element.inclusion_scale:.3f}, "
        f"effective={element.effective_font_pt:.2f} pt"
    )


__all__ = ["VisibleTextElement", "assert_minimum_text_size", "inventory_visible_text"]
