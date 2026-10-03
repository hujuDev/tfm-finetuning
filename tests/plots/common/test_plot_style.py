"""Regression tests for consistent overview publication typography."""

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

from tfm_finetuning.plots.common.plot_style import (
    FULL_WIDTH_INCHES,
    apply_plot_style,
    latex_fraction_width,
    save_figure,
    shared_y_tick_interval,
    style_y_axes,
)


def test_plot_style_standardizes_figure_titles() -> None:
    """Every overview generator should inherit the same figure-title treatment."""
    with mpl.rc_context():
        apply_plot_style()
        assert mpl.rcParams["figure.titlesize"] == 11.0
        assert mpl.rcParams["figure.titleweight"] == "bold"
        assert mpl.rcParams["xtick.labelsize"] == 9.0
        assert mpl.rcParams["legend.fontsize"] == 9.0


def test_latex_fraction_width_preserves_typographic_scale() -> None:
    """A deliberately narrow source canvas should match its LaTeX width fraction."""
    assert pytest.approx(6.06299) == FULL_WIDTH_INCHES
    assert latex_fraction_width(0.58) == pytest.approx(3.516534)


def test_shared_y_ticks_allow_panel_specific_label_intervals() -> None:
    """Panels share 0.05 guides without inheriting the widest label interval."""
    figure, axes = plt.subplots(2, 1)
    axes[0].set_ylim(-0.04, 0.20)
    axes[1].set_ylim(-0.12, 0.60)

    interval = style_y_axes(axes, tick_interval=0.05)

    assert interval == pytest.approx(0.05)
    assert axes[0].yaxis.get_major_locator()._edge.step == pytest.approx(0.05)
    assert axes[1].yaxis.get_major_locator()._edge.step == pytest.approx(0.1)
    assert axes[1].yaxis.get_minor_locator()._edge.step == pytest.approx(0.05)
    plt.close(figure)


def test_shared_y_interval_uses_one_scale_for_all_panels() -> None:
    figure, axes = plt.subplots(2, 1)
    axes[0].set_ylim(-0.1, 0.1)
    axes[1].set_ylim(-0.6, 0.6)

    assert shared_y_tick_interval(axes) == pytest.approx(0.1)
    plt.close(figure)


def test_shared_y_ticks_expand_narrow_panel_to_visible_scale_marks() -> None:
    """A narrow panel must not fall between shared ticks and show only zero."""
    figure, axes = plt.subplots(2, 1)
    axes[0].set_ylim(-0.009, 0.009)
    axes[1].set_ylim(-0.15, 0.04)

    style_y_axes(axes, tick_interval=0.02)

    assert axes[0].get_ylim() == pytest.approx((-0.02, 0.02))
    assert axes[0].yaxis.get_major_locator()._edge.step == pytest.approx(0.02)
    plt.close(figure)


def test_pdf_keeps_its_declared_canvas(tmp_path: Path) -> None:
    """Saving must not let labels silently enlarge the publication canvas."""
    figure = plt.figure(figsize=(4.0, 2.0))
    figure.suptitle("A deliberately long title that extends beyond the canvas")
    path = tmp_path / "fixed_canvas.pdf"
    save_figure(figure, path, inclusion_fraction=4.0 / FULL_WIDTH_INCHES)
    plt.close(figure)

    assert re.search(rb"/MediaBox\s*\[\s*0\s+0\s+288(?:\.0*)?\s+144(?:\.0*)?\s*\]", path.read_bytes())


def test_save_rejects_post_scaled_canvas(tmp_path: Path) -> None:
    """A source width different from its declared inclusion width must fail."""
    figure = plt.figure(figsize=(4.0, 2.0))

    with pytest.raises(AssertionError, match="does not match its LaTeX inclusion width"):
        save_figure(figure, tmp_path / "wrong_width.pdf", inclusion_fraction=0.5)
    plt.close(figure)


def test_save_rejects_sub_eight_point_visible_text(tmp_path: Path) -> None:
    """The central save path enforces the final-print hard floor."""
    figure = plt.figure(figsize=(FULL_WIDTH_INCHES, 2.0))
    figure.text(0.5, 0.5, "Too small", fontsize=7.9)

    with pytest.raises(AssertionError, match=r"below the 8\.00 pt"):
        save_figure(figure, tmp_path / "too_small.pdf")
    plt.close(figure)


def test_save_allows_explicit_figure_specific_text_floor(tmp_path: Path) -> None:
    """Dense figures may opt into a narrowly lower audited text floor."""
    figure = plt.figure(figsize=(FULL_WIDTH_INCHES, 2.0))
    figure.text(0.5, 0.5, "Dense cell value", fontsize=7.5)

    save_figure(
        figure,
        tmp_path / "explicit-text-floor.pdf",
        minimum_text_size=7.5,
    )
    plt.close(figure)


def test_save_accounts_for_latex_height_cap(tmp_path: Path) -> None:
    """A tall width-matched figure must be audited after height-cap scaling."""
    figure = plt.figure(figsize=(FULL_WIDTH_INCHES, 10.0))
    figure.text(0.5, 0.5, "Shrunk by the height cap", fontsize=9.0)

    with pytest.raises(AssertionError, match=r"effective=7\."):
        save_figure(
            figure,
            tmp_path / "too_tall.pdf",
            max_inclusion_height_fraction=0.82,
        )
    plt.close(figure)
