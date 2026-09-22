"""Tests for final-print Matplotlib text-size validation."""

import matplotlib.pyplot as plt
import pytest

from plots.figure_text_audit import assert_minimum_text_size, inventory_visible_text


def test_inventory_covers_ticks_labels_legends_annotations_and_figure_text() -> None:
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1], label="Model series")
    axis.set_title("Panel title", fontsize=11)
    axis.set_xlabel("Train-data fraction", fontsize=10)
    axis.set_ylabel("Normalized score", fontsize=10)
    axis.set_xticks([0, 1], ["0.1", "1.0"], fontsize=9)
    axis.set_yticks([0, 1], ["zero", "one"], fontsize=9)
    axis.annotate("n=12", (0.5, 0.5), fontsize=8)
    axis.legend(title="Model", fontsize=9, title_fontsize=10)
    figure.text(0.02, 0.98, "A", fontsize=11)

    elements = inventory_visible_text(figure, exact_width=True)
    by_text = {element.text: element for element in elements}

    assert by_text["Panel title"].location == "axes[0].title"
    assert by_text["Train-data fraction"].location == "axes[0].xaxis.label"
    assert by_text["0.1"].location.startswith("axes[0].xaxis.major")
    assert by_text["n=12"].location.startswith("axes[0].annotation")
    assert by_text["Model"].location == "legend[0].title"
    assert by_text["Model series"].location == "legend[0].text[0]"
    assert by_text["A"].location == "figure.text[0]"
    assert by_text["n=12"].source_font_pt == pytest.approx(8.0)
    assert by_text["n=12"].effective_font_pt == pytest.approx(8.0)
    plt.close(figure)


def test_inventory_excludes_empty_invisible_and_transparently_hidden_text() -> None:
    figure, axes = plt.subplots(1, 2)
    axes[0].text(0.5, 0.5, "visible", fontsize=8)
    axes[0].text(0.5, 0.4, "")
    axes[0].text(0.5, 0.3, "hidden").set_visible(False)
    axes[0].text(0.5, 0.2, "transparent", alpha=0)
    axes[1].text(0.5, 0.5, "hidden with parent")
    axes[1].set_visible(False)

    texts = {element.text for element in inventory_visible_text(figure, exact_width=True)}

    assert "visible" in texts
    assert "" not in texts
    assert "hidden" not in texts
    assert "transparent" not in texts
    assert "hidden with parent" not in texts
    plt.close(figure)


def test_scaled_text_below_final_print_minimum_fails_with_details() -> None:
    figure = plt.figure()
    figure.text(0.5, 0.5, "Scaled annotation", fontsize=9)

    with pytest.raises(AssertionError, match=r"Scaled annotation.*source=9\.00 pt.*effective=7\.20 pt"):
        assert_minimum_text_size(figure, inclusion_scale=0.8)
    plt.close(figure)


def test_scaled_text_at_final_print_minimum_passes() -> None:
    figure = plt.figure()
    figure.text(0.5, 0.5, "Boundary annotation", fontsize=10)

    elements = assert_minimum_text_size(figure, inclusion_scale=0.8)

    assert len(elements) == 1
    assert elements[0].effective_font_pt == pytest.approx(8.0)
    plt.close(figure)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "Specify exactly one"),
        ({"exact_width": True, "inclusion_scale": 1.0}, "Specify exactly one"),
        ({"inclusion_scale": 0.0}, "finite positive"),
    ],
)
def test_inventory_requires_an_explicit_valid_width_contract(kwargs: dict[str, object], message: str) -> None:
    figure = plt.figure()
    with pytest.raises(ValueError, match=message):
        inventory_visible_text(figure, **kwargs)
    plt.close(figure)
