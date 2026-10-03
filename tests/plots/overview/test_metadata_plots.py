"""Regression tests for Section 1 metadata-plot annotations."""


import pytest

from tfm_finetuning.plots.common.plot_style import plain_log_power_label
from tfm_finetuning.plots.overview.metadata_plots import AXIS_FEATURE_LABELS


def test_long_metadata_axis_labels_wrap_without_shrinking_text() -> None:
    """Dense three-panel metadata figures should reflow long labels."""
    assert AXIS_FEATURE_LABELS["numHighCardinalityCategoricalFeatures"] == (
        "High-cardinality\ncategorical features"
    )
    assert AXIS_FEATURE_LABELS["maxCategoricalCardinality"] == "Maximum categorical\ncardinality"
    assert AXIS_FEATURE_LABELS["trainingTableCells"] == "Training table cells"
    assert AXIS_FEATURE_LABELS["univariateSignalTop5Share"] == (
        "Top-five univariate\nsignal share"
    )
    assert AXIS_FEATURE_LABELS["numericEffectiveRankRatio"] == (
        "Normalized effective\nrank"
    )
    assert AXIS_FEATURE_LABELS["marginalShapeHeterogeneity"] == (
        "Variation in feature\ndistributions"
    )
    assert AXIS_FEATURE_LABELS["sparseRegionOutlierMass"] == "Isolated-row\ntail"
    assert AXIS_FEATURE_LABELS["categoricalRareValueShare"] == "Rare category\nvalues"
    assert AXIS_FEATURE_LABELS["categoricalCooccurrenceSparsity"] == (
        "Rare category\ncombinations"
    )
    assert AXIS_FEATURE_LABELS["localTargetVariationByTask"] == (
        "Nearest-neighbour\ntarget variation"
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.001, "10⁻³"),
        (1.0, "10⁰"),
        (1000.0, "10³"),
        (2.0, ""),
    ],
)
def test_plain_log_power_labels_do_not_use_mathtext(value: float, expected: str) -> None:
    assert plain_log_power_label(value) == expected
