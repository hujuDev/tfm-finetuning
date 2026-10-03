from __future__ import annotations

import math

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import tfm_finetuning.plots.overview.hpo_analysis as hpo_analysis
from tfm_finetuning.plots.common.plot_style import FULL_WIDTH_INCHES
from tfm_finetuning.plots.overview.hpo_analysis import _select_trials


def test_hpo_comparison_uses_matched_limits_and_equal_aspect(tmp_path, monkeypatch) -> None:
    """A narrow binary panel must not inherit the wider multiclass scale."""
    frame = pd.DataFrame(
        [
            {
                "dataset": "bin",
                "model": "mitra",
                "task_family": "binary",
                "default_gain": 0.008,
                "test_gain": 0.018,
            },
            {
                "dataset": "multi",
                "model": "mitra",
                "task_family": "multiclass",
                "default_gain": -1.0,
                "test_gain": 0.4,
            },
            {
                "dataset": "reg",
                "model": "mitra",
                "task_family": "regression",
                "default_gain": -0.1,
                "test_gain": 0.2,
            },
        ]
    )
    captured: dict[str, plt.Figure] = {}

    def capture_figure(fig, path, *, tight=True, **_kwargs):
        captured["figure"] = fig
        return path

    monkeypatch.setattr(hpo_analysis, "_save", capture_figure)

    hpo_analysis._plot_test_comparison(frame, tmp_path, models=["mitra"])

    axes = captured["figure"].axes
    assert captured["figure"].get_figwidth() == pytest.approx(FULL_WIDTH_INCHES)
    assert captured["figure"].get_figheight() <= 3.4
    assert len({round(axis.get_position().y0, 6) for axis in axes}) == 1
    for axis in axes:
        assert axis.get_xlim() == pytest.approx(axis.get_ylim())
        assert axis.get_aspect() == pytest.approx(1.0)
        assert not axis.texts
    binary_span = axes[0].get_xlim()[1] - axes[0].get_xlim()[0]
    multiclass_span = axes[1].get_xlim()[1] - axes[1].get_xlim()[0]
    assert binary_span < multiclass_span
    plt.close(captured["figure"])






def test_hpo_selection_uses_validation_then_joins_matching_test_cell() -> None:
    """Held-out test gain must not influence a lower-is-better validation choice."""
    rows = []
    for learning_rate, validation_metric, test_gain in [
        (1e-5, 0.8, math.log(2.0)),
        (1e-4, 0.6, -math.log(2.0)),
    ]:
        common = {
            "dataset": "demo",
            "model": "mitra",
            "task_family": "regression",
            "score_metric": "rmse",
            "learning_rate": learning_rate,
            "weight_decay": 0.01,
            "context_size": 5000.0,
        }
        rows.append({**common, "split": "val", "selected": validation_metric, "gain": 0.1})
        rows.append({**common, "split": "test", "selected": 1.0, "baseline": 1.0, "gain": test_gain})

    selected = _select_trials(pd.DataFrame(rows))

    assert len(selected) == 1
    assert selected.iloc[0]["learning_rate"] == pytest.approx(1e-4)
    assert selected.iloc[0]["test_gain"] == pytest.approx(-math.log(2.0))
