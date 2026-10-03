from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from plots.overview import mitra_pipeline_analysis as mitra_analysis
from plots.overview.mitra_pipeline_analysis import (
    CHUNKED_PIPELINE,
    ORIGINAL_PIPELINE,
    _plot_pipeline_comparison,
)
from tests.support.pdf_text_audit import assert_pdf_minimum_text_size


@pytest.mark.skipif(shutil.which("mutool") is None, reason="PDF glyph audit requires mutool")
def test_mitra_runtime_log_ticks_export_at_full_font_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    real_save = mitra_analysis._save

    def capture_figure(figure, path):
        captured["figure"] = figure
        return real_save(figure, path)

    monkeypatch.setattr(mitra_analysis, "_save", capture_figure)
    performance_rows = []
    runtime_rows = []
    for task_index, task in enumerate(("binary", "multiclass", "regression")):
        for dataset_index in range(3):
            dataset = f"{task}-{dataset_index}"
            for pipeline_index, pipeline in enumerate((ORIGINAL_PIPELINE, CHUNKED_PIPELINE)):
                performance_rows.append(
                    {
                        "dataset": dataset,
                        "task_family": task,
                        "pipeline": pipeline,
                        "gain": 0.01 * (dataset_index + 1) + 0.002 * pipeline_index,
                    }
                )
                seconds = 100.0 * 10**task_index * (dataset_index + 1) * (pipeline_index + 1)
                runtime_rows.append(
                    {
                        "dataset": dataset,
                        "task_family": task,
                        "pipeline": pipeline,
                        "mean_seconds": seconds,
                        "sum_seconds": seconds * 3,
                    }
                )

    output = _plot_pipeline_comparison(
        pd.DataFrame(performance_rows),
        pd.DataFrame(runtime_rows),
        tmp_path,
    )

    assert output.is_file()
    assert_pdf_minimum_text_size(output)
    figure = captured["figure"]
    figure.canvas.draw()
    title_bounds = figure.axes[3].title.get_window_extent(figure.canvas.get_renderer())
    assert title_bounds.x0 >= figure.bbox.x0
    assert title_bounds.x1 <= figure.bbox.x1
