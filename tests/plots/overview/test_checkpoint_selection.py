"""Check that checkpoint plots preserve dataset-level evidence and epoch zero."""

import matplotlib as mpl

mpl.use("Agg")

import pandas as pd
import pytest

from tfm_finetuning.plots.overview.checkpoint_selection import checkpoint_selection_data, plot_checkpoint_selection


def test_checkpoint_evidence_filters_missing_and_wrong_training_length(tmp_path):
    """Missing epochs cannot become zero and shorter runs cannot enter the plot."""
    frame = pd.DataFrame(
        {
            "dataset": ["zero", "late", "missing", "short"],
            "model": ["tabpfnv2"] * 4,
            "task_family": ["binary"] * 4,
            "selected_epoch": [0, 100, None, 25],
            "training_epochs": [100, 100, 100, 50],
            "run_count": [1, 9, 3, 3],
        }
    )
    data = checkpoint_selection_data(frame)
    assert set(data.dataset) == {"zero", "late"}
    assert data.selected_epoch.mean() == 50
    assert plot_checkpoint_selection(frame, tmp_path).is_file()
    assert len(pd.read_csv(tmp_path / "checkpoint_selection_epochs.csv")) == 2


def test_duplicate_dataset_model_requires_preprocessing_selection():
    """Two preprocessing routes must not silently double a dataset's weight."""
    row = {
        "dataset": "example",
        "model": "mitra",
        "task_family": "regression",
        "selected_epoch": 50,
        "training_epochs": 100,
    }
    with pytest.raises(ValueError, match="one preprocessing route"):
        checkpoint_selection_data(pd.DataFrame([row, row]))
