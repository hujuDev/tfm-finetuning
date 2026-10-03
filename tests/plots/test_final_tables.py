"""Check that final tables summarize their inputs rather than archived conclusions."""

import pandas as pd
import pytest

from plots.final_tables import MODELS, hpo_table, resource_table


def test_hpo_caption_counts_current_selections_and_marks_largest_context():
    trials = pd.DataFrame(
        [{"dataset": "churn", "model": model, "context_size": size} for model in MODELS for size in (5000, 10000)]
    )
    selected = pd.DataFrame(
        [
            {"dataset": "churn", "model": model, "context_size": size, "learning_rate": 1e-4, "weight_decay": 0.1}
            for model, size in zip(MODELS, (10000, 5000, 10000), strict=True)
        ]
    )
    table = hpo_table(selected, trials)
    assert "1/1 cases for TabPFN-2.5, 0/1 for TabICLv2 and 1/1 for Mitra" in table
    assert table.count(r"\textbf{10}") == 2


def test_resource_caption_uses_displayed_measured_totals():
    table = resource_table({"Study": {"mitra": 10.46}, "Cross-dataset transfer": {"mitra": 12}})
    assert "Study & -- & -- & -- & 10.5 & 10.5" in table
    assert "approximately 23 GPU-hours" in table
    with pytest.raises(ValueError, match="finite"):
        resource_table({"Study": {"mitra": float("nan")}})
