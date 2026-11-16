"""The paper project gets its figures, tables and quoted numbers from one generator."""

from __future__ import annotations

import json

from tests.plots.preprint.synthetic_results import write_config, write_results
from tfm_finetuning.plots.common.model_display import MODEL_LABELS, PAPER_MODELS
from tfm_finetuning.plots.preprint.paper import generate_paper_assets


def test_paper_assets_are_written_for_every_model(tmp_path) -> None:
    results = tmp_path / "results"
    write_results(results)
    paper = tmp_path / "paper"

    numbers = generate_paper_assets(paper_root=paper, config_path=write_config(tmp_path, results))

    for name in ("score_distributions", "dataset_wins", "metric_agreement"):
        assert (paper / "figures" / f"{name}.pdf").stat().st_size > 0
    assert sum(row["datasets"] for row in numbers["best_overall"]) == 12
    late_stage = (paper / "tables/late_stage.tex").read_text()
    assert all(MODEL_LABELS[model] in late_stage for model in PAPER_MODELS)
    assert {row["model"] for row in numbers["late_stage"]} == set(PAPER_MODELS)
    models_table = (paper / "tables/models.tex").read_text()
    cost_table = (paper / "tables/cost.tex").read_text()
    for model in PAPER_MODELS:
        assert MODEL_LABELS[model] in models_table and MODEL_LABELS[model] in cost_table
    assert "Datasets where SFT beats ICL" in models_table
    for task, count in (("binary", 4), ("multiclass", 4), ("regression", 4)):
        dataset_table = (paper / f"tables/datasets_{task}.tex").read_text()
        rows = [line for line in dataset_table.splitlines() if line.startswith(task)]
        assert len(rows) == count and all(line.count(" / ") == len(PAPER_MODELS) for line in rows)
        assert r"\textbf" in dataset_table
    assert numbers["datasets"] == 12 and numbers["tasks"] == {"Binary": 4, "Multiclass": 4, "Regression": 4}
    assert json.loads((paper / "numbers.json").read_text())["datasets"] == 12
    assert {row["model"] for row in numbers["cost"]} == set(PAPER_MODELS)
