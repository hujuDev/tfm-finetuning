"""The appendix heatmap relates fine-tuning gains to dataset characteristics for a chosen set of models."""

from __future__ import annotations

import json

from tests.plots.preprint.synthetic_results import STUDIES, write_results
from tfm_finetuning.plots.preprint.associations import plot_feature_associations


def test_heatmap_is_written_for_the_selected_models(tmp_path) -> None:
    results = tmp_path / "results"
    write_results(results, datasets_per_task=6)
    datasets = {
        f"{task}_{index}": {"baseTrainRows": 500 + 100 * index, "nFeaturesTotal": 5 + 3 * (index % 4)}
        for task in ("binary", "multiclass", "regression")
        for index in range(6)
    }
    (tmp_path / "meta.json").write_text(json.dumps({"datasets": datasets}))
    (tmp_path / "features.toml").write_text(
        '[analysis]\nfeatures = ["baseTrainRows", "nFeaturesTotal"]\nstructural_hypotheses = []\n'
    )
    analysis = {
        "bootstrap_seed": 1,
        "bootstrap_resamples": 50,
        "default_preprocessing": "shared",
        "default_context_size": 10000,
        "excluded_datasets": [],
        "associations": {
            "models": ["tabpfnv2", "tabpfnv35"],
            "meta_feature_cache": "meta.json",
            "feature_config": "features.toml",
        },
    }

    path = plot_feature_associations(
        repo_root=tmp_path, results_root=results, analysis=analysis, studies=list(STUDIES), output_dir=tmp_path
    )

    assert path == tmp_path / "feature_associations.pdf" and path.stat().st_size > 0
