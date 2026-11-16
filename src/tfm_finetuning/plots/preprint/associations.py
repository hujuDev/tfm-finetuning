"""Association of fine-tuning gains with dataset characteristics (appendix heatmap of the preprint)."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from tfm_finetuning.plots.overview.pattern_data import _analysis_subset, _canonical_frame, _dataset_metadata
from tfm_finetuning.plots.overview.pattern_diagnostic_plots import _plot_combined_association_heatmap
from tfm_finetuning.plots.overview.pattern_statistics import association_table, structural_hypothesis_association_table


def plot_feature_associations(
    *, repo_root: Path, results_root: Path, analysis: dict[str, Any], studies: list[str], output_dir: Path
) -> Path:
    """Spearman correlation of the SFT gain with dataset characteristics for the models in ``association_models``.

    Features and predeclared structural hypotheses come from ``feature_config``. Only the plotted models enter the
    multiplicity correction, so the stars in the figure refer to exactly the tests it shows.
    """
    settings = analysis["associations"]
    feature_config = tomllib.loads((repo_root / settings["feature_config"]).read_text(encoding="utf-8"))["analysis"]
    models = list(settings["models"])
    metadata = _dataset_metadata(cache_path=repo_root / settings["meta_feature_cache"])
    size = int(analysis["default_context_size"])
    frame = _canonical_frame(results_root, metadata=metadata, expected_context_size=size, studies=studies)
    frame = frame.loc[~frame["dataset"].isin([name.lower() for name in analysis["excluded_datasets"]])]
    selected = _analysis_subset(frame, preprocessing=str(analysis["default_preprocessing"]))
    selected = selected.loc[selected["model"].isin(models)]
    features = [name for name in feature_config["features"] if name in selected]
    hypotheses = [item for item in feature_config["structural_hypotheses"] if item["feature"] in selected]
    seed, resamples = int(analysis["bootstrap_seed"]), int(analysis["bootstrap_resamples"])
    associations = association_table(
        selected, feature_columns=features, bootstrap_seed=seed, bootstrap_resamples=resamples
    )
    structural = structural_hypothesis_association_table(
        selected, hypotheses=hypotheses, bootstrap_seed=seed + 2, bootstrap_resamples=resamples
    )
    return _plot_combined_association_heatmap(
        associations,
        structural,
        output_dir,
        filename="feature_associations.pdf",
        model_order=models,
    )
