"""Generate reproducible current-evidence Section 1 tables and thesis plots.

The input is the thesis-facing study detail directory configured in
``configs/plots/section1_pattern_analysis.toml``. Missing runs remain missing,
while every cell with at least one usable run contributes its measured value.

Example:
    ``python -m plots.overview.pattern_analysis``
    ``python -m plots.overview.pattern_analysis --output-root /tmp/section1-analysis``
"""

from __future__ import annotations

import argparse
import json
import shutil
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from plots.common.model_display import (
    MODEL_LABELS,
    MODEL_ORDER,
)
from plots.common.plot_style import apply_section1_plot_style
from plots.overview.checkpoint_selection import plot_checkpoint_selection
from plots.overview.normalized_distributions import plot_normalized_distributions
from plots.overview.pattern_data import (
    _analysis_subset,
    _canonical_frame,
    _dataset_metadata,
    _metric_frame,
    _resource_frame,
    _write_tables,
)
from plots.overview.pattern_diagnostic_plots import (
    _plot_checkpoint_collapses_compact,
    _plot_checkpoint_gap,
    _plot_combined_association_heatmap,
)
from plots.overview.pattern_metric_plots import (
    _plot_effect_variance,
    _plot_metric_gain_agreement,
    _plot_metric_gains,
    _plot_preprocessing_difference,
)
from plots.overview.pattern_model_comparison import (
    _critical_difference_table,
    _plot_overall_critical_differences,
    _plot_sft_dataset_wins,
)
from plots.overview.pattern_statistics import (
    association_table,
    fine_tuning_outcome_table,
    structural_hypothesis_association_table,
)
from plots.overview.result_table import write_dataset_result_table

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs/plots/section1_pattern_analysis.toml"


def generate_section1_pattern_analysis(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path | None = None,
    thesis_assets: Path | None = None,
    meta_feature_cache: Path | None = None,
) -> dict[str, Any]:
    """Generate the final 55-dataset overview and its supporting statistics."""
    apply_section1_plot_style()
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    results_root = REPO_ROOT / config["input"]["results_root"]
    output_dir = output_root or REPO_ROOT / config["output"]["root"]
    thesis_dir = thesis_assets or REPO_ROOT / config["output"]["thesis_assets"]
    output_dir.mkdir(parents=True, exist_ok=True)
    thesis_dir.mkdir(parents=True, exist_ok=True)
    metadata = _dataset_metadata(cache_path=meta_feature_cache or REPO_ROOT / config["input"]["meta_feature_cache"])
    size = int(config["analysis"]["default_context_size"])
    frame = _canonical_frame(
        results_root, metadata=metadata, expected_context_size=size, studies=config["input"]["studies"]
    )
    metrics = _metric_frame(
        results_root, metadata=metadata, expected_context_size=size, studies=config["input"]["studies"]
    )
    resources = _resource_frame(results_root, config["input"]["studies"])
    if frame.empty:
        raise RuntimeError(f"No usable result rows in {results_root}.")
    excluded = config["analysis"].get("excluded_datasets", ["airlines", "miniboone"])
    frame = frame.loc[~frame["dataset"].isin(excluded)]
    metrics = metrics.loc[~metrics["dataset"].isin(excluded)]
    protocol = str(config["analysis"]["default_preprocessing"])
    selected = _analysis_subset(frame, preprocessing=protocol)
    metrics = _analysis_subset(metrics, preprocessing=protocol)
    resources = _analysis_subset(resources, preprocessing=protocol)
    features = [name for name in config["analysis"]["features"] if name in frame]
    hypotheses = [item for item in config["analysis"].get("structural_hypotheses", []) if item["feature"] in frame]
    seed, resamples = int(config["analysis"]["bootstrap_seed"]), int(config["analysis"]["bootstrap_resamples"])
    associations = association_table(
        selected, feature_columns=features, bootstrap_seed=seed, bootstrap_resamples=resamples
    )
    final = selected.assign(effect=selected["final_effect"])
    final_associations = association_table(
        final, feature_columns=features, bootstrap_seed=seed, bootstrap_resamples=resamples
    )
    structural = structural_hypothesis_association_table(
        selected, hypotheses=hypotheses, bootstrap_seed=seed + 2, bootstrap_resamples=resamples
    )
    final_structural = structural_hypothesis_association_table(
        final, hypotheses=hypotheses, bootstrap_seed=seed + 2, bootstrap_resamples=resamples
    )
    outcomes = fine_tuning_outcome_table(selected)
    ranks = _critical_difference_table(selected, overall=True)
    _write_tables(
        output_dir,
        canonical=frame,
        metrics=metrics,
        resources=resources,
        outcomes=outcomes,
        critical_differences=ranks,
        associations=associations,
        epoch100_associations=final_associations,
        structural_associations=structural,
        epoch100_structural_associations=final_structural,
    )
    table = write_dataset_result_table(
        metrics, output_dir=output_dir, model_order=MODEL_ORDER, model_labels=MODEL_LABELS
    )
    figures = [
        plot_normalized_distributions(selected, output_dir),
        _plot_overall_critical_differences(ranks, output_dir),
        _plot_sft_dataset_wins(selected, output_dir),
        _plot_effect_variance(selected, output_dir),
        _plot_preprocessing_difference(frame, output_dir),
        _plot_checkpoint_gap(selected, output_dir),
        _plot_checkpoint_collapses_compact(selected, output_dir),
        _plot_combined_association_heatmap(associations, structural, output_dir),
        _plot_combined_association_heatmap(
            final_associations,
            final_structural,
            output_dir,
            filename="association_heatmap_epoch100_combined.pdf",
            gain_label="epoch-100 SFT gain over ICL",
        ),
        *_plot_metric_gains(metrics, output_dir),
        _plot_metric_gain_agreement(metrics, output_dir),
        plot_checkpoint_selection(selected, output_dir),
    ]
    for artifact in [table, *figures]:
        if artifact is not None:
            shutil.copy2(artifact, thesis_dir / artifact.name)
    summary = {
        "datasetCount": int(selected["dataset"].nunique()),
        "excludedDatasets": excluded,
        "figures": [path.name for path in figures if path is not None],
        "resultTable": table.name,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, help="Override the configured analysis output directory.")
    parser.add_argument("--thesis-assets", type=Path, help="Override the configured thesis-asset directory.")
    parser.add_argument("--meta-feature-cache", type=Path, help="Override the configured metadata cache.")
    args = parser.parse_args(argv)
    summary = generate_section1_pattern_analysis(
        config_path=args.config,
        output_root=args.output_root,
        thesis_assets=args.thesis_assets,
        meta_feature_cache=args.meta_feature_cache,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
