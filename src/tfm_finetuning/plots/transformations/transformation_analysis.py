"""Reproduce the controlled-transformation figures in the submitted thesis.

Example: ``python -m tfm_finetuning.plots.transformations.transformation_analysis``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from tfm_finetuning.plots.metrics import fine_tuning_gain
from tfm_finetuning.plots.transformations.class_rebalancing_data import summarize as summarize_class_rebalancing
from tfm_finetuning.plots.transformations.context_scale_figure import plot_context_interactions
from tfm_finetuning.plots.transformations.label_noise_figures import plot_label_noise_figures
from tfm_finetuning.plots.transformations.spurious_figures import plot_spurious_figures
from tfm_finetuning.plots.transformations.transformation_comparisons import (
    plot_class_tradeoff,
    plot_heavy_tail_pairs,
    plot_representation_heatmaps,
)
from tfm_finetuning.plots.transformations.transformation_data import TransformationTables, build_transformation_tables
from tfm_finetuning.plots.transformations.transformation_figures import (
    HORIZONTAL_AB_LATEX_WIDTH_FRACTION,
    plot_diamonds_mitigation,
    plot_heavy_tail_mitigation,
)
from tfm_finetuning.plots.transformations.transformation_overviews import (
    combine_representation_alignment_summaries,
    plot_subsampling_pooled_composite,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = REPO_ROOT / "configs/plots/transformations/transformation_analysis.toml"


def generate_section2_transformation_findings(
    *,
    config_path: Path = DEFAULT_CONFIG,
    repo_root: Path = REPO_ROOT,
    output_dir: Path | None = None,
    thesis_dir: Path | None = None,
) -> dict[str, Any]:
    """Write accepted scientific tables and only the final manuscript figures."""
    config = tomllib.loads(config_path.read_text())
    tables = build_transformation_tables(config, repo_root=repo_root)
    alignment = _representation_alignment_tables(config, repo_root=repo_root)
    output = output_dir or repo_root / config["output"]["root"]
    assets = thesis_dir or repo_root / config["output"]["thesis_assets"]
    output.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    _write_tables(tables, output)
    _write_alignment_tables(alignment, output)
    normalized = tables.normalized_score_summary
    primary = tables.dataset_summary.loc[lambda frame: frame.metric == frame.score_metric]
    figures = [
        plot_subsampling_pooled_composite(
            normalized[normalized.family == "subsample"],
            primary[primary.family == "subsample"],
            output_path=output / "subsample_pooled_scores_and_gains.pdf",
            condition_order=config["figures"]["subsample_order"],
            condition_labels=("1/16", "1/8", "1/4", "1/2", "Full"),
        ),
        *plot_label_noise_figures(tables.run_metrics, output),
        *plot_spurious_figures(tables.run_metrics, output),
        plot_representation_heatmaps(
            representation_summaries(normalized, alignment.normalized_score_summary),
            output / "representation_alignment_normalized_scores.pdf",
        ),
        plot_heavy_tail_pairs(
            normalized[(normalized.family == "heavy_tail") & (normalized.metric == normalized.score_metric)],
            output / "heavy_tail_pooled_scores_and_gains_horizontal.pdf",
        ),
        _diamonds_figure(tables, output_dir=output),
        plot_context_interactions(
            normalized[(normalized.family == "contextual_scale") & (normalized.metric == normalized.score_metric)],
            output / "contextual_scale_pooled_scores_and_gains.pdf",
        ),
        _heavy_tail_figure(
            tables,
            datasets=config["datasets"]["heavy_tail"],
            target_mode_order=config["figures"]["heavy_tail_target_modes"],
            output_dir=output,
        ),
    ]
    class_config = config["class_rebalancing"]
    class_datasets, class_summary = summarize_class_rebalancing(
        repo_root / config["input"]["results_root"],
        shared_study=class_config["shared_study"],
        context_study=class_config["context_study"],
    )
    class_datasets.to_csv(output / "class_rebalancing_datasets.csv", index=False)
    classes = output / "class_rebalancing_summary.csv"
    class_summary.to_csv(classes, index=False)
    figures.append(plot_class_tradeoff(class_summary, output / "class_prior_majority_downsampling_class_accuracy.pdf"))
    for figure in figures:
        shutil.copy2(figure, assets / figure.name)
    summary = {
        "studies": config["studies"],
        "alignmentStudies": config["alignment_studies"],
        "figures": [path.name for path in figures],
        "classRebalancingSummary": str(classes),
        "missingRunUnits": len(tables.missing_run_units),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def representation_summaries(update: pd.DataFrame, alignment: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Combine fine-tuning-only and inference-alignment experiments without changing pairings."""
    result = {}
    for family, title, original, transformed in [
        ("feature_order", "Feature order", "canonical", "permuted"),
        ("numeric_scale", "Numeric-feature standardization", "raw", "standardized"),
    ]:
        result[title] = combine_representation_alignment_summaries(
            update[update.family == family],
            alignment[alignment.family == family],
            value_column="normalized_score_mean",
            control_condition=f"{original}/{original}",
            fine_tuning_only_condition=f"{transformed}/{original}",
            alignment_condition_order=[
                f"{original}/{original}",
                f"{transformed}/{original}",
                f"{original}/{transformed}",
                f"{transformed}/{transformed}",
            ],
        )
    return result


def _representation_alignment_tables(config: Mapping[str, Any], *, repo_root: Path) -> TransformationTables:
    """Load the matching inference-alignment experiment configurations."""
    families = tuple(config.get("alignment_studies", {}))
    if not families:
        raise ValueError("The transformation config must define alignment_studies.")
    alignment_config = dict(config)
    alignment_config["studies"] = dict(config["alignment_studies"])
    alignment_config["datasets"] = {family: config["datasets"][family] for family in families}
    alignment_config["fit_conditions"] = {family: config["fit_conditions"][family] for family in families}
    alignment_config["controls"] = {family: config["controls"][family] for family in families}
    return build_transformation_tables(alignment_config, repo_root=repo_root)


def _write_alignment_tables(tables: TransformationTables, output_dir: Path) -> list[Path]:
    path = output_dir / "representation_alignment_normalized_scores.csv"
    tables.normalized_score_summary.to_csv(path, index=False)
    return [path]


def _write_tables(tables: TransformationTables, output_dir: Path) -> list[Path]:
    frames = {
        "run_level_metrics.csv": tables.run_metrics,
        "run_level_gains.csv": tables.run_gains,
        "normalized_run_level_scores.csv": tables.normalized_run_scores,
        "paired_control_contrasts.csv": tables.control_contrasts,
        "method_control_contrasts.csv": tables.method_contrasts,
        "dataset_condition_summary.csv": tables.dataset_summary,
        "method_condition_summary.csv": tables.method_summary,
        "method_score_summary.csv": tables.score_summary,
        "normalized_method_score_summary.csv": tables.normalized_score_summary,
        "diamonds_validation_epochs.csv": tables.epoch_metrics,
        "coverage.csv": tables.coverage,
        "missing_run_units.csv": tables.missing_run_units,
        "heavy_tail_stability_summary.csv": _heavy_tail_stability_summary(tables.run_metrics),
    }
    paths = []
    for name, frame in frames.items():
        path = output_dir / name
        frame.to_csv(path, index=False)
        paths.append(path)
    return paths


def _diamonds_figure(tables: TransformationTables, *, output_dir: Path) -> Path:
    trajectories = tables.epoch_metrics[
        (tables.epoch_metrics["family"] == "diamonds_mitigation")
        & (tables.epoch_metrics["metric"] == "rmse")
        & tables.epoch_metrics["eval_source"].isin(["zero_shot_initial", "in_memory_validation"])
    ][["condition", "seed", "epoch", "value"]].rename(columns={"condition": "target_mode", "value": "validation_rmse"})
    heldout = (
        tables.run_metrics[
            (tables.run_metrics["family"] == "diamonds_mitigation")
            & (tables.run_metrics["metric"] == "rmse")
            & (tables.run_metrics["checkpoint"].isin(["baseline", "selected", "final"]))
        ]
        .pivot_table(
            index=["condition", "seed"],
            columns="checkpoint",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename(
            columns={
                "condition": "target_mode",
                "baseline": "icl_rmse",
                "selected": "selected_rmse",
                "final": "epoch50_rmse",
            }
        )
    )
    return plot_diamonds_mitigation(
        trajectories,
        heldout,
        output_path=output_dir / "diamonds_tail_mitigation.pdf",
        target_mode_order=("none", "winsor_p01_p99", "robust_log1p"),
        target_mode_labels={
            "none": "Original target",
            "winsor_p01_p99": "Winsorized",
            "robust_log1p": "Log transformation",
        },
        inclusion_fraction=HORIZONTAL_AB_LATEX_WIDTH_FRACTION,
    )


def _heavy_tail_figure(
    tables: TransformationTables,
    *,
    datasets: Sequence[str],
    target_mode_order: Sequence[str],
    output_dir: Path,
) -> Path:
    stability = _heavy_tail_stability_summary(tables.run_metrics)
    return plot_heavy_tail_mitigation(
        stability,
        output_path=output_dir / "heavy_tail_contamination_mitigation.pdf",
        dataset_order=datasets,
        target_mode_order=target_mode_order,
        target_mode_labels={
            "none": "Untransformed",
            "winsor_p01_p99": "Winsorized",
            "robust_log1p": "Log transformation",
        },
        inclusion_fraction=0.94,
    )


def _heavy_tail_stability_summary(run_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize held-out epoch-50 drift relative to selected SFT."""
    rows = run_metrics[
        (run_metrics["family"] == "heavy_tail")
        & (run_metrics["metric"] == run_metrics["score_metric"])
        & (run_metrics["method"] == "sft")
        & run_metrics["checkpoint"].isin(["selected", "final"])
    ]
    identity = ["dataset", "model", "fold", "repeat", "sample", "seed", "condition", "metric"]
    paired = rows.pivot_table(index=identity, columns="checkpoint", values="value", aggfunc="first").reset_index()
    if not {"selected", "final"}.issubset(paired.columns):
        return pd.DataFrame(
            columns=[
                "dataset",
                "model",
                "condition",
                "contamination",
                "target_mode",
                "stability_mean",
                "stability_sem",
                "stability_n",
            ]
        )
    paired["stability"] = [
        fine_tuning_gain(selected, final, metric)
        for selected, final, metric in paired[["selected", "final", "metric"]].itertuples(index=False, name=None)
    ]
    paired = paired.dropna(subset=["stability"])
    summary = paired.groupby(["dataset", "model", "condition"], as_index=False)["stability"].agg(
        stability_mean="mean", stability_sem="sem", stability_n="count"
    )
    summary[["contamination", "target_mode"]] = summary["condition"].str.split("/", n=1, expand=True)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--thesis-dir", type=Path)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            generate_section2_transformation_findings(
                config_path=args.config,
                repo_root=args.repo_root,
                output_dir=args.output_dir,
                thesis_dir=args.thesis_dir,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
