"""Compact spurious-correlation views from canonical run-level metrics.

Example: ``python -m tfm_finetuning.plots.transformations.spurious_figures --output-dir /tmp/spurious``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from tfm_finetuning.plots.common.model_display import MODEL_COLORS, MODEL_LABELS
from tfm_finetuning.plots.common.plot_style import (
    apply_plot_style,
    latex_fraction_width,
    save_figure,
)
from tfm_finetuning.plots.transformations.transformation_data import _normalized_scores, _run_gain_table

MODELS = ("tabpfnv25", "tabiclv2", "mitra")
DATASETS = ("amazon_employee_access", "jm1", "houses")
LABELS = ("Amazon Employee Access", "JM1", "Houses")
CONDITIONS = ("0", "0.5", "0.8", "0.95")


def _curve(ax, rows, *, color, style, marker):
    """Draw equally weighted means and standard errors for each condition."""
    summary = rows.groupby("condition")["value"].agg(["mean", "sem"]).reindex(CONDITIONS)
    ax.errorbar(
        range(4),
        summary["mean"],
        yerr=summary["sem"],
        color=color,
        linestyle=style,
        marker=marker,
        markersize=3,
        capsize=2,
        linewidth=1.2,
    )


def _finish(fig, axes, path, fraction):
    """Apply common axes and save with a fixed physical canvas."""
    for ax in axes:
        ax.set_xticks(range(4), CONDITIONS)
        ax.set_xlabel("Shortcut strength")
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, path, inclusion_fraction=fraction)
    plt.close(fig)
    return path


def plot_spurious_figures(run_metrics: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Render selected-score aggregate and paired selected/final gain panels.

    Aggregate scores weight datasets equally. Gain error bars use seed-level
    means over folds. Final means the last observed checkpoint, not epoch 100.
    """
    rows = run_metrics[
        (run_metrics.family == "spurious_correlation")
        & run_metrics.dataset.isin(DATASETS)
        & (run_metrics.metric == run_metrics.score_metric)
    ].copy()
    rows["condition"] = rows.condition.astype(str)
    selected = rows[(rows.checkpoint == "baseline") | (rows.checkpoint == "selected")]
    normalized = _normalized_scores(selected)
    dataset_scores = normalized.groupby(["dataset", "model", "method", "condition"], as_index=False).value.mean()
    apply_plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    models = [Line2D([], [], color=MODEL_COLORS[m], label=MODEL_LABELS[m]) for m in MODELS]
    fig, ax = plt.subplots(figsize=(latex_fraction_width(0.5), 2.65), layout="constrained")
    for model in MODELS:
        for method, style, marker in (("icl", "--", "o"), ("sft", "-", "^")):
            _curve(
                ax,
                dataset_scores[(dataset_scores.model == model) & (dataset_scores.method == method)],
                color=MODEL_COLORS[model],
                style=style,
                marker=marker,
            )
    ax.set_ylabel("Mean normalized score")
    fig.legend(
        handles=models
        + [Line2D([], [], color="0.3", linestyle=s, label=l) for s, l in (("--", "ICL"), ("-", "SFT (selected)"))],
        loc="outside upper center",
        ncol=2,
        fontsize=8,
        frameon=False,
    )
    paths = [_finish(fig, [ax], output_dir / "spurious_correlation_aggregate_scores.pdf", 0.5)]
    gains = []
    for checkpoint in ("selected", "final"):
        paired = rows[rows.checkpoint.isin(["baseline", checkpoint])].copy()
        paired.loc[paired.checkpoint == checkpoint, "checkpoint"] = "selected"
        gain = _run_gain_table(paired)
        gain["checkpoint"] = checkpoint
        gains.append(gain)
    gains = (
        pd.concat(gains).groupby(["dataset", "model", "condition", "checkpoint", "seed"], as_index=False).value.mean()
    )
    fig, axes = plt.subplots(1, 3, figsize=(latex_fraction_width(1.0), 2.85), layout="constrained")
    for ax, dataset, label in zip(axes, DATASETS, LABELS, strict=True):
        for model in MODELS:
            for checkpoint, style, marker in (("selected", "-", "o"), ("final", "--", "s")):
                _curve(
                    ax,
                    gains[(gains.dataset == dataset) & (gains.model == model) & (gains.checkpoint == checkpoint)],
                    color=MODEL_COLORS[model],
                    style=style,
                    marker=marker,
                )
        ax.set_title(label, fontsize=8)
        ax.axhline(0, color="0.4", linewidth=0.7)
        ax.set_ylabel("SFT − ICL ROC-AUC" if dataset != "houses" else "log(ICL / SFT) RMSE", fontsize=8)
    fig.legend(
        handles=models
        + [
            Line2D([], [], color="0.3", linestyle=s, label=l)
            for s, l in (("-", "Selected"), ("--", "Final at stopping"))
        ],
        loc="outside upper center",
        ncol=5,
        fontsize=8,
        frameon=False,
    )
    paths.append(_finish(fig, axes, output_dir / "spurious_correlation_dataset_checkpoint_gains.pdf", 1.0))
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/analysis/thesis_findings/transformation_studies/run_level_metrics.csv"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plot_spurious_figures(pd.read_csv(args.input, dtype={"condition": str}), args.output_dir)
