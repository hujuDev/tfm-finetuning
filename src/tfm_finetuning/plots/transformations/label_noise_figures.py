"""Final label-noise figures from paired, validation-selected run metrics.

Example: ``plot_label_noise_figures(run_metrics, Path('results/figures'))``.
"""

from __future__ import annotations

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
DATASETS = ("jm1", "bank_marketing", "amazon_employee_access", "splice", "sdss17")
CONDITIONS = ("0", "0.05", "0.15", "0.3")
TICK_LABELS = ("0", "0.05", "0.15", "0.30")


def _curve(ax, rows, *, model, style="-", marker="o"):
    summary = rows.groupby("condition").value.agg(["mean", "sem"]).reindex(CONDITIONS)
    ax.errorbar(
        range(len(CONDITIONS)),
        summary["mean"],
        yerr=summary["sem"],
        color=MODEL_COLORS[model],
        linestyle=style,
        marker=marker,
        markersize=3,
        capsize=2,
        linewidth=1.2,
    )


def _save(fig, axes, path, *, fraction):
    for ax in axes:
        ax.set_xticks(range(len(CONDITIONS)), TICK_LABELS)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, path, inclusion_fraction=fraction)
    plt.close(fig)
    return path


def plot_label_noise_figures(run_metrics: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Plot dataset-weighted normalized scores and seed-level selected gains.

    The pooled figure includes all five thesis datasets. The individual panels
    show Bank Marketing, SDSS17, and Splice, using the thesis's metric orientation.
    Standard errors span datasets in the aggregate and seeds in individual panels.
    """
    rows = run_metrics[
        (run_metrics.family == "label_noise")
        & run_metrics.dataset.isin(DATASETS)
        & (run_metrics.metric == run_metrics.score_metric)
        & run_metrics.checkpoint.isin(["baseline", "selected"])
    ].copy()
    if rows.empty:
        raise ValueError("No paired label-noise results found.")
    rows["condition"] = rows.condition.map(lambda value: f"{float(value):g}")
    normalized = _normalized_scores(rows)
    dataset_scores = normalized.groupby(["dataset", "model", "method", "condition"], as_index=False).value.mean()
    gains = _run_gain_table(rows).groupby(["dataset", "model", "condition", "seed"], as_index=False).value.mean()
    apply_plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    handles = [Line2D([], [], color=MODEL_COLORS[model], label=MODEL_LABELS[model]) for model in MODELS]
    figure, axis = plt.subplots(figsize=(latex_fraction_width(0.5), 2.65), layout="constrained")
    for model in MODELS:
        for method, style, marker in (("icl", "--", "o"), ("sft", "-", "^")):
            _curve(
                axis,
                dataset_scores[(dataset_scores.model == model) & (dataset_scores.method == method)],
                model=model,
                style=style,
                marker=marker,
            )
    axis.set_ylabel("Mean normalized score")
    axis.set_xlabel("Fraction of training labels flipped")
    figure.legend(
        handles=handles
        + [
            Line2D([], [], color="0.3", linestyle=style, label=label) for style, label in (("--", "ICL"), ("-", "SFT"))
        ],
        loc="outside upper center",
        ncol=2,
        fontsize=8,
        frameon=False,
    )
    paths = [_save(figure, [axis], output_dir / "label_noise_aggregate_scores.pdf", fraction=0.5)]
    figure, axes = plt.subplots(1, 3, figsize=(latex_fraction_width(1.0), 2.65), layout="constrained")
    for axis, dataset, label in zip(
        axes,
        ("bank_marketing", "sdss17", "splice"),
        ("Bank Marketing", "SDSS17", "Splice"),
        strict=True,
    ):
        for model in MODELS:
            _curve(axis, gains[(gains.dataset == dataset) & (gains.model == model)], model=model)
        axis.set_title(label, fontsize=8)
        axis.axhline(0, color="0.4", linewidth=0.7)
        axis.set_ylabel("SFT − ICL ROC-AUC" if dataset == "bank_marketing" else "log(ICL / SFT) log loss", fontsize=8)
    figure.supxlabel("Fraction of training labels flipped", fontsize=9)
    figure.legend(handles=handles, loc="outside upper center", ncol=3, fontsize=8, frameon=False)
    paths.append(_save(figure, axes, output_dir / "label_noise_selected_dataset_gains.pdf", fraction=1.0))
    return paths
