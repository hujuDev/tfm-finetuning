"""Figures, tables and quoted numbers of the preprint.

Every comparison is within a model: ICL without fine-tuning against the same model after SFT.

Example: ``python -m tfm_finetuning.plots.preprint.paper`` writes into ``results/figure_assets/preprint``.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

from tfm_finetuning.plots.common.distribution_style import draw_distribution
from tfm_finetuning.plots.common.model_display import (
    MODEL_COLORS,
    MODEL_LABELS,
    PAPER_MODELS,
    PERMISSIVE_MODELS,
    TASK_LABELS,
    TASK_ORDER,
)
from tfm_finetuning.plots.common.plot_style import apply_plot_style
from tfm_finetuning.plots.figure_text_audit import assert_minimum_text_size
from tfm_finetuning.plots.preprint.associations import plot_feature_associations
from tfm_finetuning.plots.preprint.data import (
    best_overall,
    cost_frame,
    cross_model_comparison,
    generation_summary,
    late_stage_table,
    metric_gains,
    model_table,
    models_present,
    preprint_frame,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = REPO_ROOT / "configs/plots/preprint/preprint_analysis.toml"
DEFAULT_OUTPUT = REPO_ROOT / "results/figure_assets/preprint"
# ICLR text block: 397.48 TeX pt at 72.27 pt/in.
PREPRINT_WIDTH_INCHES = 397.48 / 72.27
FONT_SIZE = 8.0
# Color always identifies the model; ICL and SFT differ by hollow and filled markers only.
LEGEND_COLOR = "#4b5563"
PERMISSIVE_MARK = "*"
METRICS = {"binary": "ROC-AUC", "multiclass": "log-loss", "regression": "RMSE"}
# Binary scores sit close to 1; a floor keeps them readable, and datasets below it are marked at the edge.
AXIS_FLOORS = {"binary": 0.9}
# Minor ticks share one spacing across panels, so their density shows how far each panel is zoomed out.
MINOR_TICK_STEP = 0.05
LABELED_TICK_STEPS = {"binary": 0.05, "multiclass": 0.25, "regression": 0.1}


def _save(figure, path: Path) -> Path:
    assert_minimum_text_size(figure, minimum_pt=FONT_SIZE, exact_width=True)
    figure.savefig(path)
    plt.close(figure)
    return path


def _style_axis(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#d7dce3", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=FONT_SIZE, length=2.5)


def _display_label(model: str) -> str:
    """Model name, marked when its weights are released under a permissive license."""
    return MODEL_LABELS[model] + (PERMISSIVE_MARK if model in PERMISSIVE_MODELS else "")


def plot_score_distributions(panel: pd.DataFrame, output_dir: Path) -> Path:
    """Normalized scores over datasets per model, ICL (hollow) next to SFT (filled), task types stacked."""
    models = [m for m in PAPER_MODELS if m in set(panel["model"])]
    fig, axes = plt.subplots(3, 1, figsize=(PREPRINT_WIDTH_INCHES, 5.0), sharex=True)
    rng = np.random.default_rng(42)
    for ax, task in zip(axes, TASK_ORDER, strict=True):
        rows = panel.loc[panel["task_family"] == task]
        for index, model in enumerate(models):
            values = rows.loc[rows["model"] == model]
            for delta, column, filled in ((-0.5, "normalized_baseline", False), (0.5, "normalized_selected", True)):
                position, color = index * 2.6 + delta, MODEL_COLORS[model]
                draw_distribution(ax, values[column], position, color, rng=rng, filled=filled, width=0.42)
                below = int((values[column] < AXIS_FLOORS.get(task, -np.inf)).sum())
                if below:
                    floor = AXIS_FLOORS[task]
                    ax.scatter(position + 0.32, floor + 0.006, marker="v", s=18, color=color, zorder=5, clip_on=False)
                    ax.text(position + 0.55, floor + 0.006, str(below), color=color, fontsize=FONT_SIZE, va="center")
        ax.axhline(1, color="#8b96a8", linewidth=0.8, linestyle="--", zorder=0)
        ax.set_title(
            f"{TASK_LABELS[task]} ({METRICS[task]}), {rows['dataset'].nunique()} datasets",
            fontsize=FONT_SIZE,
            fontweight="bold",
        )
        ax.set_ylim(AXIS_FLOORS.get(task), 1.02 if task not in AXIS_FLOORS else 1.005)
        _style_axis(ax)
        ax.yaxis.set_major_locator(MultipleLocator(LABELED_TICK_STEPS[task]))
        ax.yaxis.set_minor_locator(MultipleLocator(MINOR_TICK_STEP))
        ax.tick_params(axis="y", which="minor", length=1.5)
        ax.grid(axis="y", which="minor", color="#eef1f5", linewidth=0.5)
    axes[-1].set_xticks(np.arange(len(models)) * 2.6, [_display_label(m).replace("-Fast", "-\nFast") for m in models])
    axes[-1].set_xlim(-1.0, 2.6 * (len(models) - 1) + 1.1)
    fig.supylabel("Normalized score", fontsize=FONT_SIZE, x=0.01)
    fig.legend(
        handles=[
            Patch(facecolor="none", edgecolor=LEGEND_COLOR, label="ICL"),
            Patch(facecolor=LEGEND_COLOR, alpha=0.4, edgecolor=LEGEND_COLOR, label="SFT"),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=FONT_SIZE,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.9, bottom=0.08, hspace=0.42)
    return _save(fig, output_dir / "score_distributions.pdf")


def plot_dataset_wins(table: pd.DataFrame, winners: pd.DataFrame, output_dir: Path) -> Path:
    """Datasets won by ICL (hollow) and SFT (filled) within each model, and the best model and method overall."""
    models = [m for m in PAPER_MODELS if m in set(table["model"])]
    table = table.set_index("model")
    datasets = int(table["datasets"].max())
    fig = plt.figure(figsize=(PREPRINT_WIDTH_INCHES, 2.6))
    ax = fig.add_axes([0.2, 0.15, 0.78, 0.75])
    height = 0.6

    def segment(y: float, left: float, count: int, color: str, method: str, *, label: bool) -> None:
        face = "white" if method == "icl" else to_rgba(color, 0.4)
        ax.barh(y, count, left=left, height=height, color=face, edgecolor=color, linewidth=0.9, zorder=2)
        if label:
            ax.text(left + count / 2, y, str(count), ha="center", va="center", fontsize=FONT_SIZE, zorder=3)

    for y, model in enumerate(models):
        icl, sft = int(table.loc[model, "icl_wins"]), int(table.loc[model, "sft_wins"])
        segment(y, 0, icl, MODEL_COLORS[model], "icl", label=True)
        segment(y, datasets - sft, sft, MODEL_COLORS[model], "sft", label=True)
        if datasets - sft - icl:
            # A gain of exactly zero is neither an ICL nor an SFT win.
            ax.barh(
                y,
                datasets - sft - icl,
                left=icl,
                height=height,
                color="white",
                edgecolor=MODEL_COLORS[model],
                hatch="////",
                linewidth=0.9,
                zorder=2,
            )
    overall_y = len(models) + 2.0
    counts = winners.groupby(["method", "model"])["dataset"].nunique()
    left = 0
    counts_above: list[list[float]] = []
    for method, name in (("icl", "ICL"), ("sft", "SFT")):
        start = left
        for model in models:
            count = int(counts.get((method, model), 0))
            if count:
                segment(overall_y, left, count, MODEL_COLORS[model], method, label=False)
                counts_above.append([left + count / 2, count])
                left += count
        # A square bracket over the method's segments points to their total.
        bracket_y = overall_y - 1.1
        ax.plot(
            [start + 0.15, start + 0.15, left - 0.15, left - 0.15],
            [bracket_y + 0.15, bracket_y, bracket_y, bracket_y + 0.15],
            color="black",
            linewidth=0.8,
        )
        ax.text(
            (start + left) / 2,
            bracket_y - 0.32,
            f"{name} {left - start}",
            ha="center",
            va="center",
            fontsize=FONT_SIZE,
            fontweight="bold",
        )
        if method == "icl":
            # Separate the ICL and SFT winners, which share one row.
            ax.plot([left, left], [overall_y - 0.42, overall_y + 0.42], color="black", linewidth=1.6, zorder=4)
    # Each segment's count sits above the bar, spread apart so the narrow neighbors stay readable.
    for _ in range(20):
        for first, second in zip(counts_above, counts_above[1:], strict=False):
            overlap = 1.5 - (second[0] - first[0])
            if overlap > 0:
                first[0] -= overlap / 2
                second[0] += overlap / 2
    for x, count in counts_above:
        ax.text(x, overall_y - 0.55, str(count), ha="center", va="center", fontsize=FONT_SIZE)
    ax.axhline(len(models) - 0.25, color="#aeb8c6", linewidth=0.8)
    ax.set_yticks([*range(len(models)), overall_y], [*[MODEL_LABELS[m] for m in models], "Best overall"])
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_ylim(overall_y + 0.5, -0.5)
    ax.set_xlim(0, datasets)
    ax.set_xticks([*range(0, datasets - 5, 10), datasets])
    ax.set_xlabel(f"Datasets (of {datasets})", fontsize=FONT_SIZE)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=FONT_SIZE)
    ax.tick_params(axis="x", labelsize=FONT_SIZE, length=2.5)
    fig.legend(
        handles=[
            Patch(facecolor="white", edgecolor=LEGEND_COLOR, label="ICL"),
            Patch(facecolor=to_rgba(LEGEND_COLOR, 0.4), edgecolor=LEGEND_COLOR, label="SFT"),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=FONT_SIZE,
        bbox_to_anchor=(0.59, 1.0),
    )
    return _save(fig, output_dir / "dataset_wins.pdf")


METRIC_NAMES = {"roc_auc": "ROC-AUC", "accuracy": "Accuracy", "log_loss": "Log-loss", "rmse": "RMSE", "mae": "MAE"}
# Pairs of a task type's primary metric and a secondary metric, compared through their SFT gains.
METRIC_PAIRS = (
    ("binary", "roc_auc", "accuracy"),
    ("binary", "roc_auc", "log_loss"),
    ("multiclass", "log_loss", "accuracy"),
    ("regression", "rmse", "mae"),
)


def plot_metric_agreement(gains: pd.DataFrame, output_dir: Path) -> tuple[Path, list[dict[str, Any]]]:
    """SFT gains on the primary against a secondary metric, one point per model and dataset."""
    models = [m for m in PAPER_MODELS if m in set(gains["model"])]
    fig, axes = plt.subplots(1, len(METRIC_PAIRS), figsize=(PREPRINT_WIDTH_INCHES, 2.35))
    agreement = []
    for ax, (task, primary, secondary) in zip(axes, METRIC_PAIRS, strict=True):
        rows = gains.loc[gains["task_family"] == task]
        pivot = rows.pivot_table(index=["dataset", "model"], columns="metric", values="effect", aggfunc="first")
        if primary not in pivot or secondary not in pivot:
            ax.set_visible(False)
            continue
        common = pivot[[primary, secondary]].dropna().reset_index()
        for model in models:
            points = common.loc[common["model"] == model]
            ax.scatter(points[primary], points[secondary], s=7, color=MODEL_COLORS[model], alpha=0.75, linewidths=0)
        rho = float(common[primary].corr(common[secondary], method="spearman"))
        agreement.append(
            {"task_family": task, "primary": primary, "secondary": secondary, "rho": rho, "n": len(common)}
        )
        ax.axhline(0, color="#c0c5ce", linewidth=0.6, zorder=0)
        ax.axvline(0, color="#c0c5ce", linewidth=0.6, zorder=0)
        ax.set_title(f"{TASK_LABELS[task]}\nρ = {rho:.2f}", fontsize=FONT_SIZE, fontweight="bold")
        ax.set_xlabel(f"{METRIC_NAMES[primary]} gain", fontsize=FONT_SIZE)
        ax.set_ylabel(f"{METRIC_NAMES[secondary]} gain", fontsize=FONT_SIZE, labelpad=1)
        _style_axis(ax)
        ax.locator_params(nbins=4)
    fig.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", markersize=4, color=MODEL_COLORS[m], label=MODEL_LABELS[m])
            for m in models
        ],
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=FONT_SIZE,
        handletextpad=0.1,
        columnspacing=0.6,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.subplots_adjust(left=0.1, right=0.98, top=0.66, bottom=0.18, wspace=0.75)
    return _save(fig, output_dir / "metric_agreement.pdf"), agreement


def write_model_table(table: pd.DataFrame, output_dir: Path) -> Path:
    """LaTeX table (booktabs): normalized scores, SFT gain and within-model SFT wins per task type."""
    lines = [
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Normalized score} & \multicolumn{4}{c}{Datasets where SFT beats ICL} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-8}",
        r"Model & ICL & SFT & SFT gain [CI] & Binary & Multiclass & Regression & All \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        gain = f"{row['gain']:+.3f} [{row['gain_low']:+.3f}, {row['gain_high']:+.3f}]"
        wins = " & ".join(
            f"{int(row[f'sft_wins_{task}'])}/{int(row[f'datasets_{task}'])}"
            for task in ("binary", "multiclass", "regression")
        )
        # A dagger marks models whose weights are released under a non-commercial license.
        marker = "" if row["model"] in PERMISSIVE_MODELS else r"$^\dagger$"
        lines.append(
            f"{MODEL_LABELS[row['model']]}{marker} & "
            f"{row['icl']:.3f} & {row['sft']:.3f} & {gain} & {wins} & "
            f"{int(row['sft_wins'])}/{int(row['datasets'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path = output_dir / "models.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _signed(value: float) -> str:
    """Signed difference with three decimals; differences that round to zero carry no sign."""
    return "$0.000$" if abs(value) < 0.0005 else f"${value:+.3f}$"


def write_late_stage_table(table: pd.DataFrame, output_dir: Path) -> Path:
    """LaTeX table (booktabs): validation-selected against epoch-100 checkpoints per model."""
    lines = [
        r"\begin{tabular}{lccc|cc}",
        r"\toprule",
        r" & \multicolumn{5}{c}{Normalized score} \\",
        r"\cmidrule(lr){2-6}",
        r"Model & ICL & Selected & Epoch 100 & $\Delta$ Ep.\,100\,$-$\,ICL & $\Delta$ Ep.\,100\,$-$\,Sel. \\",
        r"\midrule",
    ]
    for row in table.itertuples():
        # Bold marks the best of the three checkpoints, compared before rounding.
        scores = {"icl": row.icl, "selected": row.selected, "final": row.final}
        best = max(scores, key=scores.get)
        cells = [rf"\textbf{{{value:.3f}}}" if key == best else f"{value:.3f}" for key, value in scores.items()]
        lines.append(
            f"{MODEL_LABELS[row.model]} & {' & '.join(cells)} & "
            f"{_signed(row.final - row.icl)} & {_signed(row.final - row.selected)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path = output_dir / "late_stage.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_cost_table(cost: pd.DataFrame, output_dir: Path) -> Path:
    """LaTeX table (booktabs) of the compute needed per model over the whole panel."""
    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Model & GPU-hours & Longest run (h) \\",
        r"\midrule",
    ]
    for model in models_present(cost):
        rows = cost.loc[cost["model"] == model]
        hours = float((rows["hours_without_test_eval"] * rows["runs"]).sum())
        lines.append(f"{MODEL_LABELS[model]} & {hours:.0f} & {rows['hours_without_test_eval'].max():.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path = output_dir / "cost.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _format_metric(value: float) -> str:
    """Fixed-point with three significant digits for small values, so RMSE on any scale stays readable."""
    if not np.isfinite(value):
        return "--"
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    if magnitude >= 0.1:
        return f"{value:.3f}"
    return f"{value:.4f}"


def write_dataset_tables(panel: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Appendix tables (one per task type): raw test metric without and with SFT for every model and dataset.

    Values are means over the runs of a dataset; the best value of a dataset across all models and both methods is
    bold (every cell that shows the best value after rounding, so displayed ties are all bold).
    """
    models = [m for m in PAPER_MODELS if m in set(panel["model"])]
    tabpfn = [m for m in models if m.startswith("tabpfn")]
    others = [m for m in models if not m.startswith("tabpfn")]
    short = {"tabpfnv2": "v2", "tabpfnv25": "2.5", "tabpfnv3": "3", "tabpfnv35": "3.5", "tabpfnv35fast": "3.5-Fast"}
    paths = []
    for task in TASK_ORDER:
        rows = panel.loc[panel["task_family"] == task]
        lines = [
            # Upright tables: dataset names wrap in a fixed-width first column (needs the array package).
            r"\begin{tabular}{>{\raggedright\arraybackslash}p{1.9cm}" + "c" * len(models) + "}",
            r"\toprule",
            " & "
            + r"\multicolumn{%d}{c}{TabPFN}" % len(tabpfn)
            + "".join(f" & {MODEL_LABELS[m]}" for m in others)
            + r" \\",
            r"\cmidrule(lr){2-%d}" % (len(tabpfn) + 1),
            "Dataset" + "".join(f" & {short[m]}" for m in tabpfn) + " & " * len(others) + r" \\",
            r"\midrule",
        ]
        higher_is_better = task == "binary"
        # ROC-AUC values cluster near 1, so four decimals keep the best result distinguishable from close runners-up.
        fmt = (lambda value: f"{value:.4f}") if task == "binary" else _format_metric
        for dataset in sorted(rows["dataset"].unique()):
            subset = rows.loc[rows["dataset"] == dataset]
            values = pd.concat([subset["baseline"], subset["selected"]]).astype(float)
            best = fmt(float(values.max() if higher_is_better else values.min()))

            def _cell(value: float, best: str = best) -> str:
                text = fmt(value)
                return rf"\textbf{{{text}}}" if text == best else text

            cells = []
            for model in models:
                match = subset.loc[subset["model"] == model]
                if match.empty:
                    cells.append("--")
                    continue
                row = match.iloc[0]
                cells.append(f"{_cell(float(row['baseline']))} / {_cell(float(row['selected']))}")
            lines.append(dataset.replace("_", " ") + " & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", ""]
        path = output_dir / f"datasets_{task}.tex"
        path.write_text("\n".join(lines), encoding="utf-8")
        paths.append(path)
    return paths


def generate_paper_assets(*, paper_root: Path, config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Write figures, tables and a ``numbers.json`` of every number quoted in the text into ``paper_root``."""
    apply_plot_style()
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    analysis = config["analysis"]
    results_root = REPO_ROOT / config["input"]["results_root"]
    studies = config["input"]["studies"]
    excluded = [name.lower() for name in analysis["excluded_datasets"]]
    seed, resamples = int(analysis["bootstrap_seed"]), int(analysis["bootstrap_resamples"])
    figures, tables = paper_root / "figures", paper_root / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    preprocessing = str(analysis["default_preprocessing"])
    panel = preprint_frame(
        results_root,
        studies=studies,
        context_size=int(analysis["default_context_size"]),
        preprocessing=preprocessing,
        excluded_datasets=excluded,
    )
    summary = generation_summary(panel, seed=seed, resamples=resamples)
    models = model_table(panel, summary)
    cost = cost_frame(results_root, studies=studies, preprocessing=preprocessing, excluded_datasets=excluded)
    winners = best_overall(panel)
    plot_score_distributions(panel, figures)
    plot_dataset_wins(models, winners, figures)
    write_model_table(models, tables)
    write_cost_table(cost, tables)
    late_stage = late_stage_table(panel)
    _, agreement = plot_metric_agreement(
        metric_gains(results_root, panel, studies=studies, context_size=int(analysis["default_context_size"])), figures
    )
    write_late_stage_table(late_stage, tables)
    if "associations" in analysis:
        plot_feature_associations(
            repo_root=REPO_ROOT, results_root=results_root, analysis=analysis, studies=studies, output_dir=figures
        )
    write_dataset_tables(panel, tables)
    gains = (
        panel.groupby(["model", "task_family"])["effect"]
        .agg(median="median", mean="mean", minimum="min", maximum="max")
        .reset_index()
    )
    numbers = {
        "datasets": int(panel["dataset"].nunique()),
        "tasks": {TASK_LABELS[t]: int(panel.loc[panel["task_family"] == t, "dataset"].nunique()) for t in TASK_ORDER},
        "models": json.loads(models.to_json(orient="records")),
        "scores_by_task": json.loads(summary.to_json(orient="records")),
        "gains_by_task": json.loads(gains.to_json(orient="records")),
        "late_stage": json.loads(late_stage.to_json(orient="records")),
        "metric_agreement": agreement,
        "best_overall": json.loads(
            winners.groupby(["method", "model"])["dataset"]
            .nunique()
            .rename("datasets")
            .reset_index()
            .to_json(orient="records")
        ),
        "permissive_vs_non_commercial": json.loads(
            cross_model_comparison(
                panel,
                fine_tuned=[m for m in models_present(panel) if m in PERMISSIVE_MODELS],
                without_fine_tuning=[m for m in models_present(panel) if m not in PERMISSIVE_MODELS],
            ).to_json(orient="records")
        ),
        "cost": json.loads(
            cost.groupby("model")
            .agg(
                gpu_hours=("hours_without_test_eval", lambda s: s.mul(cost.loc[s.index, "runs"]).sum()),
                gpu_hours_with_test_eval=("hours", lambda s: s.mul(cost.loc[s.index, "runs"]).sum()),
                max_run_hours=("hours_without_test_eval", "max"),
            )
            .loc[models_present(cost)]
            .reset_index()
            .to_json(orient="records")
        ),
    }
    (paper_root / "numbers.json").write_text(json.dumps(numbers, indent=2) + "\n", encoding="utf-8")
    return numbers


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path, default=DEFAULT_OUTPUT, help="Directory to write into")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    numbers = generate_paper_assets(paper_root=args.paper_root, config_path=args.config)
    print(json.dumps({"datasets": numbers["datasets"], "tasks": numbers["tasks"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
