"""Analyse the representative hyperparameter-tuning study for thesis findings.

Validation metrics select one learning-rate, weight-decay, and context setting
per dataset/model. Held-out test metrics are joined only after that selection.
Missing or failed trial cells remain absent and never contribute zero values.

Example:
    ``python -m plots.overview.hpo_analysis``
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator

from plots.common.model_display import MODEL_COLORS, MODEL_LABELS
from plots.common.plot_style import (
    FULL_WIDTH_INCHES,
    add_section1_model_legend,
    apply_section1_plot_style,
    save_section1_figure,
    shared_y_tick_interval,
)
from plots.final_tables import hpo_table
from plots.metrics import metric_is_lower_better
from plots.overview.pattern_data import _resource_frame
from plots.results import aggregate_metrics, metric_frame

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs/plots/section1_hpo_analysis.toml"
TASK_LABELS = {"binary": "Binary", "multiclass": "Multiclass", "regression": "Regression"}
TASK_MARKERS = {"binary": "o", "multiclass": "s", "regression": "^"}
HPO_CHECKPOINT_POLICIES = {None, "", "best_val", "best_val_early_stopping"}


def generate_hpo_analysis(*, config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Generate selection tables, held-out comparisons, fit surfaces, and resources."""
    apply_section1_plot_style()
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    hpo_details = REPO_ROOT / config["input"]["results_root"]
    overview_details = REPO_ROOT / config["input"]["results_root"]
    output_dir = REPO_ROOT / config["output"]["root"]
    thesis_dir = REPO_ROOT / config["output"]["thesis_assets"]
    output_dir.mkdir(parents=True, exist_ok=True)
    thesis_dir.mkdir(parents=True, exist_ok=True)
    models = list(config["analysis"]["models"])

    trials = _trial_frame(hpo_details, models=models)
    selected = _select_trials(trials)
    defaults = _overview_gains(overview_details, models=models)
    comparison = selected.merge(defaults, on=["dataset", "model", "task_family", "score_metric"], how="left")
    comparison["gain_change_from_default"] = comparison["test_gain"] - comparison["default_gain"]
    resources = _resource_frame(hpo_details, ["foundation_regular_hpo_representative"])
    trials.to_csv(output_dir / "trials.csv", index=False)
    selected.to_csv(output_dir / "selected_configurations.csv", index=False)
    comparison.to_csv(output_dir / "default_comparison.csv", index=False)
    resources.to_csv(output_dir / "resources.csv", index=False)
    table = output_dir / "hpo_selections.tex"
    table.write_text(hpo_table(selected, trials))
    shutil.copy2(table, thesis_dir / table.name)
    figures = [_plot_test_comparison(comparison, output_dir, models=models)]
    for figure in figures:
        shutil.copy2(figure, thesis_dir / figure.name)

    summary = {
        "datasetCount": int(trials["dataset"].nunique()),
        "models": models,
        "trialAggregateCount": len(trials),
        "selectedConfigurationCount": len(selected),
        "selectedWithDefaultCount": int(comparison["default_gain"].notna().sum()),
        "figures": [path.name for path in figures],
        "resources": _resource_summary(resources),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _trial_frame(results_root: Path, *, models: list[str]) -> pd.DataFrame:
    """Aggregate HPO repetitions before selecting configurations on validation only."""
    frame = metric_frame(results_root, ["foundation_regular_hpo_representative"])
    if frame.empty:
        raise RuntimeError("No completed HPO runs found")
    frame = frame[(frame.metric == frame.score_metric) & frame.model.isin(models)]
    coordinates = ["learning_rate", "weight_decay", "context_size"]
    if frame[coordinates].isna().any().any():
        raise ValueError("HPO reports lack hyperparameter coordinates")
    return aggregate_metrics(frame, ["dataset", "model", "task_family", "score_metric", "split", *coordinates]).rename(
        columns={"effect": "gain"}
    )


def _select_trials(trials: pd.DataFrame) -> pd.DataFrame:
    selected_records: list[dict[str, Any]] = []
    val_rows = trials[trials["split"] == "val"]
    test_rows = trials[trials["split"] == "test"].set_index(
        ["dataset", "model", "learning_rate", "weight_decay", "context_size"]
    )
    for (dataset, model), group in val_rows.groupby(["dataset", "model"], sort=True):
        metric = str(group["score_metric"].iloc[0])
        chosen_index = group["selected"].idxmin() if metric_is_lower_better(metric) else group["selected"].idxmax()
        chosen = group.loc[chosen_index]
        key = (dataset, model, chosen.learning_rate, chosen.weight_decay, chosen.context_size)
        if key not in test_rows.index:
            continue
        test = test_rows.loc[key]
        if isinstance(test, pd.DataFrame):
            test = test.iloc[0]
        selected_records.append(
            {
                "dataset": dataset,
                "model": model,
                "task_family": chosen.task_family,
                "score_metric": metric,
                "learning_rate": chosen.learning_rate,
                "weight_decay": chosen.weight_decay,
                "context_size": chosen.context_size,
                "validation_gain": chosen.gain,
                "test_gain": test.gain,
                "test_baseline": test.baseline,
                "test_selected": test.selected,
            }
        )
    return pd.DataFrame(selected_records)


def _overview_gains(results_root: Path, *, models: list[str]) -> pd.DataFrame:
    """Return default native-preprocessing gains from fresh baseline runs."""
    frame = metric_frame(results_root, ["unified_foundation_baseline"])
    frame = frame[
        (frame.metric == frame.score_metric)
        & frame.model.isin(models)
        & (frame.preprocessing == "native")
        & (frame.split == "test")
    ]
    return aggregate_metrics(frame, ["dataset", "model", "task_family", "score_metric"]).rename(
        columns={"effect": "default_gain"}
    )[["dataset", "model", "task_family", "score_metric", "default_gain"]]


def _plot_test_comparison(frame: pd.DataFrame, output_dir: Path, *, models: list[str]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH_INCHES, 3.2))
    for ax, task in zip(axes, ("binary", "multiclass", "regression"), strict=True):
        rows = frame[(frame["task_family"] == task) & frame["default_gain"].notna()]
        for model in models:
            subset = rows[rows["model"] == model]
            ax.scatter(
                subset["default_gain"], subset["test_gain"], color=MODEL_COLORS[model], label=MODEL_LABELS[model]
            )
        if not rows.empty:
            low, high, interval = _comparison_axis_scale(rows)
            ax.plot([low, high], [low, high], "--", color="#6b7280", linewidth=0.8)
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)
            ax.set_aspect("equal", adjustable="box")
            decimals = max(0, int(np.ceil(-np.log10(interval)))) if interval < 1 else 0

            def format_tick(value: float, _position: float, *, places: int = decimals) -> str:
                normalized = 0.0 if np.isclose(value, 0.0) else value
                return f"{normalized:.{places}f}"

            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(MultipleLocator(interval))
                axis.set_major_formatter(FuncFormatter(format_tick))
        ax.axhline(0, color="#d1d5db", linewidth=0.6)
        ax.axvline(0, color="#d1d5db", linewidth=0.6)
        ax.set_title(TASK_LABELS[task])
        ax.grid(axis="both", color="#e5e7eb", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    add_section1_model_legend(
        fig,
        model_order=models,
        model_labels=MODEL_LABELS,
        model_colors=MODEL_COLORS,
        anchor=(0.5, 0.95),
    )
    fig.suptitle("Held-out effect of validation-selected hyperparameters")
    fig.supxlabel("Default-recipe test gain", y=0.045)
    fig.supylabel("Validation-selected HPO test gain", x=0.018)
    fig.subplots_adjust(left=0.11, right=0.985, top=0.72, bottom=0.20, wspace=0.42)
    return _save(fig, output_dir / "hpo_test_gain_vs_default.pdf", tight=False)


def _comparison_axis_scale(rows: pd.DataFrame) -> tuple[float, float, float]:
    """Return one padded, tick-aligned range for both HPO comparison axes."""
    values = pd.concat([rows["default_gain"], rows["test_gain"]]).dropna()
    if values.empty:
        return -0.05, 0.05, 0.01
    raw_low = min(0.0, float(values.min()))
    raw_high = max(0.0, float(values.max()))
    span = raw_high - raw_low
    padding = max(0.12 * span, 0.005 if span < 0.1 else 0.01)
    provisional_low = raw_low - padding
    provisional_high = raw_high + padding

    probe_figure, probe_axis = plt.subplots()
    probe_axis.set_ylim(provisional_low, provisional_high)
    interval = shared_y_tick_interval([probe_axis], max_guides=8)
    plt.close(probe_figure)
    low = math.floor(provisional_low / interval) * interval
    high = math.ceil(provisional_high / interval) * interval
    return low, high, interval


def _resource_summary(frame: pd.DataFrame) -> dict[str, Any]:
    runtime = frame[frame["resource"] == "runtime"]
    return {
        "gpuHours": float(runtime["sum_value"].dropna().sum() / 3600.0),
    }


def _save(
    fig: plt.Figure,
    path: Path,
    *,
    tight: bool = True,
    inclusion_fraction: float = 1.0,
    max_inclusion_height_fraction: float | None = None,
) -> Path:
    if tight:
        fig.tight_layout()
    save_section1_figure(
        fig,
        path,
        inclusion_fraction=inclusion_fraction,
        max_inclusion_height_fraction=max_inclusion_height_fraction,
    )
    plt.close(fig)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    print(json.dumps(generate_hpo_analysis(config_path=args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
