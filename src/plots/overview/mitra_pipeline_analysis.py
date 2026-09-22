"""Compare the original and local chunked Mitra fine-tuning pipelines.

The comparison uses the two fresh Mitra pipeline studies and writes one
appendix-ready figure containing task-specific gain agreement and full-run
runtime agreement.

Example:
    ``python -m plots.overview.mitra_pipeline_analysis``
"""

from __future__ import annotations

import argparse
import json
import shutil
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from matplotlib.ticker import FuncFormatter, NullFormatter

from plots.common.model_display import MODEL_COLORS
from plots.common.plot_style import (
    FULL_WIDTH_INCHES,
    apply_section1_plot_style,
    plain_log_power_label,
    save_section1_figure,
)
from plots.results import aggregate_metrics, metric_frame, read_runs
from utils.report_labels import normalize_name

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs/plots/section1_mitra_pipeline_analysis.toml"
ORIGINAL_PIPELINE = "mitra_original_pipeline"
CHUNKED_PIPELINE = "mitra_adapted_pipeline"
PIPELINE_LABELS = {
    ORIGINAL_PIPELINE: "Original Mitra pipeline",
    CHUNKED_PIPELINE: "Local chunked pipeline",
}
TASK_ORDER = ["binary", "multiclass", "regression"]
TASK_LABELS = {"binary": "Binary", "multiclass": "Multiclass", "regression": "Regression"}
TASK_MARKERS = {"binary": "o", "multiclass": "s", "regression": "^"}


def generate_mitra_pipeline_analysis(*, config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Write comparison tables, an appendix plot, and a compact summary."""
    apply_section1_plot_style()
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    inputs = config["input"]
    results_root = REPO_ROOT / inputs["results_root"]
    output_dir = REPO_ROOT / config["output"]["root"]
    thesis_dir = REPO_ROOT / config["output"]["thesis_assets"]
    output_dir.mkdir(parents=True, exist_ok=True)
    thesis_dir.mkdir(parents=True, exist_ok=True)

    dataset_bundle = str(config["input"].get("dataset_bundle") or "").strip()
    dataset_keys = (
        _dataset_bundle_members(
            REPO_ROOT / config["input"]["dataset_bundles_path"],
            dataset_bundle,
        )
        if dataset_bundle
        else None
    )
    performance, runtime = _comparison_frames(results_root, dataset_keys=dataset_keys)
    performance.to_csv(output_dir / "performance.csv", index=False)
    runtime.to_csv(output_dir / "runtime.csv", index=False)
    figure = _plot_pipeline_comparison(performance, runtime, output_dir)
    shutil.copy2(figure, thesis_dir / figure.name)

    runtime_pivot = runtime.pivot_table(
        index=["dataset", "task_family"], columns="pipeline", values="mean_seconds", aggfunc="first"
    )
    runtime_common = runtime_pivot[[ORIGINAL_PIPELINE, CHUNKED_PIPELINE]].dropna()
    total_seconds = runtime.groupby("pipeline")["sum_seconds"].sum()
    summary = {
        "datasetBundle": dataset_bundle or None,
        "datasetCount": int(performance["dataset"].nunique()),
        "figure": figure.name,
        "runtimeDatasetCount": len(runtime_common),
        "medianChunkedToOriginalRuntimeRatio": float(
            (runtime_common[CHUNKED_PIPELINE] / runtime_common[ORIGINAL_PIPELINE]).median()
        ),
        "summedChunkedToOriginalRuntimeRatio": float(
            total_seconds[CHUNKED_PIPELINE] / total_seconds[ORIGINAL_PIPELINE]
        ),
        "gpuHoursByPipeline": {
            PIPELINE_LABELS[pipeline]: float(group["sum_seconds"].dropna().sum() / 3600.0)
            for pipeline, group in runtime.groupby("pipeline")
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _comparison_frames(
    results_root: Path, *, dataset_keys: set[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair original and adapted Mitra runs by dataset and report physical runtime."""
    studies = [ORIGINAL_PIPELINE, CHUNKED_PIPELINE]
    frame = metric_frame(results_root, studies)
    frame = frame[(frame.metric == frame.score_metric) & (frame.split == "test")]
    if dataset_keys is not None:
        frame = frame[frame.dataset.isin(dataset_keys)]
    performance = aggregate_metrics(frame, ["dataset", "task_family", "score_metric", "study"]).rename(
        columns={"study": "pipeline", "effect": "gain"}
    )
    records = [
        {
            "dataset": normalize_name(run.dataset),
            "pipeline": run.study,
            "seconds": run.report["runtime"]["total_seconds"],
        }
        for run in read_runs(results_root, studies)
        if dataset_keys is None or normalize_name(run.dataset) in dataset_keys
    ]
    runtime = (
        pd.DataFrame(records)
        .groupby(["dataset", "pipeline"])
        .seconds.agg(mean_seconds="mean", sum_seconds="sum", run_count="count")
        .reset_index()
    )
    tasks = performance.drop_duplicates("dataset").set_index("dataset").task_family
    runtime["task_family"] = runtime.dataset.map(tasks)
    expected = {(dataset, pipeline) for dataset in performance.dataset.unique() for pipeline in studies}
    if set(performance[["dataset", "pipeline"]].itertuples(index=False, name=None)) != expected:
        raise ValueError("Incomplete original/adapted Mitra pairs")
    return performance, runtime


def _dataset_bundle_members(path: Path, bundle_id: str) -> set[str]:
    """Read one explicit dataset panel."""
    payload = yaml.safe_load(path.read_text())
    return {normalize_name(value) for value in payload[bundle_id]}


def _plot_pipeline_comparison(performance: pd.DataFrame, runtime: pd.DataFrame, output_dir: Path) -> Path:
    fig, grid = plt.subplots(2, 2, figsize=(FULL_WIDTH_INCHES, 4.8))
    axes = grid.ravel()
    color = MODEL_COLORS["mitra"]
    for ax, task in zip(axes[:3], TASK_ORDER, strict=True):
        rows = performance[performance["task_family"] == task]
        pivot = rows.pivot_table(index="dataset", columns="pipeline", values="gain", aggfunc="first")
        common = pivot[[ORIGINAL_PIPELINE, CHUNKED_PIPELINE]].dropna()
        ax.scatter(common[ORIGINAL_PIPELINE], common[CHUNKED_PIPELINE], color=color, s=24, alpha=0.78)
        _add_equality_line(ax, common[ORIGINAL_PIPELINE], common[CHUNKED_PIPELINE])
        ax.axhline(0, color="#c0c5ce", linewidth=0.6)
        ax.axvline(0, color="#c0c5ce", linewidth=0.6)
        ax.set_xlabel("Original pipeline gain")
        ax.set_ylabel("Adapted pipeline gain")
        ax.set_title(TASK_LABELS[task])
        ax.grid(alpha=0.15)

    runtime_pivot = runtime.pivot_table(
        index=["dataset", "task_family"], columns="pipeline", values="mean_seconds", aggfunc="first"
    )
    common_runtime = runtime_pivot[[ORIGINAL_PIPELINE, CHUNKED_PIPELINE]].dropna().reset_index()
    runtime_ax = axes[3]
    for task in TASK_ORDER:
        rows = common_runtime[common_runtime["task_family"] == task]
        runtime_ax.scatter(
            rows[ORIGINAL_PIPELINE],
            rows[CHUNKED_PIPELINE],
            marker=TASK_MARKERS[task],
            color=color,
            s=25,
            alpha=0.75,
            label=TASK_LABELS[task],
        )
    positive = common_runtime[(common_runtime[ORIGINAL_PIPELINE] > 0) & (common_runtime[CHUNKED_PIPELINE] > 0)]
    runtime_ax.set_xscale("log")
    runtime_ax.set_yscale("log")
    runtime_ax.xaxis.set_major_formatter(FuncFormatter(plain_log_power_label))
    runtime_ax.yaxis.set_major_formatter(FuncFormatter(plain_log_power_label))
    runtime_ax.xaxis.set_minor_formatter(NullFormatter())
    runtime_ax.yaxis.set_minor_formatter(NullFormatter())
    _add_equality_line(runtime_ax, positive[ORIGINAL_PIPELINE], positive[CHUNKED_PIPELINE])
    runtime_ax.set_xlabel("Original pipeline runtime (s)")
    runtime_ax.set_ylabel("Adapted pipeline runtime (s)")
    runtime_ax.set_title("Runtime")
    runtime_ax.legend(frameon=False, fontsize=8, loc="upper left")
    runtime_ax.grid(alpha=0.15)
    return _save(fig, output_dir / "mitra_pipeline_comparison.pdf")


def _add_equality_line(ax: plt.Axes, x: pd.Series, y: pd.Series) -> None:
    if x.empty or y.empty:
        return
    low = min(float(x.min()), float(y.min()))
    high = max(float(x.max()), float(y.max()))
    ax.plot([low, high], [low, high], "--", color="#6b7280", linewidth=0.8)


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    save_section1_figure(fig, path, max_inclusion_height_fraction=0.82)
    plt.close(fig)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    print(json.dumps(generate_mitra_pipeline_analysis(config_path=args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
