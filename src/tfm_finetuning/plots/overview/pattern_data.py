"""Build canonical evidence frames and tabular summaries for the overview study."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tfm_finetuning.finetuning.datasets.preprocessing_protocol import normalize_preprocessing_protocol
from tfm_finetuning.plots.metrics import (
    best_relative_normalized_score,
    metric_is_lower_better,
)
from tfm_finetuning.plots.results import aggregate_metrics, metric_frame, read_runs


def _canonical_frame(
    results_root: Path,
    *,
    metadata: dict[str, dict[str, Any]],
    studies: Sequence[str] = ("unified_foundation_baseline",),
    expected_context_size: int | None = None,
) -> pd.DataFrame:
    """Aggregate fresh primary metrics, preserving paired-run gains and uncertainty."""
    frame = _metric_frame(
        results_root, metadata=metadata, studies=studies, expected_context_size=expected_context_size
    )
    if frame.empty:
        return frame
    frame = frame.loc[frame.metric == frame.score_metric].copy()
    for key in {key for values in metadata.values() for key in values}:
        frame[key] = frame.dataset.map(lambda dataset: metadata.get(dataset, {}).get(key))
    frame["checkpoint_policy"] = "best_val"
    frame["trainingTableCells"] = frame["baseTrainRows"] * frame["nFeaturesTotal"]
    frame["rowToFeatureRatio"] = frame["baseTrainRows"] / frame["nFeaturesTotal"].replace(0, np.nan)
    return _add_normalized_scores(frame)


def _metric_frame(
    results_root: Path,
    *,
    metadata: dict[str, dict[str, Any]],
    studies: Sequence[str] = ("unified_foundation_baseline",),
    expected_context_size: int | None = None,
) -> pd.DataFrame:
    """Average measurements over fixed seed/split repetitions for each model."""
    frame = metric_frame(results_root, studies)
    if frame.empty:
        return frame
    frame = frame[frame.split == "test"]
    if expected_context_size is not None:
        frame = frame[frame.context_size == expected_context_size]
    return aggregate_metrics(
        frame, ["dataset", "model", "preprocessing", "task_family", "score_metric", "metric", "context_size"]
    )


def _resource_frame(results_root: Path, studies: Sequence[str] = ("unified_foundation_baseline",)) -> pd.DataFrame:
    """Summarize the wall time of physical completed runs once per run."""
    records = []
    for run in read_runs(results_root, studies):
        value = float(run.report["runtime"]["total_seconds"])
        if value < 0:
            raise ValueError(f"Negative runtime: {run.path}")
        records.append(
            {
                "dataset": run.dataset.lower().replace("-", "_").replace(" ", "_"),
                "model": run.model,
                "preprocessing": run.metadata["preprocessing"],
                "seconds": value,
            }
        )
    if not records:
        return pd.DataFrame(
            columns=["dataset", "model", "preprocessing", "resource", "mean_value", "sum_value", "run_count"]
        )
    frame = (
        pd.DataFrame(records)
        .groupby(["dataset", "model", "preprocessing"])
        .seconds.agg(mean_value="mean", sum_value="sum", run_count="count")
        .reset_index()
    )
    frame["resource"] = "runtime"
    return frame


def _analysis_subset(frame: pd.DataFrame, *, preprocessing: str) -> pd.DataFrame:
    """Return rows for the preprocessing path used by default thesis analyses."""
    if frame.empty:
        return frame.copy()
    return frame.loc[
        frame["preprocessing"].map(normalize_preprocessing_protocol) == normalize_preprocessing_protocol(preprocessing)
    ].reset_index(drop=True)


def _add_normalized_scores(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["normalized_baseline"] = np.nan
    result["normalized_selected"] = np.nan
    result["normalized_final"] = np.nan
    for indices in result.groupby(["dataset", "preprocessing"]).groups.values():
        rows = result.loc[indices]
        metric = rows["score_metric"].iloc[0]
        values = np.concatenate([rows["baseline"].to_numpy(), rows["selected"].to_numpy()])
        best = np.nanmin(values) if metric_is_lower_better(metric) else np.nanmax(values)
        result.loc[indices, "normalized_baseline"] = rows["baseline"].map(
            lambda value, best=best, metric=metric: best_relative_normalized_score(value, best, metric)
        )
        result.loc[indices, "normalized_selected"] = rows["selected"].map(
            lambda value, best=best, metric=metric: best_relative_normalized_score(value, best, metric)
        )
        result.loc[indices, "normalized_final"] = pd.to_numeric(
            rows["final"].map(
                lambda value, best=best, metric=metric: best_relative_normalized_score(value, best, metric)
            ),
            errors="coerce",
        )
    return result


def _dataset_metadata(*, cache_path: Path) -> dict[str, dict[str, Any]]:
    """Load train-only characteristics computed from the reproduction datasets."""
    if not cache_path.is_file():
        from tfm_finetuning.data.build_base_meta_cache import build_cache

        build_cache(output_path=cache_path)
    return json.loads(cache_path.read_text())["datasets"]


def _write_tables(output_dir: Path, **tables: pd.DataFrame) -> None:
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
