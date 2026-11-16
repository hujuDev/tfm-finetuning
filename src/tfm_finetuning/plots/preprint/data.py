"""Evidence frames and paired statistics for the preprint comparisons.

All scores are best-relative normalized primary scores (ROC-AUC, log-loss, RMSE by task family), computed over every
model in the panel. Uncertainty is a bootstrap over datasets, never over runs.

Example: ``frame = preprint_frame(Path("results"), studies=["overview_baseline", "preprint_tabpfn_v3"])``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from tfm_finetuning.finetuning.datasets.preprocessing_protocol import normalize_preprocessing_protocol
from tfm_finetuning.plots.common.model_display import PAPER_MODELS
from tfm_finetuning.plots.overview.pattern_data import _add_normalized_scores, _metric_frame
from tfm_finetuning.plots.overview.pattern_model_comparison import DEFAULT_DECISIVE_WINNER_MARGIN, _global_winner_table
from tfm_finetuning.plots.results import read_runs


def preprint_frame(
    results_root: Path,
    *,
    studies: Sequence[str],
    context_size: int = 10000,
    preprocessing: str = "shared",
    excluded_datasets: Sequence[str] = ("airlines", "miniboone"),
    models: Sequence[str] = PAPER_MODELS,
) -> pd.DataFrame:
    """Return one row per dataset and model with ICL/SFT scores, normalized scores and the normalized gain.

    Only datasets on which every model with results has a result are kept, so all model means share one dataset set;
    the dropped datasets are listed in ``frame.attrs["dropped_datasets"]``.
    """
    frame = _metric_frame(results_root, metadata={}, studies=studies, expected_context_size=context_size)
    if frame.empty:
        return frame
    frame = frame.loc[frame["metric"] == frame["score_metric"]]
    protocol = normalize_preprocessing_protocol(preprocessing)
    frame = frame.loc[frame["preprocessing"].map(normalize_preprocessing_protocol) == protocol]
    frame = frame.loc[~frame["dataset"].isin(excluded_datasets) & frame["model"].isin(models)]
    present = models_present(frame)
    counts = frame.groupby("dataset")["model"].nunique()
    complete = counts[counts == len(present)].index
    dropped = sorted(set(frame["dataset"]) - set(complete))
    frame = _add_normalized_scores(frame.loc[frame["dataset"].isin(complete)].reset_index(drop=True))
    frame["normalized_gain"] = frame["normalized_selected"] - frame["normalized_baseline"]
    frame.attrs["dropped_datasets"] = dropped
    return frame


def models_present(frame: pd.DataFrame) -> list[str]:
    """Return the preprint models that have results, in display order."""
    return [model for model in PAPER_MODELS if model in set(frame["model"])]


def bootstrap_mean(values: Sequence[float], *, seed: int, resamples: int) -> tuple[float, float, float]:
    """Return the mean and its 95% percentile bootstrap interval over the resampled entries."""
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if not len(data):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, len(data), size=(resamples, len(data)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(data.mean()), float(low), float(high)


def generation_summary(panel: pd.DataFrame, *, seed: int, resamples: int) -> pd.DataFrame:
    """Summarize ICL, SFT and the SFT gain per model, overall and per task family."""
    records = []
    for model in [m for m in PAPER_MODELS if m in set(panel["model"])]:
        rows = panel.loc[panel["model"] == model]
        for task, subset in [("all", rows), *rows.groupby("task_family")]:
            record = {"model": model, "task_family": task, "datasets": int(subset["dataset"].nunique())}
            for name, column in [
                ("icl", "normalized_baseline"),
                ("sft", "normalized_selected"),
                ("gain", "normalized_gain"),
            ]:
                mean, low, high = bootstrap_mean(subset[column], seed=seed, resamples=resamples)
                record.update({f"{name}_mean": mean, f"{name}_low": low, f"{name}_high": high})
            record["sft_improves_share"] = float((subset["normalized_gain"] > DEFAULT_DECISIVE_WINNER_MARGIN).mean())
            records.append(record)
    return pd.DataFrame(records)


def model_table(panel: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """One row per model: mean normalized ICL and SFT scores, SFT gain with interval, and within-model wins.

    SFT wins a dataset when the gain G, averaged over its runs, is positive, and ICL wins when it is
    negative.
    """
    overall = summary.loc[summary["task_family"] == "all"].set_index("model")
    records = []
    for model in models_present(panel):
        rows = panel.loc[panel["model"] == model]
        record: dict[str, object] = {
            "model": model,
            "datasets": int(rows["dataset"].nunique()),
            "icl": overall.loc[model, "icl_mean"],
            "sft": overall.loc[model, "sft_mean"],
            "gain": overall.loc[model, "gain_mean"],
            "gain_low": overall.loc[model, "gain_low"],
            "gain_high": overall.loc[model, "gain_high"],
            "sft_wins": int((rows["effect"] > 0).sum()),
            "icl_wins": int((rows["effect"] < 0).sum()),
        }
        for task in ("binary", "multiclass", "regression"):
            task_rows = rows.loc[rows["task_family"] == task]
            record[f"sft_wins_{task}"] = int((task_rows["effect"] > 0).sum())
            record[f"datasets_{task}"] = int(task_rows["dataset"].nunique())
        records.append(record)
    return pd.DataFrame(records)


def metric_gains(
    results_root: Path, panel: pd.DataFrame, *, studies: Sequence[str], context_size: int
) -> pd.DataFrame:
    """SFT gain over ICL on every recorded test metric, for the dataset-model pairs of the preprint panel.

    One row per dataset, model and metric, with the gain ``effect`` (difference for higher-is-better metrics, log ratio
    for losses) averaged over the runs, as for the primary metric.
    """
    frame = _metric_frame(results_root, metadata={}, studies=studies, expected_context_size=context_size)
    if frame.empty:
        return frame
    protocol = normalize_preprocessing_protocol(str(panel["preprocessing"].iloc[0]))
    frame = frame.loc[frame["preprocessing"].map(normalize_preprocessing_protocol) == protocol]
    pairs = panel[["dataset", "model"]].drop_duplicates()
    return frame.merge(pairs, on=["dataset", "model"])[["dataset", "model", "task_family", "metric", "effect"]]


def late_stage_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Per model, the validation-selected checkpoint against the last (epoch-100) checkpoint.

    SFT wins a dataset when its mean gain over ICL is positive; scores are mean normalized scores over datasets, and the
    selected epoch is the median over datasets of its run mean.
    """
    records = []
    for model in models_present(panel):
        rows = panel.loc[panel["model"] == model]
        records.append(
            {
                "model": model,
                "sft_wins_selected": int((rows["effect"] > 0).sum()),
                "sft_wins_final": int((rows["final_effect"] > 0).sum()),
                "final_below_selected": int((rows["checkpoint_gap"] > 0).sum()),
                "icl": float(rows["normalized_baseline"].mean()),
                "selected": float(rows["normalized_selected"].mean()),
                "final": float(rows["normalized_final"].mean()),
                "median_selected_epoch": float(rows["selected_epoch"].median()),
            }
        )
    return pd.DataFrame(records)


def best_overall(panel: pd.DataFrame) -> pd.DataFrame:
    """Per dataset, the model and method (``icl`` or ``sft``) with the strictly best normalized score."""
    winners = _global_winner_table(panel)
    return winners.loc[winners["strict_winner"], ["dataset", "task_family", "model", "method"]].reset_index(drop=True)


def cost_frame(
    results_root: Path,
    *,
    studies: Sequence[str],
    preprocessing: str = "shared",
    excluded_datasets: Sequence[str] = ("airlines", "miniboone"),
    models: Sequence[str] = PAPER_MODELS,
) -> pd.DataFrame:
    """Per dataset and model: mean wall time without per-epoch test evaluation, and the largest peak VRAM.

    The overview_baseline runs of TabPFN-2.5, TabICLv2 and Mitra evaluated the test set after every epoch (a diagnostic of that
    protocol); its time is removed so that the models are timed on training and validation alone. Peak VRAM is the
    largest allocated and reserved GPU memory over all phases of a run, in GiB, maximized over the splits.
    """
    protocol = normalize_preprocessing_protocol(preprocessing)
    records = []
    for run in read_runs(results_root, studies):
        dataset = run.dataset.lower().replace("-", "_").replace(" ", "_")
        if (
            run.model not in models
            or dataset in excluded_datasets
            or normalize_preprocessing_protocol(run.metadata["preprocessing"]) != protocol
        ):
            continue
        allocated: list[float] = []
        reserved: list[float] = []
        _collect_memory(run.report, allocated, reserved)
        in_fit_eval = sum(
            float(phase.get("in_fit_seconds") or 0.0)
            for phase in run.report["runtime"]["phases"]
            if phase.get("phase") == "periodic_validation_eval"
        )
        total = float(run.report["runtime"]["total_seconds"])
        records.append(
            {
                "dataset": dataset,
                "model": run.model,
                "hours": total / 3600,
                "hours_without_test_eval": (total - in_fit_eval) / 3600,
                "peak_allocated_gib": max(allocated) / 1024 if allocated else float("nan"),
                "peak_reserved_gib": max(reserved) / 1024 if reserved else float("nan"),
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    return (
        frame.groupby(["dataset", "model"])
        .agg(
            runs=("hours", "size"),
            hours=("hours", "mean"),
            hours_without_test_eval=("hours_without_test_eval", "mean"),
            peak_allocated_gib=("peak_allocated_gib", "max"),
            peak_reserved_gib=("peak_reserved_gib", "max"),
        )
        .reset_index()
    )


def _collect_memory(node: object, allocated: list[float], reserved: list[float]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "gpu_memory_peak_allocated_mib" and isinstance(value, (int, float)):
                allocated.append(float(value))
            elif key == "gpu_memory_peak_reserved_mib" and isinstance(value, (int, float)):
                reserved.append(float(value))
            else:
                _collect_memory(value, allocated, reserved)
    elif isinstance(node, list):
        for item in node:
            _collect_memory(item, allocated, reserved)


def cross_model_comparison(
    panel: pd.DataFrame,
    *,
    fine_tuned: Sequence[str],
    without_fine_tuning: Sequence[str],
    margin: float = DEFAULT_DECISIVE_WINNER_MARGIN,
) -> pd.DataFrame:
    """Normalized SFT score of one model minus the normalized ICL score of another, per task family.

    One row per (fine-tuned model, model without fine-tuning, task family or ``all``) with the mean difference and the
    number of datasets won, tied (within ``margin``) and lost by the fine-tuned model.
    """
    sft = panel.pivot(index="dataset", columns="model", values="normalized_selected")
    icl = panel.pivot(index="dataset", columns="model", values="normalized_baseline")
    task = panel.drop_duplicates("dataset").set_index("dataset")["task_family"]
    records = []
    for left in fine_tuned:
        for right in without_fine_tuning:
            difference = (sft[left] - icl[right]).dropna()
            for family in ("all", "binary", "multiclass", "regression"):
                values = difference if family == "all" else difference[task.reindex(difference.index) == family]
                records.append(
                    {
                        "fine_tuned": left,
                        "without_fine_tuning": right,
                        "task_family": family,
                        "datasets": int(len(values)),
                        "mean_difference": float(values.mean()) if len(values) else float("nan"),
                        "wins": int((values > margin).sum()),
                        "ties": int((values.abs() <= margin).sum()),
                        "losses": int((values < -margin).sum()),
                    }
                )
    return pd.DataFrame(records)
