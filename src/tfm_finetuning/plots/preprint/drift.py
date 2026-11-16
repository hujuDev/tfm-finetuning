"""Library drift check: the TabPFN-2.5 runs of overview_baseline (tabpfn 8.0.3) against the same runs with tabpfn 9.2.

Runs are paired per dataset and split. ``icl_change`` compares the zero-shot scores (inference and preprocessing
of the library) and ``sft_change`` the selected fine-tuned scores (the fine-tuning loop as well). A change is
positive when the new library is better: a metric difference for ROC-AUC, a log error ratio for log-loss and RMSE.

Example: ``python -m tfm_finetuning.plots.preprint.drift --output-root /tmp/drift``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from tfm_finetuning.finetuning.datasets.preprocessing_protocol import normalize_preprocessing_protocol
from tfm_finetuning.plots.metrics import fine_tuning_gain
from tfm_finetuning.plots.results import metric_frame, read_runs, resolve_results_root

REPO_ROOT = Path(__file__).resolve().parents[4]
SPLIT_KEYS = ["dataset", "seed", "fold", "repeat", "sample"]


def drift_table(
    results_root: Path,
    *,
    baseline_study: str = "overview_baseline",
    check_study: str = "preprint_tabpfn_drift_check",
    model: str = "tabpfnv25",
    preprocessing: str = "shared",
    context_size: int = 10000,
) -> pd.DataFrame:
    """Return one row per dataset and split present in both studies."""
    frame = metric_frame(results_root, [baseline_study, check_study])
    protocol = normalize_preprocessing_protocol(preprocessing)
    frame = frame.loc[
        (frame["split"] == "test")
        & (frame["metric"] == frame["score_metric"])
        & (frame["model"] == model)
        & (frame["context_size"] == context_size)
        & (frame["preprocessing"].map(normalize_preprocessing_protocol) == protocol)
    ]
    seconds = pd.DataFrame(
        [
            {
                "dataset": run.dataset.lower().replace("-", "_").replace(" ", "_"),
                **{key: run.metadata[key] for key in SPLIT_KEYS[1:]},
                "study": run.study,
                "seconds": float(run.report["runtime"]["total_seconds"]),
            }
            for run in read_runs(results_root, [baseline_study, check_study])
            if run.model == model and normalize_preprocessing_protocol(run.metadata["preprocessing"]) == protocol
        ]
    )
    columns = SPLIT_KEYS + ["task_family", "score_metric", "baseline", "selected", "selected_epoch"]
    old = frame.loc[frame["study"] == baseline_study, columns]
    new = frame.loc[frame["study"] == check_study, columns]
    paired = old.merge(new, on=SPLIT_KEYS + ["task_family", "score_metric"], suffixes=("_old", "_new"))
    for study, suffix in ((baseline_study, "old"), (check_study, "new")):
        times = seconds.loc[seconds["study"] == study, SPLIT_KEYS + ["seconds"]].rename(
            columns={"seconds": f"seconds_{suffix}"}
        )
        paired = paired.merge(times, on=SPLIT_KEYS, how="left")
    for name, column in (("icl", "baseline"), ("sft", "selected")):
        paired[f"{name}_change"] = [
            fine_tuning_gain(old_value, new_value, metric)
            for old_value, new_value, metric in zip(
                paired[f"{column}_old"], paired[f"{column}_new"], paired["score_metric"], strict=True
            )
        ]
    paired["speedup"] = paired["seconds_old"] / paired["seconds_new"]
    return paired.sort_values(SPLIT_KEYS).reset_index(drop=True)


def drift_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the paired changes per dataset; ``abs_*`` columns are mean absolute changes."""
    grouped = table.groupby(["dataset", "task_family", "score_metric"], sort=True)
    summary = grouped.agg(
        splits=("seed", "size"),
        icl_change=("icl_change", "mean"),
        sft_change=("sft_change", "mean"),
        abs_icl_change=("icl_change", lambda values: values.abs().mean()),
        abs_sft_change=("sft_change", lambda values: values.abs().mean()),
        seconds_old=("seconds_old", "mean"),
        seconds_new=("seconds_new", "mean"),
    )
    summary["speedup"] = summary["seconds_old"] / summary["seconds_new"]
    return summary.reset_index()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "results/figure_assets/preprint")
    args = parser.parse_args(argv)
    table = drift_table(resolve_results_root(args.results_root))
    args.output_root.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_root / "drift_by_split.csv", index=False)
    summary = drift_summary(table)
    summary.to_csv(args.output_root / "drift_summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4g}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
