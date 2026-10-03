"""Read completed fresh runs directly; no catalog or historical artifact lookup.

Example: ``rows = metric_frame(Path("results"), ["overview_baseline"])``.
"""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from tfm_finetuning.plots.metrics import fine_tuning_gain, normalize_metric_key, split_raw_metric_value

METRICS = ("roc_auc", "accuracy", "log_loss", "rmse", "mse", "mae", "r2")
TASKS = {"roc_auc": "binary", "log_loss": "multiclass", "rmse": "regression"}


REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_results_root(root: Path) -> Path:
    """Honor RESULTS_ROOT for the default input location; explicit alternatives win."""
    default = root == Path("results") or root.resolve() == REPO_ROOT / "results"
    value = os.environ.get("RESULTS_ROOT") if default else None
    if value:
        path = Path(value).expanduser()
        return path if path.is_absolute() else REPO_ROOT / path
    return root


@dataclass(frozen=True)
class Run:
    """A completed run and its adjacent training report."""

    metadata: dict[str, Any]
    report: dict[str, Any]
    path: Path

    @property
    def dataset(self) -> str:
        return self.metadata["dataset"]

    @property
    def model(self) -> str:
        return self.metadata["model"]

    @property
    def seed(self) -> int:
        return int(self.metadata["seed"])

    @property
    def study(self) -> str:
        return self.metadata["study"]

    @property
    def parameters(self) -> dict[str, Any]:
        return self.metadata["parameters"]


def read_runs(root: Path, studies: Sequence[str] | None = None) -> Iterator[Run]:
    """Yield completed runs, diagnosing failures and rejecting duplicate scientific units.

    A running/failed run is excluded with a warning. A complete run lacking its
    report is corrupt and raises an error; missing measurements are never zero.
    """
    root = resolve_results_root(root)
    if studies:
        paths = []
        for study in studies:
            found = list((root / study).glob("*/run.json"))
            if not found:
                raise FileNotFoundError(f"No runs found for study {study!r} under {root}")
            paths.extend(found)
    else:
        paths = root.glob("*/*/run.json")
    seen = set()
    for path in sorted(paths):
        metadata = json.loads(path.read_text())
        if metadata["status"] != "complete":
            warnings.warn(f"Excluded {metadata['status']} run: {path}", stacklevel=2)
            continue
        key = (
            metadata["study"],
            metadata["dataset"],
            metadata["model"],
            metadata["preprocessing"],
            metadata["seed"],
            metadata["fold"],
            metadata["repeat"],
            metadata["sample"],
            json.dumps(metadata["parameters"], sort_keys=True),
        )
        if key in seen:
            raise ValueError(f"Duplicate completed scientific run: {path}")
        seen.add(key)
        report_path = path.with_name("finetune_report.json")
        report = json.loads(report_path.read_text())
        yield Run(metadata, report, path)


def metric_frame(root: Path, studies: Sequence[str] | None = None) -> pd.DataFrame:
    """Return one row per run, split and conventional metric, with paired gains."""
    records = []
    for run in read_runs(root, studies):
        report, config = run.report, run.metadata["config"]
        metric_name = normalize_metric_key(report["score_metric"])
        finetune = config.get("finetune", {})
        evaluation = report.get("evaluation", {})
        for split in ("val", "test"):
            phases = {
                key: report.get(field, {}).get(split, {})
                for key, field in (
                    ("baseline", "zero_shot_metrics"),
                    ("selected", "selected_checkpoint_metrics"),
                    ("final", "final_checkpoint_metrics"),
                )
            }
            for metric in METRICS:
                values = {
                    key: split_raw_metric_value(payload, metric, score_metric=metric_name)
                    for key, payload in phases.items()
                }
                if values["baseline"] is None or values["selected"] is None:
                    continue
                records.append(
                    {
                        **{
                            key: run.metadata[key]
                            for key in (
                                "study",
                                "dataset",
                                "model",
                                "preprocessing",
                                "seed",
                                "fold",
                                "repeat",
                                "sample",
                                "run_id",
                            )
                        },
                        "dataset": run.dataset.lower().replace("-", "_").replace(" ", "_"),
                        "task_family": TASKS[metric_name],
                        "score_metric": metric_name,
                        "metric": metric,
                        "split": split,
                        **values,
                        "effect": fine_tuning_gain(values["baseline"], values["selected"], metric),
                        "final_effect": fine_tuning_gain(values["baseline"], values["final"], metric),
                        "selected_epoch": evaluation.get("selected_epoch"),
                        "training_epochs": evaluation.get("final_epoch"),
                        "learning_rate": finetune.get("learning_rate"),
                        "weight_decay": finetune.get("weight_decay"),
                        "context_size": finetune.get("context_size"),
                        **run.parameters,
                    }
                )
    return pd.DataFrame(records)


def aggregate_metrics(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Average seeds/splits, retaining standard errors of the paired run gains."""
    if frame.empty:
        return frame
    values = [
        name
        for name in ("baseline", "selected", "final", "effect", "final_effect", "selected_epoch", "training_epochs")
        if name in frame
    ]
    grouped = frame.groupby(keys, dropna=False, sort=True)
    result = grouped[values].mean()
    for name in ("baseline", "selected", "effect"):
        result[name + "_se"] = grouped[name].sem()
    result["run_count"] = grouped.size()
    result["checkpoint_gap"] = result["effect"] - result["final_effect"]
    return result.reset_index()


def validate_study(results_root: Path, study: str, *, repo_root: Path) -> None:
    """Reject incomplete or obsolete run grids before exporting final figures."""
    from tfm_finetuning.experiments.configuration import expand_runs
    from tfm_finetuning.utils.config import read_yaml_mapping

    recipe = read_yaml_mapping(repo_root / "configs/experiments" / f"{study}.yaml")
    expected = {run["run_id"] for run in expand_runs(recipe, repo_root=repo_root)}
    observed = {run.metadata["run_id"] for run in read_runs(results_root, [study])}
    missing, extra = expected - observed, observed - expected
    if missing or extra:
        raise ValueError(
            f"{study}: {len(missing)} missing and {len(extra)} unexpected completed runs. "
            "Complete the current recipe and keep results from different recipes in separate results directories."
        )
