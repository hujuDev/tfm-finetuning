"""Synthetic run directories in the layout read by the results loader."""

from __future__ import annotations

import json
from pathlib import Path

from tfm_finetuning.plots.common.model_display import PAPER_MODELS

TASKS = (("binary", "roc_auc"), ("multiclass", "log_loss"), ("regression", "rmse"))
STUDIES = ("overview_baseline", "preprint_tabpfn_v3")


def write_run(root: Path, *, study, dataset, model, seed, metric, baseline, selected, preprocessing="shared") -> None:
    run_id = f"{dataset}-{model}-{preprocessing}-{seed}"
    directory = root / study / run_id
    directory.mkdir(parents=True)
    config = {"finetune": {"learning_rate": 1e-5, "weight_decay": 0.01, "context_size": 10000}}
    metadata = {
        "study": study,
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "fold": seed,
        "repeat": 0,
        "sample": 0,
        "run_id": run_id,
        "preprocessing": preprocessing,
        "parameters": {"preprocessing": preprocessing},
        "config": config,
        "status": "complete",
    }
    report = {
        "config": config,
        "score_metric": metric,
        "zero_shot_metrics": {"val": {metric: baseline}, "test": {metric: baseline}},
        "selected_checkpoint_metrics": {"val": {metric: selected}, "test": {metric: selected}},
        "final_checkpoint_metrics": {"val": {metric: selected}, "test": {metric: selected}},
        "evaluation": {"selected_epoch": 3, "final_epoch": 100},
        "runtime": {
            "total_seconds": 600,
            "phases": [
                {
                    "phase": "fit_finetuned",
                    "seconds": 590,
                    "gpu_memory_peak_allocated_mib": 2048,
                    "gpu_memory_peak_reserved_mib": 4096,
                }
            ],
        },
    }
    (directory / "run.json").write_text(json.dumps(metadata))
    (directory / "finetune_report.json").write_text(json.dumps(report))


def write_results(root: Path, *, skip: set[tuple[str, str]] = frozenset(), datasets_per_task: int = 4) -> None:
    """Higher model index = better ICL; every SFT run improves by a dataset-dependent amount."""
    for task, metric in TASKS:
        for index in range(datasets_per_task):
            dataset = f"{task}_{index}"
            for model_index, model in enumerate(PAPER_MODELS):
                if (dataset, model) in skip:
                    continue
                # The new models live in their own study, as in the real runs.
                study = "preprint_tabpfn_v3" if model.startswith(("tabpfnv3",)) else "overview_baseline"
                for preprocessing in ("shared", "native"):
                    for seed in (0, 1, 2):
                        if metric == "roc_auc":
                            baseline = 0.70 + 0.01 * model_index + 0.002 * index
                            selected = baseline + 0.004 * (index + 1)
                        else:
                            baseline = 4.0 - 0.1 * model_index + 0.05 * index
                            selected = baseline / (1.02 + 0.01 * index)
                        write_run(
                            root,
                            study=study,
                            dataset=dataset,
                            model=model,
                            seed=seed,
                            metric=metric,
                            baseline=baseline,
                            selected=selected,
                            preprocessing=preprocessing,
                        )


STUDIES = ("overview_baseline", "preprint_tabpfn_v3")


def write_config(tmp_path: Path, results: Path) -> Path:
    config = tmp_path / "preprint.toml"
    config.write_text(
        f"""
[input]
results_root = "{results}"
studies = ["overview_baseline", "preprint_tabpfn_v3"]

[analysis]
bootstrap_seed = 1
bootstrap_resamples = 200
default_preprocessing = "shared"
default_context_size = 10000
excluded_datasets = ["airlines", "miniboone"]
"""
    )
    return config
