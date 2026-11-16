"""Synthetic run directories in the layout read by the results loader."""

from __future__ import annotations

import json
from pathlib import Path


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
