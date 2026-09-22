"""One local report and optional native W&B logging for each fresh run."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from finetuning.wandb_logging import (
    experiment_parameters,
    flatten_core_metrics_for_wandb,
    flatten_report_metrics_for_wandb,
    flatten_score_metrics_for_wandb,
)


def _finish_artifact(*, output: Path, report: Mapping[str, Any]) -> None:
    """Atomically publish a completed report, e.g. ``_finish_artifact(output=path, report=result)``."""
    temporary = output / "finetune_report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(output / "finetune_report.json")


def _init_wandb_run(config: dict[str, Any], *, output: Path) -> Any | None:
    """Start W&B when configured; its standard environment variables also apply."""
    settings = dict(config.get("wandb", {}))
    mode = os.environ.get("WANDB_MODE", settings.get("mode", "online"))
    if not settings or mode == "disabled":
        return None
    import wandb

    directory = Path(os.environ.get("WANDB_DIR", settings.get("dir", str(output)))).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    allowed = ("project", "entity", "name", "group", "tags", "notes")
    kwargs = {key: settings[key] for key in allowed if settings.get(key) is not None}
    for key in ("project", "entity"):
        if os.environ.get(f"WANDB_{key.upper()}"):
            kwargs[key] = os.environ[f"WANDB_{key.upper()}"]
    kwargs.setdefault("name", output.name)
    run = wandb.init(
        **kwargs,
        mode=mode,
        dir=str(directory),
        config={key: value for key, value in config.items() if key != "wandb"},
    )
    run.define_metric("train/epoch")
    run.define_metric("train/*", step_metric="train/epoch")
    run.define_metric("core/train/*", step_metric="train/epoch")
    return run


def _log_wandb_summary(
    wandb_run: Any | None,
    *,
    report: Mapping[str, Any],
    config: Mapping[str, Any],
    train_history: list[dict[str, Any]],
    periodic_rows: list[dict[str, Any]],
    output: Path,
    validation_rows: list[dict[str, Any]] | None = None,
) -> None:
    """Log canonical baseline/selected/final summaries and epoch trajectories."""
    if wandb_run is None:
        return
    wandb_run.summary.update(
        {
            **flatten_report_metrics_for_wandb(report),
            **experiment_parameters(config),
            "score/metric": report.get("score_metric"),
            "artifact/output_dir": str(output),
            "run/runtime_seconds": report.get("runtime_seconds"),
        }
    )
    timeline: dict[int, dict[str, Any]] = {}
    for row in train_history:
        epoch = int(float(row.get("train/epoch", 0)))
        timeline.setdefault(epoch, {}).update({key: value for key, value in row.items() if key != "step"})
    for row in [*(validation_rows or []), *periodic_rows]:
        epoch = int(row["epoch"])
        metrics = row.get("metrics", {})
        timeline.setdefault(epoch, {}).update(
            {
                "train/epoch": epoch,
                **flatten_score_metrics_for_wandb(metrics, prefix="train/"),
                **flatten_core_metrics_for_wandb(metrics, prefix="core/train/"),
            }
        )
    for epoch, row in sorted(timeline.items()):
        wandb_run.log({**row, "train/epoch": epoch})


def finish_run(setup: Any, report: dict[str, Any], *, validation_rows=None, periodic_rows=None) -> dict[str, Any]:
    """Save scientific results before synchronizing optional tracking; return the same report."""
    report.update(schema_version=1, config=setup.config)
    if setup.wandb_run is not None and getattr(setup.wandb_run, "url", None):
        report["wandb"] = {"url": setup.wandb_run.url}
    _finish_artifact(output=setup.output, report=report)
    if setup.wandb_run is not None:
        try:
            _log_wandb_summary(
                setup.wandb_run,
                report=report,
                config=setup.config,
                output=setup.output,
                train_history=report.get("train_history", []),
                validation_rows=validation_rows or [],
                periodic_rows=periodic_rows or [],
            )
        finally:
            setup.wandb_run.finish()
    return report
