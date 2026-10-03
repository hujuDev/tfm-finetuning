"""Train explicit transfer routes or evaluate existing weights without updates.

Example: ``run_dataset_transfer(config, Path("results/transfer/run-1"))``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from tfm_finetuning.experiments.transfer.dependencies import resolve_checkpoint_dependency
from tfm_finetuning.experiments.transfer.routes.resolve import resolve_transfer_route
from tfm_finetuning.experiments.transfer.runner.checkpoint_reporting import run_checkpoint_evaluation
from tfm_finetuning.experiments.transfer.runner.data import TransferDatasetLoader
from tfm_finetuning.experiments.transfer.runner.executor import execute_route
from tfm_finetuning.experiments.transfer.runner.reporting import training_report
from tfm_finetuning.experiments.transfer.runner.training import build_stage_trainer
from tfm_finetuning.finetuning.reporting.report import finish_run
from tfm_finetuning.finetuning.run.setup import initialize_foundation_run


def run_dataset_transfer(config: dict[str, Any], output_path: Path, *, force: bool = False) -> dict[str, Any]:
    """Execute a fresh run, requiring completed producers before any consumer."""
    started = time.monotonic()
    setup = initialize_foundation_run(config, output_path, force=force)
    try:
        # Resolve routes after split-specific seed adjustments in run setup.
        route = resolve_transfer_route(setup.config)
        root = Path(setup.config.get("results_root", setup.paths.results_root))
        if not root.is_absolute():
            root = setup.paths.repo_root / root
        checkpoint = resolve_checkpoint_dependency(
            setup.config, route, results_root=root, repo_root=setup.paths.repo_root
        )
        loader = TransferDatasetLoader(setup)
        if route.execution_mode == "checkpoint_eval":
            if checkpoint is None:
                raise ValueError("Checkpoint evaluation requires existing producer weights.")
            report = run_checkpoint_evaluation(setup, route=route, checkpoint_path=checkpoint, dataset_loader=loader)
        else:
            trainer = build_stage_trainer(setup, route=route, dataset_loader=loader)
            stages = execute_route(
                route, trainer=trainer, output=setup.output / "stages", initial_checkpoint=checkpoint
            )
            report = training_report(route, stages)
        report.update(
            {
                "pipeline_mode": "foundation_model_dataset_transfer",
                "foundation_model": {
                    "model_id": setup.spec.model_id,
                    "display_name": setup.spec.display_name,
                    "preprocessing_protocol": setup.preprocessing_protocol,
                    "resolved_model_params": setup.adapter.resolved_model_params(setup.common_config),
                },
                "split": {
                    "name": setup.split_name,
                    "base_seed": setup.base_seed,
                    "resolved_seed": setup.resolved_seed,
                    "seed_delta": setup.seed_delta,
                },
                "score_metric_source": "foundation_model.metric",
                "runtime_seconds": time.monotonic() - started,
            }
        )
        report["runtime"] = {"total_seconds": report["runtime_seconds"], "phases": loader.runtime_phases}
        finish_run(setup, report)
        return report
    except BaseException:
        if setup.wandb_run is not None:
            setup.wandb_run.finish(exit_code=1)
        raise
