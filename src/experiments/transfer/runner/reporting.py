"""Produce one current report from completed transfer stages."""

from __future__ import annotations

from typing import Any

from experiments.transfer.runner.executor import StageResult
from experiments.transfer.spec import TransferRouteSpec


def training_report(route: TransferRouteSpec, results: tuple[StageResult, ...]) -> dict[str, Any]:
    """Record selected weights and shorter-budget controls without duplicate runs."""
    terminal = results[-1]
    stage = terminal.stage
    raw = terminal.production.report
    baseline = raw.get("zero-shot-metrics", {})
    metrics = raw.get("metrics", {})
    budget_metrics = {}
    for boundary in stage.view_boundaries:
        payload = raw["checkpoint_view_metrics"][str(boundary)]
        budget_metrics[str(boundary)] = {
            "selected_checkpoint_metrics": payload["metrics"],
            "zero_shot_metrics": payload["baseline_metrics"],
            "selection": raw["checkpoint_view_selection"][str(boundary)],
            "checkpoint_path": str(terminal.production.view_checkpoints[boundary]),
        }
    budget_metrics[str(stage.update_budget)] = {
        "selected_checkpoint_metrics": metrics,
        "zero_shot_metrics": baseline,
        "selection": raw["checkpoint_selection"],
        "checkpoint_path": str(terminal.production.checkpoint_path),
    }
    # Training adapters expose a common internal result; public reports use one schema.
    report = {
        key: value
        for key, value in raw.items()
        if key
        not in {
            "metrics",
            "zero-shot-metrics",
            "checkpoint_view_metrics",
            "checkpoint_view_selection",
        }
    }
    report.update(
        {
            "dataset_transfer": {
                "family": route.family,
                "route": route.route_id,
                "execution_mode": "train",
                "context_policy": route.context_policy,
                "target": route.target_dataset,
            },
            "zero_shot_metrics": baseline,
            "selected_checkpoint_metrics": metrics,
            "target_update_metrics": budget_metrics,
            "stage_checkpoints": {result.stage.name: str(result.production.checkpoint_path) for result in results},
            "checkpoint_path": str(terminal.production.checkpoint_path),
            "stages": [
                {
                    "name": result.stage.name,
                    "role": result.stage.role,
                    "datasets": list(result.stage.datasets),
                    "updates": result.stage.update_budget,
                    "selection": result.production.report["checkpoint_selection"],
                }
                for result in results
            ],
        }
    )
    evaluations = []
    is_budget = route.family in {"dataset_transfer_data_budget", "dataset_transfer_balanced_budget"}
    is_endpoint = route.family == "dataset_transfer" and route.target_dataset == "covertype"
    if is_budget or is_endpoint:
        for budget, payload in budget_metrics.items():
            target_only = route.route_id in {
                "target_compute_matched",
                "target_budget_target_only",
                "balanced_budget_target_only",
            }
            if target_only:
                condition = f"target_only_{budget}_steps"
            elif route.route_id in {"target_budget_satimage_initialized", "sequential_satimage_then_target"}:
                condition = f"satimage_initialized_{budget}_target_steps"
            elif route.route_id == "balanced_budget_source_initialized":
                condition = f"source_initialized_{budget}_target_steps"
            else:
                continue
            evaluations.append(
                {
                    "target": route.target_dataset,
                    "route": route.route_id,
                    **payload,
                    "training_steps": int(budget),
                    "axes": {"targetTrainRows": stage.row_budget_per_dataset, "targetBudgetCondition": condition},
                }
            )
    if evaluations:
        report["evaluations"] = evaluations
    return report
