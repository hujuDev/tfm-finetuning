"""Evaluate frozen transfer checkpoints using target ICL without gradient updates."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from experiments.transfer.runner.executor import require_checkpoint
from experiments.transfer.spec import TransferRouteSpec
from finetuning.evaluation.core import _evaluate_checkpoint, _evaluate_zero_shot


def evaluate_frozen_checkpoint_bundle(
    route: TransferRouteSpec,
    *,
    checkpoint_path: Path,
    target_data: Mapping[str, Any],
    adapter_factory: Callable[[str], Any],
    common_config: Any,
    seed: int,
    device: str | None,
) -> list[dict[str, Any]]:
    """Return matched base/selected checkpoint scores on each target's test rows.

    Only estimator construction, context fitting, and prediction are reachable.
    There is deliberately no trainer, optimizer or checkpoint selection here.
    """
    if route.execution_mode != "checkpoint_eval":
        raise ValueError("Frozen evaluation requires a checkpoint_eval route.")
    checkpoint = require_checkpoint(checkpoint_path)
    if set(route.evaluation_targets).difference(target_data):
        raise ValueError("Data are missing for a declared evaluation target.")
    import torch

    evaluations = []
    with torch.inference_mode():
        for target in route.evaluation_targets:
            adapter = adapter_factory(target)
            data = target_data[target]
            common = common_config[target] if isinstance(common_config, Mapping) else common_config
            arguments = dict(
                adapter=adapter, data=data, seed=seed, device=device, common_config=common, splits=("test",)
            )
            baseline = _evaluate_zero_shot(**arguments, phase="transfer_base_icl", eval_source="base_pure_icl")
            metrics = _evaluate_checkpoint(
                **arguments,
                checkpoint_path=checkpoint,
                phase="transfer_frozen_checkpoint_icl",
                eval_source="checkpoint_pure_icl",
            )
            evaluations.append(
                {
                    "target": target,
                    "route": route.dependency.route_id,
                    "checkpoint_path": str(checkpoint),
                    "score_metric": data.metric_name,
                    "zero_shot_metrics": baseline,
                    "selected_checkpoint_metrics": metrics,
                    "axes": {"contextPolicy": route.context_policy},
                }
            )
    return evaluations
