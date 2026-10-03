"""Prepare matched target contexts and report frozen-checkpoint evaluations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.dataset_transfer.evaluation import evaluate_frozen_checkpoint_bundle
from experiments.dataset_transfer.natural_domain_context import build_route_matched_evaluation_data
from experiments.dataset_transfer.runner_data import TransferDatasetLoader
from experiments.dataset_transfer.spec import TransferRouteSpec
from finetuning.preprocessing import (
    _configure_adapter_feature_schema,
    _configure_adapter_preprocessing_protocol,
)
from finetuning.resource_policy import _apply_estimator_fallback_policy, _configure_adapter_common_config
from finetuning.run_setup import FoundationRunSetup


def run_checkpoint_evaluation(
    setup: FoundationRunSetup,
    *,
    route: TransferRouteSpec,
    checkpoint_path: Path,
    dataset_loader: TransferDatasetLoader,
) -> dict[str, Any]:
    """Evaluate the same frozen weights and base model on identical contexts."""
    data_by_target = {
        target: build_route_matched_evaluation_data(route, target=target, dataset_loader=dataset_loader)
        for target in route.evaluation_targets
    }
    adapters, common_configs = {}, {}
    for target, data in data_by_target.items():
        adapter = setup.spec.adapter_cls(setup.spec)
        _configure_adapter_preprocessing_protocol(adapter, setup.preprocessing_protocol)
        _configure_adapter_feature_schema(adapter, data)
        common, _ = _apply_estimator_fallback_policy(
            adapter=adapter, data=data, spec=setup.spec, config=setup.config, common_config=setup.common_config
        )
        _configure_adapter_common_config(adapter, common)
        adapters[target], common_configs[target] = adapter, common
    evaluations = evaluate_frozen_checkpoint_bundle(
        route,
        checkpoint_path=checkpoint_path,
        target_data=data_by_target,
        adapter_factory=adapters.__getitem__,
        common_config=common_configs,
        seed=int(setup.resolved_seed),
        device=setup.device,
    )
    primary = evaluations[0]
    return {
        "dataset_transfer": {
            "family": route.family,
            "route": route.route_id,
            "execution_mode": "checkpoint_eval",
            "context_policy": route.context_policy,
            "checkpoint_origin": route.dependency.checkpoint_origin,
            "checkpoint_route": route.dependency.route_id,
            "gradient_updates": 0,
            "context_groups": {
                target: data.metadata.get("route_matched_context", {}).get("context_groups", [target])
                for target, data in data_by_target.items()
            },
            "context_rows": {target: len(data.test_context_X) for target, data in data_by_target.items()},
        },
        "score_metric": primary["score_metric"],
        "zero_shot_metrics": primary["zero_shot_metrics"],
        "selected_checkpoint_metrics": primary["selected_checkpoint_metrics"],
        "checkpoint_path": str(checkpoint_path),
        "evaluations": evaluations,
        "train_history": [],
    }
