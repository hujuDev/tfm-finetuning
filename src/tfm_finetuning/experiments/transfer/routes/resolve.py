"""Resolve maintained Section 3 route IDs into explicit stage DAGs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tfm_finetuning.experiments.transfer.routes.balanced_budget import (
    build_balanced_budget_route,
    build_source_checkpoint_route,
)
from tfm_finetuning.experiments.transfer.routes.budget import build_budget_route
from tfm_finetuning.experiments.transfer.routes.config import checkpoint_dependency as _checkpoint_dependency
from tfm_finetuning.experiments.transfer.routes.config import clean_dataset as _clean_dataset
from tfm_finetuning.experiments.transfer.routes.config import decode_transfer_route_config
from tfm_finetuning.experiments.transfer.routes.config import dependency_data_recipe as _dependency_data_recipe
from tfm_finetuning.experiments.transfer.routes.config import dependency_training_recipe as _dependency_training_recipe
from tfm_finetuning.experiments.transfer.routes.invariants import validate_checkpoint_boundaries
from tfm_finetuning.experiments.transfer.routes.natural_domain import build_natural_domain_route
from tfm_finetuning.experiments.transfer.routes.retention import build_retention_route
from tfm_finetuning.experiments.transfer.routes.schedule import build_schedule_route, build_target_control_route
from tfm_finetuning.experiments.transfer.spec import (
    TransferDataRecipe,
    TransferRouteSpec,
    TransferStageSpec,
    TransferTrainingRecipe,
    ValidationSelector,
)


def resolve_transfer_route(config: Mapping[str, Any], *, target_dataset: str | None = None) -> TransferRouteSpec:
    """Resolve a checked-in route/config combination.

    `updates_per_dataset` is mandatory because the protocol uses explicit
    optimizer-step budgets rather than deriving a budget from epochs. Requiring
    it prevents an epoch-based implementation from silently changing the
    compute-matched comparison.

    Example:
        >>> route = resolve_transfer_route({
        ...   "data": {"path": "data/tabarena/credit-g", "split": "fold0_repeat0_sample0"},
        ...   "foundation_model": {"model_id": "tabpfn_v2_5", "classifier_checkpoint": "base.ckpt",
        ...       "test_context_includes_validation_data": True},
        ...   "dataset_transfer": {"enabled": True, "family": "dataset_transfer",
        ...       "route_id": "pooled_sources_then_target", "updates_per_dataset": 5,
        ...       "context_policy": "target_train_plus_validation"},
        ... })
        >>> [(stage.name, stage.update_budget) for stage in route.stages]
        [('source_pool', 10), ('target', 5)]
    """
    decoded = decode_transfer_route_config(config, target_dataset=target_dataset)
    if decoded.execution_mode == "checkpoint_eval":
        return _evaluation_route(
            family=decoded.family,
            route_id=decoded.route_id,
            target=decoded.target,
            transfer=decoded.transfer,
            producer_training_recipe=_dependency_training_recipe(decoded.training_recipe, transfer=decoded.transfer),
            producer_data_recipe=_dependency_data_recipe(decoded.data_recipe, transfer=decoded.transfer),
            context_policy=decoded.context_policy,
            producer_context_policy=str(decoded.transfer.get("dependency_context_policy") or decoded.context_policy),
        )
    if decoded.updates_per_dataset is None or decoded.stage_common is None:
        raise RuntimeError("Training route decoding produced no stage recipe.")
    if decoded.family == "dataset_transfer":
        route = build_schedule_route(
            route_id=decoded.route_id,
            target=decoded.target,
            steps_per_dataset=decoded.updates_per_dataset,
            row_budget=decoded.row_budget_per_dataset,
            common=decoded.stage_common,
            stage_factory=_stage,
        )
        validate_checkpoint_boundaries(route.stages)
        return route
    if decoded.family == "dataset_transfer_data_budget":
        route = build_budget_route(
            route_id=decoded.route_id,
            target=decoded.target,
            steps_per_dataset=decoded.updates_per_dataset,
            target_rows=decoded.target_train_rows,
            transfer=decoded.transfer,
            common=decoded.stage_common,
            stage_factory=_stage,
        )
        validate_checkpoint_boundaries(route.stages)
        return route
    if decoded.family == "dataset_transfer_balanced_budget":
        route = build_balanced_budget_route(
            route_id=decoded.route_id,
            target=decoded.target,
            steps_per_dataset=decoded.updates_per_dataset,
            target_rows=decoded.target_train_rows,
            transfer=decoded.transfer,
            common=decoded.stage_common,
            stage_factory=_stage,
        )
        validate_checkpoint_boundaries(route.stages)
        return route
    if decoded.family == "dataset_transfer_source_checkpoint":
        return build_source_checkpoint_route(
            route_id=decoded.route_id,
            target=decoded.target,
            steps_per_dataset=decoded.updates_per_dataset,
            row_budget=decoded.row_budget_per_dataset,
            common=decoded.stage_common,
            stage_factory=_stage,
        )
    if decoded.family == "dataset_transfer_target_controls":
        route = build_target_control_route(
            route_id=decoded.route_id,
            target=decoded.target,
            steps_per_dataset=decoded.updates_per_dataset,
            row_budget=decoded.row_budget_per_dataset,
            common=decoded.stage_common,
            stage_factory=_stage,
        )
        validate_checkpoint_boundaries(route.stages)
        return route
    if decoded.family == "dataset_transfer_natural_domain":
        route = build_natural_domain_route(
            route_id=decoded.route_id,
            target=decoded.target,
            steps_per_dataset=decoded.updates_per_dataset,
            row_budget=decoded.row_budget_per_dataset,
            common=decoded.stage_common,
            stage_factory=_stage,
        )
        validate_checkpoint_boundaries(route.stages)
        return route
    if decoded.family == "dataset_transfer_learned_retention":
        route = build_retention_route(
            route_id=decoded.route_id,
            target=decoded.target,
            steps_per_dataset=decoded.updates_per_dataset,
            row_budget=decoded.row_budget_per_dataset,
            common=decoded.stage_common,
            stage_factory=_stage,
        )
        validate_checkpoint_boundaries(route.stages)
        return route
    raise ValueError(f"Unsupported training transfer family: {decoded.family!r}")


def _evaluation_route(
    *,
    family: str,
    route_id: str,
    target: str,
    transfer: Mapping[str, Any],
    producer_training_recipe: TransferTrainingRecipe,
    producer_data_recipe: TransferDataRecipe,
    context_policy: str,
    producer_context_policy: str,
) -> TransferRouteSpec:
    raw_targets = transfer.get("evaluation_targets") or transfer.get("evaluation_target") or target
    values = raw_targets if isinstance(raw_targets, list | tuple) else [raw_targets]
    targets = tuple((_clean_dataset(value) for value in values))
    checkpoint_origin = _clean_dataset(transfer.get("checkpoint_origin") or target)
    checkpoint_route = str(transfer.get("checkpoint_route") or route_id).strip()
    producer_family = "dataset_transfer"
    if family == "dataset_transfer_natural_domain_context_eval":
        role = "terminal"
        producer_family = "dataset_transfer_natural_domain"
    elif family == "dataset_transfer_source_only_eval":
        role = str(transfer.get("checkpoint_role") or "reusable_boundary").strip()
        producer_family = str(transfer.get("checkpoint_family") or "dataset_transfer").strip()
    else:
        raise ValueError(f"Unsupported checkpoint evaluation family: {family!r}")
    return TransferRouteSpec(
        route_id=route_id,
        family=family,
        target_dataset=target,
        execution_mode="checkpoint_eval",
        evaluation_targets=targets,
        context_policy=context_policy,
        dependency=_checkpoint_dependency(
            transfer,
            family=producer_family,
            route_id=checkpoint_route,
            checkpoint_role=role,
            checkpoint_origin=checkpoint_origin,
            producer_training_recipe=producer_training_recipe,
            producer_data_recipe=producer_data_recipe,
            producer_context_policy=producer_context_policy,
        ),
    )


def _stage(
    *,
    name: str,
    role: str,
    mode: str,
    datasets: tuple[str, ...],
    row_budget: int | None,
    updates: int,
    selector: ValidationSelector,
    common: Mapping[str, Any],
    row_budget_policy: str = "cap",
    parent: str | None = None,
    view_boundaries: tuple[int, ...] = (),
) -> TransferStageSpec:
    return TransferStageSpec(
        name=name,
        role=role,
        training_mode=mode,
        datasets=datasets,
        row_budget_per_dataset=row_budget,
        row_budget_policy=row_budget_policy,
        update_budget=updates,
        sampling_weights=tuple((1.0 for _ in datasets)),
        selector=selector,
        parent_stage=parent,
        view_boundaries=view_boundaries,
        **dict(common),
    )
