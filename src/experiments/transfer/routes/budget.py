"""Construct E2 nested Covertype target-data-budget routes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from experiments.transfer.routes.config import (
    checkpoint_dependency,
    dependency_data_recipe,
    dependency_training_recipe,
)
from experiments.transfer.routes.invariants import validate_budget_row_seed_policy
from experiments.transfer.spec import TransferRouteSpec, TransferStageSpec, ValidationSelector

StageFactory = Callable[..., TransferStageSpec]
TARGET_ONLY_ROUTE = "target_budget_target_only"
SATIMAGE_INITIALIZED_ROUTE = "target_budget_satimage_initialized"
BUDGET_ROUTES = (TARGET_ONLY_ROUTE, SATIMAGE_INITIALIZED_ROUTE)
E2_TARGET = "covertype"
E2_SOURCE_ROUTE = "sequential_satimage_then_target"


def build_budget_route(
    *,
    route_id: str,
    target: str,
    steps_per_dataset: int,
    target_rows: int | None,
    transfer: Mapping[str, Any],
    common: Mapping[str, Any],
    stage_factory: StageFactory,
) -> TransferRouteSpec:
    """Build one E2 producer with exact nested target rows."""
    if target != E2_TARGET:
        raise ValueError(f"E2 is frozen to target {E2_TARGET!r}, got {target!r}.")
    if route_id not in BUDGET_ROUTES:
        raise ValueError(f"Unsupported E2 target-budget route: {route_id!r}.")
    if target_rows is None:
        raise ValueError("dataset_transfer.target_train_rows is required for E2 routes.")
    validate_budget_row_seed_policy(
        transfer, resolved_seed=int(common["row_subset_seed"]), run_seed=int(common["seed"])
    )
    dependency = None
    total_steps = 2 * steps_per_dataset if route_id == TARGET_ONLY_ROUTE else steps_per_dataset
    view_boundaries = (steps_per_dataset,) if route_id == TARGET_ONLY_ROUTE else ()
    if route_id == SATIMAGE_INITIALIZED_ROUTE:
        dependency = checkpoint_dependency(
            transfer,
            family="dataset_transfer",
            route_id=E2_SOURCE_ROUTE,
            checkpoint_role="reusable_boundary",
            checkpoint_origin=target,
            producer_training_recipe=dependency_training_recipe(common["training_recipe"], transfer=transfer),
            producer_data_recipe=dependency_data_recipe(common["data_recipe"], transfer=transfer),
            producer_context_policy=common["context_policy"],
        )
    stage = stage_factory(
        name="target_budget",
        role="terminal",
        mode="single_dataset",
        datasets=(target,),
        row_budget=target_rows,
        row_budget_policy="exact",
        updates=total_steps,
        selector=ValidationSelector("target_validation", (target,)),
        common=common,
        view_boundaries=view_boundaries,
    )
    return TransferRouteSpec(
        route_id=route_id,
        family="dataset_transfer_data_budget",
        target_dataset=target,
        execution_mode="train",
        stages=(stage,),
        dependency=dependency,
    )


__all__ = [
    "BUDGET_ROUTES",
    "E2_SOURCE_ROUTE",
    "E2_TARGET",
    "SATIMAGE_INITIALIZED_ROUTE",
    "TARGET_ONLY_ROUTE",
    "build_budget_route",
]
