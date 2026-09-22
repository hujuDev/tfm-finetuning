"""Build the balanced Section 3 target-budget extension.

The extension reuses the already selected classification source boundaries and
adds one Houses regression boundary. Keeping that mapping here makes each
source-initialized target curve reconstruct its exact producer DAG.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .route_config import checkpoint_dependency, dependency_data_recipe, dependency_training_recipe
from .route_invariants import validate_budget_row_seed_policy
from .spec import TransferRouteSpec, TransferStageSpec, ValidationSelector

StageFactory = Callable[..., TransferStageSpec]
TARGET_ONLY_ROUTE = "balanced_budget_target_only"
SOURCE_INITIALIZED_ROUTE = "balanced_budget_source_initialized"
SOURCE_CHECKPOINT_ROUTE = "source_only_checkpoint"


@dataclass(frozen=True)
class SourceBoundary:
    """Exact producer boundary used to initialize one target family."""

    label: str
    family: str
    route_id: str
    checkpoint_role: str
    checkpoint_origin: str


TARGET_SOURCE_BOUNDARIES = {
    "sdss17": SourceBoundary(
        "satimage", "dataset_transfer", "sequential_satimage_then_target", "reusable_boundary", "covertype"
    ),
    "amazon_employee_access": SourceBoundary(
        "bank_heloc", "dataset_transfer", "pooled_sources_then_target", "reusable_boundary", "credit-g"
    ),
    "diabetes130us": SourceBoundary(
        "bank_heloc", "dataset_transfer", "pooled_sources_then_target", "reusable_boundary", "credit-g"
    ),
    "physiochemical_protein": SourceBoundary(
        "houses", "dataset_transfer_source_checkpoint", SOURCE_CHECKPOINT_ROUTE, "terminal", "houses"
    ),
    "food_delivery_time": SourceBoundary(
        "houses", "dataset_transfer_source_checkpoint", SOURCE_CHECKPOINT_ROUTE, "terminal", "houses"
    ),
}


def build_balanced_budget_route(
    *,
    route_id: str,
    target: str,
    steps_per_dataset: int,
    target_rows: int | None,
    transfer: Mapping[str, Any],
    common: Mapping[str, Any],
    stage_factory: StageFactory,
) -> TransferRouteSpec:
    """Build one new target-only or source-initialized budget producer."""
    try:
        source = TARGET_SOURCE_BOUNDARIES[target]
    except KeyError as exc:
        raise ValueError(f"No balanced-budget source boundary is declared for {target!r}.") from exc
    if route_id not in {TARGET_ONLY_ROUTE, SOURCE_INITIALIZED_ROUTE}:
        raise ValueError(f"Unsupported balanced-budget route: {route_id!r}.")
    if target_rows is None:
        raise ValueError("dataset_transfer.target_train_rows is required for balanced-budget routes.")
    validate_budget_row_seed_policy(
        transfer, resolved_seed=int(common["row_subset_seed"]), run_seed=int(common["seed"])
    )
    row_budget_policy = str(transfer.get("target_row_budget_policy") or "exact").strip()
    if row_budget_policy not in {"exact", "cap"}:
        raise ValueError("dataset_transfer.target_row_budget_policy must be exact or cap.")
    target_only = route_id == TARGET_ONLY_ROUTE
    dependency = None
    if not target_only:
        dependency = checkpoint_dependency(
            transfer,
            family=source.family,
            route_id=source.route_id,
            checkpoint_role=source.checkpoint_role,
            checkpoint_origin=source.checkpoint_origin,
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
        row_budget_policy=row_budget_policy,
        updates=2 * steps_per_dataset if target_only else steps_per_dataset,
        selector=ValidationSelector("target_validation", (target,)),
        common=common,
        view_boundaries=(steps_per_dataset,) if target_only else (),
    )
    return TransferRouteSpec(
        route_id=route_id,
        family="dataset_transfer_balanced_budget",
        target_dataset=target,
        execution_mode="train",
        stages=(stage,),
        dependency=dependency,
    )


def build_source_checkpoint_route(
    *,
    route_id: str,
    target: str,
    steps_per_dataset: int,
    row_budget: int | None,
    common: Mapping[str, Any],
    stage_factory: StageFactory,
) -> TransferRouteSpec:
    """Build the reusable Houses regression source checkpoint producer."""
    if route_id != SOURCE_CHECKPOINT_ROUTE or target != "houses":
        raise ValueError(f"Unsupported source checkpoint route {route_id!r} for {target!r}.")
    stage = stage_factory(
        name="source_only",
        role="terminal",
        mode="single_dataset",
        datasets=(target,),
        row_budget=row_budget,
        row_budget_policy="cap",
        updates=steps_per_dataset,
        selector=ValidationSelector("target_validation", (target,)),
        common=common,
    )
    return TransferRouteSpec(
        route_id=route_id,
        family="dataset_transfer_source_checkpoint",
        target_dataset=target,
        execution_mode="train",
        stages=(stage,),
    )


__all__ = [
    "SOURCE_CHECKPOINT_ROUTE",
    "SOURCE_INITIALIZED_ROUTE",
    "TARGET_ONLY_ROUTE",
    "TARGET_SOURCE_BOUNDARIES",
    "build_balanced_budget_route",
    "build_source_checkpoint_route",
]
