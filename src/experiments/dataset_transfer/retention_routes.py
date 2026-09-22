"""Bank/Covertype sequential and pooled retained-capability experiments."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .spec import TransferRouteSpec, TransferStageSpec, ValidationSelector

StageFactory = Callable[..., TransferStageSpec]
BANK = "Bank_Customer_Churn"
COVERTYPE = "covertype"
BANK_THEN_COVERTYPE = "sequential_bank_then_covertype"
COVERTYPE_THEN_BANK = "sequential_covertype_then_bank"
POOLED_BANK_COVERTYPE = "pooled_bank_covertype"
RETENTION_ROUTES = (BANK_THEN_COVERTYPE, COVERTYPE_THEN_BANK, POOLED_BANK_COVERTYPE)


def build_retention_route(
    *,
    route_id: str,
    target: str,
    steps_per_dataset: int,
    row_budget: int | None,
    common: Mapping[str, Any],
    stage_factory: StageFactory,
) -> TransferRouteSpec:
    """Build a cross-domain producer with independently selected stage checkpoints."""
    expected_target = {
        BANK_THEN_COVERTYPE: COVERTYPE,
        COVERTYPE_THEN_BANK: BANK,
        POOLED_BANK_COVERTYPE: COVERTYPE,
    }.get(route_id)
    if expected_target is None:
        raise ValueError(f"Unsupported learned-retention route: {route_id!r}.")
    if target != expected_target:
        raise ValueError(f"Route {route_id!r} requires target {expected_target!r}, got {target!r}.")
    if row_budget is None:
        raise ValueError("Learned-retention routes require a numeric row cap.")
    if route_id == POOLED_BANK_COVERTYPE:
        stages = (
            stage_factory(
                name="bank_covertype_pool",
                role="terminal",
                mode="pooled_episodic",
                datasets=(BANK, COVERTYPE),
                row_budget=row_budget,
                updates=2 * steps_per_dataset,
                selector=ValidationSelector("equal_mean_validation_improvement", (BANK, COVERTYPE)),
                common=common,
            ),
        )
    elif route_id == BANK_THEN_COVERTYPE:
        stages = (
            stage_factory(
                name="source_1",
                role="intermediate",
                mode="single_dataset",
                datasets=(BANK,),
                row_budget=row_budget,
                updates=steps_per_dataset,
                selector=ValidationSelector("target_validation", (BANK,)),
                common=common,
            ),
            stage_factory(
                name="target",
                role="terminal",
                mode="single_dataset",
                datasets=(COVERTYPE,),
                row_budget=row_budget,
                updates=steps_per_dataset,
                selector=ValidationSelector("target_validation", (COVERTYPE,)),
                common=common,
                parent="source_1",
            ),
        )
    else:
        stages = (
            stage_factory(
                name="primary_source",
                role="reusable_boundary",
                mode="single_dataset",
                datasets=(COVERTYPE,),
                row_budget=row_budget,
                updates=steps_per_dataset,
                selector=ValidationSelector("target_validation", (COVERTYPE,)),
                common=common,
            ),
            stage_factory(
                name="target",
                role="terminal",
                mode="single_dataset",
                datasets=(BANK,),
                row_budget=row_budget,
                updates=steps_per_dataset,
                selector=ValidationSelector("target_validation", (BANK,)),
                common=common,
                parent="primary_source",
            ),
        )
    return TransferRouteSpec(
        route_id=route_id,
        family="dataset_transfer_learned_retention",
        target_dataset=target,
        execution_mode="train",
        stages=stages,
    )


__all__ = [
    "BANK",
    "BANK_THEN_COVERTYPE",
    "COVERTYPE",
    "COVERTYPE_THEN_BANK",
    "POOLED_BANK_COVERTYPE",
    "RETENTION_ROUTES",
    "build_retention_route",
]
