"""Construct the maintained E1 schedule routes.

The route catalog makes the related-source hierarchy explicit.  Finance keeps
its two-source design.  Remote sensing uses one primary related source and
keeps Segment only in named dilution/order ablations.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from tfm_finetuning.experiments.transfer.spec import TransferRouteSpec, TransferStageSpec, ValidationSelector

StageFactory = Callable[..., TransferStageSpec]
FINANCE_ROUTES = (
    "target_compute_matched",
    "pooled_target_sources",
    "pooled_sources_then_target",
    "sequential_bank_then_heloc_then_target",
    "sequential_heloc_then_bank_then_target",
)
REMOTE_ROUTES = {
    "covertype": (
        "target_compute_matched",
        "pooled_target_satimage",
        "sequential_satimage_then_target",
        "pooled_target_satimage_segment_ablation",
        "sequential_satimage_then_segment_then_target",
        "sequential_segment_then_satimage_then_target",
    ),
    "satimage": (
        "target_compute_matched",
        "pooled_target_covertype",
        "sequential_covertype_then_target",
        "pooled_target_covertype_segment_ablation",
        "sequential_covertype_then_segment_then_target",
        "sequential_segment_then_covertype_then_target",
    ),
}
SCHEDULE_ROUTES = {"credit-g": FINANCE_ROUTES, **REMOTE_ROUTES}
TARGET_CONTROL_DATASETS = ("credit-g", "Bank_Customer_Churn", "heloc", "segment")
PRIMARY_SOURCE = {"covertype": "satimage", "satimage": "covertype"}


def build_schedule_route(
    *,
    route_id: str,
    target: str,
    steps_per_dataset: int,
    row_budget: int | None,
    common: Mapping[str, Any],
    stage_factory: StageFactory,
) -> TransferRouteSpec:
    """Build one explicit E1 route for the concrete target."""
    allowed = SCHEDULE_ROUTES.get(target)
    if allowed is None:
        raise ValueError(f"No maintained transfer source panel for target {target!r}.")
    if route_id not in allowed:
        raise ValueError(f"Unsupported transfer schedule route {route_id!r} for {target!r}.")
    if target == "covertype" and row_budget is None:
        raise ValueError("Covertype schedule routes require a numeric row cap.")
    if route_id == "target_compute_matched":
        return _target_only(
            route_id=route_id,
            target=target,
            steps_per_dataset=steps_per_dataset,
            row_budget=row_budget,
            common=common,
            stage_factory=stage_factory,
        )
    if target == "credit-g":
        return _finance_route(
            route_id=route_id,
            target=target,
            steps_per_dataset=steps_per_dataset,
            row_budget=row_budget,
            common=common,
            stage_factory=stage_factory,
        )
    return _remote_route(
        route_id=route_id,
        target=target,
        steps_per_dataset=steps_per_dataset,
        row_budget=row_budget,
        common=common,
        stage_factory=stage_factory,
    )


def build_target_control_route(
    *,
    route_id: str,
    target: str,
    steps_per_dataset: int,
    row_budget: int | None,
    common: Mapping[str, Any],
    stage_factory: StageFactory,
) -> TransferRouteSpec:
    """Build a 300-step target-only producer with 100/200-step views."""
    if route_id != "target_control_300_steps":
        raise ValueError(f"Unsupported target-control route: {route_id!r}")
    if target not in TARGET_CONTROL_DATASETS:
        raise ValueError(f"Unsupported target-control dataset: {target!r}")
    stage = stage_factory(
        name="target_control",
        role="terminal",
        mode="single_dataset",
        datasets=(target,),
        row_budget=row_budget,
        updates=3 * steps_per_dataset,
        selector=ValidationSelector("target_validation", (target,)),
        common=common,
        view_boundaries=(steps_per_dataset, 2 * steps_per_dataset),
    )
    return TransferRouteSpec(
        route_id=route_id,
        family="dataset_transfer_target_controls",
        target_dataset=target,
        execution_mode="train",
        stages=(stage,),
    )


def _target_only(
    *,
    route_id: str,
    target: str,
    steps_per_dataset: int,
    row_budget: int | None,
    common: Mapping[str, Any],
    stage_factory: StageFactory,
) -> TransferRouteSpec:
    stage = stage_factory(
        name="target_compute_matched",
        role="terminal",
        mode="single_dataset",
        datasets=(target,),
        row_budget=row_budget,
        updates=3 * steps_per_dataset,
        selector=ValidationSelector("target_validation", (target,)),
        common=common,
        view_boundaries=(steps_per_dataset, 2 * steps_per_dataset)
        if target in PRIMARY_SOURCE
        else (steps_per_dataset,),
    )
    return TransferRouteSpec(
        route_id=route_id, family="dataset_transfer", target_dataset=target, execution_mode="train", stages=(stage,)
    )


def _finance_route(
    *,
    route_id: str,
    target: str,
    steps_per_dataset: int,
    row_budget: int | None,
    common: Mapping[str, Any],
    stage_factory: StageFactory,
) -> TransferRouteSpec:
    sources = ("Bank_Customer_Churn", "heloc")
    target_selector = ValidationSelector("target_validation", (target,))
    if route_id == "pooled_target_sources":
        stages = (
            stage_factory(
                name="target_and_sources",
                role="terminal",
                mode="pooled_episodic",
                datasets=(target, *sources),
                row_budget=row_budget,
                updates=3 * steps_per_dataset,
                selector=target_selector,
                common=common,
            ),
        )
    elif route_id == "pooled_sources_then_target":
        stages = (
            stage_factory(
                name="source_pool",
                role="reusable_boundary",
                mode="pooled_episodic",
                datasets=sources,
                row_budget=row_budget,
                updates=2 * steps_per_dataset,
                selector=ValidationSelector("equal_mean_validation_improvement", sources),
                common=common,
            ),
            stage_factory(
                name="target",
                role="terminal",
                mode="single_dataset",
                datasets=(target,),
                row_budget=row_budget,
                updates=steps_per_dataset,
                selector=target_selector,
                common=common,
                parent="source_pool",
            ),
        )
    else:
        order = {
            "sequential_bank_then_heloc_then_target": sources,
            "sequential_heloc_then_bank_then_target": tuple(reversed(sources)),
        }[route_id]
        stages = _sequential_stages(
            order=order,
            target=target,
            steps_per_dataset=steps_per_dataset,
            row_budget=row_budget,
            common=common,
            stage_factory=stage_factory,
        )
    return TransferRouteSpec(
        route_id=route_id, family="dataset_transfer", target_dataset=target, execution_mode="train", stages=stages
    )


def _remote_route(
    *,
    route_id: str,
    target: str,
    steps_per_dataset: int,
    row_budget: int | None,
    common: Mapping[str, Any],
    stage_factory: StageFactory,
) -> TransferRouteSpec:
    related = PRIMARY_SOURCE[target]
    target_selector = ValidationSelector("target_validation", (target,))
    if route_id == f"pooled_target_{related}":
        stages = (
            stage_factory(
                name="target_and_primary_source",
                role="terminal",
                mode="pooled_episodic",
                datasets=(target, related),
                row_budget=row_budget,
                updates=2 * steps_per_dataset,
                selector=target_selector,
                common=common,
            ),
        )
    elif route_id == f"sequential_{related}_then_target":
        stages = (
            stage_factory(
                name="primary_source",
                role="reusable_boundary",
                mode="single_dataset",
                datasets=(related,),
                row_budget=row_budget,
                updates=steps_per_dataset,
                selector=ValidationSelector("target_validation", (related,)),
                common=common,
            ),
            stage_factory(
                name="target",
                role="terminal",
                mode="single_dataset",
                datasets=(target,),
                row_budget=row_budget,
                updates=steps_per_dataset,
                selector=target_selector,
                common=common,
                parent="primary_source",
            ),
        )
    elif route_id == f"pooled_target_{related}_segment_ablation":
        stages = (
            stage_factory(
                name="target_related_and_segment",
                role="terminal",
                mode="pooled_episodic",
                datasets=(target, related, "segment"),
                row_budget=row_budget,
                updates=3 * steps_per_dataset,
                selector=target_selector,
                common=common,
            ),
        )
    else:
        order = (
            (related, "segment")
            if route_id == f"sequential_{related}_then_segment_then_target"
            else ("segment", related)
        )
        stages = _sequential_stages(
            order=order,
            target=target,
            steps_per_dataset=steps_per_dataset,
            row_budget=row_budget,
            common=common,
            stage_factory=stage_factory,
        )
    return TransferRouteSpec(
        route_id=route_id, family="dataset_transfer", target_dataset=target, execution_mode="train", stages=stages
    )


def _sequential_stages(
    *,
    order: tuple[str, str],
    target: str,
    steps_per_dataset: int,
    row_budget: int | None,
    common: Mapping[str, Any],
    stage_factory: StageFactory,
) -> tuple[TransferStageSpec, ...]:
    first, second = order
    return (
        stage_factory(
            name="source_1",
            role="intermediate",
            mode="single_dataset",
            datasets=(first,),
            row_budget=row_budget,
            updates=steps_per_dataset,
            selector=ValidationSelector("target_validation", (first,)),
            common=common,
        ),
        stage_factory(
            name="source_2",
            role="intermediate",
            mode="single_dataset",
            datasets=(second,),
            row_budget=row_budget,
            updates=steps_per_dataset,
            selector=ValidationSelector("target_validation", (second,)),
            common=common,
            parent="source_1",
        ),
        stage_factory(
            name="target",
            role="terminal",
            mode="single_dataset",
            datasets=(target,),
            row_budget=row_budget,
            updates=steps_per_dataset,
            selector=ValidationSelector("target_validation", (target,)),
            common=common,
            parent="source_2",
        ),
    )


__all__ = [
    "FINANCE_ROUTES",
    "REMOTE_ROUTES",
    "SCHEDULE_ROUTES",
    "TARGET_CONTROL_DATASETS",
    "build_schedule_route",
    "build_target_control_route",
]
