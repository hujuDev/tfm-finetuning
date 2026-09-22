"""Construct same-schema natural-domain transfer routes.

Example:
    >>> from data.beyondarena_natural_domain_catalog import panel_for_dataset
    >>> panel_for_dataset("covertype_neota").key
    'covertype_area'
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from data.beyondarena_natural_domain_catalog import panel_for_dataset

from .spec import TransferRouteSpec, TransferStageSpec, ValidationSelector

StageFactory = Callable[..., TransferStageSpec]
NATURAL_DOMAIN_ROUTES = ("target_control_300_steps", "source_only_pooled", "target_inclusive_pooled")


def build_natural_domain_route(
    *,
    route_id: str,
    target: str,
    steps_per_dataset: int,
    row_budget: int | None,
    common: Mapping[str, Any],
    stage_factory: StageFactory,
) -> TransferRouteSpec:
    """Build one of the three fixed routes for a rotated target domain."""
    if route_id not in NATURAL_DOMAIN_ROUTES:
        raise ValueError(f"Unsupported natural-domain route: {route_id!r}.")
    panel = panel_for_dataset(target)
    sources = tuple((dataset for dataset in panel.datasets if dataset != target))
    target_selector = ValidationSelector("target_validation", (target,))
    if route_id == "target_control_300_steps":
        stage = stage_factory(
            name="target_control",
            role="terminal",
            mode="single_dataset",
            datasets=(target,),
            row_budget=row_budget,
            updates=3 * steps_per_dataset,
            selector=target_selector,
            common=common,
            view_boundaries=(steps_per_dataset, 2 * steps_per_dataset),
        )
    elif route_id == "source_only_pooled":
        stage = stage_factory(
            name="source_pool",
            role="terminal",
            mode="pooled_episodic",
            datasets=sources,
            row_budget=row_budget,
            updates=2 * steps_per_dataset,
            selector=ValidationSelector("equal_mean_validation_improvement", sources),
            common=common,
        )
    else:
        stage = stage_factory(
            name="sources_and_target",
            role="terminal",
            mode="pooled_episodic",
            datasets=(*sources, target),
            row_budget=row_budget,
            updates=3 * steps_per_dataset,
            selector=target_selector,
            common=common,
        )
    return TransferRouteSpec(
        route_id=route_id,
        family="dataset_transfer_natural_domain",
        target_dataset=target,
        execution_mode="train",
        stages=(stage,),
    )


__all__ = ["NATURAL_DOMAIN_ROUTES", "build_natural_domain_route"]
