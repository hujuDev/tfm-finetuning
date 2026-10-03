"""Build route-matched ICL contexts for same-schema data-group transfer.

The checkpoint is always evaluated on target-group test rows. Its labeled ICL
context mirrors the groups available to the corresponding fine-tuning route:
target only, the two sources, or both sources plus the target.

Example:
    >>> context_groups("covertype_neota", "source_only_pooled")
    ('covertype_comanche_peak', 'covertype_rawah')
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from data.beyondarena.natural_domain_catalog import panel_for_dataset
from experiments.transfer.spec import TransferRouteSpec

ROUTE_MATCHED_CONTEXT_POLICY = "route_matched_train_plus_validation"


def context_groups(target: str, checkpoint_route: str) -> tuple[str, ...]:
    """Return the ordered labeled groups available to one transfer route."""
    panel = panel_for_dataset(target)
    sources = tuple(dataset for dataset in panel.datasets if dataset != target)
    if checkpoint_route == "target_control_300_steps":
        return (target,)
    if checkpoint_route == "source_only_pooled":
        return sources
    if checkpoint_route == "target_inclusive_pooled":
        return (*sources, target)
    raise ValueError(f"Route-matched natural-domain evaluation does not support producer route {checkpoint_route!r}.")


def build_route_matched_evaluation_data(
    route: TransferRouteSpec,
    *,
    target: str,
    dataset_loader: Any,
) -> Any:
    """Keep ``C_test`` as query and replace its context with the route fit pool."""
    if route.context_policy != ROUTE_MATCHED_CONTEXT_POLICY:
        return dataset_loader(target)
    dependency = route.dependency
    if dependency is None:
        raise ValueError("Route-matched context evaluation requires a checkpoint dependency.")
    if target != route.target_dataset:
        raise ValueError(
            "Natural-domain route-matched evaluation requires the declared target "
            f"{route.target_dataset!r}, got {target!r}."
        )

    groups = context_groups(target, dependency.route_id)
    group_data = {
        group: dataset_loader(
            group,
            row_budget=dependency.producer_row_budget_per_dataset,
            row_budget_policy="cap",
            row_subset_seed=dependency.producer_row_subset_seed,
        )
        for group in groups
    }
    target_data = group_data.get(target)
    if target_data is None:
        target_data = dataset_loader(
            target,
            row_budget=dependency.producer_row_budget_per_dataset,
            row_budget_policy="cap",
            row_subset_seed=dependency.producer_row_subset_seed,
        )
    _validate_compatible_groups(target_data, group_data)

    frames = [group_data[group].test_context_X for group in groups]
    labels = [np.asarray(group_data[group].test_context_y) for group in groups]
    context_ids = np.concatenate([_prefixed_ids(group, group_data[group]) for group in groups])
    components = np.concatenate([_prefixed_components(group, group_data[group]) for group in groups])
    row_counts = {group: len(group_data[group].test_context_X) for group in groups}
    metadata = {
        **dict(target_data.metadata),
        "test_context_policy": ROUTE_MATCHED_CONTEXT_POLICY,
        "route_matched_context": {
            "checkpoint_route": dependency.route_id,
            "context_groups": list(groups),
            "query_group": target,
            "rows_by_group": row_counts,
            "total_rows": sum(row_counts.values()),
            "includes_validation_per_group": True,
        },
    }
    return replace(
        target_data,
        test_context_X=pd.concat(frames, ignore_index=True),
        test_context_y=np.concatenate(labels),
        test_context_ids=context_ids,
        test_context_components=components,
        metadata=metadata,
    )


def _validate_compatible_groups(target_data: Any, groups: dict[str, Any]) -> None:
    expected_columns = list(target_data.test_X.columns)
    expected_dtype_roles = [
        "numeric" if pd.api.types.is_numeric_dtype(dtype) else "categorical" for dtype in target_data.test_X.dtypes
    ]
    for name, data in groups.items():
        if data.task_type != target_data.task_type or data.problem_type != target_data.problem_type:
            raise ValueError(f"Natural-domain context group {name!r} has an incompatible task type.")
        if list(data.test_context_X.columns) != expected_columns:
            raise ValueError(f"Natural-domain context group {name!r} has an incompatible feature schema.")
        dtype_roles = [
            "numeric" if pd.api.types.is_numeric_dtype(dtype) else "categorical"
            for dtype in data.test_context_X.dtypes
        ]
        if dtype_roles != expected_dtype_roles:
            raise ValueError(f"Natural-domain context group {name!r} has incompatible feature dtypes.")


def _prefixed_ids(group: str, data: Any) -> np.ndarray:
    raw = data.test_context_ids
    if raw is None:
        raw = np.arange(len(data.test_context_X), dtype=np.int64)
    return np.asarray([f"{group}:{value}" for value in raw], dtype=object)


def _prefixed_components(group: str, data: Any) -> np.ndarray:
    raw = data.test_context_components
    if raw is None:
        raw = np.full(len(data.test_context_X), "fit", dtype=object)
    return np.asarray([f"{group}:{value}" for value in raw], dtype=object)


__all__ = [
    "ROUTE_MATCHED_CONTEXT_POLICY",
    "build_route_matched_evaluation_data",
    "context_groups",
]
