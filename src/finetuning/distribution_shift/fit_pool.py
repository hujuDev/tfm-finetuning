"""Combined fit-pool helpers for row-budget and prior interventions.

The fit pool is the union of the fine-tuning training and validation rows.  A
row-selection intervention operates on that pool first, then this module
rebuilds the concrete training, validation, and ICL-context roles.

Example:
    >>> callable(combine_fit_pool)
    True
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from finetuning.distribution_shift._common import TRAIN_FT, VAL_FT, derive_seed, hash_array


@dataclass(frozen=True)
class FitPool:
    """Rows available to every fit-side role before the train/val partition."""

    X: pd.DataFrame
    fit_y: np.ndarray
    query_y: np.ndarray
    ids: np.ndarray


def combine_fit_pool(data: Any) -> FitPool:
    """Combine the current training and validation roles without test rows."""
    val_fit_y = data.val_fit_y if getattr(data, "val_fit_y", None) is not None else data.val_y
    ids = np.concatenate(
        [np.asarray(data.train_ids, dtype=object), np.asarray(data.val_ids, dtype=object)]
    )
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("Fit-pool row IDs must be unique across training and validation roles")
    return FitPool(
        X=pd.concat([data.train_X, data.val_X], axis=0, ignore_index=True),
        fit_y=np.concatenate([np.asarray(data.train_y), np.asarray(val_fit_y)]),
        query_y=np.concatenate([np.asarray(data.train_y), np.asarray(data.val_y)]),
        ids=ids,
    )


def nested_selection_indices(
    pool: FitPool,
    *,
    target: int,
    task_type: str,
    seed: int,
) -> np.ndarray:
    """Select a prefix of one fixed, model-independent pool ordering.

    Classification ordering interleaves independently shuffled class queues;
    therefore every smaller prefix is nested and remains approximately
    stratified. Regression uses one seeded random ordering.
    """
    if target >= len(pool.X):
        return np.arange(len(pool.X), dtype=int)
    rng = np.random.default_rng(derive_seed(seed, "fit_pool_nested_order"))
    order = _nested_order(pool.query_y, task_type=task_type, rng=rng)
    return np.sort(order[:target].astype(int))


def rebuild_fit_roles(
    data: Any,
    pool: FitPool,
    selected_indices: np.ndarray,
    *,
    task_type: str,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    """Repartition a selected pool and rebuild every concrete fit-side role."""
    selected = np.asarray(selected_indices, dtype=int)
    selected_pool = FitPool(
        X=pool.X.iloc[selected].reset_index(drop=True),
        fit_y=pool.fit_y[selected],
        query_y=pool.query_y[selected],
        ids=pool.ids[selected],
    )
    validation_fraction = len(data.val_X) / max(1, len(data.train_X) + len(data.val_X))
    val_count = max(1, min(len(selected) - 1, round(len(selected) * validation_fraction)))
    partition_rng = np.random.default_rng(derive_seed(seed, "fit_pool_partition"))
    partition_order = _nested_order(selected_pool.query_y, task_type=task_type, rng=partition_rng)
    val_pos = np.sort(partition_order[:val_count].astype(int))
    train_pos = np.sort(partition_order[val_count:].astype(int))

    train_X = selected_pool.X.iloc[train_pos].reset_index(drop=True)
    val_X = selected_pool.X.iloc[val_pos].reset_index(drop=True)
    train_y = selected_pool.fit_y[train_pos]
    val_fit_y = selected_pool.fit_y[val_pos]
    val_y = selected_pool.query_y[val_pos]
    train_ids = selected_pool.ids[train_pos]
    val_ids = selected_pool.ids[val_pos]
    test_context_X = pd.concat([train_X, val_X], axis=0, ignore_index=True)
    test_context_y = np.concatenate([train_y, val_fit_y])
    test_context_ids = np.concatenate([train_ids, val_ids])
    test_context_components = np.asarray(
        [TRAIN_FT] * len(train_ids) + [VAL_FT] * len(val_ids),
        dtype=object,
    )
    updates = {
        "train_X": train_X,
        "train_y": train_y,
        "train_ids": train_ids,
        "val_context_X": train_X.copy(),
        "val_context_y": train_y.copy(),
        "val_context_ids": train_ids.copy(),
        "val_context_components": np.asarray([TRAIN_FT] * len(train_ids), dtype=object),
        "val_X": val_X,
        "val_y": val_y,
        "val_ids": val_ids,
        "test_context_X": test_context_X,
        "test_context_y": test_context_y,
        "test_context_ids": test_context_ids,
        "test_context_components": test_context_components,
    }
    if hasattr(data, "val_fit_y"):
        updates["val_fit_y"] = val_fit_y
    return replace(data, **updates), {
        "fit_pool_size_before": len(pool.X),
        "fit_pool_size_after": len(selected_pool.X),
        "fit_pool_ids_hash": hash_array(selected_pool.ids),
        "train_size_after": len(train_pos),
        "validation_size_after": len(val_pos),
        "validation_fraction": validation_fraction,
    }


def _nested_order(labels: np.ndarray, *, task_type: str, rng: np.random.Generator) -> np.ndarray:
    size = len(labels)
    if task_type not in {"binary", "multiclass"}:
        return rng.permutation(size)
    ranked: list[tuple[float, str, int]] = []
    labels_array = np.asarray(labels)
    for label in np.unique(labels_array):
        positions = np.flatnonzero(labels_array == label)
        shuffled = rng.permutation(positions)
        denominator = max(1, len(shuffled))
        for rank, position in enumerate(shuffled):
            ranked.append(((rank + 0.5) / denominator, str(label), int(position)))
    ranked.sort()
    return np.asarray([position for _priority, _label, position in ranked], dtype=int)
