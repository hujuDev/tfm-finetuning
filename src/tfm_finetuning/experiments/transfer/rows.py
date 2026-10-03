"""Deterministic nested train-row views for transfer experiments.

The helpers in this module choose rows only from a dataset's training role.  A
single seeded permutation backs every requested budget, so smaller selections
are exact ordered prefixes of larger ones.  The resulting subset object is also
the sole source used to rebuild validation/test ICL contexts.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NestedTrainSubset:
    """One exact prefix of a deterministic train-row permutation.

    Example:
        >>> subset = build_nested_train_subsets(10, [3], seed=7)[3]
        >>> len(subset.indices)
        3
    """

    budget: int
    source_row_count: int
    indices: tuple[int, ...]

    def take_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return this subset of ``frame`` in its deterministic prefix order."""
        _require_matching_row_count(frame, expected=self.source_row_count)
        return frame.iloc[list(self.indices)].reset_index(drop=True).copy()

    def take_array(self, values: Any) -> np.ndarray:
        """Return this subset of a row-aligned array-like value."""
        array = np.asarray(values)
        _require_matching_row_count(array, expected=self.source_row_count)
        return array[np.asarray(self.indices, dtype=np.int64)].copy()


def build_nested_train_subsets(
    row_count: int,
    budgets: Iterable[int],
    *,
    seed: int,
) -> dict[int, NestedTrainSubset]:
    """Build exact nested train-row prefixes for the requested budgets.

    Budgets must be positive, unique, and no larger than ``row_count``.  Callers
    that need the full-data endpoint should pass ``row_count`` explicitly.

    Example:
        >>> subsets = build_nested_train_subsets(20, [4, 8], seed=2)
        >>> subsets[4].indices == subsets[8].indices[:4]
        True
    """
    count = _positive_int(row_count, field="row_count")
    requested = tuple(_positive_int(value, field="budget") for value in budgets)
    if not requested:
        raise ValueError("budgets must contain at least one positive row budget.")
    if len(set(requested)) != len(requested):
        raise ValueError("budgets must not contain duplicates.")
    oversized = sorted(value for value in requested if value > count)
    if oversized:
        raise ValueError(f"row budgets exceed the {count}-row training split: {oversized}.")

    permutation = np.random.default_rng(int(seed)).permutation(count)
    return {
        budget: NestedTrainSubset(
            budget=budget,
            source_row_count=count,
            indices=tuple(int(index) for index in permutation[:budget]),
        )
        for budget in sorted(requested)
    }


def apply_train_subset_to_split(
    data: Any,
    subset: NestedTrainSubset,
    *,
    include_validation_in_test_context: bool = False,
) -> Any:
    """Return a split dataclass whose training and ICL rows share ``subset``.

    Validation and test *query* rows are retained exactly.  Validation ICL uses
    the selected training rows.  Test ICL uses the same rows and may additionally
    append the unchanged outer-validation queries after checkpoint selection.

    This function expects the ``SplitEvaluationData`` field contract but remains
    independent of any adapter.
    """
    train_X = subset.take_frame(data.train_X)
    train_y = subset.take_array(data.train_y)
    train_ids = _take_optional_ids(getattr(data, "train_ids", None), subset)

    test_context_X = train_X.copy()
    test_context_y = train_y.copy()
    test_context_ids = _copy_optional(train_ids)
    test_context_components = _role_components("train", subset.budget)
    if include_validation_in_test_context:
        test_context_X = pd.concat([test_context_X, data.val_X], ignore_index=True)
        test_context_y = np.concatenate([test_context_y, np.asarray(data.val_y)])
        test_context_ids = _append_optional_ids(test_context_ids, getattr(data, "val_ids", None))
        test_context_components = np.concatenate(
            [test_context_components, _role_components("validation", len(data.val_X))]
        )

    metadata = dict(getattr(data, "metadata", {}) or {})
    metadata["nested_train_rows"] = {
        "budget": subset.budget,
        "indices": list(subset.indices),
        "test_context_includes_validation": bool(include_validation_in_test_context),
    }
    return replace(
        data,
        train_X=train_X,
        train_y=train_y,
        train_ids=train_ids,
        val_context_X=train_X.copy(),
        val_context_y=train_y.copy(),
        val_context_ids=_copy_optional(train_ids),
        val_context_components=_role_components("train", subset.budget),
        test_context_X=test_context_X,
        test_context_y=test_context_y,
        test_context_ids=test_context_ids,
        test_context_components=test_context_components,
        metadata=metadata,
    )


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be a positive integer, got {value!r}.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be a positive integer, got {value!r}.") from exc
    if parsed != value or parsed <= 0:
        raise ValueError(f"{field} must be a positive integer, got {value!r}.")
    return parsed


def _require_matching_row_count(values: Any, *, expected: int) -> None:
    if len(values) != expected:
        raise ValueError(f"Expected the original {expected}-row training split, got {len(values)} rows.")


def _take_optional_ids(values: Any, subset: NestedTrainSubset) -> np.ndarray | None:
    return None if values is None else subset.take_array(np.asarray(values, dtype=object))


def _copy_optional(values: np.ndarray | None) -> np.ndarray | None:
    return None if values is None else values.copy()


def _append_optional_ids(left: np.ndarray | None, right: Any) -> np.ndarray | None:
    if left is None and right is None:
        return None
    if left is None or right is None:
        raise ValueError("Cannot reconstruct test context IDs when only one context component has IDs.")
    return np.concatenate([left, np.asarray(right, dtype=object)])


def _role_components(role: str, count: int) -> np.ndarray:
    return np.full(int(count), role, dtype=object)


__all__ = [
    "NestedTrainSubset",
    "apply_train_subset_to_split",
    "build_nested_train_subsets",
]
