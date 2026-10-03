"""Post-finite representation transforms.

The maintained ``update_state`` studies alter only the rows used for gradient
updates.  Validation, ICL-context, and held-out query representations stay
canonical. The ``fit_state`` alignment experiments additionally vary the fit
representation and evaluate both query representations using one checkpoint.

Example:
    >>> callable(apply_numeric_scale)
    True
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from tfm_finetuning.finetuning.distribution_shift._common import hash_array

if TYPE_CHECKING:
    from collections.abc import Mapping


def apply_shuffle_feature_order(
    data: Any,
    cfg: Mapping[str, Any],
    rng: np.random.Generator,
    seed: int,
) -> tuple[Any, Any | None, dict[str, Any]]:
    """Apply a fine-tuning-only order treatment or a paired-view alignment treatment."""
    update_state = str(cfg.get("update_state") or "").strip().lower().replace("-", "_")
    if update_state:
        return _apply_finetune_only_feature_order(data, cfg, rng, seed, update_state=update_state)
    fit_state = str(cfg.get("fit_state") or "").strip().lower().replace("-", "_")
    if fit_state not in {"canonical", "permuted"}:
        raise ValueError(f"shuffle_feature_order.fit_state must be canonical or permuted, got {fit_state!r}")
    permutation = rng.permutation(data.train_X.shape[1])
    canonical = {
        role: _canonical_feature_frame(getattr(data, f"{role}_X"))
        for role in ("train", "val_context", "val", "test_context", "test")
    }
    fit_roles = ("train", "val_context", "val", "test_context")
    primary_updates = {
        f"{role}_X": (
            _positional_permute_frame(getattr(data, f"{role}_X"), permutation)
            if fit_state == "permuted" and role in fit_roles
            else canonical[role]
        )
        for role in canonical
    }
    primary_updates["test_X"] = canonical["test"]
    primary = replace(data, **primary_updates)
    alternate = replace(primary, test_X=_positional_permute_frame(data.test_X, permutation))
    primary, alternate = _mark_test_views(
        primary,
        alternate,
        family="feature_order",
        fit_state=fit_state,
        primary_view="canonical",
        alternate_view="permuted",
    )
    return (
        primary,
        alternate,
        {
            "stage": "post_finite",
            "seed": seed,
            "fit_state": fit_state,
            "primary_test_view": "canonical",
            "alternate_test_view": "permuted",
            "feature_count": int(data.train_X.shape[1]),
            "permutation_hash": hash_array(permutation.astype(int)),
            "permutation": permutation.astype(int).tolist() if bool(cfg.get("record_permutation", False)) else None,
            "same_checkpoint_test_views": True,
        },
    )


def apply_numeric_scale(
    data: Any,
    cfg: Mapping[str, Any],
    seed: int,
) -> tuple[Any, Any | None, dict[str, Any]]:
    """Apply a fine-tuning-only scale treatment or a paired-view alignment treatment."""
    update_state = str(cfg.get("update_state") or "").strip().lower().replace("-", "_")
    if update_state:
        return _apply_finetune_only_numeric_scale(data, cfg, seed, update_state=update_state)
    fit_state = str(cfg.get("fit_state") or "").strip().lower().replace("-", "_")
    if fit_state not in {"raw", "standardized"}:
        raise ValueError(f"numeric_scale.fit_state must be raw or standardized, got {fit_state!r}")
    columns = _continuous_numeric_columns(data)
    scaler = StandardScaler().fit(data.train_X.loc[:, columns]) if columns else None

    def standardize(frame: pd.DataFrame) -> pd.DataFrame:
        return _standardized_frame(frame, columns=columns, scaler=scaler)

    fit_roles = ("train", "val_context", "val", "test_context")
    updates = {
        f"{role}_X": (
            standardize(getattr(data, f"{role}_X"))
            if fit_state == "standardized" and role in fit_roles
            else getattr(data, f"{role}_X").copy()
        )
        for role in ("train", "val_context", "val", "test_context", "test")
    }
    updates["test_X"] = data.test_X.copy()
    primary = replace(data, **updates)
    alternate = replace(primary, test_X=standardize(data.test_X))
    primary, alternate = _mark_test_views(
        primary,
        alternate,
        family="numeric_scale",
        fit_state=fit_state,
        primary_view="raw",
        alternate_view="standardized",
    )
    return (
        primary,
        alternate,
        {
            "stage": "post_finite",
            "seed": seed,
            "fit_state": fit_state,
            "primary_test_view": "raw",
            "alternate_test_view": "standardized",
            "transform_fit_scope": "effective_training_numeric_columns",
            "transformed_columns": [str(column) for column in columns],
            "transformed_feature_count": len(columns),
            "binary_and_categorical_columns_preserved": True,
            "same_checkpoint_test_views": True,
        },
    )


def _apply_finetune_only_feature_order(
    data: Any,
    cfg: Mapping[str, Any],
    rng: np.random.Generator,
    seed: int,
    *,
    update_state: str,
) -> tuple[Any, None, dict[str, Any]]:
    if update_state not in {"canonical", "fixed_permutation"}:
        raise ValueError(
            f"shuffle_feature_order.update_state must be canonical or fixed_permutation, got {update_state!r}"
        )
    permutation = rng.permutation(data.train_X.shape[1])
    canonical = {
        role: _canonical_feature_frame(getattr(data, f"{role}_X"))
        for role in ("train", "val_context", "val", "test_context", "test")
    }
    updates = {f"{role}_X": frame for role, frame in canonical.items()}
    if update_state == "fixed_permutation":
        updates["train_X"] = _positional_permute_frame(data.train_X, permutation)
    transformed = replace(data, **updates)
    return (
        transformed,
        None,
        {
            "stage": "post_finite",
            "seed": seed,
            "update_state": update_state,
            "gradient_update_roles": ["train"],
            "canonical_evaluation_roles": ["val_context", "val", "test_context", "test"],
            "feature_count": int(data.train_X.shape[1]),
            "permutation_hash": hash_array(permutation.astype(int)),
            "permutation": permutation.astype(int).tolist() if bool(cfg.get("record_permutation", False)) else None,
            "same_icl_data_across_conditions": True,
            "paired_test_views": False,
        },
    )


def _apply_finetune_only_numeric_scale(
    data: Any,
    cfg: Mapping[str, Any],
    seed: int,
    *,
    update_state: str,
) -> tuple[Any, None, dict[str, Any]]:
    if update_state not in {"raw", "standardized"}:
        raise ValueError(f"numeric_scale.update_state must be raw or standardized, got {update_state!r}")
    columns = _continuous_numeric_columns(data)
    scaler = StandardScaler().fit(data.train_X.loc[:, columns]) if columns else None
    updates = {
        f"{role}_X": getattr(data, f"{role}_X").copy()
        for role in ("train", "val_context", "val", "test_context", "test")
    }
    if update_state == "standardized":
        updates["train_X"] = _standardized_frame(data.train_X, columns=columns, scaler=scaler)
    transformed = replace(data, **updates)
    return (
        transformed,
        None,
        {
            "stage": "post_finite",
            "seed": seed,
            "update_state": update_state,
            "gradient_update_roles": ["train"],
            "canonical_evaluation_roles": ["val_context", "val", "test_context", "test"],
            "transform_fit_scope": "effective_training_numeric_columns",
            "transformed_columns": [str(column) for column in columns],
            "transformed_feature_count": len(columns),
            "binary_and_categorical_columns_preserved": True,
            "same_icl_data_across_conditions": True,
            "paired_test_views": False,
        },
    )


def _continuous_numeric_columns(data: Any) -> list[Any]:
    preprocessing = data.metadata.get("feature_preprocessing") if isinstance(data.metadata, dict) else None
    source_schema = preprocessing.get("source_feature_schema") if isinstance(preprocessing, dict) else None
    source_numeric = set(source_schema.get("numeric_columns", [])) if isinstance(source_schema, dict) else set()
    candidates = [column for column in data.train_X.columns if not source_numeric or str(column) in source_numeric]
    return [
        column
        for column in candidates
        if pd.api.types.is_numeric_dtype(data.train_X[column]) and data.train_X[column].nunique(dropna=True) > 2
    ]


def _standardized_frame(
    frame: pd.DataFrame,
    *,
    columns: list[Any],
    scaler: StandardScaler | None,
) -> pd.DataFrame:
    out = frame.copy()
    if columns and scaler is not None:
        out.loc[:, columns] = scaler.transform(frame.loc[:, columns]).astype(np.float32)
    return out


def _mark_test_views(
    primary: Any,
    alternate: Any,
    *,
    family: str,
    fit_state: str,
    primary_view: str,
    alternate_view: str,
) -> tuple[Any, Any]:
    comparison = {
        "kind": "distribution_shift_test_view",
        "family": family,
        "fit_state": fit_state,
        "primary_view": primary_view,
        "alternate_view": alternate_view,
    }
    primary_metadata = {**dict(primary.metadata), "evaluation_comparison": comparison}
    alternate_metadata = {**dict(alternate.metadata), "evaluation_comparison": comparison}
    return replace(primary, metadata=primary_metadata), replace(alternate, metadata=alternate_metadata)


def _canonical_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [f"x{idx}" for idx in range(out.shape[1])]
    return out


def _positional_permute_frame(frame: pd.DataFrame, permutation: np.ndarray) -> pd.DataFrame:
    values = frame.to_numpy(copy=True)[:, permutation]
    return pd.DataFrame(values, columns=[f"x{idx}" for idx in range(len(permutation))])
