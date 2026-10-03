"""Synthetic contextualization interventions for tabular feature semantics."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from tfm_finetuning.finetuning.distribution_shift._common import resolve_calibrated_value

if TYPE_CHECKING:
    from collections.abc import Mapping


def apply_contextual_scale(
    data: Any,
    cfg: Mapping[str, Any],
    rng: np.random.Generator,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    """Apply one full-role contextual-scale condition.

    Example:
        A value ``x`` becomes ``x / multiplier(unit)``.  With the synthetic
        unit column present, the original feature meaning is recoverable.
    """
    condition = str(cfg.get("condition") or cfg.get("mode") or "none").strip().lower()
    if condition in {"", "none", "off", "disabled"}:
        return data, _original_condition_report(seed=seed, condition="none")
    scale_spread, unit_column_present, transformation_setting = _parse_condition(condition, cfg=cfg, data=data)
    if transformation_setting == "no_transformation":
        return data, _original_condition_report(seed=seed, condition=condition)
    numeric_columns = [
        column for column in data.train_X.columns if pd.api.types.is_numeric_dtype(data.train_X[column])
    ]
    if not numeric_columns:
        return data, {
            "stage": "pre_finite",
            "seed": seed,
            "condition": condition,
            "skipped": True,
            "reason": "no_numeric_features",
        }
    selected = _selected_numeric_features(
        data.train_X,
        np.asarray(data.train_y),
        numeric_columns,
        task_type=str(data.task_type),
        max_features=_resolve_feature_count(cfg, data=data),
        strategy=str(cfg.get("feature_strategy") or "importance"),
        rng=rng,
        seed=seed,
    )
    group_count = 1 if scale_spread == 1.0 else max(2, int(cfg.get("group_count") or 3))
    multipliers = np.geomspace(1.0, scale_spread, num=group_count)
    context_column = str(cfg.get("context_column") or "__synthetic_unit__")
    group_by_id: dict[Any, int] = {}
    updates = {}
    for role in ("train", "val_context", "val", "test_context", "test"):
        frame = getattr(data, f"{role}_X")
        ids = np.asarray(getattr(data, f"{role}_ids", [f"{role}:{idx}" for idx in range(len(frame))]), dtype=object)
        updates[f"{role}_X"] = _contextualized_frame(
            frame,
            ids,
            selected_features=selected,
            multipliers=multipliers,
            context_column=context_column,
            include_context=unit_column_present,
            group_by_id=group_by_id,
            rng=rng,
        )
    return replace(data, **updates), {
        "stage": "pre_finite",
        "seed": seed,
        "condition": condition,
        "transformation_setting": transformation_setting,
        "fit_side_role": "transformed",
        "test_side_role": "transformed",
        "max_multiplier": scale_spread,
        "scale_spread": float(np.max(multipliers) / np.min(multipliers)),
        "group_count": int(group_count),
        "context_column": context_column,
        "unit_visibility": "with_unit_column" if unit_column_present else "no_unit_column",
        "context_column_present": unit_column_present,
        "feature_count_before": int(data.train_X.shape[1]),
        "numeric_feature_count": int(len(numeric_columns)),
        "contextualized_feature_count": int(len(selected)),
        "affected_numeric_feature_share": float(len(selected) / len(numeric_columns)),
        "selected_features": [str(column) for column in selected],
        "multipliers": [float(value) for value in multipliers],
    }


def _original_condition_report(*, seed: int, condition: str) -> dict[str, Any]:
    return {
        "stage": "pre_finite",
        "seed": seed,
        "condition": condition,
        "transformation_setting": "no_transformation",
        "fit_side_role": "original",
        "test_side_role": "original",
        "scale_spread": 1.0,
        "unit_visibility": "no_unit_column",
        "context_column_present": False,
        "skipped": True,
    }


def _parse_condition(condition: str, *, cfg: Mapping[str, Any], data: Any) -> tuple[float, bool, str]:
    if condition == "scale_1x_no_unit_column":
        return 1.0, False, "no_transformation"
    if condition == "scale_1x_with_unit_column":
        return 1.0, True, "matched_transformation"
    if condition in {"scale_sx_no_unit_column", "scale_sx_with_unit_column"}:
        raw_spread = resolve_calibrated_value(
            cfg.get("scale_spread"),
            selections=cfg.get("scale_spread_by_dataset"),
            metadata=getattr(data, "metadata", {}) or {},
            name="contextual_scale.scale_spread",
        )
        try:
            scale_spread = float(raw_spread)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"contextual_scale.scale_spread must be a finite number >= 1, got {raw_spread!r}"
            ) from exc
        if not math.isfinite(scale_spread) or scale_spread < 1.0:
            raise ValueError(f"contextual_scale.scale_spread must be a finite number >= 1, got {raw_spread!r}")
        return scale_spread, condition.endswith("with_unit_column"), "matched_transformation"
    raise ValueError(
        "contextual_scale.condition must be none, scale_1x_no_unit_column, scale_1x_with_unit_column, "
        "scale_sx_no_unit_column, or scale_sx_with_unit_column, "
        f"got {condition!r}"
    )


def _resolve_feature_count(cfg: Mapping[str, Any], *, data: Any) -> int:
    raw_count = resolve_calibrated_value(
        cfg.get("feature_count") or 1,
        selections=cfg.get("feature_count_by_dataset"),
        metadata=getattr(data, "metadata", {}) or {},
        name="contextual_scale.feature_count",
    )
    try:
        count = int(raw_count)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"contextual_scale.feature_count must be a positive integer, got {raw_count!r}") from exc
    if count < 1:
        raise ValueError(f"contextual_scale.feature_count must be a positive integer, got {raw_count!r}")
    return count


def _selected_numeric_features(
    frame: pd.DataFrame,
    labels: np.ndarray,
    numeric_columns: list[Any],
    *,
    task_type: str,
    max_features: int,
    strategy: str,
    rng: np.random.Generator,
    seed: int,
) -> list[Any]:
    count = max(1, min(max_features, len(numeric_columns)))
    if strategy == "first":
        return numeric_columns[:count]
    if strategy == "random":
        idx = np.sort(rng.choice(len(numeric_columns), size=count, replace=False).astype(int))
        return [numeric_columns[pos] for pos in idx]
    if strategy != "importance":
        raise ValueError(f"contextual_scale.feature_strategy must be importance, random, or first, got {strategy!r}")
    x_numeric = _finite_numeric_matrix(frame, numeric_columns)
    try:
        scores = (
            mutual_info_regression(x_numeric, labels, random_state=seed)
            if task_type == "regression"
            else mutual_info_classif(x_numeric, labels, random_state=seed)
        )
    except Exception:
        scores = np.nan_to_num(np.var(x_numeric, axis=0), nan=0.0)
    selected_idx = np.sort(np.argsort(np.nan_to_num(scores, nan=0.0))[::-1][:count].astype(int))
    return [numeric_columns[pos] for pos in selected_idx]


def _finite_numeric_matrix(frame: pd.DataFrame, columns: list[Any]) -> np.ndarray:
    values = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if values.size == 0:
        return values.reshape((len(frame), 0))
    finite = np.where(np.isfinite(values), values, np.nan)
    medians = np.nanmedian(finite, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    return np.where(np.isfinite(values), values, medians)


def _contextualized_frame(
    frame: pd.DataFrame,
    ids: np.ndarray,
    *,
    selected_features: list[Any],
    multipliers: np.ndarray,
    context_column: str,
    include_context: bool,
    group_by_id: dict[Any, int],
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = frame.copy()
    groups = np.asarray([_group_for_id(row_id, group_by_id, len(multipliers), rng) for row_id in ids], dtype=int)
    row_multipliers = multipliers[groups]
    for column in selected_features:
        values = pd.to_numeric(out[column], errors="coerce").to_numpy(dtype=float)
        out[column] = np.asarray(values / row_multipliers, dtype=np.float32)
    if include_context:
        out[context_column] = pd.Series([f"unit_{idx}" for idx in groups], dtype="object", index=out.index)
    return out


def _group_for_id(row_id: Any, group_by_id: dict[Any, int], group_count: int, rng: np.random.Generator) -> int:
    if row_id not in group_by_id:
        group_by_id[row_id] = int(math.floor(rng.random() * group_count))
    return group_by_id[row_id]
