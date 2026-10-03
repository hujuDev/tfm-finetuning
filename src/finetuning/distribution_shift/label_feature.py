"""Label, feature-mask, and spurious-feature distribution interventions."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from finetuning.distribution_shift._common import (
    jsonable_label,
    label_signal,
    mixed_frame_to_numeric_matrix,
    parse_rate,
    safe_corr,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd


def apply_label_noise(
    data: Any,
    cfg: Mapping[str, Any],
    rng: np.random.Generator,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    """Inject label noise into fine-tuning/context labels while preserving query labels."""
    rate = parse_rate(cfg.get("rate"), name="label_noise.rate")
    if rate == 0.0:
        return data, {"stage": "pre_finite", "seed": seed, "rate": 0.0, "skipped": True}
    if data.task_type not in {"binary", "multiclass"}:
        return data, {
            "stage": "pre_finite",
            "seed": seed,
            "rate": rate,
            "skipped": True,
            "reason": f"unsupported_task_type:{data.task_type}",
        }
    train_size = len(data.train_y)
    fit_labels = np.concatenate([np.asarray(data.train_y), np.asarray(data.val_y)])
    fit_ids = np.concatenate(
        [np.asarray(data.train_ids, dtype=object), np.asarray(data.val_ids, dtype=object)]
    )
    classes = np.unique(fit_labels)
    fit_noisy, changed_fit, mapping = _noise_labels_by_id(
        fit_labels,
        fit_ids,
        classes=classes,
        rate=rate,
        rng=rng,
    )
    train_noisy = fit_noisy[:train_size]
    val_fit_noisy = fit_noisy[train_size:]
    val_context_noisy, changed_val_context = _mapped_context_labels(
        np.asarray(data.val_context_y),
        np.asarray(data.val_context_ids, dtype=object),
        mapping=mapping,
    )
    test_context_noisy, changed_test_context = _mapped_context_labels(
        np.asarray(data.test_context_y),
        np.asarray(data.test_context_ids, dtype=object),
        mapping=mapping,
    )
    updates = {
        "train_y": train_noisy,
        "val_context_y": val_context_noisy,
        "test_context_y": test_context_noisy,
    }
    if hasattr(data, "val_fit_y"):
        updates["val_fit_y"] = val_fit_noisy
    return replace(data, **updates), {
        "stage": "pre_finite",
        "seed": seed,
        "rate": rate,
        "transformation_setting": "fit_side_transformation",
        "fit_side_role": "corrupted_labels",
        "test_side_role": "original",
        "classes": [jsonable_label(value) for value in classes],
        "fit_pool_changed": int(changed_fit),
        "train_changed": int(np.sum(train_noisy != np.asarray(data.train_y))),
        "validation_fit_changed": int(np.sum(val_fit_noisy != np.asarray(data.val_y))),
        "val_context_changed": int(changed_val_context),
        "test_context_changed": int(changed_test_context),
        "validation_query_labels_preserved": True,
        "test_query_labels_preserved": True,
    }


def apply_feature_mask(
    data: Any,
    cfg: Mapping[str, Any],
    rng: np.random.Generator,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    """Drop selected features from every role before finite numeric preprocessing."""
    fraction = parse_rate(cfg.get("fraction"), name="feature_mask.fraction")
    if fraction == 0.0 or data.train_X.shape[1] <= 1:
        return data, {
            "stage": "pre_finite",
            "seed": seed,
            "fraction": fraction,
            "skipped": True,
            "reason": "feature_mask_noop_or_too_few_features",
        }
    strategy = str(cfg.get("strategy") or "importance").strip().lower()
    drop_count = max(1, min(data.train_X.shape[1] - 1, math.floor(data.train_X.shape[1] * fraction)))
    columns = list(data.train_X.columns)
    if strategy == "random":
        drop_idx = np.sort(rng.choice(len(columns), size=drop_count, replace=False).astype(int))
    elif strategy == "importance":
        x_numeric = mixed_frame_to_numeric_matrix(data.train_X)
        is_regression = data.task_type == "regression"
        try:
            scores = (
                mutual_info_regression(x_numeric, np.asarray(data.train_y), random_state=seed)
                if is_regression
                else mutual_info_classif(x_numeric, np.asarray(data.train_y), random_state=seed)
            )
        except Exception:
            scores = np.nan_to_num(np.var(x_numeric, axis=0), nan=0.0)
        drop_idx = np.sort(np.argsort(np.nan_to_num(scores, nan=0.0))[::-1][:drop_count].astype(int))
    else:
        raise ValueError(f"feature_mask.strategy must be random or importance, got {strategy!r}")
    drop_columns = [columns[idx] for idx in drop_idx]
    keep_columns = [column for column in columns if column not in set(drop_columns)]
    return replace(
        data,
        train_X=data.train_X.loc[:, keep_columns].copy(),
        val_context_X=data.val_context_X.loc[:, keep_columns].copy(),
        val_X=data.val_X.loc[:, keep_columns].copy(),
        test_context_X=data.test_context_X.loc[:, keep_columns].copy(),
        test_X=data.test_X.loc[:, keep_columns].copy(),
    ), {
        "stage": "pre_finite",
        "seed": seed,
        "fraction": fraction,
        "strategy": strategy,
        "feature_count_before": len(columns),
        "feature_count_after": len(keep_columns),
        "dropped_features": [str(column) for column in drop_columns],
        "selected_features": [str(column) for column in keep_columns],
    }


def apply_spurious_feature(
    data: Any,
    cfg: Mapping[str, Any],
    rng: np.random.Generator,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    """Add one same-schema shortcut that is correlated only on the fit side."""
    strength = parse_rate(cfg.get("strength"), name="spurious_feature.strength")
    column_name = str(cfg.get("column_name") or "__spurious_signal__")
    train_feature = _correlated_signal(np.asarray(data.train_y), strength=strength, rng=rng)
    val_feature = _correlated_signal(np.asarray(data.val_y), strength=strength, rng=rng)
    fit_mapping = dict(
        zip(
            np.asarray(data.train_ids, dtype=object).tolist(),
            train_feature.tolist(),
            strict=False,
        )
    )
    fit_mapping.update(
        zip(
            np.asarray(data.val_ids, dtype=object).tolist(),
            val_feature.tolist(),
            strict=False,
        )
    )
    val_context_feature = _context_spurious_signal(
        np.asarray(data.val_context_ids, dtype=object),
        rng=rng,
        fit_mapping=fit_mapping,
    )
    test_context_feature = _context_spurious_signal(
        np.asarray(data.test_context_ids, dtype=object),
        rng=rng,
        fit_mapping=fit_mapping,
    )
    test_feature = rng.normal(size=len(data.test_X))
    return replace(
        data,
        train_X=_with_feature(data.train_X, column_name, train_feature),
        val_context_X=_with_feature(data.val_context_X, column_name, val_context_feature),
        val_X=_with_feature(data.val_X, column_name, val_feature),
        test_context_X=_with_feature(data.test_context_X, column_name, test_context_feature),
        test_X=_with_feature(data.test_X, column_name, test_feature),
    ), {
        "stage": "pre_finite",
        "seed": seed,
        "strength": strength,
        "transformation_setting": "no_transformation" if strength == 0.0 else "fit_side_transformation",
        "fit_side_role": "neutral" if strength == 0.0 else "correlated",
        "test_side_role": "neutral",
        "column_name": column_name,
        "same_schema_control": True,
        "train_correlation": safe_corr(train_feature, label_signal(np.asarray(data.train_y))),
        "validation_fit_correlation": safe_corr(val_feature, label_signal(np.asarray(data.val_y))),
        "test_query_correlation": safe_corr(test_feature, label_signal(np.asarray(data.test_y))),
    }


def _noise_labels_by_id(
    labels: np.ndarray,
    ids: np.ndarray,
    *,
    classes: np.ndarray,
    rate: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, dict[Any, Any]]:
    noisy = np.asarray(labels).copy()
    mapping: dict[Any, Any] = {}
    if len(classes) < 2:
        return noisy, 0, mapping
    selected = rng.random(len(noisy)) < rate
    for pos in np.flatnonzero(selected):
        current = noisy[pos]
        choices = classes[classes != current]
        if choices.size == 0:
            continue
        replacement = rng.choice(choices)
        noisy[pos] = replacement
        mapping[ids[pos]] = replacement
    return noisy, int(np.sum(noisy != labels)), mapping


def _mapped_context_labels(
    labels: np.ndarray,
    ids: np.ndarray,
    *,
    mapping: Mapping[Any, Any],
) -> tuple[np.ndarray, int]:
    noisy = np.asarray(labels).copy()
    for pos, row_id in enumerate(ids):
        if row_id in mapping:
            noisy[pos] = mapping[row_id]
    return noisy, int(np.sum(noisy != labels))


def _with_feature(frame: pd.DataFrame, column_name: str, values: np.ndarray) -> pd.DataFrame:
    out = frame.copy()
    out[column_name] = np.asarray(values, dtype=np.float32)
    return out


def _correlated_signal(labels: np.ndarray, *, strength: float, rng: np.random.Generator) -> np.ndarray:
    signal = label_signal(labels)
    noise = rng.normal(size=len(signal))
    residual = math.sqrt(max(0.0, 1.0 - strength**2)) * noise
    return strength * signal + residual


def _context_spurious_signal(
    ids: np.ndarray,
    *,
    rng: np.random.Generator,
    fit_mapping: Mapping[Any, float],
) -> np.ndarray:
    feature = rng.normal(size=len(ids))
    for pos, row_id in enumerate(ids):
        if row_id in fit_mapping:
            feature[pos] = fit_mapping[row_id]
    return feature
