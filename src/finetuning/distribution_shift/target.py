"""Regression-target distribution interventions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np

from finetuning.distribution_shift._common import parse_rate, resolve_calibrated_value, safe_mean
from finetuning.distribution_shift.state import TargetTransformState


def apply_target_transform(data: Any, cfg: Mapping[str, Any], seed: int) -> tuple[Any, dict[str, Any]]:
    """Transform training/context regression targets and keep query targets original-scale."""
    mode = str(cfg.get("mode") or "none").strip().lower()
    if mode in {"", "none", "off", "disabled"}:
        return data, {"stage": "pre_finite", "seed": seed, "mode": "none", "skipped": True}
    if data.task_type != "regression":
        return data, {
            "stage": "pre_finite",
            "seed": seed,
            "mode": mode,
            "skipped": True,
            "reason": f"unsupported_task_type:{data.task_type}",
        }
    train_y = np.asarray(data.train_y, dtype=float)
    if mode == "robust_log1p":
        finite = train_y[np.isfinite(train_y)]
        minimum = float(np.min(finite)) if finite.size else 0.0
        shift = max(0.0, -minimum) + 1e-6

        def transform(y: Any) -> np.ndarray:
            return np.log1p(np.asarray(y, dtype=float) + shift)

        state = TargetTransformState(mode=mode, shift=shift)
        report_extra = {"shift": shift}
    elif mode == "winsor_p01_p99":
        finite = train_y[np.isfinite(train_y)]
        lower = float(np.quantile(finite, 0.01)) if finite.size else 0.0
        upper = float(np.quantile(finite, 0.99)) if finite.size else 0.0

        def transform(y: Any) -> np.ndarray:
            return np.clip(np.asarray(y, dtype=float), lower, upper)

        state = TargetTransformState(mode=mode, lower=lower, upper=upper)
        report_extra = {"lower": lower, "upper": upper, "inverse": "identity_after_clipping"}
    else:
        raise ValueError(f"target_transform.mode must be none, winsor_p01_p99, or robust_log1p, got {mode!r}")
    train_transformed = transform(data.train_y)
    val_context_transformed = transform(data.val_context_y)
    test_context_transformed = transform(data.test_context_y)
    validation_target = data.val_fit_y if getattr(data, "val_fit_y", None) is not None else data.val_y
    val_fit_y = transform(validation_target)
    report = {
        "stage": "pre_finite",
        "seed": seed,
        "mode": mode,
        "transformation_setting": "fit_side_transformation",
        "fit_side_role": "transformed_target_representation",
        "test_side_role": "original",
        "train_target_mean_before": safe_mean(data.train_y),
        "train_target_mean_after": safe_mean(train_transformed),
        "val_query_targets_preserved": True,
        "test_query_targets_preserved": True,
        **report_extra,
    }
    state = replace(state, report=report)
    updates = {
        "train_y": train_transformed,
        "val_context_y": val_context_transformed,
        "test_context_y": test_context_transformed,
    }
    if hasattr(data, "val_fit_y"):
        updates["val_fit_y"] = val_fit_y
    if hasattr(data, "target_transform_state"):
        updates["target_transform_state"] = state
    return replace(data, **updates), report


def apply_heavy_tail_contamination(
    data: Any,
    cfg: Mapping[str, Any],
    rng: np.random.Generator,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    """Contaminate every fit-side target while keeping test-side queries clean."""
    raw_rate = cfg.get("rate")
    magnitude_iqr = float(cfg.get("magnitude_iqr") or 10.0)
    if str(raw_rate).strip().lower() == "calibrated" and cfg.get("selection_by_dataset") is not None:
        selection = resolve_calibrated_value(
            raw_rate,
            selections=cfg.get("selection_by_dataset"),
            metadata=getattr(data, "metadata", {}) or {},
            name="heavy_tail_contamination.selection",
        )
        if not isinstance(selection, Mapping) or "rate" not in selection:
            raise ValueError(
                "heavy_tail_contamination.selection_by_dataset entries must contain rate and magnitude_iqr"
            )
        raw_rate = selection["rate"]
        magnitude_iqr = float(selection.get("magnitude_iqr") or magnitude_iqr)
    else:
        raw_rate = resolve_calibrated_value(
            raw_rate,
            selections=cfg.get("rate_by_dataset"),
            metadata=getattr(data, "metadata", {}) or {},
            name="heavy_tail_contamination.rate",
        )
    rate = parse_rate(raw_rate, name="heavy_tail_contamination.rate")
    if rate == 0.0:
        return data, {
            "stage": "pre_finite",
            "seed": seed,
            "rate": 0.0,
            "magnitude_iqr": magnitude_iqr,
            "transformation_setting": "no_transformation",
            "fit_side_role": "original",
            "test_side_role": "original",
            "skipped": True,
        }
    if data.task_type != "regression":
        return data, {
            "stage": "pre_finite",
            "seed": seed,
            "rate": rate,
            "skipped": True,
            "reason": f"unsupported_task_type:{data.task_type}",
        }
    scale = _iqr_scale(np.asarray(data.train_y, dtype=float))
    train_y, train_changed, train_offsets = _contaminate_targets_by_id(
        np.asarray(data.train_y, dtype=float),
        np.asarray(data.train_ids, dtype=object),
        rate=rate,
        magnitude_iqr=magnitude_iqr,
        rng=rng,
        scale=scale,
    )
    validation_target = data.val_fit_y if getattr(data, "val_fit_y", None) is not None else data.val_y
    val_fit_y, val_changed, val_offsets = _contaminate_targets_by_id(
        np.asarray(validation_target, dtype=float),
        np.asarray(data.val_ids, dtype=object),
        rate=rate,
        magnitude_iqr=magnitude_iqr,
        rng=rng,
        scale=scale,
    )
    fit_offsets = {**train_offsets, **val_offsets}
    fit_ids = set(np.asarray(data.train_ids, dtype=object).tolist()) | set(
        np.asarray(data.val_ids, dtype=object).tolist()
    )
    val_context_y, val_context_changed = _contaminate_context_targets(
        np.asarray(data.val_context_y, dtype=float),
        np.asarray(data.val_context_ids, dtype=object),
        rate=rate,
        magnitude_iqr=magnitude_iqr,
        rng=rng,
        fit_ids=fit_ids,
        fit_offsets=fit_offsets,
        scale=scale,
    )
    test_context_y, test_context_changed = _contaminate_context_targets(
        np.asarray(data.test_context_y, dtype=float),
        np.asarray(data.test_context_ids, dtype=object),
        rate=rate,
        magnitude_iqr=magnitude_iqr,
        rng=rng,
        fit_ids=fit_ids,
        fit_offsets=fit_offsets,
        scale=scale,
    )
    return replace(
        data,
        train_y=train_y,
        val_fit_y=val_fit_y,
        val_context_y=val_context_y,
        test_context_y=test_context_y,
    ), {
        "stage": "pre_finite",
        "seed": seed,
        "rate": rate,
        "transformation_setting": "fit_side_transformation",
        "fit_side_role": "transformed",
        "test_side_role": "original",
        "magnitude_iqr": magnitude_iqr,
        "training_iqr": scale,
        "train_changed": int(train_changed),
        "validation_fit_changed": int(val_changed),
        "val_context_changed": int(val_context_changed),
        "test_context_changed": int(test_context_changed),
        "val_query_targets_preserved": True,
        "test_query_targets_preserved": True,
    }


def _iqr_scale(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return 1.0
    scale = float(np.quantile(finite, 0.75) - np.quantile(finite, 0.25))
    return scale if scale > 1e-12 else 1.0


def _contaminate_targets_by_id(
    values: np.ndarray,
    ids: np.ndarray,
    *,
    rate: float,
    magnitude_iqr: float,
    rng: np.random.Generator,
    scale: float | None = None,
) -> tuple[np.ndarray, int, dict[Any, float]]:
    out = values.copy()
    effective_scale = _iqr_scale(values) if scale is None else float(scale)
    selected = rng.random(len(out)) < rate
    offsets: dict[Any, float] = {}
    for pos in np.flatnonzero(selected):
        offset = float(magnitude_iqr * effective_scale * (1.0 + rng.pareto(2.0)))
        out[pos] += offset
        offsets[ids[pos]] = offset
    return out, int(selected.sum()), offsets


def _contaminate_context_targets(
    values: np.ndarray,
    ids: np.ndarray,
    *,
    rate: float,
    magnitude_iqr: float,
    rng: np.random.Generator,
    fit_ids: set[Any],
    fit_offsets: Mapping[Any, float],
    scale: float,
) -> tuple[np.ndarray, int]:
    out = values.copy()
    changed = np.zeros(len(values), dtype=bool)
    known_fit_rows = np.asarray([row_id in fit_ids for row_id in ids], dtype=bool)
    for pos in np.flatnonzero(known_fit_rows):
        if ids[pos] in fit_offsets:
            out[pos] += float(fit_offsets[ids[pos]])
            changed[pos] = True
    other_pos = np.flatnonzero(~known_fit_rows)
    if other_pos.size:
        contaminated, _count, _offsets = _contaminate_targets_by_id(
            out[other_pos],
            ids[other_pos],
            rate=rate,
            magnitude_iqr=magnitude_iqr,
            rng=rng,
            scale=scale,
        )
        changed[other_pos] = contaminated != out[other_pos]
        out[other_pos] = contaminated
    return out, int(changed.sum())
