"""Role-aware distribution-shift facade for unified foundation-model data.

The module intentionally depends only on local data objects plus numpy/pandas/
sklearn.  It has no W&B, web, Slurm, or launch-layer dependencies.

Example:
    >>> import numpy as np
    >>> import pandas as pd
    >>> from dataclasses import dataclass
    >>> @dataclass(frozen=True)
    ... class Data:
    ...     task_type: str
    ...     train_X: pd.DataFrame
    ...     train_y: np.ndarray
    ...     val_context_X: pd.DataFrame
    ...     val_context_y: np.ndarray
    ...     val_X: pd.DataFrame
    ...     val_y: np.ndarray
    ...     test_context_X: pd.DataFrame
    ...     test_context_y: np.ndarray
    ...     test_X: pd.DataFrame
    ...     test_y: np.ndarray
    ...     metadata: dict
    >>> data = Data("binary", pd.DataFrame({"x": [0, 1]}), np.array([0, 1]),
    ...     pd.DataFrame({"x": [0, 1]}), np.array([0, 1]),
    ...     pd.DataFrame({"x": [2]}), np.array([1]),
    ...     pd.DataFrame({"x": [0, 1, 2]}), np.array([0, 1, 1]),
    ...     pd.DataFrame({"x": [3]}), np.array([0]), {})
    >>> out = apply_pre_finite_distribution_shift(
    ...     data,
    ...     {"preprocess": {"steps": ["label_noise"], "label_noise": {"rate": 0}}},
    ...     seed=0,
    ... )
    >>> out.train_y.tolist()
    [0, 1]
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ._common import (
    derive_seed,
    ensure_role_metadata,
    get_nested,
    with_distribution_shift_reports,
)
from .contextual import apply_contextual_scale
from .label_feature import (
    apply_feature_mask,
    apply_label_noise,
    apply_spurious_feature,
)
from .post_finite import (
    apply_numeric_scale,
    apply_shuffle_feature_order,
)
from .row_sampling import (
    apply_rebalance_train,
    apply_subsample_train,
)
from .state import TargetTransformState
from .target import (
    apply_heavy_tail_contamination,
    apply_target_transform,
)

PRE_FINITE_STEPS = {
    "subsample_train",
    "rebalance_train",
    "feature_mask",
    "feature_drop",
    "label_noise",
    "contextual_scale",
    "contextual_units",
    "numeric_contextualization",
    "spurious_feature",
    "spurious_correlation",
    "target_transform",
    "heavy_tail",
    "heavy_tail_contamination",
}
POST_FINITE_STEPS = {"shuffle_feature_order", "feature_order", "numeric_scale"}


def apply_pre_finite_distribution_shift(data: Any, config: Mapping[str, Any], *, seed: int) -> Any:
    """Apply row/label/target/schema transforms before finite preprocessing.

    Example:
        >>> callable(apply_pre_finite_distribution_shift)
        True
    """
    out = ensure_role_metadata(data)
    reports: dict[str, Any] = {}
    for step in _preprocess_steps(config):
        canonical = _canonical_step(step)
        if canonical not in PRE_FINITE_STEPS:
            continue
        step_seed = derive_seed(seed, canonical)
        rng = np.random.default_rng(step_seed)
        before = out
        if canonical == "subsample_train":
            out, report = apply_subsample_train(out, _step_cfg(config, "subsample_train"), rng, step_seed)
        elif canonical == "rebalance_train":
            out, report = apply_rebalance_train(out, _step_cfg(config, "rebalance_train"), rng, step_seed)
        elif canonical in {"feature_mask", "feature_drop"}:
            out, report = apply_feature_mask(out, _step_cfg(config, "feature_mask"), rng, step_seed)
        elif canonical == "label_noise":
            out, report = apply_label_noise(out, _step_cfg(config, "label_noise"), rng, step_seed)
        elif canonical == "contextual_scale":
            out, report = apply_contextual_scale(out, _step_cfg(config, "contextual_scale"), rng, step_seed)
        elif canonical in {"spurious_feature", "spurious_correlation"}:
            out, report = apply_spurious_feature(out, _step_cfg(config, "spurious_feature"), rng, step_seed)
        elif canonical in {"target_transform", "heavy_tail"}:
            out, report = apply_target_transform(out, _step_cfg(config, "target_transform"), step_seed)
        elif canonical == "heavy_tail_contamination":
            out, report = apply_heavy_tail_contamination(
                out,
                _step_cfg(config, "heavy_tail_contamination"),
                rng,
                step_seed,
            )
        else:
            report = {"stage": "pre_finite", "seed": step_seed, "skipped": True, "reason": "unknown_step"}
        reports[canonical] = report
        if out is before and not report.get("skipped"):
            raise RuntimeError(f"distribution-shift step {canonical!r} reported mutation but returned original data")
    return with_distribution_shift_reports(out, "pre_finite", reports) if reports else out


def apply_post_finite_distribution_shift(data: Any, config: Mapping[str, Any], *, seed: int) -> Any:
    """Apply positional/query transforms after finite numeric preprocessing.

    Example:
        >>> callable(apply_post_finite_distribution_shift)
        True
    """
    out, _comparison = apply_post_finite_distribution_shift_with_comparison(data, config, seed=seed)
    return out


def apply_post_finite_distribution_shift_with_comparison(
    data: Any,
    config: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[Any, Any | None]:
    """Apply post-finite transforms and return an optional paired test view."""
    out = ensure_role_metadata(data)
    comparison = None
    reports: dict[str, Any] = {}
    for step in _preprocess_steps(config):
        canonical = _canonical_step(step)
        if canonical not in POST_FINITE_STEPS:
            continue
        step_seed = derive_seed(seed, canonical)
        rng = np.random.default_rng(step_seed)
        if canonical in {"shuffle_feature_order", "feature_order"}:
            out, step_comparison, report = apply_shuffle_feature_order(
                out,
                _step_cfg(config, "shuffle_feature_order"),
                rng,
                step_seed,
            )
        elif canonical == "numeric_scale":
            out, step_comparison, report = apply_numeric_scale(
                out,
                _step_cfg(config, "numeric_scale"),
                step_seed,
            )
        else:
            step_comparison = None
            report = {"stage": "post_finite", "seed": step_seed, "skipped": True, "reason": "unknown_step"}
        if step_comparison is not None:
            if comparison is not None:
                raise ValueError("Only one paired post-finite test-view transform may be active in a run")
            comparison = step_comparison
        reports[canonical] = report
    if reports:
        out = with_distribution_shift_reports(out, "post_finite", reports)
        if comparison is not None:
            comparison = with_distribution_shift_reports(comparison, "post_finite", reports)
    return out, comparison


def inverse_transform_metric_prediction(
    prediction: Any,
    *,
    task_type: str,
    target_transform_state: TargetTransformState | None,
) -> Any:
    """Inverse-transform regression predictions before original-scale metrics."""
    if task_type != "regression" or target_transform_state is None:
        return prediction
    return target_transform_state.inverse(prediction)


def _preprocess_steps(config: Mapping[str, Any]) -> list[str]:
    steps = get_nested(config, "preprocess.steps")
    if not isinstance(steps, list):
        return []
    return [str(step).strip() for step in steps if str(step).strip()]


def _canonical_step(step: str) -> str:
    text = str(step).strip().lower().replace("-", "_")
    aliases = {
        "feature_order": "shuffle_feature_order",
        "feature_drop": "feature_mask",
        "spurious_correlation": "spurious_feature",
        "contextual_units": "contextual_scale",
        "numeric_contextualization": "contextual_scale",
        "heavy_tail": "target_transform",
    }
    return aliases.get(text, text)


def _step_cfg(config: Mapping[str, Any], step: str) -> Mapping[str, Any]:
    candidates = [f"preprocess.{step}"]
    if step == "feature_mask":
        candidates.append("preprocess.feature_drop")
    if step == "shuffle_feature_order":
        candidates.append("preprocess.feature_order")
    if step == "spurious_feature":
        candidates.append("preprocess.spurious_correlation")
    if step == "contextual_scale":
        candidates.extend(("preprocess.contextual_units", "preprocess.numeric_contextualization"))
    if step == "target_transform":
        candidates.append("preprocess.heavy_tail")
    for path in candidates:
        value = get_nested(config, path)
        if isinstance(value, Mapping):
            return value
    return {}
