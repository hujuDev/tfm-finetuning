"""Run data-preparation phase for foundation-model fine-tuning."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from finetuning.data import (
    SplitEvaluationData,
    _apply_foundation_model_row_limits,
    _densify_sparse_feature_frames,
    load_split_evaluation_data,
)
from finetuning.distribution_shift import (
    apply_post_finite_distribution_shift_with_comparison,
    apply_pre_finite_distribution_shift,
)
from finetuning.evaluation import _test_context_policy_for_data
from finetuning.fit import _record_runtime_phase
from finetuning.models.base import FoundationModelAdapter
from finetuning.preprocessing import (
    _apply_preprocessing_protocol,
    _configure_adapter_feature_schema,
    _mark_primary_test_context_comparison_data,
    _prepare_alternate_test_context_data,
    _refresh_adapter_feature_schema_metadata,
)


def _prepare_foundation_run_data(
    *,
    config: Mapping[str, Any],
    repo_root: Path,
    adapter: FoundationModelAdapter,
    runtime_phases: list[dict[str, Any]],
    resolved_seed: int,
    preprocessing_protocol: str,
    model_id: str,
) -> tuple[SplitEvaluationData, Any, SplitEvaluationData | None]:
    """Load and transform run data before estimator construction.

    Example:
        >>> callable(_prepare_foundation_run_data)
        True
    """
    phase_started = time.monotonic()
    data = load_split_evaluation_data(config=config, repo_root=repo_root)
    _record_runtime_phase(
        runtime_phases,
        phase="load_data",
        started_at=phase_started,
        train_rows=len(data.train_X),
        val_rows=len(data.val_X),
        test_rows=len(data.test_X),
        task_type=data.task_type,
    )
    phase_started = time.monotonic()
    data = apply_pre_finite_distribution_shift(data, config, seed=int(resolved_seed))
    pre_finite_shift = data.metadata.get("distribution_shift")
    pre_finite_reports = (
        pre_finite_shift.get("pre_finite")
        if isinstance(pre_finite_shift, Mapping)
        else None
    )
    _record_runtime_phase(
        runtime_phases,
        phase="distribution_shift_pre_finite",
        started_at=phase_started,
        applied=bool(pre_finite_reports),
        step_count=len(pre_finite_reports) if isinstance(pre_finite_reports, Mapping) else 0,
        train_rows=len(data.train_X),
        val_context_rows=len(data.val_context_X),
        test_context_rows=len(data.test_context_X),
    )
    phase_started = time.monotonic()
    data = _apply_foundation_model_row_limits(data, config=config, seed=int(resolved_seed))
    row_limits = data.metadata.get("row_limits")
    effective_rows = row_limits.get("effective_rows") if isinstance(row_limits, Mapping) else {}
    if not isinstance(effective_rows, Mapping):
        effective_rows = {}
    _record_runtime_phase(
        runtime_phases,
        phase="row_limits",
        started_at=phase_started,
        applied=isinstance(row_limits, Mapping),
        train_rows=effective_rows.get("train", len(data.train_X)),
        val_context_rows=effective_rows.get("val_context", len(data.val_context_X)),
        val_query_rows=effective_rows.get("val_query", len(data.val_X)),
        test_context_rows=effective_rows.get("test_context", len(data.test_context_X)),
        test_query_rows=effective_rows.get("test_query", len(data.test_X)),
    )
    phase_started = time.monotonic()
    data = _densify_sparse_feature_frames(data)
    sparse_densification = data.metadata.get("sparse_feature_densification")
    _record_runtime_phase(
        runtime_phases,
        phase="sparse_feature_densification",
        started_at=phase_started,
        applied=isinstance(sparse_densification, Mapping),
    )
    phase_started = time.monotonic()
    data = _apply_preprocessing_protocol(
        data,
        protocol=preprocessing_protocol,
        model_id=model_id,
        seed=int(resolved_seed),
    )
    feature_preprocessing = data.metadata.get("feature_preprocessing")
    _record_runtime_phase(
        runtime_phases,
        phase="feature_preprocessing",
        started_at=phase_started,
        preprocessing_protocol=preprocessing_protocol,
        method=feature_preprocessing.get("method") if isinstance(feature_preprocessing, Mapping) else None,
        raw_feature_count=(
            feature_preprocessing.get("raw_feature_count") if isinstance(feature_preprocessing, Mapping) else None
        ),
        effective_feature_count=(
            feature_preprocessing.get("effective_feature_count") if isinstance(feature_preprocessing, Mapping) else None
        ),
    )
    phase_started = time.monotonic()
    data, shift_comparison_data = apply_post_finite_distribution_shift_with_comparison(
        data,
        config,
        seed=int(resolved_seed),
    )
    data = _refresh_adapter_feature_schema_metadata(data)
    if shift_comparison_data is not None:
        shift_comparison_data = _refresh_adapter_feature_schema_metadata(shift_comparison_data)
    _configure_adapter_feature_schema(adapter, data)
    post_finite_shift = data.metadata.get("distribution_shift")
    post_finite_reports = (
        post_finite_shift.get("post_finite")
        if isinstance(post_finite_shift, Mapping)
        else None
    )
    _record_runtime_phase(
        runtime_phases,
        phase="distribution_shift_post_finite",
        started_at=phase_started,
        applied=bool(post_finite_reports),
        step_count=len(post_finite_reports) if isinstance(post_finite_reports, Mapping) else 0,
        train_features=data.train_X.shape[1],
        test_query_rows=len(data.test_X),
    )
    feature_preprocessing = data.metadata.get("feature_preprocessing")
    phase_started = time.monotonic()
    data = _mark_primary_test_context_comparison_data(data, config)
    context_comparison_data = _prepare_alternate_test_context_data(
        config=config,
        repo_root=repo_root,
        primary_data=data,
        seed=int(resolved_seed),
        preprocessing_protocol=preprocessing_protocol,
        model_id=model_id,
    )
    if shift_comparison_data is not None and context_comparison_data is not None:
        raise ValueError(
            "A run cannot combine a distribution-shift test-view comparison with "
            "foundation_model.test_context_comparison"
        )
    comparison_data = shift_comparison_data or context_comparison_data
    if context_comparison_data is not None:
        _record_runtime_phase(
            runtime_phases,
            phase="prepare_test_context_comparison",
            started_at=phase_started,
            primary_policy=_test_context_policy_for_data(data),
            alternate_policy=_test_context_policy_for_data(context_comparison_data),
            primary_test_context_rows=len(data.test_context_X),
            alternate_test_context_rows=len(context_comparison_data.test_context_X),
        )
    elif shift_comparison_data is not None:
        comparison = shift_comparison_data.metadata.get("evaluation_comparison", {})
        _record_runtime_phase(
            runtime_phases,
            phase="prepare_distribution_shift_test_views",
            started_at=phase_started,
            family=comparison.get("family"),
            fit_state=comparison.get("fit_state"),
            primary_view=comparison.get("primary_view"),
            alternate_view=comparison.get("alternate_view"),
        )
    return data, feature_preprocessing, comparison_data
