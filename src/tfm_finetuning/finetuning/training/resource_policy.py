"""Estimator and memory fallback policy for runs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from tfm_finetuning.finetuning.datasets.loading import SplitEvaluationData
from tfm_finetuning.finetuning.models.base import (
    CommonConfigAdapter,
    CommonFineTuneConfig,
    EstimatorMinimumAdapter,
    FoundationModelAdapter,
)
from tfm_finetuning.finetuning.models.registry import ModelSpec
from tfm_finetuning.utils.config import get_nested

_RESOURCE_FALLBACK_STRICT = "strict"
_RESOURCE_FALLBACK_OOM_CONTEXT_RETRY = "oom_context_retry"
_ESTIMATOR_FALLBACK_STRICT = "strict"
_ESTIMATOR_FALLBACK_RETRY = "estimator_retry"


def _is_off_like(value: Any) -> bool:
    if isinstance(value, bool):
        return not value
    return str(value).strip().lower() in {"", "0", "false", "off", "none", "disabled", "no"}


def _normalize_resource_fallback_policy(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"strict", "off", "none", "disabled", "false", "0", "no"}:
        return _RESOURCE_FALLBACK_STRICT
    if text in {"oom_context_retry", "context_retry", "retry", "auto", "true", "1", "yes", "on"}:
        return _RESOURCE_FALLBACK_OOM_CONTEXT_RETRY
    raise SystemExit(
        "finetune.resource_fallback_policy must be 'strict' or 'oom_context_retry', "
        f"got {value!r}."
    )


def _resolve_resource_fallback_policy(config: Mapping[str, Any], *, model_id: str) -> str:
    raw_policy = get_nested(config, "finetune.resource_fallback_policy")
    if raw_policy is not None:
        return _normalize_resource_fallback_policy(raw_policy)
    raw_sizes = get_nested(config, "finetune.oom_retry_context_sizes")
    if raw_sizes is None:
        raw_sizes = get_nested(config, "finetune.oom_retry_context_size")
    if raw_sizes is not None:
        parsed = _coerce_context_sizes(raw_sizes)
        return _RESOURCE_FALLBACK_STRICT if parsed == () else _RESOURCE_FALLBACK_OOM_CONTEXT_RETRY
    return _RESOURCE_FALLBACK_OOM_CONTEXT_RETRY if model_id == "tabicl_v2" else _RESOURCE_FALLBACK_STRICT


def _normalize_estimator_fallback_policy(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"strict", "off", "none", "disabled", "false", "0", "no"}:
        return _ESTIMATOR_FALLBACK_STRICT
    if text in {"estimator_retry", "retry", "auto", "true", "1", "yes", "on"}:
        return _ESTIMATOR_FALLBACK_RETRY
    raise SystemExit(
        "finetune.estimator_fallback_policy must be 'strict' or 'estimator_retry', "
        f"got {value!r}."
    )


def _resolve_estimator_fallback_policy(config: Mapping[str, Any], *, model_id: str) -> str:
    raw_policy = get_nested(config, "finetune.estimator_fallback_policy")
    if raw_policy is not None:
        return _normalize_estimator_fallback_policy(raw_policy)
    return _ESTIMATOR_FALLBACK_RETRY if model_id.startswith("tabpfn_v2") else _ESTIMATOR_FALLBACK_STRICT


def _coerce_context_sizes(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None if value else ()
    if isinstance(value, str):
        if _is_off_like(value):
            return ()
        raw_values = [part for part in re.split(r"[\s,]+", value.strip()) if part]
    elif isinstance(value, list | tuple | set):
        raw_values = list(value)
    else:
        raw_values = [value]

    sizes: list[int] = []
    for raw_value in raw_values:
        if _is_off_like(raw_value):
            continue
        try:
            size = int(raw_value)
        except Exception as exc:
            message = f"finetune.oom_retry_context_sizes must contain integers, got {raw_value!r}."
            raise SystemExit(message) from exc
        if size <= 0:
            raise SystemExit("finetune.oom_retry_context_sizes must contain positive integers.")
        sizes.append(size)
    return tuple(sizes)


def _dedupe_context_sizes(sizes: list[int]) -> tuple[int, ...]:
    return tuple(dict.fromkeys(size for size in sizes if size > 0))


def _default_oom_retry_context_sizes(context_size: int) -> tuple[int, ...]:
    base = max(int(context_size), 1)
    if base <= 500:
        return (base,)
    candidates = {base, max(base // 2, 500), max(base // 4, 500), 1000, 500}
    return tuple(sorted((size for size in candidates if 0 < size <= base), reverse=True))


def _resolve_oom_retry_context_sizes(
    config: Mapping[str, Any],
    *,
    model_id: str,
    common_config: CommonFineTuneConfig,
) -> tuple[int, ...]:
    """Return ordered context-size attempts for one model fit."""
    configured = int(common_config.context_size)
    if _resolve_resource_fallback_policy(config, model_id=model_id) == _RESOURCE_FALLBACK_STRICT:
        return (configured,)
    raw_value = get_nested(config, "finetune.oom_retry_context_sizes")
    if raw_value is None:
        raw_value = get_nested(config, "finetune.oom_retry_context_size")
    if raw_value is None:
        return _default_oom_retry_context_sizes(configured)

    parsed = _coerce_context_sizes(raw_value)
    if parsed is None:
        return _default_oom_retry_context_sizes(configured)
    if not parsed:
        return (configured,)
    return _dedupe_context_sizes([configured, *(size for size in parsed if size <= configured)])


def _minimum_required_estimators(adapter: FoundationModelAdapter, *, feature_count: int) -> int:
    if not isinstance(adapter, EstimatorMinimumAdapter):
        return 1
    required = adapter.minimum_required_estimators(int(feature_count))
    return 1 if required is None else max(1, int(required))


def _estimator_config_payload(config: CommonFineTuneConfig) -> dict[str, int]:
    return {
        "n_estimators_finetune": int(config.n_estimators_finetune),
        "n_estimators_validation": int(config.n_estimators_validation),
        "n_estimators_final_inference": int(config.n_estimators_final_inference),
    }


def _apply_estimator_fallback_policy(
    *,
    adapter: FoundationModelAdapter,
    data: SplitEvaluationData,
    spec: ModelSpec,
    config: Mapping[str, Any],
    common_config: CommonFineTuneConfig,
) -> tuple[CommonFineTuneConfig, dict[str, Any]]:
    """Apply the explicit strict/retry estimator-count policy."""
    feature_count = int(data.train_X.shape[1])
    required_minimum = _minimum_required_estimators(adapter, feature_count=feature_count)
    configured = _estimator_config_payload(common_config)
    below_required = {key: value for key, value in configured.items() if value < required_minimum}
    policy = _resolve_estimator_fallback_policy(config, model_id=spec.model_id)
    if below_required and policy == _ESTIMATOR_FALLBACK_STRICT:
        details = ", ".join(f"{key}={value}" for key, value in sorted(below_required.items()))
        raise SystemExit(
            f"{spec.model_id} requires at least {required_minimum} estimator(s) for {feature_count} features, "
            f"but strict estimator fallback policy received {details}. Set "
            "finetune.estimator_fallback_policy='estimator_retry' or configure larger estimator counts."
        )

    effective_config = common_config
    if below_required:
        effective_config = replace(
            common_config,
            n_estimators_finetune=max(int(common_config.n_estimators_finetune), required_minimum),
            n_estimators_validation=max(int(common_config.n_estimators_validation), required_minimum),
            n_estimators_final_inference=max(int(common_config.n_estimators_final_inference), required_minimum),
        )
    metadata = {
        "estimator_fallback_policy": policy,
        "estimator_fallback_triggered": bool(below_required),
        "required_min_n_estimators": required_minimum,
        "feature_count_for_estimator_requirement": feature_count,
        "configured_n_estimators": configured,
        "effective_n_estimators": _estimator_config_payload(effective_config),
        "below_required_n_estimators": below_required,
        "estimator_fallback_reason": (
            f"{spec.model_id} feature count {feature_count} requires at least {required_minimum} estimator(s)."
            if below_required
            else None
        ),
    }
    return effective_config, metadata


def _configure_adapter_common_config(
    adapter: FoundationModelAdapter,
    common_config: CommonFineTuneConfig,
) -> None:
    """Pass shared runtime options only to adapters that consume them."""
    if isinstance(adapter, CommonConfigAdapter):
        adapter.set_common_config(common_config)


def _runtime_option_payload(common_config: CommonFineTuneConfig) -> dict[str, Any]:
    """Return common runtime options recorded in final reports."""
    return {"ignore_pretraining_limits": bool(common_config.ignore_pretraining_limits)}
