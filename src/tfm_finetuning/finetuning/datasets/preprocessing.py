"""Preprocessing protocol helpers for fine-tuning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from tfm_finetuning.finetuning.datasets.loading import (
    TEST_CONTEXT_POLICY_FULL_ORIGINAL,
    SplitEvaluationData,
    _apply_model_row_limits,
    _densify_sparse_feature_frames,
    _resolve_test_context_comparison_enabled,
    _test_context_flag_config,
    load_split_evaluation_data,
)
from tfm_finetuning.finetuning.datasets.preprocessing_protocol import (
    PREPROCESSING_NATIVE,
    PREPROCESSING_SHARED,
    SUPPORTED_PREPROCESSING_PROTOCOLS,
    normalize_preprocessing_protocol,
)
from tfm_finetuning.finetuning.distribution_shift import (
    apply_post_finite_distribution_shift,
    apply_pre_finite_distribution_shift,
    apply_test_context_majority_downsampling,
)
from tfm_finetuning.finetuning.models.base import (
    FeatureSchemaAdapter,
    FoundationModelAdapter,
    PreprocessingProtocolAdapter,
)
from tfm_finetuning.finetuning.models.preprocessing import (
    FiniteFeaturePreprocessor,
    feature_schema_metadata,
)
from tfm_finetuning.utils.config import get_nested

TEST_CONTEXT_COMPARISON_TOGGLE_VALIDATION = "toggle_validation_inclusion"
TEST_CONTEXT_COMPARISON_MAJORITY_DOWNSAMPLING = "original_majority_downsampling"


def _resolve_test_context_comparison_mode(config: Mapping[str, Any]) -> str:
    raw_mode = get_nested(config, "finetune.test_context_comparison.mode")
    mode = str(raw_mode or TEST_CONTEXT_COMPARISON_TOGGLE_VALIDATION).strip().lower().replace("-", "_")
    aliases = {
        "toggle_validation": TEST_CONTEXT_COMPARISON_TOGGLE_VALIDATION,
        "majority_downsampling": TEST_CONTEXT_COMPARISON_MAJORITY_DOWNSAMPLING,
    }
    mode = aliases.get(mode, mode)
    supported = {
        TEST_CONTEXT_COMPARISON_TOGGLE_VALIDATION,
        TEST_CONTEXT_COMPARISON_MAJORITY_DOWNSAMPLING,
    }
    if mode not in supported:
        raise ValueError(f"finetune.test_context_comparison.mode must be one of {sorted(supported)}, got {raw_mode!r}")
    return mode


def _mark_primary_test_context_comparison_data(
    data: SplitEvaluationData,
    config: Mapping[str, Any],
) -> SplitEvaluationData:
    """Attach an explicit primary-policy label for paired representation comparisons."""
    if not _resolve_test_context_comparison_enabled(config):
        return data
    if _resolve_test_context_comparison_mode(config) != TEST_CONTEXT_COMPARISON_MAJORITY_DOWNSAMPLING:
        return data
    metadata = dict(data.metadata)
    metadata["test_context_comparison_policy"] = TEST_CONTEXT_POLICY_FULL_ORIGINAL
    return replace(data, metadata=metadata)


def _resolve_preprocessing_protocol(config: Mapping[str, Any]) -> str:
    """Resolve the feature preprocessing protocol for a run.

    Example:
        >>> _resolve_preprocessing_protocol({})
        'native'
    """
    if get_nested(config, "finetune.feature_preprocessing") is not None:
        raise ValueError("Use finetune.preprocessing_protocol = native or shared.")
    raw_protocol = get_nested(config, "finetune.preprocessing_protocol")
    protocol = normalize_preprocessing_protocol(raw_protocol)
    if protocol in {None, ""}:
        return PREPROCESSING_NATIVE
    if protocol not in SUPPORTED_PREPROCESSING_PROTOCOLS:
        supported = ", ".join(sorted(SUPPORTED_PREPROCESSING_PROTOCOLS))
        raise SystemExit(
            f"finetune.preprocessing_protocol={raw_protocol!r} is unsupported; expected one of {supported}."
        )
    return protocol


def _configure_adapter_preprocessing_protocol(adapter: FoundationModelAdapter, protocol: str) -> None:
    if isinstance(adapter, PreprocessingProtocolAdapter):
        adapter.set_preprocessing_protocol(protocol)


def _configure_adapter_feature_schema(adapter: FoundationModelAdapter, data: SplitEvaluationData) -> None:
    if not isinstance(adapter, FeatureSchemaAdapter):
        return
    preprocessing = data.metadata.get("feature_preprocessing")
    schema = None
    if isinstance(preprocessing, Mapping):
        schema = preprocessing.get("adapter_feature_schema")
    adapter.set_feature_schema(schema if isinstance(schema, Mapping) else {})


def _refresh_adapter_feature_schema_metadata(data: SplitEvaluationData) -> SplitEvaluationData:
    preprocessing = data.metadata.get("feature_preprocessing")
    if not isinstance(preprocessing, Mapping):
        return data
    updated_preprocessing = dict(preprocessing)
    protocol = str(updated_preprocessing.get("protocol") or data.metadata.get("preprocessing_protocol") or "")
    protocol = normalize_preprocessing_protocol(protocol)
    categorical_indices = None if protocol == PREPROCESSING_NATIVE else ()
    updated_schema = feature_schema_metadata(data.train_X, categorical_feature_indices=categorical_indices)
    refreshed_categorical_indices = list(updated_schema["categorical_feature_indices"])
    updated_preprocessing["adapter_feature_schema"] = updated_schema
    updated_preprocessing["categorical_feature_indices"] = refreshed_categorical_indices
    metadata = dict(data.metadata)
    metadata["feature_preprocessing"] = updated_preprocessing
    if isinstance(metadata.get("preprocessing"), Mapping):
        metadata["preprocessing"] = dict(metadata["preprocessing"])
        metadata["preprocessing"]["adapter_feature_schema"] = updated_schema
        metadata["preprocessing"]["categorical_feature_indices"] = refreshed_categorical_indices
    return replace(data, metadata=metadata)


def _native_preprocessing_metadata(data: SplitEvaluationData, *, model_id: str) -> dict[str, Any]:
    columns = [str(column) for column in data.train_X.columns]
    schema = feature_schema_metadata(data.train_X)
    metadata = {
        "protocol": PREPROCESSING_NATIVE,
        "method": f"{model_id}_native",
        "fit_scope": "adapter_or_model_internal",
        "raw_feature_count": len(columns),
        "effective_feature_count": len(columns),
        "dropped_features": [],
        "selected_features": columns,
        "missing_unknown_handling": "native",
        "output_dtype": "native",
        "is_shared_preprocessing_ablation": False,
        "source_feature_schema": schema,
        "adapter_feature_schema": schema,
        "categorical_feature_indices": list(schema["categorical_feature_indices"]),
    }
    if model_id.startswith("tabpfn_v2"):
        metadata.update(_tabpfn_native_validation_encoding_metadata(schema))
    return metadata


def _tabpfn_native_validation_encoding_metadata(schema: Mapping[str, Any]) -> dict[str, Any]:
    columns = schema.get("columns")
    dtypes = schema.get("dtypes")
    categorical_feature_indices = schema.get("categorical_feature_indices")
    if (
        not isinstance(columns, list)
        or not isinstance(dtypes, Mapping)
        or not isinstance(categorical_feature_indices, list)
    ):
        return {}

    encoded_columns: list[str] = []
    for index in categorical_feature_indices:
        if not isinstance(index, int) or index < 0 or index >= len(columns):
            continue
        column = str(columns[index])
        dtype = str(dtypes.get(column, ""))
        if dtype == "category" or dtype == "object" or dtype.startswith("string"):
            encoded_columns.append(column)

    if not encoded_columns:
        return {}
    return {
        "adapter_validation_encoding": "tabpfn_native_categorical_ordinal_bridge",
        "adapter_validation_encoding_scope": "adapter_boundary_only",
        "adapter_validation_encoded_columns": encoded_columns,
        "adapter_validation_unknown_handling": "unknown_or_missing_category_to_nan",
    }


def _fit_preprocessing_protocol(
    data: SplitEvaluationData,
    *,
    protocol: str,
    seed: int,
    model_id: str,
) -> Any:
    if protocol == PREPROCESSING_NATIVE:
        return None
    if protocol == PREPROCESSING_SHARED:
        return FiniteFeaturePreprocessor.fit(data.train_X)
    supported = ", ".join(sorted(SUPPORTED_PREPROCESSING_PROTOCOLS))
    raise ValueError(
        f"Unsupported preprocessing protocol for model_id={model_id}: {protocol!r}; expected {supported}."
    )


def _preprocessing_metadata(
    *,
    preprocessor: Any,
    protocol: str,
    data: SplitEvaluationData,
    model_id: str,
) -> dict[str, Any]:
    if protocol == PREPROCESSING_NATIVE:
        return _native_preprocessing_metadata(data, model_id=model_id)
    metadata = preprocessor.metadata(protocol=protocol)
    metadata["is_shared_preprocessing_ablation"] = protocol != PREPROCESSING_NATIVE
    return metadata


def _apply_preprocessing_protocol(
    data: SplitEvaluationData,
    *,
    protocol: str,
    model_id: str,
    seed: int,
) -> SplitEvaluationData:
    """Apply the configured preprocessing protocol to every context/query split.

    Example:
        >>> data = SplitEvaluationData(
        ...     task_type="binary", problem_type="binary", metric_name="roc_auc",
        ...     train_X=pd.DataFrame({"x": [1.0, np.nan]}), train_y=np.array([0, 1]),
        ...     val_context_X=pd.DataFrame({"x": [1.0, np.nan]}), val_context_y=np.array([0, 1]),
        ...     val_X=pd.DataFrame({"x": [np.nan]}), val_y=np.array([1]),
        ...     test_context_X=pd.DataFrame({"x": [1.0]}), test_context_y=np.array([0]),
        ...     test_X=pd.DataFrame({"x": [np.inf]}), test_y=np.array([1]),
        ...     metadata={},
        ... )
        >>> out = _apply_preprocessing_protocol(data, protocol="shared", model_id="tabicl_v2", seed=0)
        >>> out.test_X.isna().any().any()
        False
    """
    protocol = normalize_preprocessing_protocol(protocol)
    preprocessor = _fit_preprocessing_protocol(data, protocol=protocol, seed=seed, model_id=model_id)
    metadata = _preprocessing_metadata(preprocessor=preprocessor, protocol=protocol, data=data, model_id=model_id)
    source_schema = feature_schema_metadata(data.train_X)
    metadata.setdefault("source_feature_schema", source_schema)
    if preprocessor is None:
        return replace(
            data,
            metadata={
                **data.metadata,
                "preprocessing_protocol": protocol,
                "feature_preprocessing": metadata,
                "preprocessing": metadata,
            },
        )
    train_X = preprocessor.transform(data.train_X)
    val_context_X = preprocessor.transform(data.val_context_X)
    val_X = preprocessor.transform(data.val_X)
    test_context_X = preprocessor.transform(data.test_context_X)
    test_X = preprocessor.transform(data.test_X)
    adapter_schema = feature_schema_metadata(train_X, categorical_feature_indices=())
    metadata["adapter_feature_schema"] = adapter_schema
    metadata["categorical_feature_indices"] = []
    return replace(
        data,
        train_X=train_X,
        val_context_X=val_context_X,
        val_X=val_X,
        test_context_X=test_context_X,
        test_X=test_X,
        metadata={
            **data.metadata,
            "preprocessing_protocol": protocol,
            "feature_preprocessing": metadata,
            "preprocessing": metadata,
        },
    )


def _prepare_alternate_test_context_data(
    *,
    config: Mapping[str, Any],
    repo_root: Path,
    primary_data: SplitEvaluationData,
    seed: int,
    preprocessing_protocol: str,
    model_id: str,
) -> SplitEvaluationData | None:
    if not _resolve_test_context_comparison_enabled(config):
        return None
    comparison_mode = _resolve_test_context_comparison_mode(config)
    if comparison_mode == TEST_CONTEXT_COMPARISON_MAJORITY_DOWNSAMPLING:
        comparison_cfg = get_nested(config, "finetune.test_context_comparison")
        if not isinstance(comparison_cfg, Mapping):
            comparison_cfg = {}
        alternate, _report = apply_test_context_majority_downsampling(
            primary_data,
            comparison_cfg,
            seed=seed,
        )
        return _refresh_adapter_feature_schema_metadata(alternate)
    primary_includes_validation = bool(primary_data.metadata.get("test_context_includes_validation_data"))
    alternate_config = _test_context_flag_config(
        config,
        includes_validation_data=not primary_includes_validation,
    )
    alternate = load_split_evaluation_data(config=alternate_config, repo_root=repo_root)
    alternate = apply_pre_finite_distribution_shift(alternate, config, seed=seed)
    alternate = _apply_model_row_limits(alternate, config=config, seed=seed)
    alternate = _densify_sparse_feature_frames(alternate)
    alternate = _apply_preprocessing_protocol(
        alternate,
        protocol=preprocessing_protocol,
        model_id=model_id,
        seed=seed,
    )
    alternate = apply_post_finite_distribution_shift(alternate, config, seed=seed)
    return _refresh_adapter_feature_schema_metadata(alternate)
