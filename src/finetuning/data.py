"""Split data loading and row-shaping helpers for foundation-model fine-tuning."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from finetuning import pipeline_eval_shared as _pipeline_eval_shared
from finetuning import pipeline_shared as _pipeline_shared
from finetuning.distribution_shift import TargetTransformState
from finetuning.models.base import TaskKind, task_kind_from_value
from finetuning.runner import load_foundation_split_data
from utils.config_utils import get_nested, set_nested

TEST_CONTEXT_POLICY_TRAIN_ONLY = "train_only_for_test_icl"
TEST_CONTEXT_POLICY_TRAIN_PLUS_VALIDATION = "train_plus_validation_for_test_icl"
TEST_CONTEXT_POLICY_FULL_ORIGINAL = "full_original_for_test_icl"
TEST_CONTEXT_POLICY_BALANCED_ORIGINAL_MAJORITY = "balanced_original_majority_downsampled_for_test_icl"


@dataclass(frozen=True)
class SplitEvaluationData:
    """Data needed to fit and evaluate one foundation-model run.

    Example:
        >>> SplitEvaluationData.__name__
        'SplitEvaluationData'
    """

    task_type: TaskKind
    problem_type: str
    metric_name: str
    train_X: pd.DataFrame
    train_y: np.ndarray
    val_context_X: pd.DataFrame
    val_context_y: np.ndarray
    val_X: pd.DataFrame
    val_y: np.ndarray
    test_context_X: pd.DataFrame
    test_context_y: np.ndarray
    test_X: pd.DataFrame
    test_y: np.ndarray
    metadata: dict[str, Any]
    train_ids: np.ndarray | None = None
    val_context_ids: np.ndarray | None = None
    val_ids: np.ndarray | None = None
    test_context_ids: np.ndarray | None = None
    test_ids: np.ndarray | None = None
    val_context_components: np.ndarray | None = None
    test_context_components: np.ndarray | None = None
    val_fit_y: np.ndarray | None = None
    target_transform_state: TargetTransformState | None = None


def _coerce_fraction(value: Any, *, dotted_key: str) -> float:
    try:
        fraction = float(value)
    except Exception as exc:
        raise SystemExit(f"{dotted_key} must be a fraction in (0, 1), got {value!r}.") from exc
    if not 0.0 < fraction < 1.0:
        raise SystemExit(f"{dotted_key} must be a fraction in (0, 1), got {value!r}.")
    return fraction


def _coerce_bool_config(value: Any, *, dotted_key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off", ""}:
            return False
    raise SystemExit(f"{dotted_key} must be a boolean, got {value!r}.")


def _resolve_validation_fraction(config: Mapping[str, Any]) -> float:
    for dotted_key in (
        "foundation_model.validation_split_ratio",
        "evaluation.val_fraction",
    ):
        value = get_nested(config, dotted_key)
        if value is not None:
            return _coerce_fraction(value, dotted_key=dotted_key)
    return 0.2


def _resolve_validation_seed(config: Mapping[str, Any]) -> int:
    for dotted_key in (
        "foundation_model.validation_split_seed",
        "evaluation.val_seed",
        "data.seed",
        "seed",
    ):
        value = get_nested(config, dotted_key)
        if value is None:
            continue
        try:
            return int(value)
        except Exception as exc:
            raise SystemExit(f"{dotted_key} must be an integer, got {value!r}.") from exc
    return 0


def _resolve_test_context_includes_validation_data(config: Mapping[str, Any]) -> bool:
    for dotted_key in (
        "foundation_model.test_context_includes_validation_data",
        "foundation_model.test_context_includes_explicit_validation_data",
    ):
        value = get_nested(config, dotted_key)
        if value is not None:
            return _coerce_bool_config(value, dotted_key=dotted_key)
    return False


def _resolve_test_context_comparison_enabled(config: Mapping[str, Any]) -> bool:
    for dotted_key in (
        "foundation_model.test_context_comparison.enabled",
        "foundation_model.compare_test_contexts",
    ):
        value = get_nested(config, dotted_key)
        if value is not None:
            return _coerce_bool_config(value, dotted_key=dotted_key)
    return False


def _test_context_policy_name(*, includes_validation_data: bool) -> str:
    return (
        TEST_CONTEXT_POLICY_TRAIN_PLUS_VALIDATION
        if includes_validation_data
        else TEST_CONTEXT_POLICY_TRAIN_ONLY
    )


def _test_context_flag_config(config: Mapping[str, Any], *, includes_validation_data: bool) -> dict[str, Any]:
    copied = json.loads(json.dumps(config))
    set_nested(
        copied,
        "foundation_model.test_context_includes_validation_data",
        bool(includes_validation_data),
    )
    return copied


def _resolve_optional_positive_int(config: Mapping[str, Any], dotted_keys: tuple[str, ...]) -> int | None:
    for dotted_key in dotted_keys:
        value = get_nested(config, dotted_key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "off", "false"}:
            return None
        try:
            parsed = int(value)
        except Exception as exc:
            raise SystemExit(f"{dotted_key} must be a positive integer or null, got {value!r}.") from exc
        if parsed <= 0:
            raise SystemExit(f"{dotted_key} must be a positive integer or null, got {value!r}.")
        return parsed
    return None


def _row_limit_indices(*, row_count: int, max_rows: int | None, seed: int, offset: int) -> np.ndarray | None:
    if max_rows is None or row_count <= int(max_rows):
        return None
    rng = np.random.default_rng(int(seed) + int(offset))
    return np.sort(rng.choice(row_count, size=int(max_rows), replace=False))


def _take_limited_role(
    data: SplitEvaluationData,
    role: str,
    *,
    max_rows: int | None,
    seed: int,
    offset: int,
) -> dict[str, Any]:
    features = getattr(data, f"{role}_X")
    labels = np.asarray(getattr(data, f"{role}_y"))
    indices = _row_limit_indices(row_count=len(features), max_rows=max_rows, seed=seed, offset=offset)
    if indices is None:
        return {}
    updates: dict[str, Any] = {
        f"{role}_X": features.iloc[indices].reset_index(drop=True),
        f"{role}_y": labels[indices],
    }
    ids_name = f"{role}_ids"
    if hasattr(data, ids_name):
        ids = getattr(data, ids_name)
        if ids is not None:
            updates[ids_name] = np.asarray(ids, dtype=object)[indices]
    components_name = f"{role}_components"
    if hasattr(data, components_name):
        components = getattr(data, components_name)
        if components is not None:
            updates[components_name] = np.asarray(components, dtype=object)[indices]
    if role == "val" and data.val_fit_y is not None:
        updates["val_fit_y"] = np.asarray(data.val_fit_y)[indices]
    return updates


def _dense_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a dense feature frame when OpenML/TabArena yielded sparse columns.

    Example:
        >>> sparse = pd.DataFrame({"x": pd.arrays.SparseArray([0.0, 1.0])})
        >>> isinstance(_dense_feature_frame(sparse)["x"].dtype, pd.SparseDtype)
        False
    """
    sparse_columns = [
        column
        for column in frame.columns
        if isinstance(frame[column].dtype, pd.SparseDtype)
    ]
    if not sparse_columns:
        return frame
    dense = frame.copy()
    for column in sparse_columns:
        dense[column] = frame[column].sparse.to_dense()
    return dense


def _densify_sparse_feature_frames(data: SplitEvaluationData) -> SplitEvaluationData:
    """Materialize sparse feature frames before model adapters see them.

    Example:
        >>> data = SplitEvaluationData(
        ...     task_type="binary", problem_type="binary", metric_name="roc_auc",
        ...     train_X=pd.DataFrame({"x": pd.arrays.SparseArray([0.0, 1.0])}),
        ...     train_y=np.array([0, 1]),
        ...     val_context_X=pd.DataFrame({"x": pd.arrays.SparseArray([0.0, 1.0])}),
        ...     val_context_y=np.array([0, 1]),
        ...     val_X=pd.DataFrame({"x": pd.arrays.SparseArray([1.0])}),
        ...     val_y=np.array([1]),
        ...     test_context_X=pd.DataFrame({"x": pd.arrays.SparseArray([0.0, 1.0])}),
        ...     test_context_y=np.array([0, 1]),
        ...     test_X=pd.DataFrame({"x": pd.arrays.SparseArray([0.0])}),
        ...     test_y=np.array([0]),
        ...     metadata={},
        ... )
        >>> isinstance(_densify_sparse_feature_frames(data).train_X["x"].dtype, pd.SparseDtype)
        False
    """
    frames = {
        "train": data.train_X,
        "val_context": data.val_context_X,
        "val_query": data.val_X,
        "test_context": data.test_context_X,
        "test_query": data.test_X,
    }
    sparse_columns_by_split = {
        split: [
            str(column)
            for column in frame.columns
            if isinstance(frame[column].dtype, pd.SparseDtype)
        ]
        for split, frame in frames.items()
    }
    if not any(sparse_columns_by_split.values()):
        return data
    return replace(
        data,
        train_X=_dense_feature_frame(data.train_X),
        val_context_X=_dense_feature_frame(data.val_context_X),
        val_X=_dense_feature_frame(data.val_X),
        test_context_X=_dense_feature_frame(data.test_context_X),
        test_X=_dense_feature_frame(data.test_X),
        metadata={
            **data.metadata,
            "sparse_feature_densification": {
                "applied": True,
                "sparse_columns_by_split": sparse_columns_by_split,
            },
        },
    )


def _apply_foundation_model_row_limits(
    data: SplitEvaluationData,
    *,
    config: Mapping[str, Any],
    seed: int,
) -> SplitEvaluationData:
    """Apply explicit train/context/query row caps for large direct runs.

    Example:
        >>> data = SplitEvaluationData(
        ...     task_type="binary", problem_type="binary", metric_name="roc_auc",
        ...     train_X=pd.DataFrame({"x": range(3)}), train_y=np.array([0, 1, 0]),
        ...     val_context_X=pd.DataFrame({"x": range(3)}), val_context_y=np.array([0, 1, 0]),
        ...     val_X=pd.DataFrame({"x": range(2)}), val_y=np.array([0, 1]),
        ...     test_context_X=pd.DataFrame({"x": range(3)}), test_context_y=np.array([0, 1, 0]),
        ...     test_X=pd.DataFrame({"x": range(2)}), test_y=np.array([0, 1]),
        ...     metadata={},
        ... )
        >>> _apply_foundation_model_row_limits(
        ...     data, config={"foundation_model": {"train_max_rows": 2}}, seed=0
        ... ).train_X.shape[0]
        2
    """
    train_max_rows = _resolve_optional_positive_int(
        config,
        (
            "foundation_model.train_max_rows",
            "foundation_model.fit_train_max_rows",
            "foundation_model.max_train_rows",
            "preprocess.clamp_train_pool.max_size",
        ),
    )
    inference_context_max_rows = _resolve_optional_positive_int(
        config,
        (
            "foundation_model.inference_context_max_rows",
            "foundation_model.eval_context_max_rows",
        ),
    )
    eval_query_max_rows = _resolve_optional_positive_int(
        config,
        (
            "foundation_model.eval_query_max_rows",
            "foundation_model.inference_query_max_rows",
        ),
    )
    if train_max_rows is None and inference_context_max_rows is None and eval_query_max_rows is None:
        return data

    updates: dict[str, Any] = {}
    updates.update(_take_limited_role(data, "train", max_rows=train_max_rows, seed=seed, offset=11))
    updates.update(
        _take_limited_role(
            data,
            "val_context",
            max_rows=inference_context_max_rows,
            seed=seed,
            offset=23,
        )
    )
    updates.update(
        _take_limited_role(
            data,
            "test_context",
            max_rows=inference_context_max_rows,
            seed=seed,
            offset=31,
        )
    )
    updates.update(_take_limited_role(data, "val", max_rows=eval_query_max_rows, seed=seed, offset=43))
    updates.update(_take_limited_role(data, "test", max_rows=eval_query_max_rows, seed=seed, offset=47))
    limited = replace(data, **updates) if updates else data
    metadata = dict(data.metadata)
    metadata["row_limits"] = {
        "train_max_rows": train_max_rows,
        "inference_context_max_rows": inference_context_max_rows,
        "eval_query_max_rows": eval_query_max_rows,
        "raw_rows": {
            "train": len(data.train_X),
            "val_context": len(data.val_context_X),
            "val_query": len(data.val_X),
            "test_context": len(data.test_context_X),
            "test_query": len(data.test_X),
        },
        "effective_rows": {
            "train": len(limited.train_X),
            "val_context": len(limited.val_context_X),
            "val_query": len(limited.val_X),
            "test_context": len(limited.test_context_X),
            "test_query": len(limited.test_X),
        },
    }
    return replace(
        limited,
        metadata=metadata,
    )


def _resolve_dataset_path(config: Mapping[str, Any], *, repo_root: Path) -> Path:
    data_cfg = config.get("data") if isinstance(config.get("data"), Mapping) else {}
    raw_path = str(data_cfg.get("path") or "").strip()
    raw_path = raw_path.removeprefix(":")
    if not raw_path:
        raise SystemExit("data.path is required for foundation_model finetuning.")
    path = Path(raw_path)
    return path if path.is_absolute() else (repo_root / path).resolve()


def _load_dataset_info(dataset_path: Path) -> dict[str, Any]:
    info_path = dataset_path / "info.json"
    if not info_path.exists():
        return {}
    try:
        payload = json.loads(info_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _split_tuple(config: Mapping[str, Any]) -> tuple[int, int, int]:
    data_cfg = config.get("data") if isinstance(config.get("data"), Mapping) else {}
    split_name = data_cfg.get("split")
    split_tuple = _pipeline_shared.parse_split_tuple_name(split_name)
    if split_tuple is None:
        raise SystemExit("data.split must be fold<k>_repeat<r>_sample<s> for foundation_model finetuning.")
    return split_tuple


def _to_task_kind(problem_type: str) -> TaskKind:
    return task_kind_from_value("binary" if problem_type == "binclass" else problem_type)


def _row_ids_for_frame(frame: pd.DataFrame, *, source: str) -> np.ndarray:
    return np.asarray([f"{source}:{idx}" for idx in frame.index], dtype=object)


def _context_components_for_ids(
    ids: np.ndarray,
    *,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
) -> np.ndarray:
    train_lookup = set(np.asarray(train_ids, dtype=object).tolist())
    val_lookup = set(np.asarray(val_ids, dtype=object).tolist())
    components: list[str] = []
    for row_id in np.asarray(ids, dtype=object).tolist():
        if row_id in train_lookup:
            components.append("train_ft")
        elif row_id in val_lookup:
            components.append("val_ft")
        else:
            components.append("ctx_icl")
    return np.asarray(components, dtype=object)


def load_split_evaluation_data(
    *,
    config: Mapping[str, Any],
    repo_root: Path,
) -> SplitEvaluationData:
    """Load train/validation/test data for the generated direct-run config.

    Example:
        >>> callable(load_split_evaluation_data)
        True
    """
    dataset_path = _resolve_dataset_path(config, repo_root=repo_root)
    info_payload = _load_dataset_info(dataset_path)
    split_tuple = _split_tuple(config)
    val_fraction = _resolve_validation_fraction(config)
    val_seed = _resolve_validation_seed(config)
    test_context_includes_validation_data = _resolve_test_context_includes_validation_data(config)
    task_id = info_payload.get("task_id")
    if task_id is not None:
        bundle = _pipeline_eval_shared.load_tabarena_aligned_eval_bundle(
            repo_root=repo_root,
            task_id=int(task_id),
            split_tuple=split_tuple,
            val_fraction=val_fraction,
            val_seed=val_seed,
        )
        problem_type = str(bundle["problem_type"])
        task_type = _to_task_kind(problem_type)
        test_context_key = "test" if test_context_includes_validation_data else "val"
        train_X_raw = bundle["context_features"]["val"]
        val_X_raw = bundle["query_features"]["val"]
        test_context_X_raw = bundle["context_features"][test_context_key]
        test_X_raw = bundle["query_features"]["test"]
        train_ids = _row_ids_for_frame(train_X_raw, source="outer_train")
        val_ids = _row_ids_for_frame(val_X_raw, source="outer_train")
        test_context_ids = _row_ids_for_frame(test_context_X_raw, source="outer_train")
        test_ids = _row_ids_for_frame(test_X_raw, source="outer_test")
        return SplitEvaluationData(
            task_type=task_type,
            problem_type=problem_type,
            metric_name=str(bundle["metric_name"]),
            train_X=train_X_raw.reset_index(drop=True),
            train_y=np.asarray(bundle["context_labels"]["val"]),
            val_context_X=train_X_raw.reset_index(drop=True),
            val_context_y=np.asarray(bundle["context_labels"]["val"]),
            val_X=val_X_raw.reset_index(drop=True),
            val_y=np.asarray(bundle["query_labels"]["val"]),
            test_context_X=test_context_X_raw.reset_index(drop=True),
            test_context_y=np.asarray(bundle["context_labels"][test_context_key]),
            test_X=test_X_raw.reset_index(drop=True),
            test_y=np.asarray(bundle["query_labels"]["test"]),
            metadata={
                "data_backend": "tabarena_openml_task_wrapper",
                "dataset_id": dataset_path.name,
                "task_id": int(task_id),
                "dataset_path": str(dataset_path),
                "fold": split_tuple[0],
                "repeat": split_tuple[1],
                "sample": split_tuple[2],
                "validation_fraction": val_fraction,
                "validation_seed": val_seed,
                "test_context_includes_validation_data": bool(
                    test_context_includes_validation_data
                ),
                "test_context_source": (
                    "outer_train"
                    if test_context_includes_validation_data
                    else "inner_train"
                ),
            },
            train_ids=train_ids,
            val_context_ids=train_ids,
            val_ids=val_ids,
            test_context_ids=test_context_ids,
            test_ids=test_ids,
            val_context_components=np.asarray(["train_ft"] * len(train_ids), dtype=object),
            test_context_components=_context_components_for_ids(
                test_context_ids,
                train_ids=train_ids,
                val_ids=val_ids,
            ),
        )

    hpo_split = load_foundation_split_data(
        repo_root=repo_root,
        dataset_id=dataset_path.name,
        split_tuple=split_tuple,
        inner_validation_fraction=val_fraction,
        inner_validation_seed=val_seed,
        dataset_path=dataset_path,
    )
    problem_type = str(hpo_split.task_type)
    metric_name = _pipeline_eval_shared.benchmark_metric_for_problem_type(problem_type)
    test_context_X = (
        hpo_split.context_X if test_context_includes_validation_data else hpo_split.train_X
    )
    test_context_y = (
        hpo_split.context_y if test_context_includes_validation_data else hpo_split.train_y
    )
    test_context_ids = (
        hpo_split.context_ids if test_context_includes_validation_data else hpo_split.train_ids
    )
    test_context_components = (
        hpo_split.context_components
        if test_context_includes_validation_data
        else np.asarray(["train_ft"] * len(hpo_split.train_X), dtype=object)
    )
    return SplitEvaluationData(
        task_type=hpo_split.task_type,
        problem_type=problem_type,
        metric_name=metric_name,
        train_X=hpo_split.train_X,
        train_y=hpo_split.train_y,
        val_context_X=hpo_split.train_X,
        val_context_y=hpo_split.train_y,
        val_X=hpo_split.val_X,
        val_y=hpo_split.val_y,
        test_context_X=test_context_X,
        test_context_y=test_context_y,
        test_X=hpo_split.test_X,
        test_y=hpo_split.test_y,
        metadata={
            "data_backend": "local_exported_arrays",
            "dataset_id": dataset_path.name,
            "dataset_path": str(dataset_path),
            "fold": split_tuple[0],
            "repeat": split_tuple[1],
            "sample": split_tuple[2],
            "validation_fraction": val_fraction,
            "validation_seed": val_seed,
            "test_context_includes_validation_data": bool(
                test_context_includes_validation_data
            ),
            "test_context_source": (
                "outer_train"
                if test_context_includes_validation_data
                else "inner_train"
            ),
        },
        train_ids=hpo_split.train_ids,
        val_context_ids=hpo_split.train_ids,
        val_ids=hpo_split.val_ids,
        test_context_ids=test_context_ids,
        test_ids=hpo_split.test_ids,
        val_context_components=np.asarray(["train_ft"] * len(hpo_split.train_X), dtype=object),
        test_context_components=test_context_components,
    )
