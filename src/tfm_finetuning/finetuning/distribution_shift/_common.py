"""Shared helpers for role-aware distribution interventions."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

TRAIN_FT = "train_ft"
VAL_FT = "val_ft"
CTX_OTHER = "ctx_icl"


def get_nested(mapping: Mapping[str, Any], path: str) -> Any | None:
    cursor: Any = mapping
    for part in path.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def derive_seed(base_seed: int, step_name: str) -> int:
    digest = hashlib.sha256(step_name.encode("utf-8")).hexdigest()
    return int((int(base_seed) + int(digest[:8], 16)) % (2**32))


def parse_fraction(value: Any) -> float | None:
    result: float | None = None
    if isinstance(value, bool):
        result = None
    elif isinstance(value, (int, float)):
        numeric = float(value)
        result = numeric if math.isfinite(numeric) else None
    elif isinstance(value, str):
        text = value.strip()
        if "/" in text:
            try:
                numerator, denominator = text.split("/", 1)
                result = float(numerator) / float(denominator)
            except Exception:
                result = None
        else:
            try:
                numeric = float(text)
            except ValueError:
                result = None
            else:
                result = numeric if math.isfinite(numeric) else None
    return result


def is_noop_token(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "off", "disabled", "false", "original", "all", "full"}
    return False


def hash_array(values: np.ndarray) -> str:
    arr = np.asarray(values)
    payload = np.array([str(value) for value in arr], dtype=object)
    digest = hashlib.sha256("\n".join(payload.tolist()).encode("utf-8")).hexdigest()
    return digest[:12]


def ensure_role_metadata(data: Any) -> Any:
    train_ids = optional_array(getattr(data, "train_ids", None), len(data.train_X), "outer_train")
    val_ids = optional_array(getattr(data, "val_ids", None), len(data.val_X), "outer_train_val")
    test_ids = optional_array(getattr(data, "test_ids", None), len(data.test_X), "outer_test")
    val_context_ids = optional_array(
        getattr(data, "val_context_ids", None),
        len(data.val_context_X),
        "val_context",
        fallback=train_ids if len(data.val_context_X) == len(train_ids) else None,
    )
    test_context_ids = optional_array(
        getattr(data, "test_context_ids", None),
        len(data.test_context_X),
        "test_context",
    )
    val_context_components = optional_components(
        getattr(data, "val_context_components", None),
        val_context_ids,
        train_ids=train_ids,
        val_ids=val_ids,
    )
    test_context_components = optional_components(
        getattr(data, "test_context_components", None),
        test_context_ids,
        train_ids=train_ids,
        val_ids=val_ids,
    )
    updates: dict[str, Any] = {
        name: value
        for name, value in (
            ("train_ids", train_ids),
            ("val_ids", val_ids),
            ("test_ids", test_ids),
            ("val_context_ids", val_context_ids),
            ("test_context_ids", test_context_ids),
            ("val_context_components", val_context_components),
            ("test_context_components", test_context_components),
        )
        if hasattr(data, name)
    }
    return replace(data, **updates) if updates else data


def optional_array(value: Any, length: int, prefix: str, fallback: np.ndarray | None = None) -> np.ndarray:
    if value is not None:
        arr = np.asarray(value, dtype=object)
        if len(arr) != length:
            raise ValueError(f"{prefix} ids length {len(arr)} does not match role length {length}")
        return arr
    if fallback is not None:
        if len(fallback) != length:
            raise ValueError("fallback row IDs must match role length")
        return np.asarray(fallback, dtype=object)
    return np.asarray([f"{prefix}:{idx}" for idx in range(length)], dtype=object)


def optional_components(
    value: Any,
    ids: np.ndarray,
    *,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
) -> np.ndarray:
    if value is not None:
        arr = np.asarray(value, dtype=object)
        if len(arr) != len(ids):
            raise ValueError("role component length does not match context length")
        return arr
    train_lookup = set(train_ids.tolist())
    val_lookup = set(val_ids.tolist())
    components = []
    for row_id in ids.tolist():
        if row_id in train_lookup:
            components.append(TRAIN_FT)
        elif row_id in val_lookup:
            components.append(VAL_FT)
        else:
            components.append(CTX_OTHER)
    return np.asarray(components, dtype=object)


def with_distribution_shift_reports(data: Any, stage: str, reports: Mapping[str, Any]) -> Any:
    if not reports:
        return data
    metadata = dict(getattr(data, "metadata", {}) or {})
    existing = metadata.get("distribution_shift")
    payload = dict(existing if isinstance(existing, Mapping) else {})
    stage_payload = dict(payload.get(stage) if isinstance(payload.get(stage), Mapping) else {})
    stage_payload.update(reports)
    payload[stage] = stage_payload
    metadata["distribution_shift"] = payload
    return replace(data, metadata=metadata)


def subset_role(
    data: Any,
    role: str,
    mask: np.ndarray,
    *,
    transformed_y: np.ndarray | None = None,
) -> dict[str, Any]:
    x_name = f"{role}_X"
    y_name = f"{role}_y"
    ids_name = f"{role}_ids"
    components_name = f"{role}_components"
    x_value = getattr(data, x_name).loc[mask].reset_index(drop=True)
    y_value = np.asarray(transformed_y if transformed_y is not None else getattr(data, y_name))[mask]
    updates = {x_name: x_value, y_name: y_value}
    if hasattr(data, ids_name):
        updates[ids_name] = np.asarray(getattr(data, ids_name), dtype=object)[mask]
    if hasattr(data, components_name):
        updates[components_name] = np.asarray(getattr(data, components_name), dtype=object)[mask]
    if role == "val" and hasattr(data, "val_fit_y"):
        val_fit_y = getattr(data, "val_fit_y", None)
        if val_fit_y is not None:
            updates["val_fit_y"] = np.asarray(val_fit_y)[mask]
    return updates


def parse_rate(value: Any, *, name: str) -> float:
    if is_noop_token(value):
        return 0.0
    rate = parse_fraction(value)
    if rate is None or not 0.0 <= rate <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    return float(rate)


def resolve_calibrated_value(
    value: Any,
    *,
    selections: Any,
    metadata: Mapping[str, Any],
    name: str,
) -> Any:
    """Resolve a dataset-specific value selected by a prior calibration run.

    Example:
        >>> resolve_calibrated_value(
        ...     "calibrated",
        ...     selections={"demo": 3},
        ...     metadata={"dataset_id": "demo"},
        ...     name="scale_spread",
        ... )
        3
    """
    if str(value).strip().lower() != "calibrated":
        return value
    if not isinstance(selections, Mapping):
        raise ValueError(f"{name}='calibrated' requires a dataset selection mapping")
    dataset_id = str(metadata.get("dataset_id") or "").strip()
    if not dataset_id:
        dataset_path = str(metadata.get("dataset_path") or "").rstrip("/")
        dataset_id = dataset_path.rsplit("/", 1)[-1] if dataset_path else ""
    normalized = dataset_id.lower().replace("-", "_")
    for key, selected in selections.items():
        if str(key).strip().lower().replace("-", "_") == normalized:
            return selected
    raise ValueError(f"No calibrated {name} is configured for dataset {dataset_id or '<unknown>'!r}")


def jsonable_label(label: Any) -> int | float | str:
    if isinstance(label, np.generic):
        label = label.item()
    if isinstance(label, (int, float, str)):
        return label
    return str(label)


def mixed_frame_to_numeric_matrix(frame: pd.DataFrame) -> np.ndarray:
    columns: list[np.ndarray] = []
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        else:
            values = pd.factorize(series, sort=True)[0].astype(float)
            values[values < 0] = np.nan
        finite = values[np.isfinite(values)]
        fill = float(np.median(finite)) if finite.size else 0.0
        values = np.where(np.isfinite(values), values, fill)
        columns.append(values.reshape(-1, 1))
    if not columns:
        return np.zeros((len(frame), 0), dtype=float)
    return np.hstack(columns)


def label_signal(labels: np.ndarray) -> np.ndarray:
    if labels.size == 0:
        return np.zeros((0,), dtype=float)
    if np.issubdtype(labels.dtype, np.number):
        signal = labels.astype(float)
    else:
        signal = pd.factorize(labels, sort=True)[0].astype(float)
    finite = signal[np.isfinite(signal)]
    mean = float(np.mean(finite)) if finite.size else 0.0
    std = float(np.std(finite)) if finite.size else 0.0
    if std <= 1e-12:
        return np.zeros_like(signal, dtype=float)
    return (signal - mean) / std


def safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or len(y) < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if math.isfinite(value) else None


def safe_mean(values: Any) -> float | None:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))
