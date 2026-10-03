"""TabICL fine-tuning input preparation for native and shared protocols."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd


def prepare_native_finetune_inputs(
    train_X: pd.DataFrame,
    val_X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    preprocessing = importlib.import_module("tabicl._sklearn.preprocessing")
    transformer = preprocessing.TransformToNumerical()
    train = _as_finite_numeric_array(transformer.fit_transform(train_X))
    val = _as_finite_numeric_array(transformer.transform(val_X))
    return (
        train,
        val,
        {
            "method": "tabicl_transform_to_numerical",
            "fit_scope": "train_features_only",
            "raw_feature_count": int(train_X.shape[1]),
            "effective_feature_count": int(train.shape[1]),
            "output_dtype": str(train.dtype),
            "applied_to": "native_finetune_inputs",
        },
    )


def prepare_shared_finetune_inputs(
    train_X: pd.DataFrame,
    val_X: pd.DataFrame,
    *,
    protocol: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train = _as_finite_numeric_array(train_X)
    val = _as_finite_numeric_array(val_X)
    return (
        train,
        val,
        {
            "method": "shared_preprocessed_numeric_passthrough",
            "protocol": protocol,
            "fit_scope": "shared_protocol_pre_fitted",
            "raw_feature_count": int(train_X.shape[1]),
            "effective_feature_count": int(train.shape[1]),
            "output_dtype": str(train.dtype),
            "applied_to": "native_finetune_inputs",
        },
    )


def _as_finite_numeric_array(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
