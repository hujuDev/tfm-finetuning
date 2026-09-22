"""Mitra's explicit finite-numeric and label adapter boundary."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from finetuning.models.preprocessing import (
    FiniteFeaturePreprocessor,
)
from finetuning.preprocessing_protocol import (
    PREPROCESSING_NATIVE,
)


def is_classifier(estimator: Any) -> bool:
    return estimator.__class__.__name__.lower().endswith("classifier")


def has_trained_ensemble(estimator: Any) -> bool:
    return bool(getattr(estimator, "trainers", None))


def adapter_input_preprocessing_metadata(
    *,
    preprocessor: FiniteFeaturePreprocessor,
    protocol: str,
    train_X: pd.DataFrame,
    train_X_fit: pd.DataFrame,
    applied_to: str,
) -> dict[str, Any]:
    return {
        "method": "mitra_adapter_finite_numeric_guard",
        "protocol": protocol,
        "fit_scope": "train_features_only",
        "raw_feature_count": int(train_X.shape[1]),
        "effective_feature_count": int(train_X_fit.shape[1]),
        "numeric_feature_count": len(preprocessor.numeric_columns),
        "categorical_feature_count": len(preprocessor.categorical_columns),
        "output_dtype": "float32",
        "applied_to": applied_to,
        "input_already_shared_preprocessed": protocol != PREPROCESSING_NATIVE,
        "reason": "direct_mitra_sklearn_interface_requires_finite_numeric_inputs",
    }


def context_features(estimator: Any, context_X: pd.DataFrame) -> pd.DataFrame:
    """Fit a fresh outer preprocessor on the active target ICL context."""
    preprocessor = FiniteFeaturePreprocessor.fit(context_X)
    estimator._foundation_mitra_feature_preprocessor = preprocessor
    return preprocessor.transform(context_X)


def predict_features(estimator: Any, query_X: pd.DataFrame) -> pd.DataFrame:
    preprocessor = getattr(estimator, "_foundation_mitra_feature_preprocessor", None)
    return query_X if preprocessor is None else preprocessor.transform(query_X)


def fit_dense_label_encoder(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes, dense = np.unique(np.asarray(y), return_inverse=True)
    return classes, dense.astype(np.int64, copy=False)


def encode_with_classes(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(classes.tolist())}
    encoded: list[int] = []
    for value in np.asarray(y).tolist():
        try:
            encoded.append(lookup[value])
        except KeyError as exc:
            raise ValueError(f"Mitra context/validation label {value!r} was not seen during fitting.") from exc
    return np.asarray(encoded, dtype=np.int64)


def frame_values(frame: pd.DataFrame) -> np.ndarray:
    return frame.to_numpy(copy=False)


def refit_trainer_preprocessors(estimator: Any, context_X: pd.DataFrame, context_y: np.ndarray) -> None:
    """Reset every learned ensemble member's preprocessing to target context rows."""
    X = frame_values(context_X)
    y = np.asarray(context_y)
    trainers = list(getattr(estimator, "trainers", []) or [])
    seeds = {int(getattr(trainer.cfg, "seed", 0)) for trainer in trainers if getattr(trainer, "cfg", None) is not None}
    if len(seeds) > 1:
        raise RuntimeError("Mitra ensemble members disagree on their shared RNG seed.")
    shared_rng = np.random.RandomState(next(iter(seeds), 0))
    for trainer in trainers:
        cfg = getattr(trainer, "cfg", None)
        if cfg is None:
            continue
        preprocessor = _new_trainer_preprocessor(trainer)
        preprocessor.fit(X, y)
        trainer.preprocessor = preprocessor
        # Mitra constructs every ensemble member with one shared, progressively
        # advanced RNG. Preserve that ensemble diversity after target rebinding.
        trainer.rng = shared_rng


def _new_trainer_preprocessor(trainer: Any) -> Any:
    module = importlib.import_module("autogluon.tabular.models.mitra._internal.data.preprocessor")
    cls = module.Preprocessor
    cfg = trainer.cfg
    params = cfg.hyperparams
    return cls(
        dim_embedding=params["dim_embedding"],
        n_classes=int(trainer.n_classes),
        dim_output=params["dim_output"],
        use_quantile_transformer=params["use_quantile_transformer"],
        use_feature_count_scaling=params["use_feature_count_scaling"],
        use_random_transforms=params["use_random_transforms"],
        shuffle_classes=params["shuffle_classes"],
        shuffle_features=params["shuffle_features"],
        random_mirror_regression=params["random_mirror_regression"],
        random_mirror_x=params["random_mirror_x"],
        task=cfg.task,
    )
