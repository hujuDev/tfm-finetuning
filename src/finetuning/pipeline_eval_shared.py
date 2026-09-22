"""Load the official OpenML partition and create the thesis validation split."""
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

PartKey = Literal["train", "val", "test"]


def benchmark_metric_for_problem_type(problem_type: str) -> str:
    metric_map = {
        "binary": "roc_auc",
        "multiclass": "log_loss",
        "regression": "rmse",
    }
    metric = metric_map.get(str(problem_type))
    if metric is None:
        raise ValueError(f"Unsupported benchmark problem_type={problem_type!r}")
    return metric



def split_train_for_synthetic_val(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    problem_type: str,
    val_fraction: float,
    val_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    val_fraction = float(val_fraction)
    if not (0.0 < val_fraction < 1.0):
        val_fraction = 0.2
    train_size = 1.0 - val_fraction
    if len(X_train) <= 2:
        return X_train, X_train, y_train, y_train

    try:
        from autogluon.core.utils import generate_train_test_split

        X_fit, X_val, y_fit, y_val = generate_train_test_split(
            X=X_train,
            y=y_train,
            problem_type=problem_type,
            train_size=train_size,
            random_state=int(val_seed),
        )
        return X_fit, X_val, y_fit, y_val
    except Exception:
        rng = np.random.default_rng(int(val_seed))
        perm = rng.permutation(len(X_train))
        n_val = max(1, min(len(X_train) - 1, int(round(len(X_train) * val_fraction))))
        val_pos = perm[:n_val]
        fit_pos = perm[n_val:]
        fit_idx = X_train.index.take(fit_pos)
        val_idx = X_train.index.take(val_pos)
        return X_train.loc[fit_idx], X_train.loc[val_idx], y_train.loc[fit_idx], y_train.loc[val_idx]



def load_tabarena_aligned_eval_bundle(
    *,
    repo_root: Path,
    task_id: int,
    split_tuple: tuple[int, int, int],
    val_fraction: float,
    val_seed: int,
) -> dict[str, Any]:
    fold, repeat, sample = int(split_tuple[0]), int(split_tuple[1]), int(split_tuple[2])

    with contextlib.suppress(Exception):
        import openml

        openml_cache = repo_root / "data" / "cache" / "openml"
        openml_cache.mkdir(parents=True, exist_ok=True)
        openml.config.set_root_cache_directory(str(openml_cache))

    from tabarena.benchmark.task.openml.task_wrapper import OpenMLTaskWrapper

    task_wrapper = OpenMLTaskWrapper.from_task_id(int(task_id))
    X_train_full, y_train_full, X_test, y_test = task_wrapper.get_train_test_split(
        fold=fold,
        repeat=repeat,
        sample=sample,
    )
    problem_type = str(task_wrapper.problem_type)
    metric_name = benchmark_metric_for_problem_type(problem_type)
    X_train_fit, X_val, y_train_fit, y_val = split_train_for_synthetic_val(
        X_train=X_train_full,
        y_train=y_train_full,
        problem_type=problem_type,
        val_fraction=val_fraction,
        val_seed=val_seed,
    )

    context_features: dict[PartKey, pd.DataFrame] = {
        "train": X_train_full,
        "val": X_train_fit,
        "test": X_train_full,
    }
    context_labels: dict[PartKey, pd.Series] = {
        "train": y_train_full,
        "val": y_train_fit,
        "test": y_train_full,
    }
    query_features: dict[PartKey, pd.DataFrame] = {
        "train": X_train_full,
        "val": X_val,
        "test": X_test,
    }
    query_labels: dict[PartKey, pd.Series] = {
        "train": y_train_full,
        "val": y_val,
        "test": y_test,
    }

    return {
        "problem_type": problem_type,
        "metric_name": metric_name,
        "task_id": int(task_id),
        "fold": fold,
        "repeat": repeat,
        "sample": sample,
        "context_features": context_features,
        "context_labels": context_labels,
        "query_features": query_features,
        "query_labels": query_labels,
    }
