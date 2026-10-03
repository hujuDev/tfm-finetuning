"""Data loading helpers for unified runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tfm_finetuning.finetuning.datasets.tabarena import Dataset, build_dataset_from_local_path
from tfm_finetuning.finetuning.models.base import TaskKind, task_kind_from_value


@dataclass(frozen=True)
class SplitData:
    """Concrete train/validation/test data for one split.

    Example:
        >>> SplitData.__name__
        'SplitData'
    """

    task_type: TaskKind
    train_X: pd.DataFrame
    train_y: np.ndarray
    val_X: pd.DataFrame
    val_y: np.ndarray
    context_X: pd.DataFrame
    context_y: np.ndarray
    test_X: pd.DataFrame
    test_y: np.ndarray
    train_ids: np.ndarray | None = None
    val_ids: np.ndarray | None = None
    context_ids: np.ndarray | None = None
    context_components: np.ndarray | None = None
    test_ids: np.ndarray | None = None


def dataset_id_to_local_path(repo_root: Path, dataset_id: str, catalog_record: dict[str, Any] | None = None) -> Path:
    """Resolve a dataset id under any maintained local export root."""
    candidates = [dataset_id]
    if catalog_record is not None:
        metadata = catalog_record.get("metadata")
        if isinstance(metadata, dict):
            for key in ("dataset_slug", "dataset_dir", "dataset_name"):
                value = metadata.get(key)
                if value:
                    candidates.append(str(value))
    candidates.extend(
        [
            dataset_id.replace("-", "_"),
            dataset_id.replace("-", "_").lower(),
            dataset_id.lower(),
        ]
    )
    for root_name in ("tabarena", "openml", "other", "beyondarena"):
        for candidate in dict.fromkeys(candidates):
            path = repo_root / "data" / root_name / candidate
            if (path / "info.json").exists():
                return path
    raise FileNotFoundError(f"Could not resolve local export for dataset {dataset_id!r}.")


def dataset_part_to_frame(dataset: Dataset, part: str) -> pd.DataFrame:
    """Convert one loaded local dataset part into a mixed-type DataFrame."""
    frames: list[pd.DataFrame] = []
    x_num = dataset.data.get("x_num")
    if isinstance(x_num, dict) and part in x_num:
        arr_num = np.asarray(x_num[part])
        frames.append(pd.DataFrame(arr_num, columns=[f"num_{idx}" for idx in range(arr_num.shape[1])]))

    x_cat = dataset.data.get("x_cat")
    if isinstance(x_cat, dict) and part in x_cat:
        arr_cat = np.asarray(x_cat[part])
        if arr_cat.ndim == 1:
            arr_cat = arr_cat.reshape(-1, 1)
        cat_payload: dict[str, Any] = {}
        for idx in range(arr_cat.shape[1]):
            column = arr_cat[:, idx]
            if np.issubdtype(column.dtype, np.integer):
                cat_payload[f"cat_{idx}"] = pd.Categorical(column.astype(np.int64))
            elif np.issubdtype(column.dtype, np.number):
                cat_payload[f"cat_{idx}"] = column.astype(np.float32)
            else:
                cat_payload[f"cat_{idx}"] = pd.Categorical(column)
        frames.append(pd.DataFrame(cat_payload))

    x_bin = dataset.data.get("x_bin")
    if isinstance(x_bin, dict) and part in x_bin:
        arr_bin = np.asarray(x_bin[part])
        frames.append(pd.DataFrame(arr_bin, columns=[f"bin_{idx}" for idx in range(arr_bin.shape[1])]))

    if not frames:
        return pd.DataFrame(index=np.arange(dataset.size(part)))  # type: ignore[arg-type]
    return pd.concat(frames, axis=1)


def load_split_data(
    *,
    repo_root: Path,
    dataset_id: str,
    split_tuple: tuple[int, int, int],
    inner_validation_fraction: float,
    inner_validation_seed: int,
    catalog_record: dict[str, Any] | None = None,
    dataset_path: Path | None = None,
) -> SplitData:
    """Load a dataset split and derive inner train/validation data."""
    dataset_path = (
        Path(dataset_path)
        if dataset_path is not None
        else dataset_id_to_local_path(repo_root, dataset_id, catalog_record)
    )
    dataset, _metadata = build_dataset_from_local_path(
        dataset_path=dataset_path,
        split_tuple=split_tuple,
        num_policy=None,
        cat_policy=None,
        seed=int(inner_validation_seed),
    )
    task_type = task_kind_from_value(dataset.task.type_)
    outer_train_X = pd.concat(
        [dataset_part_to_frame(dataset, "train"), dataset_part_to_frame(dataset, "val")],
        axis=0,
        ignore_index=True,
    )
    outer_train_y = np.concatenate([dataset.task.labels["train"], dataset.task.labels["val"]])
    train_X_raw, val_X_raw, train_y, val_y = _split_inner_train_val(
        outer_train_X,
        outer_train_y,
        task_type=task_type,
        validation_fraction=inner_validation_fraction,
        seed=inner_validation_seed,
    )
    train_ids = np.asarray([f"outer_train:{idx}" for idx in train_X_raw.index], dtype=object)
    val_ids = np.asarray([f"outer_train:{idx}" for idx in val_X_raw.index], dtype=object)
    train_X = train_X_raw.reset_index(drop=True)
    val_X = val_X_raw.reset_index(drop=True)
    context_X = pd.concat([train_X, val_X], axis=0, ignore_index=True)
    context_y = np.concatenate([np.asarray(train_y), np.asarray(val_y)])
    context_ids = np.concatenate([train_ids, val_ids])
    context_components = np.asarray(
        ["train_ft"] * len(train_ids) + ["val_ft"] * len(val_ids),
        dtype=object,
    )
    test_X = dataset_part_to_frame(dataset, "test").reset_index(drop=True)
    return SplitData(
        task_type=task_type,
        train_X=train_X,
        train_y=np.asarray(train_y),
        val_X=val_X,
        val_y=np.asarray(val_y),
        context_X=context_X,
        context_y=context_y,
        test_X=test_X,
        test_y=np.asarray(dataset.task.labels["test"]),
        train_ids=train_ids,
        val_ids=val_ids,
        context_ids=context_ids,
        context_components=context_components,
        test_ids=np.asarray([f"outer_test:{idx}" for idx in range(len(test_X))], dtype=object),
    )


def _split_inner_train_val(
    data: pd.DataFrame,
    labels: np.ndarray,
    *,
    task_type: TaskKind,
    validation_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    stratify = None
    if task_type in {"binary", "multiclass"}:
        _, counts = np.unique(labels, return_counts=True)
        if counts.size > 1 and int(counts.min()) >= 2:
            stratify = labels
    try:
        return train_test_split(
            data,
            labels,
            test_size=float(validation_fraction),
            random_state=int(seed),
            stratify=stratify,
        )
    except ValueError:
        return train_test_split(
            data,
            labels,
            test_size=float(validation_fraction),
            random_state=int(seed),
            stratify=None,
        )
