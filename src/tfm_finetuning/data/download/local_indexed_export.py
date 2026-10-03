"""Write DataFrames as the repository's indexed local-array dataset format.

Example:
    >>> callable(write_indexed_local_export)
    True
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SplitKey = tuple[int, int, int]
OuterSplit = tuple[np.ndarray, np.ndarray]


def write_indexed_local_export(
    *,
    output_dir: Path,
    name: str,
    features: pd.DataFrame,
    labels: np.ndarray,
    categorical_columns: Sequence[str],
    outer_splits: Mapping[SplitKey, OuterSplit],
    task_type: str,
    tabarena_metric: str,
    label_mapping: Mapping[str, int],
    provenance: Mapping[str, Any],
    validation_fraction: float = 0.1,
    validation_seed: int = 0,
) -> dict[str, Any]:
    """Write one deterministic local export and return its ``info.json`` payload."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(features) != len(labels):
        raise ValueError("Feature and label row counts must match.")
    categorical = tuple(str(column) for column in categorical_columns)
    missing = sorted(set(categorical).difference(features.columns))
    if missing:
        raise ValueError(f"Unknown categorical columns: {missing}")
    numeric = tuple(column for column in features.columns if column not in categorical)

    _save_array(output_dir / "Y.npy", labels)
    if numeric:
        numeric_values = features.loc[:, numeric].apply(pd.to_numeric, errors="coerce")
        _save_array(output_dir / "X_num.npy", numeric_values.to_numpy(dtype=np.float32))
    if categorical:
        _save_array(output_dir / "X_cat.npy", features.loc[:, categorical].to_numpy(dtype=object))

    split_payloads: dict[str, dict[str, int]] = {}
    first_sizes: tuple[int, int, int] | None = None
    for (fold, repeat, sample), (outer_train, test) in sorted(outer_splits.items()):
        train, validation = _inner_train_validation_split(
            np.asarray(outer_train, dtype=np.int64),
            labels=labels,
            validation_fraction=validation_fraction,
            seed=validation_seed,
        )
        test = np.asarray(test, dtype=np.int64)
        _validate_partition(len(labels), train=train, validation=validation, test=test)
        split_name = f"fold{fold}_repeat{repeat}_sample{sample}"
        split_dir = output_dir / f"split-{split_name}"
        split_dir.mkdir(parents=True, exist_ok=True)
        _save_array(split_dir / "train_idx.npy", train)
        _save_array(split_dir / "val_idx.npy", validation)
        _save_array(split_dir / "test_idx.npy", test)
        split_payloads[split_name] = {
            "fold": int(fold),
            "repeat": int(repeat),
            "sample": int(sample),
            "train_size": len(train),
            "val_size": len(validation),
            "test_size": len(test),
        }
        first_sizes = first_sizes or (len(train), len(validation), len(test))
    if first_sizes is None:
        raise ValueError("At least one outer split is required.")

    info = {
        "name": name,
        "id": f"beyondarena:{name}",
        "source": "beyondarena",
        "task_type": task_type,
        "n_classes": len(label_mapping),
        "n_rows": len(labels),
        "n_num_features": len(numeric),
        "n_bin_features": 0,
        "n_cat_features": len(categorical),
        "train_size": first_sizes[0],
        "val_size": first_sizes[1],
        "test_size": first_sizes[2],
        "tabarena_metric": tabarena_metric,
        "feature_names": {"numeric": list(numeric), "categorical": list(categorical)},
        "label_mapping": dict(label_mapping),
        "splits": split_payloads,
        "provenance": dict(provenance),
    }
    _write_json(output_dir / "info.json", info)
    return info


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of ``path`` without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inner_train_validation_split(
    outer_train: np.ndarray,
    *,
    labels: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    stratify = labels[outer_train]
    try:
        train, validation = train_test_split(
            outer_train,
            test_size=float(validation_fraction),
            random_state=int(seed),
            stratify=stratify,
        )
    except ValueError:
        train, validation = train_test_split(
            outer_train,
            test_size=float(validation_fraction),
            random_state=int(seed),
            stratify=None,
        )
    return np.sort(train), np.sort(validation)


def _validate_partition(
    row_count: int,
    *,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
) -> None:
    roles = [set(values.tolist()) for values in (train, validation, test)]
    if any(left.intersection(right) for index, left in enumerate(roles) for right in roles[index + 1 :]):
        raise ValueError("Local export split roles must be disjoint.")
    combined = set.union(*roles)
    if combined != set(range(row_count)):
        raise ValueError("Local export split roles must cover every row exactly once.")


def _save_array(path: Path, values: np.ndarray) -> None:
    np.save(path, np.asarray(values), allow_pickle=values.dtype == object)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = ["OuterSplit", "SplitKey", "sha256_file", "write_indexed_local_export"]
