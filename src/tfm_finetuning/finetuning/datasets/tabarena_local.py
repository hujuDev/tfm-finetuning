"""Local TabArena/OpenML export loading."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tfm_finetuning.data.dataset_utils import map_tabarena_metric_to_score
from tfm_finetuning.finetuning.datasets.tabarena_common import (
    DataKey,
    Dataset,
    PartKey,
    Score,
    Task,
    TaskType,
)


def _normalize_local_problem_type(problem_type: Any, *, n_classes: Any) -> str:
    text = str(problem_type or "").strip().lower()
    if text in {"binary", "binclass"}:
        return "binary"
    if text in {"multiclass", "multi"}:
        return "multiclass"
    if text == "regression":
        return "regression"
    if text in {"classification", "class"}:
        try:
            n_classes_int = int(n_classes)
        except Exception as exc:
            raise ValueError("classification task_type requires n_classes in info.json") from exc
        return "binary" if n_classes_int == 2 else "multiclass"
    try:
        n_classes_int = int(n_classes)
    except Exception as exc:
        raise ValueError(f"Unsupported task_type={problem_type!r}") from exc
    if n_classes_int <= 0:
        return "regression"
    return "binary" if n_classes_int == 2 else "multiclass"


def _default_score_for_problem_type(problem_type: str) -> Score:
    if problem_type == "binary":
        return Score.ROC_AUC
    if problem_type == "multiclass":
        return Score.CROSS_ENTROPY
    return Score.RMSE


def _load_feature_block(path: Path) -> np.ndarray:
    values = np.asarray(np.load(path, allow_pickle=True))
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return values


def _load_label_block(path: Path) -> np.ndarray:
    values = np.asarray(np.load(path, allow_pickle=True))
    return values.reshape(-1)


def _split_name_from_tuple(split_tuple: tuple[int, int, int] | None, info_payload: dict[str, Any]) -> str:
    if split_tuple is None:
        fold = int(info_payload.get("fold", 0))
        repeat = int(info_payload.get("repeat", 0))
        sample = int(info_payload.get("sample", 0))
    else:
        fold, repeat, sample = (int(split_tuple[0]), int(split_tuple[1]), int(split_tuple[2]))
    return f"fold{fold}_repeat{repeat}_sample{sample}"


def _indexed_split_export_exists(dataset_dir: Path, split_name: str) -> bool:
    split_dir = dataset_dir / f"split-{split_name}"
    return (dataset_dir / "Y.npy").exists() and all(
        (split_dir / f"{part}_idx.npy").exists() for part in ("train", "val", "test")
    )


def _local_split_cardinality(info_payload: dict[str, Any]) -> tuple[int, int, int]:
    splits = info_payload.get("splits")
    if not isinstance(splits, dict):
        return 1, 1, 1
    folds: set[int] = set()
    repeats: set[int] = set()
    samples: set[int] = set()
    for split_payload in splits.values():
        if not isinstance(split_payload, dict):
            continue
        with contextlib.suppress(Exception):
            folds.add(int(split_payload.get("fold")))
        with contextlib.suppress(Exception):
            repeats.add(int(split_payload.get("repeat")))
        with contextlib.suppress(Exception):
            samples.add(int(split_payload.get("sample")))
    return max(1, len(folds)), max(1, len(repeats)), max(1, len(samples))


def _load_indexed_split_blocks(
    *,
    dataset_dir: Path,
    split_name: str,
) -> tuple[dict[PartKey, np.ndarray], dict[DataKey, dict[PartKey, np.ndarray]]]:
    split_dir = dataset_dir / f"split-{split_name}"
    indices = {
        part: np.asarray(np.load(split_dir / f"{part}_idx.npy", allow_pickle=True), dtype=np.int64).reshape(-1)
        for part in ("train", "val", "test")
    }
    y_full = _load_label_block(dataset_dir / "Y.npy")
    labels: dict[PartKey, np.ndarray] = {part: y_full[idx] for part, idx in indices.items()}
    data: dict[DataKey, dict[PartKey, np.ndarray]] = {"y": labels}

    for key, prefix, dtype in (
        ("x_num", "X_num", np.float32),
        ("x_bin", "X_bin", np.float32),
        ("x_cat", "X_cat", None),
    ):
        block_path = dataset_dir / f"{prefix}.npy"
        if not block_path.exists():
            continue
        values = _load_feature_block(block_path)
        if values.shape[0] != y_full.shape[0]:
            raise ValueError(
                f"Row count mismatch for {block_path}: {values.shape[0]} features vs {y_full.shape[0]} labels."
            )
        if dtype is not None:
            values = values.astype(dtype, copy=False)
        data[key] = {part: values[idx] for part, idx in indices.items()}

    return labels, data


def build_dataset_from_local_path(
    *,
    dataset_path: str | Path,
    split_tuple: tuple[int, int, int] | None,
    num_policy: str | None,
    cat_policy: str | None,
    seed: int,
    fillna: bool = True,
) -> tuple[Dataset, dict[str, Any]]:
    dataset_dir = Path(dataset_path).expanduser().resolve()
    info_path = dataset_dir / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Local dataset info.json not found: {info_path}")
    info_payload = json.loads(info_path.read_text(encoding="utf-8"))
    if not isinstance(info_payload, dict):
        raise ValueError(f"Expected mapping payload in {info_path}, got {type(info_payload).__name__}.")

    split_name = _split_name_from_tuple(split_tuple, info_payload)
    if _indexed_split_export_exists(dataset_dir, split_name):
        labels, data = _load_indexed_split_blocks(dataset_dir=dataset_dir, split_name=split_name)
    else:
        labels = {part: _load_label_block(dataset_dir / f"Y_{part}.npy") for part in ("train", "val", "test")}
        data = {"y": labels}

        for key, prefix, dtype in (
            ("x_num", "X_num", np.float32),
            ("x_bin", "X_bin", np.float32),
            ("x_cat", "X_cat", None),
        ):
            block_paths = {part: dataset_dir / f"{prefix}_{part}.npy" for part in ("train", "val", "test")}
            if not any(path.exists() for path in block_paths.values()):
                continue
            if not all(path.exists() for path in block_paths.values()):
                missing = ", ".join(part for part, path in block_paths.items() if not path.exists())
                raise FileNotFoundError(f"Missing {prefix} block(s) for local dataset {dataset_dir.name}: {missing}")
            block_payload: dict[PartKey, np.ndarray] = {}
            for part, block_path in block_paths.items():
                values = _load_feature_block(block_path)
                if values.shape[0] != labels[part].shape[0]:
                    raise ValueError(
                        f"Row count mismatch for {block_path}: "
                        f"{values.shape[0]} features vs {labels[part].shape[0]} labels."
                    )
                if dtype is not None:
                    values = values.astype(dtype, copy=False)
                block_payload[part] = values
            data[key] = block_payload

    problem_type = _normalize_local_problem_type(
        info_payload.get("task_type"),
        n_classes=info_payload.get("n_classes"),
    )
    if num_policy is not None or cat_policy is not None:
        raise ValueError("Load raw data here; select native/shared preprocessing in the model configuration.")

    task_type_map = {
        "regression": TaskType.REGRESSION,
        "binary": TaskType.BINCLASS,
        "multiclass": TaskType.MULTICLASS,
    }
    score_name = info_payload.get("score")
    if score_name is None:
        score_name = map_tabarena_metric_to_score(info_payload.get("tabarena_metric"))
    try:
        score = Score(score_name) if score_name is not None else _default_score_for_problem_type(problem_type)
    except Exception:
        score = _default_score_for_problem_type(problem_type)

    dataset = Dataset(
        data=data,
        task=Task(labels=labels, type_=task_type_map[problem_type], score=score),
    )
    if split_tuple is None:
        fold = int(info_payload.get("fold", 0))
        repeat = int(info_payload.get("repeat", 0))
        sample = int(info_payload.get("sample", 0))
    else:
        fold, repeat, sample = (int(split_tuple[0]), int(split_tuple[1]), int(split_tuple[2]))
    dataset_name = str(info_payload.get("name") or dataset_dir.name).strip() or dataset_dir.name
    n_folds, n_repeats, n_samples = _local_split_cardinality(info_payload)
    metadata = {
        "task_id": info_payload.get("task_id"),
        "name": dataset_name,
        "dataset": dataset_name,
        "dataset_dir": dataset_dir.name,
        "id": info_payload.get("id"),
        "fold": fold,
        "repeat": repeat,
        "sample": sample,
        "split": f"fold{fold}_repeat{repeat}_sample{sample}",
        "n_folds": n_folds,
        "n_repeats": n_repeats,
        "n_samples": n_samples,
        "problem_type": problem_type,
        "tabarena_metric": info_payload.get("tabarena_metric"),
        "score": score.value,
        "data_path": str(dataset_dir),
        "source": "local_export",
    }
    for key in (
        "n_classes",
        "n_rows",
        "n_num_features",
        "n_bin_features",
        "n_cat_features",
        "train_size",
        "val_size",
        "test_size",
    ):
        value = info_payload.get(key)
        if value is not None:
            metadata[key] = value
    return dataset, metadata


