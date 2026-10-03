#!/usr/bin/env python3
"""Download an OpenML task (into the OpenML cache) and export to the local indexed format.

This script:
  - Downloads/loads a task and OpenML split indices for a given task id.
  - Exports a dynamic split grid by default: one fixed split for large
    datasets and one compact 3-fold CV repeat for small datasets.
  - Creates a validation split from the OpenML train split (stratified for classification).
  - Writes full X_num/X_cat/X_bin and Y arrays once, plus split indices under split-<name>/.
  - Supports source=tabarena (existing behavior) and source=openml (arbitrary task ids).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import openml
import pandas as pd
from pandas.api.types import is_bool_dtype, is_object_dtype, is_string_dtype
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from tabarena.benchmark.task.openml import OpenMLTaskWrapper
from tabarena.nips2025_utils.fetch_metadata import load_task_metadata

try:
    from data.dataset_utils import (
        map_tabarena_metric_to_score,
        parse_subset_spec,
        slugify_dataset_name,
    )
except (ImportError, ModuleNotFoundError):
    from dataset_utils import (
        map_tabarena_metric_to_score,
        parse_subset_spec,
        slugify_dataset_name,
    )

try:
    from data.dataset_utils import find_dataset_matches
except (ImportError, ModuleNotFoundError):
    from dataset_utils import find_dataset_matches

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download an OpenML task and export to local indexed dataset files. "
            "Supports TabArena tasks or arbitrary OpenML tasks by id."
        )
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--task-id", type=int, help="OpenML task id.")
    group.add_argument(
        "--dataset-name",
        type=str,
        help="Dataset name from task_metadata_tabarena51.csv (column 'name').",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="tabarena",
        choices=["tabarena", "openml"],
        help="Dataset source. Use 'openml' to allow non-TabArena tasks by --task-id.",
    )
    parser.add_argument(
        "--openml-split-policy",
        type=str,
        default="official",
        choices=["official", "tabarena"],
        help=(
            "Default split policy for --source openml when split args are omitted. "
            "'official' uses OpenML task estimation splits; 'tabarena' uses post-hoc TabArena-style splits."
        ),
    )
    parser.add_argument(
        "--resampling-protocol",
        type=str,
        default="cv_auto",
        choices=["cv_auto", "cv-auto", "fixed_split", "cross_validation", "legacy_tabarena"],
        help=(
            "Default split grid when fold/repeat/sample args are omitted. "
            "cv_auto exports one fixed split for datasets with at least 2500 rows "
            "and a compact 3-fold CV grid for smaller datasets. legacy_tabarena "
            "restores the old 3 folds x {10 small, 3 large} repeats default."
        ),
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="TabPFNv2",
        help=(
            "Limit datasets to a TabArena subset (default: TabPFNv2). "
            "Examples: TabPFNv2, TabICL, TabPFNv2+binary."
        ),
    )
    parser.add_argument("--fold", type=int, default=None, help="Fold index to export.")
    parser.add_argument("--repeat", type=int, default=None, help="Repeat index to export.")
    parser.add_argument("--sample", type=int, default=None, help="Sample index to export.")
    parser.add_argument(
        "--folds",
        type=str,
        default=None,
        help=(
            "Comma-separated fold indices to export (overrides --fold). "
            "When omitted with other split args, defaults follow source policy."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=str,
        default=None,
        help=(
            "Comma-separated repeat indices to export (overrides --repeat). "
            "When omitted with other split args, defaults follow source policy."
        ),
    )
    parser.add_argument(
        "--samples",
        type=str,
        default=None,
        help=(
            "Comma-separated sample indices to export (overrides --sample). "
            "When omitted with other split args, defaults follow source policy."
        ),
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default=None,
        help="Override split directory name (default: fold{fold}_repeat{repeat}_sample{sample}).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of the TabArena train split to use for validation.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for train/val splitting.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory for the dataset. Defaults to "
            "$REPO_ROOT/data/<source>/<dataset_slug>."
        ),
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=None,
        help="Optional path to task_metadata_tabarena51.csv.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the output directory.",
    )
    return parser.parse_args()


def _load_metadata(subset: str, metadata_path: str | None) -> pd.DataFrame:
    subset = None if subset == "all" else subset
    base_subset, problem_filter = parse_subset_spec(subset)
    if metadata_path is None:
        metadata = load_task_metadata(paper=True, subset=base_subset)
    else:
        metadata = load_task_metadata(paper=True, subset=base_subset, path=metadata_path)
    if problem_filter:
        if problem_filter == "classification":
            metadata = metadata[metadata["problem_type"] != "regression"]
        else:
            metadata = metadata[metadata["problem_type"] == problem_filter]
    return metadata


def _repo_root() -> Path:
    return Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[3]))


def _configure_openml_cache() -> Path:
    cache_dir = Path(os.environ.get("OPENML_CACHE_DIR", _repo_root() / "data" / "cache" / "openml"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    openml.config.set_root_cache_directory(str(cache_dir))
    return cache_dir


def _resolve_output_dir(output_dir: str | None, *, source: str, dataset_name: str) -> Path:
    if output_dir:
        return Path(output_dir)
    return _repo_root() / "data" / source / slugify_dataset_name(dataset_name)


def _resolve_task(
    metadata: pd.DataFrame | None,
    task_id: int | None,
    dataset_name: str | None,
    *,
    source: str,
) -> tuple[int, str]:
    if source == "openml":
        if task_id is None:
            raise SystemExit("--task-id is required for --source openml.")
        task = openml.tasks.get_task(int(task_id))
        resolved_dataset_name = str(task.get_dataset().name or f"openml_task_{task_id}")
        return int(task_id), resolved_dataset_name

    if metadata is None:
        raise SystemExit("TabArena metadata is required when --source=tabarena.")

    if task_id is not None:
        match = metadata[metadata["tid"] == task_id]
        if match.empty:
            raise SystemExit(f"Task id {task_id} not found in the selected metadata subset.")
        dataset_name = str(match.iloc[0]["name"])
        return int(task_id), dataset_name

    if dataset_name is None:
        raise SystemExit("Either --task-id or --dataset-name must be provided.")
    match = find_dataset_matches(metadata, dataset_name)
    if match.empty:
        available = ", ".join(sorted(metadata["name"].unique())[:20])
        raise SystemExit(f"Dataset name '{dataset_name}' not found. Example names: {available}")
    if len(match) > 1:
        raise SystemExit(f"Dataset name '{dataset_name}' is ambiguous.")
    task_id = int(match.iloc[0]["tid"])
    dataset_name = str(match.iloc[0]["name"])
    return task_id, dataset_name


def _infer_openml_problem_type(task: object, y: pd.Series) -> str:
    task_type_text = (
        f"{getattr(task, 'task_type_id', '')} {getattr(task, 'task_type', '')}".strip().lower()
    )
    if "regression" in task_type_text:
        return "regression"
    if "classification" in task_type_text:
        return "binary" if pd.Series(y).dropna().nunique() <= 2 else "multiclass"
    raise SystemExit(f"Unsupported OpenML task type for export: {task_type_text or 'unknown'}.")


def _default_score_metric_for_openml(problem_type: str) -> str:
    mapping = {
        "binary": "roc-auc",
        "multiclass": "cross-entropy",
        "regression": "rmse",
    }
    return mapping[problem_type]


def _split_train_val(
    y: np.ndarray, *, val_fraction: float, seed: int, problem_type: str
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < val_fraction < 1:
        raise SystemExit("--val-fraction must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(y))
    if problem_type in {"binary", "multiclass"}:
        val_indices = []
        for label in np.unique(y):
            label_indices = indices[y == label]
            rng.shuffle(label_indices)
            n_val = round(len(label_indices) * val_fraction)
            n_val = min(n_val, len(label_indices) - 1) if len(label_indices) > 1 else 0
            if n_val > 0:
                val_indices.append(label_indices[:n_val])
        val_idx = np.concatenate(val_indices) if val_indices else np.array([], dtype=int)
        train_idx = np.setdiff1d(indices, val_idx, assume_unique=False)
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        return train_idx, val_idx

    rng.shuffle(indices)
    n_val = round(len(indices) * val_fraction)
    n_val = min(max(n_val, 1), len(indices) - 1)
    return indices[n_val:], indices[:n_val]


def _is_categorical(series: pd.Series) -> bool:
    return (
        isinstance(series.dtype, pd.CategoricalDtype)
        or is_object_dtype(series)
        or is_string_dtype(series)
        or is_bool_dtype(series)
    )


def _categorical_schema(X_train: pd.DataFrame, cat_cols: list[str]) -> list[dict]:
    schema = []
    for col in cat_cols:
        series = X_train[col]
        categories = pd.Series(series).dropna().unique().tolist()
        is_binary = len(categories) <= 2
        schema.append({"name": col, "categories": categories, "is_binary": is_binary})
    return schema


def _encode_categories(series: pd.Series, categories: list) -> np.ndarray:
    cat_type = pd.CategoricalDtype(categories=categories)
    return series.astype(cat_type).cat.codes.to_numpy()


def _encode_features(
    X: pd.DataFrame, num_cols: list[str], cat_schema: list[dict]
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    X_num = X[num_cols].to_numpy(dtype=np.float32, copy=False) if num_cols else None

    bin_arrays: list[np.ndarray] = []
    cat_arrays: list[np.ndarray] = []
    for col_info in cat_schema:
        col = col_info["name"]
        codes = _encode_categories(X[col], col_info["categories"])
        if col_info["is_binary"]:
            bin_arrays.append(codes.astype(np.float32))
        else:
            cat_arrays.append(codes.astype(np.int64))

    X_bin = np.column_stack(bin_arrays) if bin_arrays else None
    X_cat = np.column_stack(cat_arrays) if cat_arrays else None
    return X_num, X_cat, X_bin


def _encode_labels(y: pd.Series, categories: list) -> np.ndarray:
    cat_type = pd.CategoricalDtype(categories=categories)
    codes = y.astype(cat_type).cat.codes.to_numpy()
    if (codes < 0).any():
        raise SystemExit("Found labels not seen in the training split.")
    return codes.astype(np.int64, copy=False)


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_array(path: Path, arr: np.ndarray | None) -> None:
    if arr is None:
        return
    np.save(path, arr)


def _parse_int_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return [int(item) for item in value.split(",")]


def _to_positive_int(value: object) -> int | None:
    try:
        parsed = int(str(value))
    except Exception:
        return None
    if parsed <= 0:
        return None
    return parsed


def _resolve_openml_default_split_grid(task: object) -> tuple[list[int], list[int], list[int]]:
    estimation = getattr(task, "estimation_procedure", None)
    if not isinstance(estimation, dict):
        return [0], [0], [0]
    parameters = estimation.get("parameters")
    if not isinstance(parameters, dict):
        return [0], [0], [0]

    n_folds = _to_positive_int(parameters.get("number_folds")) or 1
    n_repeats = _to_positive_int(parameters.get("number_repeats")) or 1
    n_samples = _to_positive_int(parameters.get("number_samples")) or 1
    return list(range(n_folds)), list(range(n_repeats)), list(range(n_samples))


def _resolve_split_grid(
    args: argparse.Namespace,
    *,
    source: str,
    n_instances: int,
    task: object,
    openml_split_policy: str,
) -> tuple[list[int], list[int], list[int], bool]:
    folds = _parse_int_list(args.folds)
    repeats = _parse_int_list(args.repeats)
    samples = _parse_int_list(args.samples)
    use_default_policy = (
        folds is None
        and repeats is None
        and samples is None
        and args.fold is None
        and args.repeat is None
        and args.sample is None
    )

    if use_default_policy:
        protocol = str(getattr(args, "resampling_protocol", "cv_auto")).strip().lower()
        if protocol == "fixed_split":
            folds, repeats, samples = [0], [0], [0]
        elif protocol == "cross_validation":
            folds, repeats, samples = [0, 1, 2], [0], [0]
        elif protocol in {"cv_auto", "cv-auto", "paper_auto", "paper-auto"}:
            if n_instances < 2500:
                folds, repeats, samples = [0, 1, 2], [0], [0]
            else:
                folds, repeats, samples = [0], [0], [0]
        elif protocol in {"legacy_tabarena", "legacy"} and (
            source == "tabarena" or openml_split_policy == "tabarena"
        ):
            default_repeats = 10 if n_instances < 2500 else 3
            folds, repeats, samples = list(range(3)), list(range(default_repeats)), [0]
        elif protocol in {"legacy_tabarena", "legacy"}:
            folds, repeats, samples = _resolve_openml_default_split_grid(task)
        else:
            raise SystemExit(
                f"Unknown --resampling-protocol={protocol!r}. Expected one of: "
                "cv_auto, fixed_split, cross_validation, legacy_tabarena."
            )
        return folds, repeats, samples, True

    fold = 0 if args.fold is None else args.fold
    repeat = 0 if args.repeat is None else args.repeat
    sample = 0 if args.sample is None else args.sample
    return folds or [fold], repeats or [repeat], samples or [sample], False


def _build_posthoc_tabarena_split_map(
    y: np.ndarray,
    *,
    folds: list[int],
    repeats: list[int],
    samples: list[int],
    seed: int,
    problem_type: str,
) -> dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]]:
    if any(sample != 0 for sample in samples):
        raise SystemExit("Post-hoc TabArena split policy supports only sample=0.")
    if any(fold not in {0, 1, 2} for fold in folds):
        raise SystemExit("Post-hoc TabArena split policy supports only folds 0,1,2.")
    if any(repeat < 0 for repeat in repeats):
        raise SystemExit("Repeat indices must be non-negative.")
    n_repeats = max(repeats) + 1
    n_splits = 3

    if problem_type in {"binary", "multiclass"}:
        splitter = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=seed,
        )
        split_iter = splitter.split(np.zeros(len(y), dtype=np.float32), y)
    else:
        splitter = RepeatedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=seed,
        )
        split_iter = splitter.split(np.zeros(len(y), dtype=np.float32))

    split_map: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}
    requested = {(repeat, fold, sample) for repeat in repeats for fold in folds for sample in samples}
    for split_idx, (train_idx, test_idx) in enumerate(split_iter):
        repeat = split_idx // n_splits
        fold = split_idx % n_splits
        key = (repeat, fold, 0)
        if key in requested:
            split_map[key] = (
                np.asarray(train_idx, dtype=np.int64),
                np.asarray(test_idx, dtype=np.int64),
            )
    missing = requested.difference(split_map.keys())
    if missing:
        missing_txt = ", ".join(f"(repeat={r}, fold={f}, sample={s})" for r, f, s in sorted(missing))
        raise SystemExit(f"Could not generate requested post-hoc TabArena split(s): {missing_txt}")
    return split_map


def _resolve_split_name(*, fold: int, repeat: int, sample: int, override: str | None) -> str:
    if override:
        return override
    return f"fold{fold}_repeat{repeat}_sample{sample}"


def _load_existing_info(path: Path) -> dict | None:
    info_path = path / "info.json"
    if not info_path.exists():
        return None
    try:
        return json.loads(info_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    """Download one TabArena dataset and export it for TabPFN finetuning."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    openml_cache_dir = _configure_openml_cache()
    logger.info("Using OpenML cache: %s", openml_cache_dir)

    metadata = _load_metadata(args.subset, args.metadata_path) if args.source == "tabarena" else None
    task_id, dataset_name = _resolve_task(
        metadata=metadata,
        task_id=args.task_id,
        dataset_name=args.dataset_name,
        source=args.source,
    )

    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    X_full, y_full, _, _ = dataset.get_data(
        dataset_format="dataframe",
        target=task.target_name,
    )
    folds, repeats, samples, use_default_split_policy = _resolve_split_grid(
        args,
        source=args.source,
        n_instances=len(y_full),
        task=task,
        openml_split_policy=args.openml_split_policy,
    )
    if args.split_name and len(folds) * len(repeats) * len(samples) > 1:
        raise SystemExit("--split-name is only supported when exporting a single fold/repeat/sample.")

    # Use full feature set for encoding so all splits share the same arrays.
    num_cols = [col for col in X_full.columns if not _is_categorical(X_full[col])]
    cat_cols = [col for col in X_full.columns if col not in num_cols]
    cat_schema = _categorical_schema(X_full, cat_cols)

    X_num_full, X_cat_full, X_bin_full = _encode_features(X_full, num_cols, cat_schema)

    task_type_map = {"binary": "binclass", "multiclass": "multiclass", "regression": "regression"}
    if args.source == "tabarena":
        wrapper = OpenMLTaskWrapper.from_task_id(task_id)
        problem_type = wrapper.problem_type
        tabarena_metric = wrapper.eval_metric
        score_metric = map_tabarena_metric_to_score(tabarena_metric)
        if score_metric is None:
            logger.warning(
                "Unsupported TabArena eval metric %r for %s; leaving score unset.",
                tabarena_metric,
                dataset_name,
            )
    else:
        problem_type = _infer_openml_problem_type(task, y_full)
        tabarena_metric = None
        score_metric = _default_score_metric_for_openml(problem_type)

    task_type = task_type_map[problem_type]

    if problem_type == "regression":
        y_full_arr = y_full.to_numpy(dtype=np.float32, copy=False)
        n_classes = None
    else:
        label_categories = pd.Series(y_full).dropna().unique().tolist()
        y_full_arr = _encode_labels(y_full, label_categories)
        n_classes = len(label_categories)

    posthoc_tabarena_splits: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] | None = None
    if args.source == "openml" and args.openml_split_policy == "tabarena":
        y_split_source = y_full.to_numpy(copy=False)
        posthoc_tabarena_splits = _build_posthoc_tabarena_split_map(
            y_split_source,
            folds=folds,
            repeats=repeats,
            samples=samples,
            seed=args.seed,
            problem_type=problem_type,
        )

    output_dir = _resolve_output_dir(args.output_dir, source=args.source, dataset_name=dataset_name)
    _ensure_output_dir(output_dir)
    info_payload = _load_existing_info(output_dir)
    if info_payload and not args.overwrite:
        existing_task = info_payload.get("task_id")
        if existing_task not in (None, task_id):
            raise SystemExit(
                f"Output dir '{output_dir}' already contains task_id={existing_task}; "
                f"refusing to write task_id={task_id} without --overwrite."
            )
    elif info_payload is None:
        info_payload = {}

    if args.overwrite or not (output_dir / "Y.npy").exists():
        _save_array(output_dir / "X_num.npy", X_num_full)
        _save_array(output_dir / "X_cat.npy", X_cat_full)
        _save_array(output_dir / "X_bin.npy", X_bin_full)
        _save_array(output_dir / "Y.npy", y_full_arr)

    info_payload.update(
        {
            "name": dataset_name,
            "id": f"{args.source}-{task_id}",
            "source": args.source,
            "task_type": task_type,
            "n_num_features": X_num_full.shape[1] if X_num_full is not None else 0,
            "n_bin_features": X_bin_full.shape[1] if X_bin_full is not None else 0,
            "n_cat_features": X_cat_full.shape[1] if X_cat_full is not None else 0,
            "n_rows": len(y_full_arr),
            "task_id": task_id,
        }
    )
    if tabarena_metric is not None:
        info_payload["tabarena_metric"] = tabarena_metric
    else:
        info_payload.pop("tabarena_metric", None)
    if score_metric is not None:
        info_payload["score"] = score_metric
    splits = info_payload.get("splits")
    if not isinstance(splits, dict):
        splits = {}

    for fold in folds:
        for repeat in repeats:
            for sample in samples:
                split_name = _resolve_split_name(
                    fold=fold,
                    repeat=repeat,
                    sample=sample,
                    override=args.split_name,
                )
                split_dir = output_dir / f"split-{split_name}"
                if all((split_dir / f"{part}_idx.npy").is_file() for part in ("train", "val", "test")) and not args.overwrite:
                    logger.info("Split %s already exists at %s; skipping.", split_name, split_dir)
                    continue
                split_dir.mkdir(parents=True, exist_ok=True)

                if posthoc_tabarena_splits is not None:
                    key = (repeat, fold, sample)
                    train_idx_global, test_idx_global = posthoc_tabarena_splits[key]
                else:
                    try:
                        train_idx_global, test_idx_global = task.get_train_test_split_indices(
                            fold=fold,
                            repeat=repeat,
                            sample=sample,
                        )
                    except ValueError:
                        if not use_default_split_policy:
                            raise
                        logger.warning(
                            "Skipping unavailable split fold=%d repeat=%d sample=%d for task_id=%d.",
                            fold,
                            repeat,
                            sample,
                            task_id,
                        )
                        continue
                y_train_full = y_full.iloc[train_idx_global]
                train_idx, val_idx = _split_train_val(
                    y_train_full.to_numpy(),
                    val_fraction=args.val_fraction,
                    seed=args.seed,
                    problem_type=problem_type,
                )
                train_idx_global = np.asarray(train_idx_global)
                test_idx_global = np.asarray(test_idx_global)
                train_idx = train_idx_global[train_idx]
                val_idx = train_idx_global[val_idx]

                np.save(split_dir / "train_idx.npy", train_idx)
                np.save(split_dir / "val_idx.npy", val_idx)
                np.save(split_dir / "test_idx.npy", test_idx_global)

                splits[split_name] = {
                    "fold": fold,
                    "repeat": repeat,
                    "sample": sample,
                    "train_size": len(train_idx),
                    "val_size": len(val_idx),
                    "test_size": len(test_idx_global),
                    "val_fraction": float(args.val_fraction),
                    "split_seed": int(args.seed),
                }

    info_payload["splits"] = splits
    if n_classes is not None:
        info_payload["n_classes"] = n_classes

    (output_dir / "info.json").write_text(json.dumps(info_payload, indent=4))
    (output_dir / "READY").touch()

    logger.info("Wrote dataset to %s with %d split(s)", output_dir, len(splits))


if __name__ == "__main__":
    main()
