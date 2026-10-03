from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import pytest

from finetuning.datasets.tabarena import Dataset, build_dataset_from_local_path

REPO_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("REPO_ROOT", str(REPO_ROOT))

logger = logging.getLogger(__name__)


def _write_dummy_dataset(
    root: Path, *, splits: dict[str, dict[str, list[int]]]
) -> tuple[np.ndarray, np.ndarray]:
    all_indices = []
    for split in splits.values():
        all_indices.extend(split["train"])
        all_indices.extend(split["val"])
        all_indices.extend(split["test"])
    n_samples = max(all_indices) + 1 if all_indices else 0
    x_num = np.arange(n_samples * 2, dtype=np.float32).reshape(n_samples, 2)
    y = np.arange(n_samples, dtype=np.int64)

    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "X_num.npy", x_num)
    np.save(root / "X_bin.npy", np.zeros((n_samples, 0), dtype=np.float32))
    np.save(root / "X_cat.npy", np.zeros((n_samples, 0), dtype=np.int64))
    np.save(root / "X_meta.npy", np.zeros((n_samples, 0), dtype=np.float32))
    np.save(root / "Y.npy", y)
    (root / "info.json").write_text(json.dumps({"task_type": "multiclass"}))

    for split_name, split in splits.items():
        split_dir = root / f"split-{split_name}"
        split_dir.mkdir(parents=True, exist_ok=True)
        np.save(split_dir / "train_idx.npy", np.asarray(split["train"], dtype=np.int64))
        np.save(split_dir / "val_idx.npy", np.asarray(split["val"], dtype=np.int64))
        np.save(split_dir / "test_idx.npy", np.asarray(split["test"], dtype=np.int64))

    return x_num, y


def _resolve_integration_dataset() -> tuple[Path, str] | None:
    dataset_name = os.environ.get("TABARENA_INTEGRATION_DATASET", "anneal")
    dataset_dir = REPO_ROOT / "data" / "tabarena" / dataset_name
    if not dataset_dir.exists():
        return None

    requested_split = os.environ.get("TABARENA_INTEGRATION_SPLIT")
    if requested_split:
        split_dir = dataset_dir / f"split-{requested_split}"
        if split_dir.exists():
            return dataset_dir, requested_split

    info_path = dataset_dir / "info.json"
    if info_path.exists():
        payload = json.loads(info_path.read_text())
        splits = payload.get("splits")
        if isinstance(splits, dict) and splits:
            return dataset_dir, sorted(splits.keys())[0]

    for split_dir in sorted(dataset_dir.glob("split-*")):
        if split_dir.is_dir():
            return dataset_dir, split_dir.name.replace("split-", "", 1)

    return None


def _iter_split_names(dataset_dir: Path) -> list[str]:
    info_path = dataset_dir / "info.json"
    if info_path.exists():
        payload = json.loads(info_path.read_text())
        splits = payload.get("splits")
        if isinstance(splits, dict) and splits:
            return sorted(splits.keys())

    split_names: list[str] = []
    for split_dir in sorted(dataset_dir.glob("split-*")):
        if split_dir.is_dir():
            split_names.append(split_dir.name.replace("split-", "", 1))
    return split_names


def _parse_split_triplet(split_name: str) -> tuple[int | None, int | None, int | None]:
    match = re.fullmatch(r"fold(\d+)_repeat(\d+)_sample(\d+)", split_name)
    if not match:
        return None, None, None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _load_local_dataset(dataset_dir: Path, split_name: str) -> Dataset:
    fold, repeat, sample = _parse_split_triplet(split_name)
    if fold is None or repeat is None or sample is None:
        raise AssertionError(f"Unexpected split name in test fixture: {split_name}")
    dataset, _metadata = build_dataset_from_local_path(
        dataset_path=dataset_dir,
        split_tuple=(fold, repeat, sample),
        num_policy=None,
        cat_policy=None,
        seed=0,
    )
    return dataset


def test_split_partitions_are_disjoint(tmp_path: Path) -> None:
    split_name = "fold0_repeat0_sample0"
    splits = {
        split_name: {
            "train": [0, 1, 2, 3],
            "val": [4, 5],
            "test": [6, 7, 8],
        }
    }
    _write_dummy_dataset(tmp_path / "dataset", splits=splits)

    dataset = _load_local_dataset(tmp_path / "dataset", split_name)
    y_train = set(dataset.data["y"]["train"].tolist())
    y_val = set(dataset.data["y"]["val"].tolist())
    y_test = set(dataset.data["y"]["test"].tolist())

    assert y_train.isdisjoint(y_val)
    assert y_train.isdisjoint(y_test)
    assert y_val.isdisjoint(y_test)


def test_finetune_loader_uses_train_split_indices(tmp_path: Path) -> None:
    split_a = "fold0_repeat0_sample0"
    split_b = "fold1_repeat0_sample0"
    splits = {
        split_a: {
            "train": [0, 1, 2],
            "val": [3],
            "test": [4, 5],
        },
        split_b: {
            "train": [6, 7, 8],
            "val": [9],
            "test": [10, 11],
        },
    }
    x_num, _ = _write_dummy_dataset(tmp_path / "dataset", splits=splits)

    dataset = _load_local_dataset(tmp_path / "dataset", split_b)
    np.testing.assert_array_equal(dataset.data["x_num"]["train"], x_num[splits[split_b]["train"]])


def test_validation_split_loaded_from_val_indices(tmp_path: Path) -> None:
    split_name = "fold0_repeat0_sample0"
    splits = {
        split_name: {
            "train": [0, 1, 2],
            "val": [3, 4],
            "test": [5, 6],
        }
    }
    x_num, _ = _write_dummy_dataset(tmp_path / "dataset", splits=splits)

    dataset = _load_local_dataset(tmp_path / "dataset", split_name)
    np.testing.assert_array_equal(dataset.data["x_num"]["val"], x_num[splits[split_name]["val"]])


def test_benchmark_loader_uses_test_split_indices(tmp_path: Path) -> None:
    split_name = "fold0_repeat0_sample0"
    splits = {
        split_name: {
            "train": [0, 1, 2],
            "val": [3, 4],
            "test": [5, 6, 7],
        }
    }
    x_num, _ = _write_dummy_dataset(tmp_path / "dataset", splits=splits)

    dataset = _load_local_dataset(tmp_path / "dataset", split_name)
    np.testing.assert_array_equal(dataset.data["x_num"]["test"], x_num[splits[split_name]["test"]])


@pytest.mark.integration
def test_tabarena_split_integrity_integration() -> None:
    resolved = _resolve_integration_dataset()
    if resolved is None:
        pytest.skip("TabArena dataset not found under data/tabarena or no split folders.")
    dataset_dir, split_name = resolved
    split_dir = dataset_dir / f"split-{split_name}"

    train_idx = np.load(split_dir / "train_idx.npy")
    val_idx = np.load(split_dir / "val_idx.npy")
    test_idx = np.load(split_dir / "test_idx.npy")

    assert len(train_idx) > 0
    assert len(val_idx) > 0
    assert len(test_idx) > 0
    assert set(train_idx).isdisjoint(val_idx)
    assert set(train_idx).isdisjoint(test_idx)
    assert set(val_idx).isdisjoint(test_idx)


@pytest.mark.integration
def test_tabarena_split_loader_integration() -> None:
    resolved = _resolve_integration_dataset()
    if resolved is None:
        pytest.skip("TabArena dataset not found under data/tabarena or no split folders.")
    dataset_dir, split_name = resolved
    split_dir = dataset_dir / f"split-{split_name}"

    train_idx = np.load(split_dir / "train_idx.npy")
    val_idx = np.load(split_dir / "val_idx.npy")
    test_idx = np.load(split_dir / "test_idx.npy")

    dataset = _load_local_dataset(dataset_dir, split_name)
    y = np.load(dataset_dir / "Y.npy")

    assert dataset.size("train") == len(train_idx)
    assert dataset.size("val") == len(val_idx)
    assert dataset.size("test") == len(test_idx)
    np.testing.assert_array_equal(dataset.data["y"]["train"], y[train_idx])
    np.testing.assert_array_equal(dataset.data["y"]["val"], y[val_idx])
    np.testing.assert_array_equal(dataset.data["y"]["test"], y[test_idx])


@pytest.mark.integration
def test_tabarena_cross_fold_repeat_train_val_overlap_integration() -> None:
    dataset_dir = REPO_ROOT / "data" / "tabarena" / "anneal"
    if not dataset_dir.exists():
        pytest.skip("TabArena anneal dataset not found under data/tabarena.")

    split_names = _iter_split_names(dataset_dir)
    if len(split_names) < 2:
        pytest.skip("Need at least two TabArena splits to compare folds.")

    split_info: list[tuple[str, int, int, int]] = []
    for name in split_names:
        fold_id, repeat_id, sample_id = _parse_split_triplet(name)
        if fold_id is None or repeat_id is None or sample_id is None:
            continue
        split_info.append((name, fold_id, repeat_id, sample_id))
    if len(split_info) < 2:
        pytest.skip("No usable split names with fold/repeat identifiers found.")

    desired = {(0, 0), (0, 1), (1, 0), (1, 1)}
    selected: dict[tuple[int, int], tuple[str, int]] = {}
    for name, fold_id, repeat_id, sample_id in split_info:
        key = (fold_id, repeat_id)
        if key not in desired:
            continue
        current = selected.get(key)
        if current is None or sample_id == 0:
            selected[key] = (name, sample_id)

    if len(selected) < len(desired):
        missing = sorted(desired - set(selected.keys()))
        pytest.skip(f"Missing split(s) for fold/repeat combinations: {missing}")

    split_payload: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for (fold_id, repeat_id), (split_name, _sample_id) in selected.items():
        split_dir = dataset_dir / f"split-{split_name}"
        train_idx = np.load(split_dir / "train_idx.npy")
        val_idx = np.load(split_dir / "val_idx.npy")
        test_idx = np.load(split_dir / "test_idx.npy")
        assert len(train_idx) > 0
        assert len(val_idx) > 0
        assert len(test_idx) > 0
        assert set(train_idx).isdisjoint(val_idx)
        assert set(train_idx).isdisjoint(test_idx)
        assert set(val_idx).isdisjoint(test_idx)
        split_payload[(fold_id, repeat_id)] = {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        }

    split_keys = sorted(split_payload.keys())
    for i, key_a in enumerate(split_keys):
        for key_b in split_keys[i + 1 :]:
            train_a = split_payload[key_a]["train"]
            val_b = split_payload[key_b]["val"]
            train_b = split_payload[key_b]["train"]
            val_a = split_payload[key_a]["val"]
            logger.info(
                "Comparing splits: a=fold%d_repeat%d b=fold%d_repeat%d",
                key_a[0],
                key_a[1],
                key_b[0],
                key_b[1],
            )
            assert not set(train_a).isdisjoint(val_b)
            assert not set(train_b).isdisjoint(val_a)


@pytest.mark.integration
def test_tabarena_benchmark_splits_match_exports_integration() -> None:
    dataset_name = os.environ.get("TABARENA_INTEGRATION_DATASET", "anneal")
    dataset_dir = REPO_ROOT / "data" / "tabarena" / dataset_name
    if not dataset_dir.exists():
        pytest.skip("TabArena dataset not found under data/tabarena.")

    info_path = dataset_dir / "info.json"
    if not info_path.exists():
        pytest.skip("TabArena dataset info.json not found.")
    info = json.loads(info_path.read_text())
    task_id = info.get("task_id")
    if task_id is None:
        pytest.skip("TabArena dataset info.json missing task_id.")

    split_names = _iter_split_names(dataset_dir)
    desired = {(0, 0), (0, 1), (1, 0), (1, 1)}
    selected: dict[tuple[int, int], str] = {}
    for name in split_names:
        fold_id, repeat_id, sample_id = _parse_split_triplet(name)
        if fold_id is None or repeat_id is None or sample_id is None:
            continue
        key = (fold_id, repeat_id)
        if key not in desired:
            continue
        if key not in selected or sample_id == 0:
            selected[key] = name

    if len(selected) < len(desired):
        missing = sorted(desired - set(selected.keys()))
        pytest.skip(f"Missing split(s) for fold/repeat combinations: {missing}")

    try:
        import openml
        from tabarena.benchmark.task.openml import OpenMLTaskWrapper
    except Exception as exc:  # pragma: no cover - optional dependency
        pytest.skip(f"TabArena/OpenML dependencies unavailable: {exc}")

    openml_cache_dir = REPO_ROOT / "data" / "cache" / "openml"
    openml_cache_dir.mkdir(parents=True, exist_ok=True)
    openml.config.set_root_cache_directory(str(openml_cache_dir))

    task = OpenMLTaskWrapper.from_task_id(int(task_id))

    for (fold_id, repeat_id), split_name in sorted(selected.items()):
        split_dir = dataset_dir / f"split-{split_name}"
        train_idx = np.load(split_dir / "train_idx.npy")
        val_idx = np.load(split_dir / "val_idx.npy")
        test_idx = np.load(split_dir / "test_idx.npy")

        _fold_id, _repeat_id, sample_id = _parse_split_triplet(split_name)
        if sample_id is None:
            pytest.skip(f"Split name missing sample id: {split_name}")

        train_idx_global, test_idx_global = task.get_split_indices(
            fold=fold_id,
            repeat=repeat_id,
            sample=sample_id,
        )
        train_idx_global = np.asarray(train_idx_global)
        test_idx_global = np.asarray(test_idx_global)

        logger.info(
            "Benchmark split check: fold=%d repeat=%d split=%s",
            fold_id,
            repeat_id,
            split_name,
        )

        export_train_union = np.concatenate([train_idx, val_idx])
        assert set(export_train_union) == set(train_idx_global)
        assert set(test_idx) == set(test_idx_global)
