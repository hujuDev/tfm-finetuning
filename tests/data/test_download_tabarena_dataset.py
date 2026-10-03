from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from data.download import tabarena_dataset as exporter

if TYPE_CHECKING:
    from pathlib import Path


class _DummyTask:
    def __init__(self, *, task_type_id: str, task_type: str) -> None:
        self.task_type_id = task_type_id
        self.task_type = task_type


class _DummySplitTask:
    def __init__(self, *, estimation_procedure: dict[str, object] | None) -> None:
        self.estimation_procedure = estimation_procedure


def _build_split_args(**overrides: object) -> argparse.Namespace:
    payload = {
        "fold": None,
        "repeat": None,
        "sample": None,
        "folds": None,
        "repeats": None,
        "samples": None,
    }
    payload.update(overrides)
    return argparse.Namespace(**payload)


def test_resolve_split_grid_uses_tabarena_defaults_for_small_datasets() -> None:
    """No split arguments default to compact 3-fold CV for n<2500."""
    args = _build_split_args()
    task = _DummySplitTask(estimation_procedure=None)
    folds, repeats, samples, used_default = exporter._resolve_split_grid(
        args,
        source="tabarena",
        n_instances=2499,
        task=task,
        openml_split_policy="official",
    )
    assert used_default is True
    assert folds == [0, 1, 2]
    assert repeats == [0]
    assert samples == [0]


def test_resolve_split_grid_uses_tabarena_defaults_for_large_datasets() -> None:
    """No split arguments default to one fixed split for n>=2500."""
    args = _build_split_args()
    task = _DummySplitTask(estimation_procedure=None)
    folds, repeats, samples, used_default = exporter._resolve_split_grid(
        args,
        source="tabarena",
        n_instances=2500,
        task=task,
        openml_split_policy="official",
    )
    assert used_default is True
    assert folds == [0]
    assert repeats == [0]
    assert samples == [0]


def test_resolve_split_grid_respects_explicit_scalar_split_args() -> None:
    """Explicit scalar fold/repeat/sample values override default split policy."""
    args = _build_split_args(fold=1, repeat=2, sample=3)
    task = _DummySplitTask(estimation_procedure=None)
    folds, repeats, samples, used_default = exporter._resolve_split_grid(
        args,
        source="tabarena",
        n_instances=1000,
        task=task,
        openml_split_policy="official",
    )
    assert used_default is False
    assert folds == [1]
    assert repeats == [2]
    assert samples == [3]


def test_resolve_split_grid_respects_explicit_list_split_args() -> None:
    """Explicit fold/repeat/sample lists override default split policy."""
    args = _build_split_args(folds="0,2", repeats="1,4", samples="0")
    task = _DummySplitTask(estimation_procedure=None)
    folds, repeats, samples, used_default = exporter._resolve_split_grid(
        args,
        source="tabarena",
        n_instances=1000,
        task=task,
        openml_split_policy="official",
    )
    assert used_default is False
    assert folds == [0, 2]
    assert repeats == [1, 4]
    assert samples == [0]


def test_resolve_split_grid_openml_defaults_follow_task_estimation() -> None:
    """Legacy OpenML defaults follow the task estimation procedure split cardinality."""
    args = _build_split_args(resampling_protocol="legacy_tabarena")
    task = _DummySplitTask(
        estimation_procedure={
            "type": "crossvalidation",
            "parameters": {"number_repeats": "1", "number_folds": "10", "number_samples": ""},
        }
    )
    folds, repeats, samples, used_default = exporter._resolve_split_grid(
        args,
        source="openml",
        n_instances=581012,
        task=task,
        openml_split_policy="official",
    )
    assert used_default is True
    assert folds == list(range(10))
    assert repeats == [0]
    assert samples == [0]


def test_resolve_split_grid_tabarena_defaults_use_dataset_size_policy() -> None:
    """TabArena paper defaults keep the 2500-row cutoff policy."""
    args = _build_split_args()
    task = _DummySplitTask(estimation_procedure=None)
    folds, repeats, samples, used_default = exporter._resolve_split_grid(
        args,
        source="tabarena",
        n_instances=2499,
        task=task,
        openml_split_policy="official",
    )
    assert used_default is True
    assert folds == [0, 1, 2]
    assert repeats == [0]
    assert samples == [0]


def test_resolve_split_grid_openml_tabarena_policy_uses_dataset_size_cutoff() -> None:
    """OpenML can opt into the same paper split defaults via openml_split_policy=tabarena."""
    args = _build_split_args()
    task = _DummySplitTask(
        estimation_procedure={
            "type": "crossvalidation",
            "parameters": {"number_repeats": "1", "number_folds": "10", "number_samples": ""},
        }
    )
    folds, repeats, samples, used_default = exporter._resolve_split_grid(
        args,
        source="openml",
        n_instances=581012,
        task=task,
        openml_split_policy="tabarena",
    )
    assert used_default is True
    assert folds == [0]
    assert repeats == [0]
    assert samples == [0]


def test_resolve_split_grid_legacy_tabarena_protocol_restores_old_defaults() -> None:
    """legacy_tabarena keeps the old 3 folds x repeat-count default."""
    args = _build_split_args(resampling_protocol="legacy_tabarena")
    task = _DummySplitTask(estimation_procedure=None)
    folds, repeats, samples, used_default = exporter._resolve_split_grid(
        args,
        source="tabarena",
        n_instances=2499,
        task=task,
        openml_split_policy="official",
    )
    assert used_default is True
    assert folds == [0, 1, 2]
    assert repeats == list(range(10))
    assert samples == [0]


def test_build_posthoc_tabarena_split_map_returns_requested_grid() -> None:
    """Post-hoc TabArena split map returns requested repeat/fold/sample keys."""
    y = np.asarray([0, 1] * 30, dtype=np.int64)
    split_map = exporter._build_posthoc_tabarena_split_map(
        y,
        folds=[0, 1, 2],
        repeats=[0, 1],
        samples=[0],
        seed=0,
        problem_type="binary",
    )
    assert sorted(split_map.keys()) == [
        (0, 0, 0),
        (0, 1, 0),
        (0, 2, 0),
        (1, 0, 0),
        (1, 1, 0),
        (1, 2, 0),
    ]


def test_infer_openml_problem_type_binary() -> None:
    """Binary classification tasks are inferred as binary."""
    task = _DummyTask(
        task_type_id="TaskType.SUPERVISED_CLASSIFICATION",
        task_type="Supervised Classification",
    )
    y = pd.Series(["a", "b", "a", "b"])
    assert exporter._infer_openml_problem_type(task, y) == "binary"


def test_infer_openml_problem_type_multiclass() -> None:
    """Classification tasks with more than two labels map to multiclass."""
    task = _DummyTask(
        task_type_id="TaskType.SUPERVISED_CLASSIFICATION",
        task_type="Supervised Classification",
    )
    y = pd.Series(["a", "b", "c", "a"])
    assert exporter._infer_openml_problem_type(task, y) == "multiclass"


def test_infer_openml_problem_type_regression() -> None:
    """Regression tasks are inferred as regression."""
    task = _DummyTask(
        task_type_id="TaskType.SUPERVISED_REGRESSION",
        task_type="Supervised Regression",
    )
    y = pd.Series([1.0, 2.0, 3.0])
    assert exporter._infer_openml_problem_type(task, y) == "regression"


@pytest.mark.parametrize(
    ("problem_type", "expected_score"),
    [("binary", "roc-auc"), ("multiclass", "cross-entropy"), ("regression", "rmse")],
)
def test_default_score_metric_for_openml(problem_type: str, expected_score: str) -> None:
    """OpenML defaults map each task family to the expected score key."""
    assert exporter._default_score_metric_for_openml(problem_type) == expected_score


def test_resolve_default_output_dir_for_openml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """OpenML exports default to REPO_ROOT/data/openml/<dataset_slug>."""
    monkeypatch.setenv("REPO_ROOT", str(tmp_path))
    output_dir = exporter._resolve_output_dir(None, source="openml", dataset_name="Credit G")
    assert output_dir == tmp_path / "data" / "openml" / "credit_g"


def test_resolve_task_openml_by_task_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenML source resolution pulls dataset name from the task."""
    class _DummyDataset:
        name = "My OpenML Dataset"

    class _DummyOpenMLTask:
        def get_dataset(self) -> _DummyDataset:
            return _DummyDataset()

    monkeypatch.setattr(
        exporter.openml.tasks,
        "get_task",
        lambda task_id: _DummyOpenMLTask(),
    )
    metadata = pd.DataFrame([{"tid": 1, "name": "tabarena_only"}])
    task_id, dataset_name = exporter._resolve_task(
        metadata=metadata,
        task_id=42,
        dataset_name=None,
        source="openml",
    )
    assert task_id == 42
    assert dataset_name == "My OpenML Dataset"


def test_resolve_task_openml_requires_task_id() -> None:
    """OpenML source rejects missing task ids."""
    metadata = pd.DataFrame([{"tid": 1, "name": "tabarena_only"}])
    with pytest.raises(SystemExit, match="--task-id is required for --source openml"):
        exporter._resolve_task(
            metadata=metadata,
            task_id=None,
            dataset_name=None,
            source="openml",
        )


def test_configure_openml_cache_uses_repo_local_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenML exports should not depend on a writable home-directory cache."""
    configured_paths: list[str] = []

    monkeypatch.delenv("OPENML_CACHE_DIR", raising=False)
    monkeypatch.setenv("REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(
        exporter.openml.config,
        "set_root_cache_directory",
        configured_paths.append,
    )

    cache_dir = exporter._configure_openml_cache()

    assert cache_dir == tmp_path / "data" / "cache" / "openml"
    assert cache_dir.is_dir()
    assert configured_paths == [str(cache_dir)]
