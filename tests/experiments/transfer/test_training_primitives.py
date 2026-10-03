from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tfm_finetuning.experiments.transfer.pooled import (
    build_balanced_episode_schedule,
    run_balanced_pooled_training,
)
from tfm_finetuning.experiments.transfer.rows import (
    apply_train_subset_to_split,
    build_nested_train_subsets,
)
from tfm_finetuning.experiments.transfer.selection import (
    CheckpointValidation,
    ValidationMetric,
    select_source_mean_checkpoint,
    select_target_checkpoint,
)
from tfm_finetuning.finetuning.datasets.loading import SplitEvaluationData


def _split_data() -> SplitEvaluationData:
    train_X = pd.DataFrame({"feature": np.arange(12), "other": np.arange(100, 112)})
    val_X = pd.DataFrame({"feature": [20, 21], "other": [120, 121]})
    test_X = pd.DataFrame({"feature": [30, 31], "other": [130, 131]})
    return SplitEvaluationData(
        task_type="binary",
        problem_type="binary",
        metric_name="roc_auc",
        train_X=train_X,
        train_y=np.arange(12) % 2,
        val_context_X=train_X.copy(),
        val_context_y=np.arange(12) % 2,
        val_X=val_X,
        val_y=np.array([0, 1]),
        test_context_X=train_X.copy(),
        test_context_y=np.arange(12) % 2,
        test_X=test_X,
        test_y=np.array([1, 0]),
        metadata={"dataset": "target"},
        train_ids=np.array([f"train-{index}" for index in range(12)], dtype=object),
        val_context_ids=np.array([f"train-{index}" for index in range(12)], dtype=object),
        val_ids=np.array(["val-0", "val-1"], dtype=object),
        test_context_ids=np.array([f"train-{index}" for index in range(12)], dtype=object),
        test_ids=np.array(["test-0", "test-1"], dtype=object),
    )


def test_nested_train_rows_are_deterministic_exact_prefixes() -> None:
    first = build_nested_train_subsets(20_000, [64, 256, 1_024, 4_096, 16_384], seed=3)
    repeated = build_nested_train_subsets(20_000, [16_384, 64, 4_096, 256, 1_024], seed=3)

    assert first == repeated
    assert first[64].indices == first[256].indices[:64]
    assert first[256].indices == first[1_024].indices[:256]
    assert len(set(first[16_384].indices)) == 16_384
    assert first[64] != build_nested_train_subsets(20_000, [64], seed=4)[64]


def test_nested_train_view_reuses_exact_rows_for_finetuning_and_icl() -> None:
    original = _split_data()
    subset = build_nested_train_subsets(len(original.train_X), [4], seed=11)[4]
    view = apply_train_subset_to_split(original, subset)

    pd.testing.assert_frame_equal(view.train_X, view.val_context_X)
    pd.testing.assert_frame_equal(view.train_X, view.test_context_X)
    np.testing.assert_array_equal(view.train_y, view.val_context_y)
    np.testing.assert_array_equal(view.train_y, view.test_context_y)
    np.testing.assert_array_equal(view.train_ids, view.val_context_ids)
    np.testing.assert_array_equal(view.train_ids, view.test_context_ids)
    pd.testing.assert_frame_equal(view.val_X, original.val_X)
    pd.testing.assert_frame_equal(view.test_X, original.test_X)
    np.testing.assert_array_equal(view.val_y, original.val_y)
    np.testing.assert_array_equal(view.test_y, original.test_y)
    assert original.metadata == {"dataset": "target"}
    assert view.metadata["nested_train_rows"]["indices"] == list(subset.indices)


def test_nested_train_view_can_add_untouched_validation_only_to_test_icl() -> None:
    original = _split_data()
    subset = build_nested_train_subsets(len(original.train_X), [4], seed=11)[4]
    view = apply_train_subset_to_split(original, subset, include_validation_in_test_context=True)

    pd.testing.assert_frame_equal(view.test_context_X.iloc[:4].reset_index(drop=True), view.train_X)
    pd.testing.assert_frame_equal(view.test_context_X.iloc[4:].reset_index(drop=True), original.val_X)
    np.testing.assert_array_equal(view.test_context_y[:4], view.train_y)
    np.testing.assert_array_equal(view.test_context_y[4:], original.val_y)
    np.testing.assert_array_equal(view.test_context_ids[4:], original.val_ids)
    assert view.test_context_components.tolist() == ["train"] * 4 + ["validation"] * 2
    pd.testing.assert_frame_equal(view.val_X, original.val_X)
    pd.testing.assert_frame_equal(view.test_X, original.test_X)


def test_nested_train_rows_reject_implicit_clamping() -> None:
    with pytest.raises(ValueError, match="exceed"):
        build_nested_train_subsets(63, [64], seed=0)


@dataclass
class _EpisodeSource:
    columns: tuple[str, ...]
    label_values: tuple[int, ...]

    def sample_episode(self, *, rng: np.random.Generator) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "labels": self.label_values,
            "draw": int(rng.integers(0, 2**31)),
        }


class _RecordingSession:
    def __init__(self, schemas: dict[str, tuple[str, ...]]) -> None:
        self.schemas = schemas
        self.optimizer_step = 0
        self.session_identity = id(self)
        self.calls: list[tuple[int, str, int, int]] = []

    def train_episode(
        self,
        episode: dict[str, Any],
        *,
        dataset_name: str,
        global_update_index: int,
        dataset_update_index: int,
    ) -> dict[str, int]:
        assert episode["columns"] == self.schemas[dataset_name]
        self.optimizer_step += 1
        self.calls.append((self.session_identity, dataset_name, global_update_index, dataset_update_index))
        return {"optimizer_step": self.optimizer_step, "draw": episode["draw"]}


def _sources() -> dict[str, _EpisodeSource]:
    return {
        "credit": _EpisodeSource(("income", "age"), (0, 1)),
        "satellite": _EpisodeSource(("band_1", "band_2", "band_3"), (1, 3, 7)),
        "software": _EpisodeSource(("loc",), ("low", "high")),
    }


def test_balanced_schedule_is_exact_deterministic_and_prefix_balanced() -> None:
    schedule = build_balanced_episode_schedule(list(_sources()), updates_per_dataset=7, seed=5)

    assert schedule == build_balanced_episode_schedule(list(_sources()), updates_per_dataset=7, seed=5)
    assert Counter(schedule) == {"credit": 7, "satellite": 7, "software": 7}
    for offset in range(0, len(schedule), 3):
        assert set(schedule[offset : offset + 3]) == set(_sources())


def test_pooled_training_keeps_native_schemas_and_one_persistent_session() -> None:
    sources = _sources()
    session = _RecordingSession({name: source.columns for name, source in sources.items()})
    result = run_balanced_pooled_training(session, sources, updates_per_dataset=5, seed=9)

    assert result.total_updates == 15
    assert dict(result.update_counts) == {"credit": 5, "satellite": 5, "software": 5}
    assert session.optimizer_step == 15
    assert {call[0] for call in session.calls} == {session.session_identity}
    assert [record.dataset_name for record in result.records] == list(result.schedule)
    assert [record.result["optimizer_step"] for record in result.records] == list(range(1, 16))

    repeated = run_balanced_pooled_training(
        _RecordingSession({name: source.columns for name, source in sources.items()}),
        _sources(),
        updates_per_dataset=5,
        seed=9,
    )
    assert repeated.schedule == result.schedule
    assert [record.result["draw"] for record in repeated.records] == [
        record.result["draw"] for record in result.records
    ]


def _candidate(
    checkpoint_id: str,
    step: int,
    *,
    target: float,
    high_source: float,
    low_source: float,
) -> CheckpointValidation:
    return CheckpointValidation(
        checkpoint_id=checkpoint_id,
        step=step,
        metrics={
            "target": ValidationMetric(target, baseline=0.5, higher_is_better=True),
            "high": ValidationMetric(high_source, baseline=0.8, higher_is_better=True),
            "low": ValidationMetric(low_source, baseline=10.0, higher_is_better=False),
        },
    )


def test_target_only_selector_ignores_source_performance() -> None:
    candidates = [
        _candidate("best-target", 2, target=0.7, high_source=0.81, low_source=9.0),
        _candidate("best-sources", 3, target=0.6, high_source=0.95, low_source=1.0),
    ]

    selected = select_target_checkpoint(candidates, target_dataset="target")

    assert selected.checkpoint_id == "best-target"
    assert selected.selection_score == pytest.approx(0.2)
    assert dict(selected.dataset_improvements) == {"target": pytest.approx(0.2)}


def test_source_selector_equal_weights_orientation_corrected_improvements() -> None:
    candidates = [
        _candidate("large-low-metric-gain", 2, target=0.5, high_source=0.81, low_source=1.0),
        _candidate("high-metric-gain", 3, target=0.5, high_source=0.90, low_source=9.0),
    ]

    selected = select_source_mean_checkpoint(candidates, source_datasets=["high", "low"])

    assert selected.checkpoint_id == "large-low-metric-gain"
    assert selected.dataset_improvements["high"] == pytest.approx(0.01)
    assert selected.dataset_improvements["low"] == pytest.approx(9.0)
    assert selected.selection_score == pytest.approx((0.01 + 9.0) / 2)


def test_selector_requires_matched_baselines_and_all_declared_datasets() -> None:
    first = _candidate("first", 1, target=0.6, high_source=0.9, low_source=9.0)
    mismatched = CheckpointValidation(
        checkpoint_id="mismatched",
        step=2,
        metrics={
            **first.metrics,
            "target": ValidationMetric(0.7, baseline=0.55, higher_is_better=True),
        },
    )

    with pytest.raises(ValueError, match="unmatched"):
        select_target_checkpoint([first, mismatched], target_dataset="target")
    with pytest.raises(ValueError, match="lacks validation"):
        select_source_mean_checkpoint([first], source_datasets=["missing"])
