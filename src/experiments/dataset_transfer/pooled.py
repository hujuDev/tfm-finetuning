"""Dataset-native balanced episode scheduling for pooled transfer stages.

Pooled datasets never share a row matrix here.  Each source owns its episode
sampler and therefore keeps its own schema and label vocabulary.  A single
``EpisodicFineTuneSession`` consumes every sampled episode, which represents one
persistent model/optimizer/scheduler session across the complete stage.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np


class DatasetEpisodeSource(Protocol):
    """A dataset-owned sampler that emits one schema-native training episode."""

    def sample_episode(self, *, rng: np.random.Generator) -> Any:
        """Sample one support/query episode without accessing another dataset."""
        ...


class EpisodicFineTuneSession(Protocol):
    """One persistent model and optimizer used for an entire pooled stage.

    Implementations must not recreate optimizer or scheduler state in
    ``train_episode``.  A new session is constructed only at a stage boundary.
    """

    def train_episode(
        self,
        episode: Any,
        *,
        dataset_name: str,
        global_update_index: int,
        dataset_update_index: int,
    ) -> Any:
        """Apply one optimizer update using a dataset-native episode."""
        ...


@dataclass(frozen=True)
class PooledUpdateRecord:
    """Audit record for one completed optimizer update."""

    global_update_index: int
    dataset_name: str
    dataset_update_index: int
    result: Any


@dataclass(frozen=True)
class PooledTrainingResult:
    """Completed balanced schedule, counts, and per-update adapter results."""

    schedule: tuple[str, ...]
    update_counts: Mapping[str, int]
    records: tuple[PooledUpdateRecord, ...]

    @property
    def total_updates(self) -> int:
        """Return the number of optimizer updates completed by the session."""
        return len(self.records)


def build_balanced_episode_schedule(
    dataset_names: Sequence[str],
    *,
    updates_per_dataset: int,
    seed: int,
) -> tuple[str, ...]:
    """Return a deterministic schedule with the requested steps per dataset.

    Each round contains every dataset once in a seeded random order.  This gives
    exact global balance and keeps prefix counts within one update of each other.

    Example:
        >>> schedule = build_balanced_episode_schedule(["a", "b"], updates_per_dataset=3, seed=0)
        >>> Counter(schedule) == {"a": 3, "b": 3}
        True
    """
    names = _dataset_names(dataset_names)
    update_count = _positive_int(updates_per_dataset, field="updates_per_dataset")
    rng = np.random.default_rng(_non_negative_int(seed, field="seed"))
    schedule: list[str] = []
    for _ in range(update_count):
        order = rng.permutation(len(names))
        schedule.extend(names[int(index)] for index in order)
    return tuple(schedule)


def run_balanced_pooled_training(
    session: EpisodicFineTuneSession,
    episode_sources: Mapping[str, DatasetEpisodeSource],
    *,
    updates_per_dataset: int,
    seed: int,
) -> PooledTrainingResult:
    """Run exactly balanced dataset-native episodes through one session.

    The mapping order is the declared scientific dataset order.  Episode RNGs
    are derived from ``(seed, dataset_name, dataset-local update index)`` so a
    dataset's sampled episode stream is stable even when interleaving changes.
    """
    if not episode_sources:
        raise ValueError("episode_sources must contain at least one dataset.")
    source_names = tuple(episode_sources)
    schedule = build_balanced_episode_schedule(
        source_names,
        updates_per_dataset=updates_per_dataset,
        seed=seed,
    )
    completed: Counter[str] = Counter()
    records: list[PooledUpdateRecord] = []
    for global_index, dataset_name in enumerate(schedule):
        dataset_index = completed[dataset_name]
        episode_rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    _non_negative_int(seed, field="seed"),
                    _stable_dataset_token(dataset_name),
                    dataset_index,
                ]
            )
        )
        episode = episode_sources[dataset_name].sample_episode(rng=episode_rng)
        step_result = session.train_episode(
            episode,
            dataset_name=dataset_name,
            global_update_index=global_index,
            dataset_update_index=dataset_index,
        )
        records.append(
            PooledUpdateRecord(
                global_update_index=global_index,
                dataset_name=dataset_name,
                dataset_update_index=dataset_index,
                result=step_result,
            )
        )
        completed[dataset_name] += 1

    counts = {name: int(completed[name]) for name in source_names}
    expected = _positive_int(updates_per_dataset, field="updates_per_dataset")
    if any(count != expected for count in counts.values()):
        raise RuntimeError(f"Balanced episode schedule completed with unexpected counts: {counts}.")
    return PooledTrainingResult(
        schedule=schedule,
        update_counts=MappingProxyType(counts),
        records=tuple(records),
    )


def _dataset_names(values: Sequence[str]) -> tuple[str, ...]:
    names = tuple(str(value).strip() for value in values)
    if not names or any(not name for name in names):
        raise ValueError("dataset_names must contain non-empty names.")
    if len(set(names)) != len(names):
        raise ValueError("dataset_names must be unique.")
    return names


def _non_negative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be a non-negative integer, got {value!r}.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be a non-negative integer, got {value!r}.") from exc
    if parsed != value or parsed < 0:
        raise ValueError(f"{field} must be a non-negative integer, got {value!r}.")
    return parsed


def _positive_int(value: Any, *, field: str) -> int:
    parsed = _non_negative_int(value, field=field)
    if parsed == 0:
        raise ValueError(f"{field} must be a positive integer, got {value!r}.")
    return parsed


def _stable_dataset_token(dataset_name: str) -> int:
    digest = hashlib.sha256(dataset_name.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


__all__ = [
    "DatasetEpisodeSource",
    "EpisodicFineTuneSession",
    "PooledTrainingResult",
    "PooledUpdateRecord",
    "build_balanced_episode_schedule",
    "run_balanced_pooled_training",
]
