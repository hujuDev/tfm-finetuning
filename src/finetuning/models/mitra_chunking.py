"""Deterministic row chunking and support/query splits for local Mitra training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from finetuning.models.mitra_common import (
    MITRA_INTERFACE_LOCAL_CHUNKED,
    MITRA_SCHEDULER_COSINE_WARMUP,
)


@dataclass
class LocalMitraFineTuneDiagnostics:
    """Fit-time diagnostics for the thesis-owned Mitra fine-tune loop."""

    interface: str = MITRA_INTERFACE_LOCAL_CHUNKED
    context_size: int = 0
    query_ratio: float = 0.0
    min_chunk_size: int = 50
    max_samples_support: int = 0
    max_samples_query: int = 0
    use_lr_scheduler: bool = True
    scheduler_policy: str = MITRA_SCHEDULER_COSINE_WARMUP
    warmup_proportion: float = 0.1
    warmup_steps: int = 0
    total_optimizer_steps_planned: int = 0
    chunks_per_epoch: int = 0
    epochs_configured: int = 0
    epochs_completed: int = 0
    optimizer_steps_completed: int = 0
    skipped_chunks: int = 0
    dropped_tail_rows: int = 0
    query_size_clamped_count: int = 0
    rare_class_fallback_count: int = 0
    singleton_chunk_count: int = 0
    grad_clip: float | None = None
    grad_clip_applied_steps: int = 0
    grad_norm_max: float | None = None
    budget_reached: bool = False
    early_stopped: bool = False

    def as_payload(self) -> dict[str, Any]:
        """Return JSON-serializable diagnostics."""
        return asdict(self)


@dataclass(frozen=True)
class SplitResult:
    support_idx: np.ndarray
    query_idx: np.ndarray
    query_size_clamped: bool = False
    rare_class_fallback: bool = False
    singleton_chunk: bool = False


def count_mitra_epoch_chunks(n_samples: int, context_size: int, min_chunk_size: int = 50) -> int:
    if n_samples <= 0:
        return 0
    context_size = max(2, int(context_size))
    min_chunk_size = max(2, int(min_chunk_size))
    if n_samples <= context_size:
        return 1
    full_chunks, remainder = divmod(int(n_samples), context_size)
    return full_chunks + (1 if remainder >= min_chunk_size else 0)


def dropped_mitra_tail_rows(n_samples: int, context_size: int, min_chunk_size: int = 50) -> int:
    if n_samples <= 0:
        return 0
    context_size = max(2, int(context_size))
    min_chunk_size = max(2, int(min_chunk_size))
    if n_samples <= context_size:
        return 0
    remainder = int(n_samples) % context_size
    return remainder if 0 < remainder < min_chunk_size else 0


def iter_mitra_epoch_chunks(
    n_samples: int,
    *,
    context_size: int,
    epoch_seed: int,
    min_chunk_size: int = 50,
) -> list[np.ndarray]:
    if n_samples <= 0:
        return []
    context_size = max(2, int(context_size))
    min_chunk_size = max(2, int(min_chunk_size))
    rng = np.random.default_rng(int(epoch_seed))
    perm = rng.permutation(int(n_samples))
    if n_samples <= context_size:
        return [perm]
    full_chunks, remainder = divmod(int(n_samples), context_size)
    chunks = [perm[index * context_size : (index + 1) * context_size] for index in range(full_chunks)]
    if remainder >= min_chunk_size:
        chunks.append(perm[full_chunks * context_size :])
    return chunks


def split_mitra_support_query_indices(
    y_chunk: np.ndarray,
    *,
    query_ratio: float,
    seed: int,
    classification: bool,
) -> SplitResult:
    """Split one chunk while retaining query classes in support when possible."""
    n_rows = len(y_chunk)
    if n_rows < 2:
        raise ValueError("Local Mitra fine-tuning requires at least two rows per chunk.")
    requested_query = max(1, round(n_rows * float(query_ratio)))
    query_size = min(requested_query, n_rows - 1)
    query_size_clamped = query_size != requested_query
    rng = np.random.default_rng(int(seed))
    if not classification:
        perm = rng.permutation(n_rows)
        return SplitResult(
            support_idx=perm[query_size:],
            query_idx=perm[:query_size],
            query_size_clamped=query_size_clamped,
        )

    y_chunk = np.asarray(y_chunk)
    labels, counts = np.unique(y_chunk, return_counts=True)
    singleton_chunk = bool(np.any(counts == 1))
    max_safe_query = int(np.sum(np.maximum(counts - 1, 0)))
    if max_safe_query <= 0:
        perm = rng.permutation(n_rows)
        return SplitResult(
            support_idx=perm[query_size:],
            query_idx=perm[:query_size],
            query_size_clamped=query_size_clamped,
            rare_class_fallback=True,
            singleton_chunk=singleton_chunk,
        )
    if query_size > max_safe_query:
        query_size = max_safe_query
        query_size_clamped = True

    selected: list[int] = []
    per_label_selected = dict.fromkeys(labels.tolist(), 0)
    per_label_limit = {label: int(count - 1) for label, count in zip(labels.tolist(), counts.tolist(), strict=True)}
    for candidate in rng.permutation(n_rows):
        label = y_chunk[candidate].item() if hasattr(y_chunk[candidate], "item") else y_chunk[candidate]
        if per_label_selected[label] >= per_label_limit[label]:
            continue
        selected.append(int(candidate))
        per_label_selected[label] += 1
        if len(selected) >= query_size:
            break
    if not selected:
        perm = rng.permutation(n_rows)
        return SplitResult(
            support_idx=perm[query_size:],
            query_idx=perm[:query_size],
            query_size_clamped=query_size_clamped,
            rare_class_fallback=True,
            singleton_chunk=singleton_chunk,
        )
    query_idx = np.asarray(selected, dtype=np.int64)
    query_mask = np.zeros(n_rows, dtype=bool)
    query_mask[query_idx] = True
    return SplitResult(
        support_idx=np.flatnonzero(~query_mask),
        query_idx=query_idx,
        query_size_clamped=query_size_clamped,
        singleton_chunk=singleton_chunk,
    )
