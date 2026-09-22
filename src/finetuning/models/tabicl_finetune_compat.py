from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class TabICLMetaBatchSplitDiagnostics:
    """Diagnostics for TabICL fine-tune meta-batch split compatibility.

    Example:
        >>> diagnostics = TabICLMetaBatchSplitDiagnostics()
        >>> diagnostics.as_dict()["fallback_count"]
        0
    """

    total_chunk_count: int = 0
    classification_chunk_count: int = 0
    dense_label_remap_count: int = 0
    fallback_count: int = 0
    singleton_chunk_count: int = 0
    singleton_sample_count: int = 0
    query_size_clamped_count: int = 0
    unsupported_chunk_count: int = 0

    def as_dict(self) -> dict[str, int]:
        """Return a JSON/W&B-safe payload."""
        return asdict(self)


def _is_local_dense_labels(y: np.ndarray) -> bool:
    labels = np.unique(y)
    if labels.dtype.kind not in {"i", "u"}:
        return False
    return bool(np.array_equal(labels, np.arange(len(labels), dtype=labels.dtype)))


def _remap_to_local_dense_labels(y: np.ndarray) -> np.ndarray:
    _labels, inverse = np.unique(y, return_inverse=True)
    return inverse.astype(np.int64, copy=False)


def _safe_supported_query_split(
    y_chunk: np.ndarray,
    *,
    query_size: int,
    seed: int,
    diagnostics: TabICLMetaBatchSplitDiagnostics,
) -> tuple[np.ndarray, np.ndarray]:
    y_chunk = np.asarray(y_chunk)
    n_samples = len(y_chunk)
    query_size = max(1, min(int(query_size), n_samples - 1))
    labels, inverse, counts = np.unique(y_chunk, return_inverse=True, return_counts=True)
    singleton_mask = counts[inverse] == 1
    if singleton_mask.any():
        diagnostics.singleton_chunk_count += 1
        diagnostics.singleton_sample_count += int(singleton_mask.sum())

    eligible_query_indices = np.flatnonzero(~singleton_mask)
    if len(eligible_query_indices) == 0:
        diagnostics.unsupported_chunk_count += 1
        raise ValueError("TabICL cannot build a classification query from a chunk with only singleton classes.")

    candidate_groups: list[np.ndarray] = []
    rng = np.random.default_rng(int(seed))
    for label in labels[counts >= 2]:
        label_indices = np.flatnonzero(y_chunk == label)
        shuffled = rng.permutation(label_indices)
        candidate_groups.append(shuffled[1:])

    query_candidates = np.concatenate(candidate_groups) if candidate_groups else np.array([], dtype=np.int64)
    if len(query_candidates) == 0:
        diagnostics.unsupported_chunk_count += 1
        raise ValueError("TabICL cannot build a supported classification query for this chunk.")

    safe_query_size = min(query_size, len(query_candidates))
    if safe_query_size < query_size:
        diagnostics.query_size_clamped_count += 1

    query_idx = np.sort(rng.choice(query_candidates, size=safe_query_size, replace=False))
    query_mask = np.zeros(n_samples, dtype=bool)
    query_mask[query_idx] = True
    context_idx = np.flatnonzero(~query_mask)
    return context_idx, query_idx


@contextlib.contextmanager
def patch_tabicl_finetune_meta_batches() -> Iterator[TabICLMetaBatchSplitDiagnostics]:
    """Patch TabICL fine-tune chunk splits for rare-class classification chunks.

    The patch is process-local and scoped to the context manager. It keeps the
    upstream stratified split when it is feasible, remaps classification chunk
    labels to dense local ids, and falls back to a context-supported query split
    when a random chunk contains singleton classes.

    Example:
        >>> with patch_tabicl_finetune_meta_batches() as diagnostics:
        ...     diagnostics.fallback_count
        0
    """
    from tabicl._finetune import data as tabicl_data

    diagnostics = TabICLMetaBatchSplitDiagnostics()
    original_build_meta_batch = tabicl_data._build_meta_batch
    original_split_ctx_query = tabicl_data._split_ctx_query

    def split_ctx_query(y_chunk: np.ndarray, *, query_size: int, seed: int, stratify: bool):
        if not stratify:
            return original_split_ctx_query(y_chunk, query_size=query_size, seed=seed, stratify=False)
        try:
            return original_split_ctx_query(y_chunk, query_size=query_size, seed=seed, stratify=True)
        except ValueError as exc:
            message = str(exc)
            stratification_error = (
                "least populated class" in message or "test_size" in message or "train_size" in message
            )
            if not stratification_error:
                raise
            diagnostics.fallback_count += 1
            return _safe_supported_query_split(
                np.asarray(y_chunk),
                query_size=query_size,
                seed=seed,
                diagnostics=diagnostics,
            )

    def build_meta_batch(X_chunk: np.ndarray, y_chunk: np.ndarray, *, classification: bool, **kwargs: Any):
        diagnostics.total_chunk_count += 1
        if classification:
            diagnostics.classification_chunk_count += 1
            y_chunk = np.asarray(y_chunk)
            if not _is_local_dense_labels(y_chunk):
                diagnostics.dense_label_remap_count += 1
                y_chunk = _remap_to_local_dense_labels(y_chunk)
        return original_build_meta_batch(X_chunk, y_chunk, classification=classification, **kwargs)

    tabicl_data._split_ctx_query = split_ctx_query
    tabicl_data._build_meta_batch = build_meta_batch
    try:
        yield diagnostics
    finally:
        tabicl_data._build_meta_batch = original_build_meta_batch
        tabicl_data._split_ctx_query = original_split_ctx_query
