from __future__ import annotations

import numpy as np

from finetuning.models.tabicl.finetune_compat import patch_tabicl_finetune_meta_batches


def test_tabicl_safe_split_keeps_singleton_classes_in_context() -> None:
    from tabicl._finetune import data as tabicl_data

    y_chunk = np.array([0, 0, 0, 0, 1, 2, 2, 2, 2])
    singleton_index = int(np.flatnonzero(y_chunk == 1)[0])

    with patch_tabicl_finetune_meta_batches() as diagnostics:
        context_idx, query_idx = tabicl_data._split_ctx_query(
            y_chunk,
            query_size=4,
            seed=7,
            stratify=True,
        )

    assert singleton_index in set(context_idx.tolist())
    assert singleton_index not in set(query_idx.tolist())
    assert set(y_chunk[query_idx]).issubset(set(y_chunk[context_idx]))
    assert diagnostics.fallback_count == 1
    assert diagnostics.singleton_chunk_count == 1
    assert diagnostics.singleton_sample_count == 1


def test_tabicl_meta_batch_remaps_sparse_chunk_labels_to_local_dense_ids() -> None:
    from tabicl._finetune import data as tabicl_data

    X_chunk = np.arange(24, dtype=np.float32).reshape(12, 2)
    y_chunk = np.array([1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3])

    with patch_tabicl_finetune_meta_batches() as diagnostics:
        batch = tabicl_data._build_meta_batch(
            X_chunk,
            y_chunk,
            classification=True,
            n_estimators=1,
            query_size=4,
            epoch_seed=11,
            chunk_idx=0,
            norm_methods="none",
            feat_shuffle_method="none",
            class_shuffle_method="none",
            outlier_threshold=4.0,
            preprocessing_seed=11,
        )

    assert diagnostics.classification_chunk_count == 1
    assert diagnostics.dense_label_remap_count == 1
    assert int(batch.y_train.max().item()) <= 1
    assert int(batch.y_query.max().item()) <= 1
