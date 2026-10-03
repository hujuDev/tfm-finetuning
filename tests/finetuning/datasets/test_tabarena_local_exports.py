from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tfm_finetuning.finetuning.datasets.tabarena import TaskType, build_dataset_from_local_path


def test_build_dataset_from_local_path_uses_indexed_openml_split_export(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "higgs"
    split_dir = dataset_dir / "split-fold1_repeat0_sample0"
    split_dir.mkdir(parents=True)

    np.save(dataset_dir / "X_num.npy", np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float32))
    np.save(dataset_dir / "Y.npy", np.asarray([0, 1, 0, 1, 1], dtype=np.int64))
    np.save(split_dir / "train_idx.npy", np.asarray([0, 2], dtype=np.int64))
    np.save(split_dir / "val_idx.npy", np.asarray([1], dtype=np.int64))
    np.save(split_dir / "test_idx.npy", np.asarray([3, 4], dtype=np.int64))
    (dataset_dir / "info.json").write_text(
        json.dumps(
            {
                "name": "Higgs",
                "id": "openml-360114",
                "source": "openml",
                "task_id": 360114,
                "task_type": "binclass",
                "n_classes": 2,
                "score": "roc-auc",
                "splits": {
                    "fold0_repeat0_sample0": {"fold": 0, "repeat": 0, "sample": 0},
                    "fold1_repeat0_sample0": {
                        "fold": 1,
                        "repeat": 0,
                        "sample": 0,
                        "train_size": 2,
                        "val_size": 1,
                        "test_size": 2,
                    },
                    "fold2_repeat0_sample0": {"fold": 2, "repeat": 0, "sample": 0},
                    "fold0_repeat1_sample0": {"fold": 0, "repeat": 1, "sample": 0},
                    "fold1_repeat1_sample0": {"fold": 1, "repeat": 1, "sample": 0},
                    "fold2_repeat1_sample0": {"fold": 2, "repeat": 1, "sample": 0},
                },
            }
        ),
        encoding="utf-8",
    )

    dataset, metadata = build_dataset_from_local_path(
        dataset_path=dataset_dir,
        split_tuple=(1, 0, 0),
        num_policy=None,
        cat_policy=None,
        seed=0,
    )

    assert dataset.task.type_ == TaskType.BINCLASS
    assert dataset.data["x_num"]["train"].tolist() == [[0.0], [2.0]]
    assert dataset.data["x_num"]["val"].tolist() == [[1.0]]
    assert dataset.data["x_num"]["test"].tolist() == [[3.0], [4.0]]
    assert dataset.data["y"]["train"].tolist() == [0, 0]
    assert metadata["split"] == "fold1_repeat0_sample0"
    assert metadata["n_folds"] == 3
    assert metadata["n_repeats"] == 2
