"""Raw local dataset containers; preprocessing belongs to the model protocol."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Literal

import numpy as np

PartKey = Literal["train", "val", "test"]
DataKey = Literal["x_num", "x_bin", "x_cat", "y"]


class TaskType(enum.Enum):
    REGRESSION = "regression"
    BINCLASS = "binclass"
    MULTICLASS = "multiclass"


class Score(enum.Enum):
    ACCURACY = "accuracy"
    CROSS_ENTROPY = "cross-entropy"
    MAE = "mae"
    R2 = "r2"
    RMSE = "rmse"
    ROC_AUC = "roc-auc"


@dataclass(frozen=True)
class Task:
    """Prediction task metadata and split labels."""

    labels: dict[PartKey, np.ndarray]
    type_: TaskType
    score: Score

    @property
    def is_regression(self) -> bool:
        """Return whether this is a regression task."""
        return self.type_ == TaskType.REGRESSION

    @property
    def is_binclass(self) -> bool:
        """Return whether this is a binary classification task."""
        return self.type_ == TaskType.BINCLASS

    @property
    def is_multiclass(self) -> bool:
        """Return whether this is a multiclass task."""
        return self.type_ == TaskType.MULTICLASS

    @property
    def is_classification(self) -> bool:
        """Return whether this is any classification task."""
        return self.is_binclass or self.is_multiclass


@dataclass
class Dataset:
    """Encoded feature blocks and task labels for train/val/test splits."""

    data: dict[DataKey, dict[PartKey, np.ndarray]]
    task: Task

    def __contains__(self, key: DataKey) -> bool:
        """Return whether a feature block exists."""
        return key in self.data

    def __getitem__(self, key: DataKey) -> dict[PartKey, np.ndarray]:
        """Return one feature block across splits."""
        return self.data[key]

    def __setitem__(self, key: DataKey, value: dict[PartKey, np.ndarray]) -> None:
        """Replace one feature block across splits."""
        self.data[key] = value

    @property
    def n_num_features(self) -> int:
        """Return the number of numerical features."""
        return self.data["x_num"]["train"].shape[1] if "x_num" in self.data else 0

    @property
    def n_bin_features(self) -> int:
        """Return the number of binary features."""
        return self.data["x_bin"]["train"].shape[1] if "x_bin" in self.data else 0

    @property
    def n_cat_features(self) -> int:
        """Return the number of categorical features."""
        return self.data["x_cat"]["train"].shape[1] if "x_cat" in self.data else 0

    @property
    def n_features(self) -> int:
        """Return the total encoded feature count."""
        return self.n_num_features + self.n_bin_features + self.n_cat_features

    def size(self, part: None | PartKey) -> int:
        """Return one split size or the total number of rows."""
        return sum(map(len, self.task.labels.values())) if part is None else len(self.task.labels[part])
