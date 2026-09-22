"""TabPFN-native categorical encoding at the adapter boundary."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NativeCategoricalEncoder:
    """Encode native pandas categorical columns for TabPFN validation calls."""

    columns: tuple[Hashable, ...]
    encoder: Any | None

    @classmethod
    def empty(cls) -> NativeCategoricalEncoder:
        """Return a no-op encoder for shared-preprocessed inputs."""
        return cls(columns=(), encoder=None)

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        *,
        categorical_feature_indices: tuple[int, ...],
    ) -> NativeCategoricalEncoder:
        """Fit train-only ordinal encoders for native categorical columns."""
        columns = _validation_encode_columns(frame, categorical_feature_indices)
        if not columns:
            return cls.empty()
        from sklearn.preprocessing import OrdinalEncoder

        encoder = OrdinalEncoder(
            dtype=np.float64,
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
            encoded_missing_value=np.nan,
        )
        encoder.fit(_category_frame(frame, columns))
        return cls(columns=columns, encoder=encoder)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Transform known columns while mapping missing or unknown values to NaN."""
        if not self.columns or self.encoder is None:
            return frame
        missing_columns = [column for column in self.columns if column not in frame.columns]
        if missing_columns:
            raise ValueError(f"TabPFN native categorical encoder missing columns: {missing_columns!r}")
        transformed = frame.copy()
        encoded = self.encoder.transform(_category_frame(frame, self.columns))
        for index, column in enumerate(self.columns):
            transformed[column] = encoded[:, index]
        return transformed


def _validation_encode_columns(
    frame: pd.DataFrame,
    categorical_feature_indices: tuple[int, ...],
) -> tuple[Hashable, ...]:
    columns = tuple(frame.columns)
    selected: list[Hashable] = []
    for index in categorical_feature_indices:
        if index < 0 or index >= len(columns):
            continue
        column = columns[index]
        if _needs_validation_encoding(frame[column]):
            selected.append(column)
    return tuple(selected)


def _needs_validation_encoding(series: pd.Series) -> bool:
    dtype = series.dtype
    return (
        isinstance(dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
    )


def _category_frame(frame: pd.DataFrame, columns: tuple[Hashable, ...]) -> pd.DataFrame:
    category_frame = frame.loc[:, list(columns)].astype(object)
    return category_frame.where(pd.notna(category_frame), np.nan)
