from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Hashable

SHARED_FEATURE_PREPROCESSING = "shared"



def feature_schema_metadata(
    frame: pd.DataFrame,
    *,
    categorical_feature_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Return adapter-facing feature schema metadata for a feature frame.

    Example:
        >>> schema = feature_schema_metadata(pd.DataFrame({"cat": ["a"], "x": [1.0]}))
        >>> schema["categorical_feature_indices"]
        [0]
    """
    columns = tuple(frame.columns)
    if categorical_feature_indices is None:
        categorical_indices = tuple(
            index for index, column in enumerate(columns) if _is_categorical_feature(frame[column])
        )
    else:
        categorical_indices = tuple(int(index) for index in categorical_feature_indices)

    categorical_index_set = set(categorical_indices)
    categorical_columns = [str(columns[index]) for index in categorical_indices if 0 <= index < len(columns)]
    numeric_columns = [
        str(column)
        for index, column in enumerate(columns)
        if index not in categorical_index_set and _is_numeric_feature(frame[column])
    ]
    boolean_columns = [
        str(column)
        for index, column in enumerate(columns)
        if index in categorical_index_set and pd.api.types.is_bool_dtype(frame[column].dtype)
    ]
    non_numeric_columns = [
        str(column)
        for index, column in enumerate(columns)
        if index in categorical_index_set and str(column) not in boolean_columns
    ]
    return {
        "columns": [str(column) for column in columns],
        "dtypes": {str(column): str(frame[column].dtype) for column in columns},
        "feature_count": len(columns),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "categorical_feature_indices": list(categorical_indices),
        "boolean_columns": boolean_columns,
        "non_numeric_columns": non_numeric_columns,
    }


@dataclass(frozen=True)
class FiniteFeaturePreprocessor:
    """Train-fitted transformer that returns finite numeric feature frames.

    Example:
        >>> frame = pd.DataFrame({"x": [1.0, np.nan], "label": ["a", None]})
        >>> preprocessor = FiniteFeaturePreprocessor.fit(frame)
        >>> preprocessor.transform(frame).isna().any().any()
        False
    """

    columns: tuple[Hashable, ...]
    numeric_columns: tuple[Hashable, ...]
    categorical_columns: tuple[Hashable, ...]
    numeric_fill_values: dict[Hashable, float]
    categorical_maps: dict[Hashable, dict[str, int]]

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> FiniteFeaturePreprocessor:
        """Fit finite-feature preprocessing from training features only.

        Example:
            >>> preprocessor = FiniteFeaturePreprocessor.fit(pd.DataFrame({"x": [1.0, np.nan]}))
            >>> preprocessor.numeric_fill_values["x"]
            1.0
        """
        columns = tuple(frame.columns)
        numeric_columns: list[Hashable] = []
        categorical_columns: list[Hashable] = []
        numeric_fill_values: dict[Hashable, float] = {}
        categorical_maps: dict[Hashable, dict[str, int]] = {}

        for column in columns:
            series = frame[column]
            if _is_numeric_feature(series):
                numeric_columns.append(column)
                numeric = _coerce_numeric(series)
                finite_mean = numeric.mean(skipna=True)
                numeric_fill_values[column] = 0.0 if pd.isna(finite_mean) else float(finite_mean)
            else:
                categorical_columns.append(column)
                categorical_maps[column] = _fit_category_map(series)

        return cls(
            columns=columns,
            numeric_columns=tuple(numeric_columns),
            categorical_columns=tuple(categorical_columns),
            numeric_fill_values=numeric_fill_values,
            categorical_maps=categorical_maps,
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Transform a feature frame to finite float32 columns.

        Example:
            >>> train = pd.DataFrame({"x": [1.0, np.nan], "cat": ["a", None]})
            >>> out = FiniteFeaturePreprocessor.fit(train).transform(train)
            >>> str(out.dtypes["x"])
            'float32'
        """
        missing_columns = [column for column in self.columns if column not in frame.columns]
        if missing_columns:
            raise ValueError(f"Feature frame is missing columns required by preprocessing: {missing_columns!r}")

        transformed_columns = {
            column: _coerce_numeric(frame[column]).fillna(self.numeric_fill_values[column])
            for column in self.numeric_columns
        }
        transformed_columns.update(
            {
                column: _encode_categories(frame[column], self.categorical_maps[column])
                for column in self.categorical_columns
            }
        )
        transformed = pd.DataFrame(
            {column: transformed_columns[column] for column in self.columns},
            index=frame.index,
        )
        return transformed.astype(np.float32, copy=False)

    def metadata(self, *, protocol: str = SHARED_FEATURE_PREPROCESSING) -> dict[str, Any]:
        """Return report metadata for the fitted preprocessing step.

        Example:
            >>> FiniteFeaturePreprocessor.fit(pd.DataFrame({"x": [1]})).metadata()["output_dtype"]
            'float32'
        """
        selected = [str(column) for column in self.columns]
        return {
            "protocol": protocol,
            "method": SHARED_FEATURE_PREPROCESSING,
            "fit_scope": "train_features_only",
            "raw_feature_count": len(self.columns),
            "effective_feature_count": len(self.columns),
            "numeric_feature_count": len(self.numeric_columns),
            "categorical_feature_count": len(self.categorical_columns),
            "dropped_features": [],
            "selected_features": selected,
            "missing_numeric_handling": "train_mean_impute_or_zero",
            "missing_category_code": 0,
            "unknown_category_code": 1,
            "output_dtype": "float32",
        }








def _is_numeric_feature(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series.dtype):
        return False
    return bool(pd.api.types.is_numeric_dtype(series.dtype))


def _is_categorical_feature(series: pd.Series) -> bool:
    dtype = series.dtype
    return bool(
        isinstance(dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
    )


def _coerce_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan).astype("float64")


def _fit_category_map(series: pd.Series) -> dict[str, int]:
    observed = sorted({_category_token(value) for value in series.to_numpy(dtype=object) if not pd.isna(value)})
    return {token: index + 2 for index, token in enumerate(observed)}


def _encode_categories(series: pd.Series, category_map: dict[str, int]) -> pd.Series:
    return pd.Series(
        (
            0 if pd.isna(value) else category_map.get(_category_token(value), 1)
            for value in series.to_numpy(dtype=object)
        ),
        index=series.index,
        dtype="float64",
    )


def _category_token(value: object) -> str:
    return str(value)
