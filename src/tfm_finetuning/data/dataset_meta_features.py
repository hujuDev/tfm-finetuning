"""Train-distribution dataset meta-features for thesis experiment design plots.

The functions in this module intentionally operate on already-resolved training
frames so tests and future pipelines can reuse the same definitions without
going through thesis-web.

Example:
    >>> import pandas as pd
    >>> X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    >>> y = [0, 0, 1, 1]
    >>> compute_base_meta_features(X, y, "binary")["labelEntropyNormalized"]
    1.0
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from scipy import sparse
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

BASE_META_FEATURE_KEYS = (
    "baseTrainRows",
    "nFeaturesTotal",
    "trainingTableCells",
    "featureToRowRatio",
    "numericFeatureShare",
    "numHighCardinalityCategoricalFeatures",
    "maxCategoricalCardinality",
    "missingValueShare",
    "labelEntropyNormalized",
    "targetP99OverMedian",
    "targetRobustTailIndex",
    "univariateSignalTop5Share",
    "numericEffectiveRankRatio",
    "numericMeanAbsoluteSpearman",
    "numericTailWeightMedian",
    "categoricalRareValueShare",
    "missingnessPatternEntropyNormalized",
    "duplicateRowShare",
    "schemaTypeEntropyNormalized",
    "localClassDisagreementRate",
    "localTargetVariationIqr",
    "marginalShapeHeterogeneity",
    "sparseRegionOutlierMass",
    "categoricalCooccurrenceSparsity",
)

CLASSIFICATION_TASK_FAMILIES = {"binary", "binclass", "classification", "multiclass"}
REGRESSION_TASK_FAMILIES = {"regression"}
LOCAL_DATA_ROOTS = ("tabarena", "openml", "other", "beyondarena")
DEFAULT_SPLIT_TUPLE = (0, 0, 0)
DEFAULT_META_FEATURE_SEED = 0
DEFAULT_MAX_MI_ROWS = 50_000
DEFAULT_MAX_STRUCTURE_ROWS = 50_000
DEFAULT_MAX_DEPENDENCE_FEATURES = 256
DEFAULT_MAX_LOCAL_ROWS = 1_000
DEFAULT_MAX_CATEGORICAL_LEVELS = 32


def empty_base_meta_features() -> dict[str, float | int | None]:
    """Return the overview base-meta fields with unavailable values.

    Example:
        >>> set(empty_base_meta_features()) == set(BASE_META_FEATURE_KEYS)
        True
    """
    return dict.fromkeys(BASE_META_FEATURE_KEYS)


def clean_finite_number(value: object) -> float | None:
    """Return a finite float, or ``None`` for missing/non-finite values.

    Example:
        >>> clean_finite_number("2.5")
        2.5
        >>> clean_finite_number(float("nan")) is None
        True
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def normalized_task_family(value: object) -> str:
    """Normalize task-family aliases used by TabArena and local configs.

    Example:
        >>> normalized_task_family("binclass")
        'binary'
    """
    text = str(getattr(value, "value", value) or "").strip().lower().replace("-", "_")
    if text == "binclass":
        return "binary"
    return text


def fallback_base_meta_features(
    *,
    n_train_rows: object,
    n_val_rows: object,
    n_features_total: object,
) -> dict[str, float | int | None]:
    """Return dataset-scale fields from catalog split-size metadata.

    Example:
        >>> features = fallback_base_meta_features(n_train_rows=8, n_val_rows=2, n_features_total=5)
        >>> features["baseTrainRows"], features["trainingTableCells"]
        (10, 50)
    """
    features = empty_base_meta_features()
    train_rows = clean_finite_number(n_train_rows)
    val_rows = clean_finite_number(n_val_rows)
    total_features = clean_finite_number(n_features_total)
    if total_features is not None:
        features["nFeaturesTotal"] = int(total_features)
    if train_rows is not None and val_rows is not None:
        base_train_rows = int(train_rows + val_rows)
        features["baseTrainRows"] = base_train_rows
        if base_train_rows > 0 and total_features is not None:
            features["trainingTableCells"] = base_train_rows * int(total_features)
            features["featureToRowRatio"] = float(total_features) / float(base_train_rows)
    return features


def compute_label_entropy_normalized(y_train: Any) -> float | None:
    """Compute class-label entropy normalized by the maximum entropy for ``k`` labels.

    Example:
        >>> compute_label_entropy_normalized([0, 0, 1, 1])
        1.0
    """
    labels = pd.Series(y_train).dropna().to_numpy()
    if labels.size == 0:
        return None
    _classes, counts = np.unique(labels, return_counts=True)
    if counts.size < 2:
        return None
    probabilities = counts.astype(float) / float(counts.sum())
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    normalized = entropy / float(np.log(counts.size))
    return clean_finite_number(normalized)


def compute_target_p99_over_median(y_train: Any) -> float | None:
    """Return the regression target p99 divided by the target median.

    The ratio is only defined when the median is finite and positive.

    Example:
        >>> compute_target_p99_over_median([1, 2, 3, 4])
        1.5879999999999999
    """
    values = pd.to_numeric(pd.Series(y_train), errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    median = float(np.median(values))
    if not np.isfinite(median) or median <= 0:
        return None
    return clean_finite_number(float(np.percentile(values, 99)) / median)


def compute_target_robust_tail_index(y_train: Any) -> float | None:
    """Return the central 98% target range divided by the interquartile range.

    Unlike p99/median, this shift-invariant regression-tail descriptor remains
    defined for zero-centred and negative targets.

    Example:
        >>> compute_target_robust_tail_index([-2, -1, 0, 1, 2])
        1.96
    """
    values = pd.to_numeric(pd.Series(y_train), errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    q01, q25, q75, q99 = np.percentile(values, [1, 25, 75, 99])
    iqr = float(q75 - q25)
    if not np.isfinite(iqr) or iqr <= 0:
        return None
    return clean_finite_number(float(q99 - q01) / iqr)


def compute_base_meta_features(
    X_train: pd.DataFrame,
    y_train: Any,
    task_family: object,
    *,
    max_mi_rows: int = DEFAULT_MAX_MI_ROWS,
    seed: int = DEFAULT_META_FEATURE_SEED,
    include_univariate_signal: bool = True,
) -> dict[str, float | int | None]:
    """Compute overview base meta-features from the outer training distribution.

    ``X_train`` and ``y_train`` should represent the exported outer train and
    validation rows concatenated together; test rows are intentionally absent.

    Example:
        >>> import pandas as pd
        >>> X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
        >>> compute_base_meta_features(X, [0, 0, 1, 1], "binary")["baseTrainRows"]
        4
    """
    if len(X_train) != len(y_train):
        raise ValueError(f"X_train/y_train length mismatch: {len(X_train)} != {len(y_train)}")

    task = normalized_task_family(task_family)
    features = empty_base_meta_features()
    base_train_rows = len(X_train)
    features["baseTrainRows"] = base_train_rows
    features["nFeaturesTotal"] = int(X_train.shape[1])
    features["trainingTableCells"] = base_train_rows * int(X_train.shape[1])
    features["featureToRowRatio"] = float(X_train.shape[1]) / float(base_train_rows) if base_train_rows > 0 else None
    features["numericFeatureShare"] = (
        sum(is_numeric_dtype(dtype) and not is_bool_dtype(dtype) for dtype in X_train.dtypes) / float(X_train.shape[1])
        if X_train.shape[1] > 0
        else None
    )
    categorical_cardinalities = [
        int(series.nunique(dropna=True))
        for _column, series in X_train.items()
        if not (is_numeric_dtype(series.dtype) and not is_bool_dtype(series.dtype))
    ]
    features["numHighCardinalityCategoricalFeatures"] = sum(
        cardinality > 50 for cardinality in categorical_cardinalities
    )
    features["maxCategoricalCardinality"] = max(categorical_cardinalities) if categorical_cardinalities else None
    features["missingValueShare"] = (
        float(X_train.isna().to_numpy().mean()) if X_train.shape[0] > 0 and X_train.shape[1] > 0 else None
    )
    features.update(
        compute_distribution_structure_features(
            X_train,
            max_rows=DEFAULT_MAX_STRUCTURE_ROWS,
            max_numeric_features=DEFAULT_MAX_DEPENDENCE_FEATURES,
            seed=seed,
        )
    )
    if task in CLASSIFICATION_TASK_FAMILIES:
        features["labelEntropyNormalized"] = compute_label_entropy_normalized(y_train)
    elif task in REGRESSION_TASK_FAMILIES:
        features["targetP99OverMedian"] = compute_target_p99_over_median(y_train)
        features["targetRobustTailIndex"] = compute_target_robust_tail_index(y_train)
    if include_univariate_signal:
        features["univariateSignalTop5Share"] = compute_univariate_signal_top5_share(
            X_train,
            y_train,
            task,
            max_rows=max_mi_rows,
            seed=seed,
        )
    features.update(
        compute_local_target_structure_features(
            X_train,
            y_train,
            task,
            max_rows=DEFAULT_MAX_LOCAL_ROWS,
            seed=seed,
        )
    )
    return features


def compute_distribution_structure_features(
    X_train: pd.DataFrame,
    *,
    max_rows: int = DEFAULT_MAX_STRUCTURE_ROWS,
    max_numeric_features: int = DEFAULT_MAX_DEPENDENCE_FEATURES,
    seed: int = DEFAULT_META_FEATURE_SEED,
) -> dict[str, float | None]:
    """Summarize multivariate dependence, marginals, schema sparsity, and row redundancy.

    The summaries are label-free and use training rows only.  Row and numeric
    feature caps are deterministic safeguards for very large or wide tables.
    """
    sampled = _deterministic_frame_sample(X_train, max_rows=max_rows, seed=seed)
    numeric = sampled.loc[
        :,
        [
            column
            for column in sampled.columns
            if is_numeric_dtype(sampled[column].dtype) and not is_bool_dtype(sampled[column].dtype)
        ],
    ]
    if numeric.shape[1] > max_numeric_features:
        rng = np.random.default_rng(int(seed))
        positions = np.sort(rng.choice(numeric.shape[1], size=int(max_numeric_features), replace=False))
        numeric = numeric.iloc[:, positions]

    effective_rank, mean_abs_spearman = _numeric_dependence_summary(numeric)
    categorical = sampled.loc[
        :,
        [
            column
            for column in sampled.columns
            if not (is_numeric_dtype(sampled[column].dtype) and not is_bool_dtype(sampled[column].dtype))
        ],
    ]
    return {
        "numericEffectiveRankRatio": effective_rank,
        "numericMeanAbsoluteSpearman": mean_abs_spearman,
        "numericTailWeightMedian": _numeric_tail_weight_median(numeric),
        "marginalShapeHeterogeneity": _marginal_shape_heterogeneity(numeric),
        "sparseRegionOutlierMass": _sparse_region_outlier_mass(
            sampled,
            max_rows=DEFAULT_MAX_LOCAL_ROWS,
            max_features=DEFAULT_MAX_DEPENDENCE_FEATURES,
            seed=seed,
        ),
        "categoricalRareValueShare": _categorical_rare_value_share(categorical),
        "categoricalCooccurrenceSparsity": _categorical_cooccurrence_sparsity(categorical),
        "missingnessPatternEntropyNormalized": _missingness_pattern_entropy(sampled),
        "duplicateRowShare": clean_finite_number(float(sampled.duplicated().mean())) if len(sampled) else None,
        "schemaTypeEntropyNormalized": _schema_type_entropy(sampled),
    }


def _schema_type_entropy(frame: pd.DataFrame) -> float | None:
    """Return normalized entropy of the numeric/categorical feature mix.

    Zero means a single-type schema and one means an even mix. The descriptor
    is intentionally based on columns rather than cells so wide columns do not
    receive extra weight merely because the dataset has more rows.
    """
    if frame.shape[1] == 0:
        return None
    numeric_count = sum(is_numeric_dtype(dtype) and not is_bool_dtype(dtype) for dtype in frame.dtypes)
    probabilities = np.asarray(
        [numeric_count / frame.shape[1], (frame.shape[1] - numeric_count) / frame.shape[1]],
        dtype=float,
    )
    probabilities = probabilities[probabilities > 0]
    if probabilities.size == 1:
        return 0.0
    return clean_finite_number(-float(np.sum(probabilities * np.log(probabilities))) / np.log(2.0))


def _deterministic_frame_sample(frame: pd.DataFrame, *, max_rows: int, seed: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame.reset_index(drop=True)
    rng = np.random.default_rng(int(seed))
    positions = np.sort(rng.choice(len(frame), size=int(max_rows), replace=False))
    return frame.iloc[positions].reset_index(drop=True)


def _numeric_dependence_summary(numeric: pd.DataFrame) -> tuple[float | None, float | None]:
    if numeric.shape[1] < 2 or len(numeric) < 3:
        return None, None
    filled = numeric.apply(pd.to_numeric, errors="coerce")
    varying = filled.max(skipna=True).ne(filled.min(skipna=True))
    filled = filled.loc[:, varying]
    if filled.shape[1] < 2:
        return None, None
    filled = filled.fillna(filled.median()).fillna(0.0)
    correlation = filled.corr(method="spearman").to_numpy(dtype=float)
    if not np.isfinite(correlation).all():
        return None, None
    off_diagonal = np.abs(correlation[np.triu_indices(correlation.shape[0], k=1)])
    mean_abs = clean_finite_number(float(np.mean(off_diagonal))) if off_diagonal.size else None
    eigenvalues = np.clip(np.linalg.eigvalsh(correlation), 0.0, None)
    eigenvalue_sum = float(eigenvalues.sum())
    if eigenvalue_sum <= 0:
        return None, mean_abs
    probabilities = eigenvalues[eigenvalues > 0] / eigenvalue_sum
    effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    return clean_finite_number(effective_rank / correlation.shape[0]), mean_abs


def _numeric_tail_weight_median(numeric: pd.DataFrame) -> float | None:
    tail_weights: list[float] = []
    for column in numeric.columns:
        series = numeric[column]
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size < 4:
            continue
        q01, q25, q75, q99 = np.percentile(values, [1, 25, 75, 99])
        iqr = float(q75 - q25)
        if iqr > 0:
            tail_weights.append(float(q99 - q01) / iqr)
    return clean_finite_number(float(np.median(tail_weights))) if tail_weights else None


def _marginal_shape_heterogeneity(numeric: pd.DataFrame) -> float | None:
    """Return the mean distance between robust numeric marginal-shape signatures.

    Each column is represented by bounded versions of its skewness, robust
    tail weight, exact-zero share, and discreteness. Pairwise Euclidean
    distances are divided by their maximum possible value, yielding a
    scale-free value in ``[0, 1]``. Skewness is therefore one component rather
    than a standalone hypothesis.
    """
    signatures: list[list[float]] = []
    for column in numeric.columns:
        values = pd.to_numeric(numeric[column], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size < 8 or np.unique(values).size < 2:
            continue
        q01, q25, q75, q99 = np.percentile(values, [1, 25, 75, 99])
        iqr = float(q75 - q25)
        tail_weight = float(q99 - q01) / iqr if iqr > 0 else 0.0
        skewness = float(pd.Series(values).skew())
        if not np.isfinite(skewness):
            continue
        signatures.append(
            [
                0.5 * (np.tanh(skewness / 3.0) + 1.0),
                np.tanh(np.log1p(max(tail_weight, 0.0)) / 3.0),
                float(np.mean(values == 0)),
                1.0 - float(np.unique(values).size) / float(values.size),
            ]
        )
    if len(signatures) < 2:
        return None
    matrix = np.asarray(signatures, dtype=float)
    distances = []
    for left in range(len(matrix) - 1):
        distances.extend(np.linalg.norm(matrix[left + 1 :] - matrix[left], axis=1) / 2.0)
    return clean_finite_number(float(np.mean(distances)))


def _sparse_region_outlier_mass(
    frame: pd.DataFrame,
    *,
    max_rows: int,
    max_features: int,
    seed: int,
) -> float | None:
    """Return the 95th/median nearest-neighbour distance ratio.

    The mixed-type representation matches the local-target descriptor. A high
    value means that the sparsest five percent of training rows are far from
    ordinary local support. Duplicate rows have zero distance and are retained;
    the reference median uses positive distances so duplicates do not make the
    ratio undefined.
    """
    sampled = _deterministic_frame_sample(frame, max_rows=max_rows, seed=seed)
    if len(sampled) < 3 or sampled.shape[1] == 0:
        return None
    if sampled.shape[1] > max_features:
        rng = np.random.default_rng(int(seed) + 1)
        positions = np.sort(rng.choice(sampled.shape[1], size=int(max_features), replace=False))
        sampled = sampled.iloc[:, positions]
    matrix = _mixed_local_distance_matrix(sampled)
    if matrix.shape[1] == 0:
        return None
    distances, indices = (
        NearestNeighbors(
            n_neighbors=min(4, len(sampled)),
            metric="euclidean",
            algorithm="brute",
        )
        .fit(matrix)
        .kneighbors(matrix, return_distance=True)
    )
    nearest_distances = []
    for row_index, (row_distances, row_indices) in enumerate(zip(distances, indices, strict=True)):
        non_self = row_distances[row_indices != row_index]
        if non_self.size == 0:
            return None
        nearest_distances.append(float(non_self[0]))
    nearest = np.asarray(nearest_distances, dtype=float)
    positive = nearest[nearest > 0]
    if positive.size == 0:
        return 0.0
    reference = float(np.median(positive))
    return clean_finite_number(float(np.percentile(nearest, 95)) / reference)


def _categorical_rare_value_share(categorical: pd.DataFrame) -> float | None:
    rare_cells = 0
    observed_cells = 0
    for column in categorical.columns:
        series = categorical[column]
        observed = series.dropna()
        if observed.empty:
            continue
        counts = observed.value_counts(dropna=True)
        rare_count_threshold = max(2, int(np.floor(0.01 * len(observed))))
        rare_values = set(counts[counts <= rare_count_threshold].index)
        rare_cells += int(observed.isin(rare_values).sum())
        observed_cells += len(observed)
    if observed_cells == 0:
        return None
    return clean_finite_number(float(rare_cells) / float(observed_cells))


def _categorical_cooccurrence_sparsity(categorical: pd.DataFrame) -> float | None:
    """Return the share of rows belonging to rare joint categorical patterns.

    At least two categorical columns are required so this does not simply
    duplicate the rare-value descriptor. Missing values are explicit pattern
    components. A pattern is rare when it occurs at most twice or in at most
    one percent of rows, whichever count is larger.
    """
    if len(categorical) == 0 or categorical.shape[1] < 2:
        return None
    patterns = categorical.astype("string").fillna("<missing>")
    hashes = pd.util.hash_pandas_object(patterns, index=False)
    counts = hashes.value_counts()
    rare_count_threshold = max(2, int(np.floor(0.01 * len(patterns))))
    rare_hashes = counts[counts <= rare_count_threshold].index
    return clean_finite_number(float(hashes.isin(rare_hashes).mean()))


def _missingness_pattern_entropy(frame: pd.DataFrame) -> float | None:
    if len(frame) == 0 or frame.shape[1] == 0:
        return None
    pattern_hashes = pd.util.hash_pandas_object(frame.isna(), index=False)
    probabilities = pattern_hashes.value_counts(normalize=True).to_numpy(dtype=float)
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    maximum_entropy = float(np.log(len(frame)))
    if maximum_entropy <= 0:
        return 0.0
    return clean_finite_number(entropy / maximum_entropy)


def compute_local_target_structure_features(
    X_train: pd.DataFrame,
    y_train: Any,
    task_family: object,
    *,
    max_rows: int = DEFAULT_MAX_LOCAL_ROWS,
    max_features: int = DEFAULT_MAX_DEPENDENCE_FEATURES,
    seed: int = DEFAULT_META_FEATURE_SEED,
) -> dict[str, float | None]:
    """Measure target variation between neighbouring training rows.

    Classification reports the fraction of nearest-neighbour pairs with
    different labels. Regression reports their median absolute target
    difference divided by the target IQR. Numeric columns are robust-scaled,
    categorical columns are capped one-hot encodings, and the two schema parts
    receive equal total distance weight when both are present.
    """
    result = {
        "localClassDisagreementRate": None,
        "localTargetVariationIqr": None,
    }
    task = normalized_task_family(task_family)
    if task not in CLASSIFICATION_TASK_FAMILIES | REGRESSION_TASK_FAMILIES:
        return result
    X, y = _valid_signal_rows(X_train, y_train, task)
    if len(X) < 3 or X.shape[1] == 0:
        return result
    if len(X) > max_rows:
        rng = np.random.default_rng(int(seed))
        positions = np.sort(rng.choice(len(X), size=int(max_rows), replace=False))
        X = X.iloc[positions].reset_index(drop=True)
        y = y[positions]
    if X.shape[1] > max_features:
        rng = np.random.default_rng(int(seed) + 1)
        positions = np.sort(rng.choice(X.shape[1], size=int(max_features), replace=False))
        X = X.iloc[:, positions]

    matrix = _mixed_local_distance_matrix(X)
    if matrix.shape[1] == 0:
        return result
    neighbour_count = min(4, len(X))
    fitted_neighbours = NearestNeighbors(
        n_neighbors=neighbour_count,
        metric="euclidean",
        algorithm="brute",
    ).fit(matrix)
    indices = fitted_neighbours.kneighbors(matrix, return_distance=False)
    nearest = np.empty(len(X), dtype=int)
    for row_index, candidates in enumerate(indices):
        non_self = candidates[candidates != row_index]
        if non_self.size == 0:
            return result
        nearest[row_index] = int(non_self[0])

    if task in CLASSIFICATION_TASK_FAMILIES:
        result["localClassDisagreementRate"] = clean_finite_number(float(np.mean(y != y[nearest])))
        return result

    target = np.asarray(y, dtype=float)
    q25, q75 = np.percentile(target, [25, 75])
    iqr = float(q75 - q25)
    if iqr > 0:
        variation = float(np.median(np.abs(target - target[nearest]))) / iqr
        result["localTargetVariationIqr"] = clean_finite_number(variation)
    return result


def _mixed_local_distance_matrix(frame: pd.DataFrame) -> sparse.csr_matrix:
    numeric_columns = [
        column
        for column in frame.columns
        if is_numeric_dtype(frame[column].dtype) and not is_bool_dtype(frame[column].dtype)
    ]
    categorical_columns = [column for column in frame.columns if column not in numeric_columns]
    parts: list[sparse.csr_matrix] = []
    if numeric_columns:
        numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        medians = numeric.median()
        scales = numeric.quantile(0.75) - numeric.quantile(0.25)
        scales = scales.where(scales > 0, 1.0)
        numeric = ((numeric.fillna(medians).fillna(0.0) - medians.fillna(0.0)) / scales).clip(-10, 10)
        numeric_values = numeric.to_numpy(dtype=np.float32) / np.sqrt(len(numeric_columns))
        parts.append(sparse.csr_matrix(numeric_values))
    if categorical_columns:
        categorical = frame[categorical_columns].astype("string").fillna("<missing>")
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            max_categories=DEFAULT_MAX_CATEGORICAL_LEVELS,
            sparse_output=True,
            dtype=np.float32,
        )
        encoded = encoder.fit_transform(categorical).tocsr()
        encoded /= np.sqrt(2.0 * len(categorical_columns))
        parts.append(encoded)
    if not parts:
        return sparse.csr_matrix((len(frame), 0), dtype=np.float32)
    if len(parts) == 1:
        return parts[0]
    return sparse.hstack(parts, format="csr")


def compute_univariate_signal_top5_share(
    X_train: pd.DataFrame,
    y_train: Any,
    task_family: object,
    *,
    max_rows: int = DEFAULT_MAX_MI_ROWS,
    seed: int = DEFAULT_META_FEATURE_SEED,
) -> float | None:
    """Return the share of positive univariate signal concentrated in the top five features.

    Example:
        >>> import pandas as pd
        >>> X = pd.DataFrame({"signal": [0, 0, 1, 1], "constant": [1, 1, 1, 1]})
        >>> compute_univariate_signal_top5_share(X, [0, 0, 1, 1], "binary")
        1.0
    """
    scores = compute_univariate_signal_scores(
        X_train,
        y_train,
        task_family,
        max_rows=max_rows,
        seed=seed,
    )
    return _univariate_signal_top5_share(scores)


def compute_univariate_signal_scores(
    X_train: pd.DataFrame,
    y_train: Any,
    task_family: object,
    *,
    max_rows: int = DEFAULT_MAX_MI_ROWS,
    seed: int = DEFAULT_META_FEATURE_SEED,
) -> np.ndarray | None:
    """Return non-negative univariate mutual-information scores."""
    if len(X_train) != len(y_train) or len(X_train) < 3 or X_train.shape[1] == 0:
        return None

    task = normalized_task_family(task_family)
    X, y = _valid_signal_rows(X_train, y_train, task)
    if len(X) < 3:
        return None
    if len(X) > max_rows:
        rng = np.random.default_rng(int(seed))
        positions = np.sort(rng.choice(len(X), size=int(max_rows), replace=False))
        X = X.iloc[positions].reset_index(drop=True)
        y = y[positions]

    X_numeric, discrete_features = _mixed_frame_to_numeric_matrix(X)
    if X_numeric.shape[1] == 0:
        return None
    variable_features = np.nanstd(X_numeric, axis=0) > 0
    if not bool(np.any(variable_features)):
        return None
    X_numeric = X_numeric[:, variable_features]
    discrete_features = discrete_features[variable_features]

    try:
        if task in REGRESSION_TASK_FAMILIES:
            scores = mutual_info_regression(
                X_numeric,
                y.astype(float, copy=False),
                discrete_features=discrete_features,
                random_state=int(seed),
            )
        elif task in CLASSIFICATION_TASK_FAMILIES:
            labels, _uniques = pd.factorize(pd.Series(y), sort=True)
            if np.unique(labels).size < 2:
                return None
            scores = mutual_info_classif(
                X_numeric,
                labels,
                discrete_features=discrete_features,
                random_state=int(seed),
            )
        else:
            return None
    except (TypeError, ValueError, FloatingPointError):
        return None

    return np.clip(
        np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        None,
    )


def _univariate_signal_top5_share(scores: np.ndarray | None) -> float | None:
    if scores is None:
        return None
    positive_scores = scores[scores > 0]
    total_signal = float(np.sum(positive_scores))
    if total_signal <= 0 or not np.isfinite(total_signal):
        return None
    top5_signal = float(np.sum(np.sort(positive_scores)[-5:]))
    return clean_finite_number(top5_signal / total_signal)


def resolve_local_dataset_path(
    repo_root: Path,
    dataset_id: str,
    catalog_record: dict[str, Any] | None = None,
) -> Path | None:
    """Resolve a dataset id or catalog slug to a local export directory.

    Example:
        >>> resolve_local_dataset_path(Path("/missing"), "jm1") is None
        True
    """
    available_paths = _local_dataset_export_dirs(str(repo_root))
    for candidate in _candidate_dataset_names(dataset_id, catalog_record):
        path = available_paths.get(candidate)
        if path is not None:
            return path
    return None


def load_local_dataset_base_meta_features(
    *,
    repo_root: Path,
    dataset_id: str,
    task_family: object,
    catalog_record: dict[str, Any] | None = None,
    split_tuple: tuple[int, int, int] = DEFAULT_SPLIT_TUPLE,
    include_univariate_signal: bool = True,
) -> dict[str, float | int | None]:
    """Load local train+validation rows and compute overview base meta-features.

    Missing or incompatible local exports return unavailable computed fields
    instead of failing study payload generation.

    Example:
        >>> features = load_local_dataset_base_meta_features(
        ...     repo_root=Path("/missing"),
        ...     dataset_id="jm1",
        ...     task_family="binary",
        ... )
        >>> features["baseTrainRows"] is None
        True
    """
    dataset_path = resolve_local_dataset_path(repo_root, dataset_id, catalog_record)
    if dataset_path is None:
        return empty_base_meta_features()
    try:
        from tfm_finetuning.finetuning.datasets.runner_data import dataset_part_to_frame
        from tfm_finetuning.finetuning.datasets.tabarena import build_dataset_from_local_path

        dataset, _metadata = build_dataset_from_local_path(
            dataset_path=dataset_path,
            split_tuple=split_tuple,
            num_policy=None,
            cat_policy=None,
            seed=DEFAULT_META_FEATURE_SEED,
        )
        X_train = pd.concat(
            [dataset_part_to_frame(dataset, "train"), dataset_part_to_frame(dataset, "val")],
            axis=0,
            ignore_index=True,
        )
        y_train = np.concatenate([np.asarray(dataset.task.labels["train"]), np.asarray(dataset.task.labels["val"])])
        return compute_base_meta_features(
            X_train,
            y_train,
            task_family or dataset.task.type_,
            include_univariate_signal=include_univariate_signal,
        )
    except (OSError, ValueError, TypeError, KeyError):
        return empty_base_meta_features()


def merge_base_meta_features(
    primary: dict[str, float | int | None],
    fallback: dict[str, float | int | None],
) -> dict[str, float | int | None]:
    """Fill unavailable primary base-meta values from a fallback mapping.

    Example:
        >>> merge_base_meta_features({"baseTrainRows": None}, {"baseTrainRows": 3})["baseTrainRows"]
        3
    """
    merged = empty_base_meta_features()
    for key in BASE_META_FEATURE_KEYS:
        value = primary.get(key)
        merged[key] = value if value is not None else fallback.get(key)
    return merged


def _candidate_dataset_names(dataset_id: str, catalog_record: dict[str, Any] | None) -> list[str]:
    raw_candidates: list[str] = [str(dataset_id or "").strip()]
    if catalog_record is not None:
        metadata = catalog_record.get("metadata") if isinstance(catalog_record, dict) else None
        if isinstance(metadata, dict):
            raw_candidates.extend(
                str(metadata.get(key) or "").strip()
                for key in ("dataset_id", "dataset_slug", "dataset_dir", "dataset_name", "name", "id")
            )
        raw_candidates.append(str(catalog_record.get("dataset_id") or "").strip())

    candidates: list[str] = []
    for value in raw_candidates:
        if not value:
            continue
        variants = {
            value,
            value.lower(),
            value.replace("-", "_"),
            value.replace("-", "_").lower(),
            value.replace("_", "-"),
            value.replace("_", "-").lower(),
        }
        for variant in variants:
            if variant and variant not in candidates:
                candidates.append(variant)
    return candidates


@lru_cache(maxsize=8)
def _local_dataset_export_dirs(repo_root: str) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for root_name in LOCAL_DATA_ROOTS:
        root = Path(repo_root) / "data" / root_name
        try:
            children = list(root.iterdir())
        except OSError:
            continue
        for child in children:
            if child.name in paths:
                continue
            try:
                if child.is_dir() and (child / "info.json").is_file():
                    paths[child.name] = child
            except OSError:
                continue
    return paths


def local_dataset_ids(repo_root: Path) -> tuple[str, ...]:
    """Return locally exported dataset identifiers in deterministic order.

    Example:
        >>> local_dataset_ids(Path("/missing"))
        ()
    """
    return tuple(sorted(_local_dataset_export_dirs(str(repo_root))))


def _valid_signal_rows(
    X_train: pd.DataFrame,
    y_train: Any,
    task_family: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    y_series = pd.Series(y_train)
    if task_family in REGRESSION_TASK_FAMILIES:
        y_numeric = pd.to_numeric(y_series, errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(y_numeric)
        return X_train.loc[valid].reset_index(drop=True), y_numeric[valid]
    valid = y_series.notna().to_numpy()
    return X_train.loc[valid].reset_index(drop=True), y_series.loc[valid].to_numpy()


def _mixed_frame_to_numeric_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    columns: list[np.ndarray] = []
    discrete_features: list[bool] = []
    for _name, series in frame.items():  # noqa: PERF102
        values, is_discrete = _series_to_numeric_values(series)
        columns.append(values)
        discrete_features.append(is_discrete)
    if not columns:
        return np.empty((len(frame), 0), dtype=float), np.empty(0, dtype=bool)
    return np.column_stack(columns).astype(float, copy=False), np.asarray(discrete_features, dtype=bool)


def _series_to_numeric_values(series: pd.Series) -> tuple[np.ndarray, bool]:
    if _is_discrete_series(series):
        encoded, _uniques = pd.factorize(series.astype("string").fillna("<missing>"), sort=True)
        return encoded.astype(float), True

    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(numeric)
    fill_value = float(np.median(numeric[finite])) if bool(np.any(finite)) else 0.0
    numeric[~finite] = fill_value
    return numeric, False


def _is_discrete_series(series: pd.Series) -> bool:
    if is_bool_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
        return True
    if not is_numeric_dtype(series):
        return True
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        return False
    unique_values = np.unique(finite)
    return unique_values.size <= 2 and set(unique_values.tolist()).issubset({0.0, 1.0})
