"""Dataset naming and TabArena metric helpers shared by export and training."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def parse_subset_spec(subset: str | None) -> tuple[str | None, str | None]:
    """Parse subset spec into a TabArena base subset and an optional problem filter."""
    if subset is None:
        return None, None
    subset = str(subset).strip()
    if subset == "" or subset.lower() in {"all", "any", "*"}:
        return None, None
    tokens = [t for t in re.split(r"[^A-Za-z0-9]+", subset.lower()) if t]
    if not tokens:
        return None, None
    base_map = {
        "tabpfnv2": "TabPFNv2",
        "tabpfn": "TabPFNv2",
        "tabicl": "TabICL",
    }
    base_subset: str | None = None
    problem_filter: str | None = None
    for token in tokens:
        if token in {"compatible", "tabarena", "subset"}:
            continue
        if token in base_map:
            candidate = base_map[token]
            if base_subset is None:
                base_subset = candidate
            elif base_subset != candidate:
                raise SystemExit(f"Conflicting subset filters: {base_subset} vs {candidate}")
            continue
        if token in {"binary", "bin"}:
            if problem_filter and problem_filter != "binary":
                raise SystemExit(f"Conflicting problem filters: {problem_filter} vs binary")
            problem_filter = "binary"
            continue
        if token in {"multiclass", "multi"}:
            if problem_filter and problem_filter != "multiclass":
                raise SystemExit(f"Conflicting problem filters: {problem_filter} vs multiclass")
            problem_filter = "multiclass"
            continue
        if token in {"regression", "regress"}:
            if problem_filter and problem_filter != "regression":
                raise SystemExit(f"Conflicting problem filters: {problem_filter} vs regression")
            problem_filter = "regression"
            continue
        if token in {"classification", "class", "cls"}:
            if problem_filter is None:
                problem_filter = "classification"
            continue
    return base_subset, problem_filter


def find_dataset_matches(metadata: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Match a dataset by official TabArena name or slugified export directory name."""
    dataset_name = str(dataset).strip()
    if not dataset_name:
        return metadata.iloc[0:0]
    names = metadata["name"].astype(str)
    matches = metadata[names.str.lower() == dataset_name.lower()]
    if not matches.empty:
        return matches
    dataset_slug = slugify_dataset_name(dataset_name)
    slug_lookup = names.map(slugify_dataset_name)
    return metadata[slug_lookup == dataset_slug]


def slugify_dataset_name(name: str) -> str:
    """Normalize public dataset labels into export directory names."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", name.strip())
    cleaned = cleaned.strip("_")
    return cleaned.lower() if cleaned else "dataset"


def map_tabarena_metric_to_score(metric: str | None) -> str | None:
    if not metric:
        return None
    metric = str(metric).strip().lower()
    mapping = {
        "roc_auc": "roc-auc",
        "roc-auc": "roc-auc",
        "auc": "roc-auc",
        "log_loss": "cross-entropy",
        "cross_entropy": "cross-entropy",
        "cross-entropy": "cross-entropy",
        "rmse": "rmse",
        "root_mean_squared_error": "rmse",
        "mae": "mae",
        "r2": "r2",
        "accuracy": "accuracy",
    }
    return mapping.get(metric)
