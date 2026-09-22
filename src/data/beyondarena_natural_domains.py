"""Materialize the frozen BeyondArena natural-domain experiment datasets.

The output uses the same indexed local-array contract as the existing OpenML
exports, while preserving BeyondArena's exact grouped Covertype folds.

Example:
    >>> callable(materialize_natural_domain_datasets)
    True
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data.beyondarena_natural_domain_catalog import (
    BEYONDARENA_REPO_ID,
    BEYONDARENA_REVISION,
    CONTAINERS,
    GROUPED_COVERTYPE_DATASET,
    NATURAL_DOMAIN_PANELS,
    WIDS_HOSPITAL_IDS,
    WILDERNESS_DOMAIN_LABELS,
    BeyondArenaContainer,
)
from data.local_indexed_export import OuterSplit, SplitKey, sha256_file, write_indexed_local_export

_TASK_METADATA = "task_metadata.predictive-ml-task-mold-v1.json"
_EXPERIMENT_METADATA = "experiment_metadata.predictive-ml-splits-mold-v1.json"
_DATASET_FILE = "dataset.parquet"
_DTYPES_FILE = "dtypes.json"


@dataclass(frozen=True)
class LoadedContainer:
    """Loaded official container bytes needed for logical-domain export."""

    spec: BeyondArenaContainer
    path: Path
    frame: pd.DataFrame
    dtypes: Mapping[str, str]
    task: Mapping[str, Any]
    experiment: Mapping[str, Any]


def download_selected_snapshot(*, local_dir: Path) -> Path:
    """Download only the five pinned containers used by the experiment."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=BEYONDARENA_REPO_ID,
        repo_type="dataset",
        revision=BEYONDARENA_REVISION,
        local_dir=Path(local_dir),
        allow_patterns=[f"{spec.relative_path}/*" for spec in CONTAINERS.values()],
        ignore_patterns=["**/tabarena_text_cache_*"],
    )
    return Path(path)


def materialize_natural_domain_datasets(
    *,
    snapshot_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Write all target-aware domains plus official grouped Covertype."""
    containers = {
        name: load_container(snapshot_root=Path(snapshot_root), spec=spec) for name, spec in CONTAINERS.items()
    }
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    records.extend(_materialize_heart(containers, output_root=output_root))
    records.extend(_materialize_wids(containers["wids_diabetes_mellitus"], output_root=output_root))
    records.extend(_materialize_covertype(containers["covertype"], output_root=output_root))
    manifest = {
        "schema_version": 1,
        "protocol_version": "beyondarena_natural_domains_v1",
        "source": {
            "repo_id": BEYONDARENA_REPO_ID,
            "revision": BEYONDARENA_REVISION,
        },
        "split_protocol": {
            "heart": "official_beyondarena_iid_splits",
            "wids_hospital": "custom_stratified_80_20_seed0_with_identifiers_dropped",
            "covertype_area": "custom_stratified_80_20_seed0_with_area_dropped",
            "covertype_grouped_ood": "official_leave_one_wilderness_area_out",
            "inner_validation_fraction": 0.1,
            "inner_validation_seed": 0,
        },
        "panels": [
            {
                "key": panel.key,
                "domain_kind": panel.domain_kind,
                "datasets": list(panel.datasets),
            }
            for panel in NATURAL_DOMAIN_PANELS
        ],
        "datasets": sorted(records, key=lambda record: str(record["name"])),
    }
    manifest_path = output_root / "natural_domain_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_container(*, snapshot_root: Path, spec: BeyondArenaContainer) -> LoadedContainer:
    """Load one pinned container and its Data Foundry metadata."""
    path = Path(snapshot_root) / spec.relative_path
    required = (_DATASET_FILE, _DTYPES_FILE, _TASK_METADATA, _EXPERIMENT_METADATA)
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"BeyondArena container {spec.relative_path!r} is missing {missing}.")
    return LoadedContainer(
        spec=spec,
        path=path,
        frame=pd.read_parquet(path / _DATASET_FILE),
        dtypes=_read_mapping(path / _DTYPES_FILE),
        task=_read_mapping(path / _TASK_METADATA),
        experiment=_read_mapping(path / _EXPERIMENT_METADATA),
    )


def _materialize_heart(
    containers: Mapping[str, LoadedContainer],
    *,
    output_root: Path,
) -> list[dict[str, Any]]:
    names = NATURAL_DOMAIN_PANELS[0].datasets
    loaded = [containers[name] for name in names]
    _require_shared_container_schema(loaded)
    label_mapping = _shared_label_mapping(loaded)
    records = []
    for container in loaded:
        records.append(
            _write_logical_dataset(
                container=container,
                output_root=output_root,
                name=container.spec.dataset,
                frame=container.frame,
                dropped_columns=(),
                outer_splits=_official_outer_splits(container.experiment),
                label_mapping=label_mapping,
                domain={"kind": "separate_cohort", "value": container.spec.dataset},
            )
        )
    return records


def _materialize_wids(container: LoadedContainer, *, output_root: Path) -> list[dict[str, Any]]:
    frame = container.frame
    target = _target_name(container)
    hospital_counts = frame.groupby("hospital_id", dropna=False).size()
    selected = tuple(int(value) for value in hospital_counts.sort_values(ascending=False).index[:3])
    if selected != WIDS_HOSPITAL_IDS:
        raise ValueError(f"Pinned WiDS hospital selection changed: expected {WIDS_HOSPITAL_IDS}, observed {selected}.")
    label_mapping = _label_mapping(frame[target])
    records = []
    for hospital_id in WIDS_HOSPITAL_IDS:
        domain = frame.loc[frame["hospital_id"] == hospital_id].reset_index(drop=True)
        _require_class_support(domain[target], minimum=100, domain=f"hospital {hospital_id}")
        records.append(
            _write_logical_dataset(
                container=container,
                output_root=output_root,
                name=f"wids_diabetes_hospital_{hospital_id}",
                frame=domain,
                dropped_columns=("hospital_id", "icu_id"),
                outer_splits=_custom_outer_split(domain[target], seed=0),
                label_mapping=label_mapping,
                domain={
                    "kind": "custom_hospital",
                    "column": "hospital_id",
                    "value": hospital_id,
                    "selection": "three_largest_hospitals_each_class_at_least_100",
                },
            )
        )
    return records


def _materialize_covertype(container: LoadedContainer, *, output_root: Path) -> list[dict[str, Any]]:
    frame = container.frame
    target = _target_name(container)
    label_mapping = _label_mapping(frame[target])
    records = []
    for name, area in WILDERNESS_DOMAIN_LABELS.items():
        domain = frame.loc[frame["Wilderness_Area"] == area].reset_index(drop=True)
        records.append(
            _write_logical_dataset(
                container=container,
                output_root=output_root,
                name=name,
                frame=domain,
                dropped_columns=("Wilderness_Area",),
                outer_splits=_custom_outer_split(domain[target], seed=0),
                label_mapping=label_mapping,
                domain={"kind": "wilderness_area", "column": "Wilderness_Area", "value": area},
            )
        )
    records.append(
        _write_logical_dataset(
            container=container,
            output_root=output_root,
            name=GROUPED_COVERTYPE_DATASET,
            frame=frame,
            dropped_columns=(),
            outer_splits=_official_outer_splits(container.experiment),
            label_mapping=label_mapping,
            domain={"kind": "official_grouped_ood", "column": "Wilderness_Area"},
        )
    )
    return records


def _write_logical_dataset(
    *,
    container: LoadedContainer,
    output_root: Path,
    name: str,
    frame: pd.DataFrame,
    dropped_columns: tuple[str, ...],
    outer_splits: Mapping[SplitKey, OuterSplit],
    label_mapping: Mapping[str, int],
    domain: Mapping[str, Any],
) -> dict[str, Any]:
    target = _target_name(container)
    feature_columns = [column for column in frame.columns if column != target and column not in dropped_columns]
    categorical = [
        column for column in feature_columns if str(container.dtypes.get(column, "")).lower() in {"category", "string"}
    ]
    labels = _encode_labels(frame[target], label_mapping=label_mapping)
    source_file = container.path / _DATASET_FILE
    info = write_indexed_local_export(
        output_dir=output_root / name,
        name=name,
        features=frame.loc[:, feature_columns],
        labels=labels,
        categorical_columns=categorical,
        outer_splits=outer_splits,
        task_type=_task_type(container),
        tabarena_metric=str(container.task["objective_metric_name"]),
        label_mapping=label_mapping,
        provenance={
            "repo_id": BEYONDARENA_REPO_ID,
            "revision": BEYONDARENA_REVISION,
            "container": container.spec.relative_path,
            "dataset_sha256": sha256_file(source_file),
            "domain": dict(domain),
            "dropped_columns": list(dropped_columns),
        },
    )
    return {
        "name": name,
        "rows": len(frame),
        "features": len(feature_columns),
        "task_type": info["task_type"],
        "split_count": len(outer_splits),
        "domain": dict(domain),
    }


def _official_outer_splits(experiment: Mapping[str, Any]) -> dict[SplitKey, OuterSplit]:
    raw = experiment.get("splits")
    if not isinstance(raw, Mapping):
        raise ValueError("BeyondArena experiment metadata has no split mapping.")
    splits: dict[SplitKey, OuterSplit] = {}
    for repeat, folds in raw.items():
        if not isinstance(folds, Mapping):
            raise ValueError("BeyondArena split repeats must contain fold mappings.")
        for fold, pair in folds.items():
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("BeyondArena folds must contain [train_indices, test_indices].")
            splits[(int(fold), int(repeat), 0)] = (
                np.asarray(pair[0], dtype=np.int64),
                np.asarray(pair[1], dtype=np.int64),
            )
    return splits


def _custom_outer_split(labels: pd.Series, *, seed: int) -> dict[SplitKey, OuterSplit]:
    indices = np.arange(len(labels), dtype=np.int64)
    train, test = train_test_split(
        indices,
        test_size=0.2,
        random_state=int(seed),
        stratify=_encode_labels(labels, label_mapping=_label_mapping(labels)),
    )
    return {(0, 0, 0): (np.sort(train), np.sort(test))}


def _shared_label_mapping(containers: list[LoadedContainer]) -> dict[str, int]:
    values = pd.concat([container.frame[_target_name(container)] for container in containers], ignore_index=True)
    return _label_mapping(values)


def _require_shared_container_schema(containers: list[LoadedContainer]) -> None:
    schemas = [
        tuple(
            (
                column,
                str(container.dtypes.get(column) or "").lower() in {"category", "string"},
            )
            for column in container.frame.columns
            if column != _target_name(container)
        )
        for container in containers
    ]
    if any(schema != schemas[0] for schema in schemas[1:]):
        raise ValueError("Heart cohorts no longer expose one shared ordered feature schema and semantic type roles.")


def _label_mapping(labels: pd.Series) -> dict[str, int]:
    values = sorted({_label_token(value) for value in labels.dropna().tolist()})
    if len(values) < 2:
        raise ValueError("Classification datasets require at least two labels.")
    return {value: index for index, value in enumerate(values)}


def _encode_labels(labels: pd.Series, *, label_mapping: Mapping[str, int]) -> np.ndarray:
    if labels.isna().any():
        raise ValueError("Labels contain missing values.")
    tokens = labels.map(_label_token)
    if not set(tokens).issubset(label_mapping):
        raise ValueError("Labels contain unknown values.")
    return tokens.map(label_mapping).to_numpy(dtype=np.int64)


def _label_token(value: Any) -> str:
    if pd.isna(value):
        return "<missing>"
    return str(value)


def _target_name(container: LoadedContainer) -> str:
    target = str(container.task.get("target_column_name") or "").strip()
    if not target or target not in container.frame:
        raise ValueError(f"Container {container.spec.dataset!r} has no valid target column.")
    return target


def _task_type(container: LoadedContainer) -> str:
    raw = str(container.task.get("problem_type") or "").strip().lower()
    mapping = {
        "binary_classification": "binary",
        "multiclass_classification": "multiclass",
    }
    try:
        return mapping[raw]
    except KeyError as exc:
        raise ValueError(f"Unsupported BeyondArena problem type: {raw!r}.") from exc


def _require_class_support(labels: pd.Series, *, minimum: int, domain: str) -> None:
    counts = labels.value_counts(dropna=False)
    if len(counts) < 2 or int(counts.min()) < minimum:
        raise ValueError(f"WiDS {domain} does not satisfy the frozen per-class support threshold.")


def _read_mapping(path: Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a mapping in {path}.")
    return payload


__all__ = [
    "LoadedContainer",
    "download_selected_snapshot",
    "load_container",
    "materialize_natural_domain_datasets",
]
