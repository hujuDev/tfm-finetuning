from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.beyondarena_natural_domain_catalog import CONTAINERS, WIDS_HOSPITAL_IDS
from data.beyondarena_natural_domains import materialize_natural_domain_datasets
from finetuning.data import load_split_evaluation_data


def test_materializer_writes_ten_leakage_safe_local_exports(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    _write_heart(snapshot)
    _write_wids(snapshot)
    _write_covertype(snapshot)
    output = tmp_path / "data" / "beyondarena"

    manifest = materialize_natural_domain_datasets(snapshot_root=snapshot, output_root=output)

    assert len(manifest["datasets"]) == 10
    assert manifest["source"]["revision"]
    wids_info = json.loads((output / "wids_diabetes_hospital_118" / "info.json").read_text())
    assert wids_info["n_num_features"] + wids_info["n_cat_features"] == 2
    assert wids_info["provenance"]["dropped_columns"] == ["hospital_id", "icu_id"]
    area_info = json.loads((output / "covertype_comanche_peak" / "info.json").read_text())
    area_features = {feature for feature_group in area_info["feature_names"].values() for feature in feature_group}
    assert "Wilderness_Area" not in area_features

    config = {
        "data": {
            "path": str(output / "heart_disease_cleveland"),
            "split": "fold0_repeat0_sample0",
        },
        "foundation_model": {
            "validation_split_ratio": 0.1,
            "validation_split_seed": 0,
            "test_context_includes_validation_data": True,
        },
    }
    data = load_split_evaluation_data(config=config, repo_root=tmp_path)
    assert data.train_X.shape[1] == 2
    assert str(data.train_X["cat_0"].dtype) == "category"
    assert len(data.test_X) == 10


def _write_heart(root: Path) -> None:
    for index, name in enumerate(
        (
            "heart_disease_cleveland",
            "heart_disease_hungary",
            "heart_disease_va_long_beach",
        )
    ):
        frame = pd.DataFrame(
            {
                "age": np.arange(30, dtype=np.float64 if index == 0 else np.int64),
                "sex": pd.Series(["F", "M"] * 15, dtype="category"),
                "heart_disease_diagnosis": pd.Series([0, 1] * 15, dtype="category"),
            }
        )
        _write_container(
            root,
            name,
            frame,
            target="heart_disease_diagnosis",
            dtypes={
                "age": "float64" if index == 0 else "int64",
                "sex": "category",
                "heart_disease_diagnosis": "category",
            },
            splits={"0": {"0": [list(range(20)), list(range(20, 30))]}},
        )


def _write_wids(root: Path) -> None:
    sizes = {118: 240, 19: 220, 188: 200, 4: 180}
    rows = []
    for hospital, size in sizes.items():
        for index in range(size):
            rows.append(
                {
                    "hospital_id": hospital,
                    "icu_id": hospital * 10 + index % 2,
                    "age": 30.0 + index % 50,
                    "gender": "F" if index % 2 else "M",
                    "diabetes_mellitus": index % 2,
                }
            )
    frame = pd.DataFrame(rows)
    frame[["hospital_id", "icu_id", "gender", "diabetes_mellitus"]] = frame[
        ["hospital_id", "icu_id", "gender", "diabetes_mellitus"]
    ].astype("category")
    _write_container(
        root,
        "wids_diabetes_mellitus",
        frame,
        target="diabetes_mellitus",
        dtypes={
            "hospital_id": "category",
            "icu_id": "category",
            "age": "float64",
            "gender": "category",
            "diabetes_mellitus": "category",
        },
        splits={"0": {"0": [list(range(560)), list(range(560, len(frame)))]}},
    )


def _write_covertype(root: Path) -> None:
    areas = (
        "Comanche Peak Wilderness Area",
        "Neota Wilderness Area",
        "Rawah Wilderness Area",
    )
    frame = pd.DataFrame(
        [
            {
                "Elevation": 2000 + index,
                "Wilderness_Area": area,
                "Soil_Type": f"soil-{index % 4}",
                "Cover_Type": index % 3,
            }
            for area in areas
            for index in range(60)
        ]
    )
    frame[["Wilderness_Area", "Soil_Type", "Cover_Type"]] = frame[
        ["Wilderness_Area", "Soil_Type", "Cover_Type"]
    ].astype("category")
    folds = {}
    for fold in range(3):
        test = list(range(fold * 60, (fold + 1) * 60))
        train = sorted(set(range(180)).difference(test))
        folds[str(fold)] = [train, test]
    _write_container(
        root,
        "covertype",
        frame,
        target="Cover_Type",
        dtypes={
            "Elevation": "int64",
            "Wilderness_Area": "category",
            "Soil_Type": "category",
            "Cover_Type": "category",
        },
        splits={"0": folds},
        problem_type="multiclass_classification",
        metric="log_loss",
    )


def _write_container(
    root: Path,
    name: str,
    frame: pd.DataFrame,
    *,
    target: str,
    dtypes: dict[str, str],
    splits: dict[str, object],
    problem_type: str = "binary_classification",
    metric: str = "roc_auc",
) -> None:
    path = root / CONTAINERS[name].relative_path
    path.mkdir(parents=True)
    frame.to_parquet(path / "dataset.parquet", index=False)
    (path / "dtypes.json").write_text(json.dumps(dtypes), encoding="utf-8")
    (path / "task_metadata.predictive-ml-task-mold-v1.json").write_text(
        json.dumps(
            {
                "target_column_name": target,
                "problem_type": problem_type,
                "objective_metric_name": metric,
            }
        ),
        encoding="utf-8",
    )
    (path / "experiment_metadata.predictive-ml-splits-mold-v1.json").write_text(
        json.dumps({"splits": splits}), encoding="utf-8"
    )


def test_wids_selection_is_frozen_before_transfer_results() -> None:
    assert WIDS_HOSPITAL_IDS == (118, 19, 188)
