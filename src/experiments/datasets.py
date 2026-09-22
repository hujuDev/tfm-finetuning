"""Prepare missing datasets automatically, with a shared-filesystem download lock.

Example: ``ensure_dataset('churn', config, repo_root=root)`` returns its local export.
"""

from __future__ import annotations

import fcntl
import json
import re
import subprocess
import sys
from pathlib import Path

from experiments.configuration import dataset_key
from utils.config_io import read_yaml_mapping


def _ready(path: Path, split: str) -> bool:
    """Observe complete exports without racing another worker's metadata write."""
    if not (path / "info.json").is_file() or not (path / "Y.npy").is_file():
        return False
    if not any((path / name).is_file() for name in ("X_num.npy", "X_cat.npy", "X_bin.npy")):
        return False
    try:
        info = json.loads((path / "info.json").read_text())
    except (OSError, ValueError):
        return False
    return split in info.get("splits", {}) and all(
        (path / f"split-{split}" / f"{part}_idx.npy").is_file() for part in ("train", "val", "test")
    )


def ensure_dataset(dataset: str, config: dict, *, repo_root: Path) -> Path:
    """Reuse complete exported splits or fetch the pinned public source before training."""
    key = dataset_key(dataset)
    split = config.get("data", {}).get("split", "fold0_repeat0_sample0")
    match = re.fullmatch(r"fold(\d+)_repeat(\d+)_sample(\d+)", split)
    if not match:
        raise ValueError(f"Invalid data.split: {split}")
    candidates = [repo_root / "data" / family / key for family in ("tabarena", "openml", "beyondarena")]
    declared = str(config.get("data", {}).get("path", ""))
    if declared:
        path = Path(declared.removeprefix(":"))
        path = path if path.is_absolute() else repo_root / path
        if dataset_key(path.name) == key:
            candidates.insert(0, path)
    for path in candidates:
        if _ready(path, split):
            return path
    from data.beyondarena_natural_domain_catalog import GROUPED_COVERTYPE_DATASET, NATURAL_DOMAIN_PANELS

    derived = {d for panel in NATURAL_DOMAIN_PANELS for d in panel.datasets} | {GROUPED_COVERTYPE_DATASET}
    locks = repo_root / "data" / ".locks"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / ("beyondarena.lock" if key in derived else key + ".lock")).open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for path in candidates:
            if _ready(path, split):
                return path
        if key in derived:
            from data.beyondarena_natural_domains import (
                download_selected_snapshot,
                materialize_natural_domain_datasets,
            )

            snapshot = download_selected_snapshot(local_dir=repo_root / "data/cache/beyondarena")
            materialize_natural_domain_datasets(snapshot_root=snapshot, output_root=repo_root / "data/beyondarena")
            output = repo_root / "data/beyondarena" / key
        else:
            registry = {dataset_key(k): v for k, v in read_yaml_mapping(repo_root / "configs/datasets.yaml").items()}
            task = registry.get(key, {}).get("task_id")
            output = repo_root / "data" / ("openml" if task else "tabarena") / key
            # The exporter uses official OpenML folds, and creates validation only from training rows.
            command = [sys.executable, "-m", "data.download_tabarena_dataset_single", "--output-dir", str(output)]
            command += ["--source", "openml", "--task-id", str(task)] if task else ["--dataset-name", dataset]
            if output.exists():
                command.append("--overwrite")  # Repair a partial export; the exporter otherwise skips existing files.
            for flag, value in zip(("--fold", "--repeat", "--sample"), match.groups(), strict=True):
                command += [flag, value]
            command += [
                "--val-fraction",
                str(config.get("foundation_model", {}).get("validation_split_ratio", 0.1)),
                "--seed",
                str(config.get("foundation_model", {}).get("validation_split_seed", 0)),
            ]
            subprocess.run(command, cwd=repo_root, check=True)
        if not _ready(output, split):
            raise RuntimeError(f"Dataset preparation did not produce complete split {split}: {output}")
        return output
