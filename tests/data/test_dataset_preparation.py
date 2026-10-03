"""Automatic export retries and feature/split completeness, without network access."""

import json
from pathlib import Path

import pytest

from experiments.datasets import ensure_dataset


def complete_export(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "Y.npy").write_bytes(b"labels")
    (path / "X_num.npy").write_bytes(b"features")
    split = "fold0_repeat0_sample0"
    (path / "info.json").write_text(json.dumps({"splits": {split: {}}}))
    directory = path / f"split-{split}"
    directory.mkdir(exist_ok=True)
    for part in ("train", "val", "test"):
        (directory / f"{part}_idx.npy").write_bytes(b"indices")


@pytest.mark.parametrize("partial", ["metadata", "features", "split"])
def test_partial_export_is_repaired_instead_of_skipped(tmp_path, monkeypatch, partial):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/datasets.yaml").write_text("demo:\n  task_id: 123\n")
    output = tmp_path / "data/openml/demo"
    complete_export(output)
    if partial == "metadata":
        (output / "info.json").write_text("{")
    elif partial == "features":
        (output / "X_num.npy").unlink()
    else:
        (output / "split-fold0_repeat0_sample0/test_idx.npy").unlink()
    commands = []

    def export(command, **kwargs):
        commands.append(command)
        assert "--overwrite" in command
        assert command[command.index("--task-id") + 1] == "123"
        complete_export(output)

    monkeypatch.setattr("experiments.datasets.subprocess.run", export)
    assert ensure_dataset("demo", {}, repo_root=tmp_path) == output
    assert len(commands) == 1
    assert ensure_dataset("demo", {}, repo_root=tmp_path) == output
    assert len(commands) == 1
