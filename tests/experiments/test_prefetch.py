"""Prefetch exports each distinct dataset split once, in a fixed order."""

from __future__ import annotations

from pathlib import Path

from tfm_finetuning.experiments import prefetch as prefetch_module

ROOT = Path(__file__).resolve().parents[2]


def test_prefetch_exports_each_dataset_split_once(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        prefetch_module,
        "ensure_dataset",
        lambda dataset, config, repo_root: calls.append((dataset, config["data"]["split"])),
    )

    ready = prefetch_module.prefetch("overview_baseline", repo_root=ROOT, datasets=["credit-g", "jm1"])

    assert calls == ready
    assert len(set(ready)) == len(ready)
    assert {dataset for dataset, _ in ready} == {"credit-g", "jm1"}
    # credit-g has a 3 fold x 3 seed grid of splits; jm1 one fixed split.
    assert sum(dataset == "credit-g" for dataset, _ in ready) == 3
    assert sum(dataset == "jm1" for dataset, _ in ready) == 1
