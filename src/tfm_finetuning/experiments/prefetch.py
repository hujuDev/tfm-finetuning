"""Export every dataset split an experiment needs, one after another, before its jobs are submitted.

Jobs export missing datasets themselves, guarded by a file lock. On a shared file system without cluster-wide
locks (BeeGFS by default) parallel array tasks can corrupt each other's export, so run this once first.

Example: ``python -m tfm_finetuning.experiments.prefetch overview_baseline --dataset jm1``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from tfm_finetuning.experiments.configuration import expand_runs
from tfm_finetuning.experiments.datasets import ensure_dataset
from tfm_finetuning.utils.config import read_yaml_mapping

REPO_ROOT = Path(__file__).resolve().parents[3]


def prefetch(
    experiment: str, *, repo_root: Path = REPO_ROOT, datasets: Sequence[str] | None = None
) -> list[tuple[str, str]]:
    """Export each distinct (dataset, split) of the experiment's runs and return them in order."""
    recipe = read_yaml_mapping(repo_root / "configs/experiments" / f"{experiment}.yaml")
    seen: dict[tuple[str, str], None] = {}
    for run in expand_runs(recipe, repo_root=repo_root, datasets=list(datasets) if datasets else None):
        config = run["config"]
        key = (run["dataset"], config.get("data", {}).get("split", "fold0_repeat0_sample0"))
        if key not in seen:
            ensure_dataset(run["dataset"], config, repo_root=repo_root)
            seen[key] = None
            print(f"Ready: {key[0]} {key[1]}", flush=True)
    return list(seen)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    parser.add_argument("--dataset", action="append", help="Dataset name; repeat to select multiple")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    prefetch(args.experiment, repo_root=args.repo_root, datasets=args.dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
