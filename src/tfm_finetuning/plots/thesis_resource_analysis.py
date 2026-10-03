"""Summarize physical run wall time from the current reproduction.

Example: ``python -m tfm_finetuning.plots.thesis_resource_analysis``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

from tfm_finetuning.plots.final_tables import resource_table
from tfm_finetuning.plots.results import REPO_ROOT, read_runs, resolve_results_root

DEFAULT_CONFIG = Path("configs/plots/thesis_resource_analysis.toml")


def generate(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Count each completed run once, including supporting training and evaluation."""
    config = tomllib.loads(config_path.read_text())
    output = Path(config["output"]["analysis_dir"])
    assets = Path(config["output"]["thesis_asset_dir"])
    output.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    results_root = resolve_results_root(Path(config["input"]["results_root"]))
    # Only shipped thesis recipes contribute; notebook and custom run directories do not.
    studies = sorted(
        path.stem for path in (REPO_ROOT / "configs/experiments").glob("*.yaml") if (results_root / path.stem).is_dir()
    )
    if not studies:
        raise ValueError("No results for retained thesis experiment recipes")
    groups = defaultdict(lambda: defaultdict(float))
    for run in read_runs(results_root, studies):
        if run.study == "overview_hpo_representative":
            group = "HPO"
        elif run.study.startswith("distribution_shift_"):
            group = "Controlled transformations"
        elif run.study.startswith("dataset_transfer"):
            group = "Cross-dataset transfer"
        else:
            group = "Default in-distribution"
        seconds = float(run.report["runtime"]["total_seconds"])
        if seconds < 0:
            raise ValueError(f"Negative runtime: {run.path}")
        groups[group][run.model] += seconds / 3600
    if not groups:
        raise ValueError("No completed runs with measured runtime")
    table = output / "resource_table.tex"
    table.write_text(resource_table(groups))
    shutil.copy2(table, assets / table.name)
    summary = {"studies": dict(groups), "table": str(table)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    """Render the resource appendix for this run collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    print(json.dumps(generate(parser.parse_args().config), indent=2))


if __name__ == "__main__":
    main()
