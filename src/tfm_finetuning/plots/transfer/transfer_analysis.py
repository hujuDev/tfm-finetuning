"""Reproduce the final schedule, target-budget and source-checkpoint comparisons.

Example: ``python -m tfm_finetuning.plots.transfer.transfer_analysis``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tomllib
from pathlib import Path
from typing import Any

import pandas as pd

from tfm_finetuning.plots.transfer.balanced_transfer_data import (
    build_balanced_transfer_tables,
    write_balanced_transfer_tables,
)
from tfm_finetuning.plots.transfer.balanced_transfer_figures import render_balanced_transfer_figures
from tfm_finetuning.plots.transfer.followup_data import build_budget_contrasts
from tfm_finetuning.plots.transfer.source_transfer_figure import plot_transfer_pairs
from tfm_finetuning.plots.transfer.transfer_data import build_transfer_tables, write_transfer_tables
from tfm_finetuning.plots.transfer.transfer_figures import render_transfer_figures

DEFAULT_CONFIG = Path("configs/plots/section3_transfer_analysis.toml")


def generate(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Build only evidence and figures included in the submitted findings."""
    config = tomllib.loads(config_path.read_text())
    output = Path(config["output"]["analysis_dir"])
    assets = Path(config["output"]["thesis_asset_dir"])
    schedules = build_transfer_tables(Path(config["input"]["results_root"]))
    budgets = build_budget_contrasts(config["input"], schedules)
    balanced = build_balanced_transfer_tables(config["input"], budgets)
    tables = (*write_transfer_tables(schedules, output), *write_balanced_transfer_tables(balanced, output))
    figures = [
        *render_transfer_figures(schedules, output, assets),
        *render_balanced_transfer_figures(balanced, output, assets),
    ]
    # The submitted source-only figure uses the approved paired gain layout.
    source = plot_transfer_pairs(
        pd.read_csv(output / "balanced_source_only_results.csv"), output / "source_only_checkpoint_transfer.pdf"
    )
    shutil.copy2(source, assets / source.name)
    summary = {"tables": [str(path) for path in tables], "figures": [str(path) for path in figures]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    """CLI entrypoint for final transfer figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(generate(args.config), indent=2))


if __name__ == "__main__":
    main()
