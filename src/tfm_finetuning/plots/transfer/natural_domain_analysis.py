"""Generate the Section 3 natural-domain transfer findings figure.

Example:
    ``python -m tfm_finetuning.plots.transfer.natural_domain_analysis``
"""

from __future__ import annotations

import argparse
import json
import shutil
import tomllib
from pathlib import Path
from typing import Any

from tfm_finetuning.plots.transfer.natural_domain_data import (
    TARGET_100_ROUTE,
    NaturalDomainRow,
    load_natural_domain_rows,
    summarize_mixed_contrasts,
    summarize_natural_domain_rows,
    write_natural_domain_tables,
)

DEFAULT_CONFIG = Path("configs/plots/transfer/natural_domain_analysis.toml")
FIGURE_NAME = "natural_domain_route_matched_transfer.pdf"


def generate(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Generate tables, one publication figure, and a machine-readable summary."""
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    results_root = Path(config["input"]["results_root"])
    output_dir = Path(config["output"]["analysis_dir"])
    thesis_asset_dir = Path(config["output"]["thesis_asset_dir"])
    rows = load_natural_domain_rows(results_root)
    fixed_target_rows = load_natural_domain_rows(results_root, fixed_target_context=True)
    table_paths = write_natural_domain_tables(rows, output_dir)
    fixed_table_paths = write_natural_domain_tables(
        fixed_target_rows,
        output_dir / "fixed_target_context",
    )
    table_paths = (*table_paths, *fixed_table_paths)
    figure_path = render_natural_domain_figure(
        rows,
        fixed_target_rows,
        output_dir / FIGURE_NAME,
    )
    thesis_asset_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(figure_path, thesis_asset_dir / FIGURE_NAME)
    shutil.copy2(config_path, output_dir / "analysis_config.toml")
    summary = {
        "coverage_complete": True,
        "target_rows": len(rows),
        "fixed_target_context_rows": len(fixed_target_rows),
        "scientific_seed_results": sum(row.seed_count for row in rows)
        + sum(row.seed_count for row in fixed_target_rows if row.route != TARGET_100_ROUTE),
        "panel_summary": list(summarize_natural_domain_rows(rows)),
        "fixed_target_context_panel_summary": list(summarize_natural_domain_rows(fixed_target_rows)),
        "mixed_vs_target_control": list(summarize_mixed_contrasts(rows)),
        "tables": [str(path) for path in table_paths],
        "figure": str(figure_path),
        "thesis_figure": str(thesis_asset_dir / FIGURE_NAME),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def render_natural_domain_figure(
    rows: tuple[NaturalDomainRow, ...],
    fixed_target_rows: tuple[NaturalDomainRow, ...],
    output: Path,
) -> Path:
    """Render the shared-reference, five-setting normalized-score comparison."""
    from tfm_finetuning.plots.transfer.natural_domain_grouped_figure import render_grouped_natural_domains

    return render_grouped_natural_domains(rows, fixed_target_rows, output)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(generate(args.config), indent=2))


if __name__ == "__main__":
    main()
