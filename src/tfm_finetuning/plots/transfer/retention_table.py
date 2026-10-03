"""Render the final retained-capability appendix table.

Example: ``python -m tfm_finetuning.plots.transfer.retention_table --config configs/plots/transfer/retention.toml``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tomllib
from pathlib import Path

import numpy as np

from tfm_finetuning.plots.metrics import fine_tuning_gain
from tfm_finetuning.plots.transfer.retention_data import (
    BANK,
    COVERTYPE,
    RETENTION_ROUTES,
    LearnedRetentionResult,
    SingleDatasetReference,
    _write_csv,
    load_retention,
)
from tfm_finetuning.plots.transfer.transfer_data import MODEL_ORDER

DEFAULT_CONFIG = Path("configs/plots/transfer/retention.toml")
ROUTE_LABELS = (r"Bank $\to$ Covertype", r"Covertype $\to$ Bank", "Alternating updates")


def table_text(
    references: tuple[SingleDatasetReference, ...],
    evaluations: tuple[LearnedRetentionResult, ...],
) -> str:
    """Compare same-seed final checkpoints with independent 100-update references."""
    lookup = {(row.dataset, row.model, row.seed): row for row in references}
    if len(lookup) != len(references):
        raise ValueError("Duplicate single-dataset retention references")
    lines = [
        r"\begin{table}[H]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.25}",
        r"\begin{tabularx}{\textwidth}{@{}>{\RaggedRight\arraybackslash}X*{3}{>{\centering\arraybackslash}p{0.215\textwidth}}@{}}",
        r"\toprule",
        r"\textbf{Fine-tuning schedule} & \textbf{TabPFN-2.5} & \textbf{TabICLv2} & \textbf{Mitra} \\",
        r"\midrule",
    ]
    for dataset, heading in ((BANK, "Bank: ROC-AUC difference"), (COVERTYPE, "Covertype: log-loss ratio")):
        lines.extend(
            [
                r"\addlinespace[3pt]",
                rf"\multicolumn{{4}}{{@{{}}l}}{{\textit{{Evaluated on {heading}}}}} \\",
                r"\addlinespace[2pt]",
            ]
        )
        for route, label in zip(RETENTION_ROUTES, ROUTE_LABELS, strict=True):
            cells = []
            for model in MODEL_ORDER:
                rows = [
                    row
                    for row in evaluations
                    if row.evaluation_target == dataset and row.route == route and row.model == model
                ]
                if len(rows) != 3 or {row.seed for row in rows} != {0, 1, 2}:
                    raise ValueError(f"Expected three matched retention seeds for {dataset}/{model}/{route}")
                gains = []
                for row in rows:
                    reference = lookup[(dataset, model, row.seed)]
                    if reference.metric != row.metric:
                        raise ValueError("Retention reference and evaluation use different metrics")
                    value = fine_tuning_gain(reference.checkpoint_value, row.checkpoint_value, row.metric)
                    if value is None:
                        raise ValueError("Non-finite retained-capability contrast")
                    gains.append(value)
                cells.append(rf"${np.mean(gains):+.4f} \pm {np.std(gains, ddof=1) / np.sqrt(3):.4f}$")
            lines.append(" & ".join([label, *cells]) + r" \\")
        if dataset == BANK:
            lines.append(r"\addlinespace[5pt]")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\caption[Performance after fine-tuning on both datasets.]{\textbf{Performance after fine-tuning on both datasets.} Values are mean $\pm$ standard error across three seeds. Each model is compared with an independent 100-update run on the evaluation dataset using the same seed. Bank reports $\mathrm{AUC}_{\mathrm{both}}-\mathrm{AUC}_{\mathrm{single}}$ and Covertype $\log(\mathrm{loss}_{\mathrm{single}}/\mathrm{loss}_{\mathrm{both}})$. Positive values favour training on both datasets. Sequential schedules use 100 updates per dataset. Only evaluation on the first dataset checks for loss of earlier gains. Separately trained reference models and possible differences in inference context make this an indirect check.}",
            r"\label{tab:appendix-transfer-learned-retention}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def generate(config_path: Path = DEFAULT_CONFIG) -> dict[str, object]:
    """Load corrected evaluation reports, write evidence CSVs and the final table."""
    config = tomllib.loads(config_path.read_text())
    references, evaluations = load_retention(config["input"])
    output = Path(config["output"]["analysis_dir"])
    assets = Path(config["output"]["thesis_assets"])
    output.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "single_dataset_references.csv", references)
    _write_csv(output / "bank_covertype_retention.csv", evaluations)
    table = output / "learned_capability_changes.tex"
    table.write_text(table_text(references, evaluations))
    shutil.copy2(table, assets / table.name)
    summary = {
        "input": config["input"],
        "referenceCount": len(references),
        "evaluationCount": len(evaluations),
        "table": str(table),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    """Generate the appendix table from an explicit evidence configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(generate(args.config), indent=2))


if __name__ == "__main__":
    main()
