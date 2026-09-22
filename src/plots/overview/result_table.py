"""Render the submitted per-dataset primary-metric appendix table.

Example: ``write_dataset_result_table(frame, output_dir=out,
model_order=models, model_labels=labels)`` keeps one row per dataset.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from plots.metrics import metric_is_lower_better

PRIMARY_METRICS = {"binary": "roc_auc", "multiclass": "log_loss", "regression": "rmse"}
TASK_LABELS = {
    "binary": "Binary classification",
    "multiclass": "Multiclass classification",
    "regression": "Regression",
}
METRIC_LABELS = {"roc_auc": "ROC-AUC", "log_loss": "log-loss", "rmse": "RMSE"}
DATASET_LABELS = {
    "apsfailure": "APS Failure",
    "aps_failure": "APS Failure",
    "heloc": "HELOC",
    "jm1": "JM1",
    "mic": "MIC",
    "sdss17": "SDSS17",
    "diabetes130us": "Diabetes130US",
    "credit_g": "Credit-G",
    "credit-g": "Credit-G",
    "qsar_biodeg": "QSAR Biodeg",
    "qsar_fish_toxicity": "QSAR Fish Toxicity",
    "qsar_tid_11": "QSAR TID 11",
    "e_commereshippingdata": r"E Commerce\\Shipping Data",
    "givemesomecredit": r"Give Me Some\\Credit",
    "superconductivity": r"Super-\\conductivity",
}

PREAMBLE = r"""\begingroup
\scriptsize
\setlength{\LTcapwidth}{\textwidth}
\setlength{\tabcolsep}{0.7pt}
\renewcommand{\arraystretch}{1}
\newlength{\resultdatasetwidth}
\setlength{\resultdatasetwidth}{0.155\linewidth}
\newlength{\resultmetricwidth}
\setlength{\resultmetricwidth}{0.057\linewidth}
\newcommand{\resultdataset}[1]{\parbox[c][3.6em][c]{\resultdatasetwidth}{\RaggedRight #1}}
\newcommand{\resultmetric}[1]{\parbox[c][3.6em][c]{\resultmetricwidth}{\centering #1}}
\newcommand{\resultheader}[1]{\parbox[t][3.2em][t]{\linewidth}{\centering #1}}
\newcommand{\resultft}[2]{\shortstack[c]{{\tiny\strut}\\#1\\{\tiny #2}}}
\newcommand{\bestresult}[1]{\textbf{#1}}
\newcommand{\resultrowrule}{\\\specialrule{0.12pt}{0pt}{0pt}}"""

CAPTION = r"""\caption[Fine-tuning results for individual datasets]{\textbf{Fine-tuning results for individual datasets.} This table lists the detailed results for each dataset behind the aggregate result figures in \cref{fig:findings-overview-norm-score,fig:findings-overview-gain-outcomes}. Entries are mean test metrics, with gains over the same model's ICL result beneath SFT values. Positive gains indicate improvement. Bold marks the best observed mean per dataset.}\label{tab:unified-foundation-baseline-scorebars-primary-score-taskfamily} \\"""


def write_dataset_result_table(
    frame: pd.DataFrame,
    *,
    output_dir: Path,
    model_order: Sequence[str],
    model_labels: Mapping[str, str],
) -> Path:
    """Write primary-metric ICL/SFT values and gains while preserving missing cells."""
    path = output_dir / "default_overview_dataset_table.tex"
    path.write_text(
        dataset_result_table_latex(frame, model_order=model_order, model_labels=model_labels),
        encoding="utf-8",
    )
    return path


def dataset_result_table_latex(
    frame: pd.DataFrame,
    *,
    model_order: Sequence[str],
    model_labels: Mapping[str, str],
) -> str:
    """Show the task's primary metric, keeping missing cells and best-value ties."""
    required = {"dataset", "model", "task_family", "metric", "baseline", "selected", "effect"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Dataset result table is missing columns: {sorted(missing)}")
    primary = frame[frame["metric"] == frame["task_family"].map(PRIMARY_METRICS)]
    if primary.duplicated(["dataset", "model"]).any():
        raise ValueError("Expected one primary-metric row per dataset and model")
    count = 2 + 2 * len(model_order)
    columns = (
        r"@{}>{\RaggedRight\arraybackslash}m{\resultdatasetwidth}"
        r">{\centering\arraybackslash}m{\resultmetricwidth}"
        rf"*{{{len(model_order)}}}{{@{{\hspace{{1.2pt}}}}*{{2}}{{>{{\centering\arraybackslash}}m{{0.091\linewidth}}}}}}@{{}}"
    )
    header = (
        " & ".join(
            [
                r"\resultheader{Dataset}",
                r"\resultheader{Metric}",
                *[rf"\multicolumn{{2}}{{c}}{{{_latex_escape(model_labels[m])}}}" for m in model_order],
            ]
        )
        + r" \\"
    )
    rules = "".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(model_order)))
    subheader = " & ".join(["", "", *[value for _ in model_order for value in ("ICL", "SFT")]]) + r" \\"
    lines = [PREAMBLE, rf"\begin{{longtable}}{{{columns}}}", CAPTION]
    for boundary in (r"\endfirsthead", r"\endhead"):
        lines.extend([r"\toprule", header, rules, subheader, r"\midrule", boundary])
    lines.extend(
        [
            r"\midrule",
            rf"\multicolumn{{{count}}}{{r}}{{Continued on next page}} \\",
            r"\midrule",
            r"\endfoot",
            r"\bottomrule",
            r"\endlastfoot",
        ]
    )
    for task, metric in PRIMARY_METRICS.items():
        task_rows = primary[primary["task_family"] == task]
        if task_rows.empty:
            continue
        lines.extend(
            [
                r"\addlinespace[2pt]",
                rf"\multicolumn{{{count}}}{{@{{}}l}}{{\textbf{{{TASK_LABELS[task]}}}}} \\",
                r"\addlinespace[1pt]",
            ]
        )
        for dataset in sorted(task_rows["dataset"].unique(), key=lambda d: _dataset_label(d).lower()):
            rows = task_rows[task_rows["dataset"] == dataset].set_index("model")
            values = [
                float(v) for v in rows[["baseline", "selected"]].to_numpy().flat if pd.notna(v) and math.isfinite(v)
            ]
            best = (min(values) if metric_is_lower_better(metric) else max(values)) if values else None
            cells = [rf"\resultdataset{{{_dataset_label(dataset)}}}", rf"\resultmetric{{{METRIC_LABELS[metric]}}}"]
            for model in model_order:
                if model not in rows.index:
                    cells.extend(["--", "--"])
                    continue
                row = rows.loc[model]
                cells.extend(
                    [
                        _metric_cell(row["baseline"], best),
                        rf"\resultft{{{_metric_cell(row['selected'], best)}}}{{{_gain_value(row['effect'])}}}",
                    ]
                )
            lines.append(" & ".join(cells) + r" \resultrowrule")
    return "\n".join([*lines, r"\end{longtable}", r"\endgroup", ""])


def _dataset_label(dataset: object) -> str:
    name = str(dataset)
    return DATASET_LABELS.get(name, _latex_escape(name.replace("_", " ").replace("-", " ").title()))


def _metric_cell(value: object, best: float | None) -> str:
    if pd.isna(value) or not math.isfinite(float(value)):
        return "--"
    text = _raw_value(float(value))
    if best is not None and math.isclose(float(value), best, rel_tol=1e-10, abs_tol=1e-12):
        return rf"\bestresult{{{text}}}"
    return text


def _raw_value(value: float) -> str:
    magnitude = abs(value)
    decimals = 1 if magnitude >= 1000 else 2 if magnitude >= 100 else 3 if magnitude >= 10 else 4
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")


def _gain_value(value: object) -> str:
    if pd.isna(value) or not math.isfinite(float(value)):
        return "--"
    numeric = float(value)
    return "0.0000" if abs(numeric) < 0.00005 else f"{numeric:+.4f}"


def _latex_escape(value: object) -> str:
    text = str(value)
    for original, replacement in [("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_")]:
        text = text.replace(original, replacement)
    return text


__all__ = ["dataset_result_table_latex", "write_dataset_result_table"]
