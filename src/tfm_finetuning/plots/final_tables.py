"""Render final appendix tables from analysis values, without embedding results.

Example: ``hpo_table(selected_configurations, trial_table)`` returns LaTeX.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

MODELS = ["tabpfnv25", "tabiclv2", "mitra"]
LABELS = {
    "airfoil_self_noise": "Airfoil Self Noise",
    "maternal_health_risk": "Maternal Health Risk",
    "miami_housing": "Miami Housing",
    "jm1": "JM1",
}


def hpo_table(selected: pd.DataFrame, trials: pd.DataFrame) -> str:
    """Render labelled parameter columns, grouped by genuine context eligibility."""
    eligible = {}
    for dataset in selected.dataset.unique():
        by_model = [
            sorted(trials.loc[(trials.dataset == dataset) & (trials.model == model), "context_size"].unique())
            for model in MODELS
        ]
        if not by_model[0] or not all(values == by_model[0] for values in by_model):
            raise ValueError(f"Missing or model-dependent context eligibility: {dataset}")
        eligible[dataset] = by_model[0]
    lines = [
        r"\begin{table}[H]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{0pt}",
        r"\renewcommand{\arraystretch}{1.22}",
        # Empty spacer columns keep model headings, rules and values on the same grid.
        r"\begin{tabularx}{\textwidth}{@{}>{\RaggedRight\arraybackslash}X@{\hspace{6pt}}w{c}{46pt}*{3}{w{c}{10pt}w{c}{25pt}w{c}{24pt}w{c}{37pt}}@{}}",
        r"\toprule",
        r"& & & \multicolumn{3}{@{}c@{}}{\textbf{TabPFN-2.5}} & & \multicolumn{3}{@{}c@{}}{\textbf{TabICLv2}} & & \multicolumn{3}{@{}c@{}}{\textbf{Mitra}} \\",
        r"\cmidrule{4-6}\cmidrule{8-10}\cmidrule{12-14}",
        r"\textbf{Dataset} & \shortstack{Eligible\\contexts} & & LR & WD & Context & & LR & WD & Context & & LR & WD & Context \\",
        r"\midrule",
    ]
    for multiple, heading in [(True, "Multiple context sizes tested"), (False, "One eligible context size")]:
        lines.extend(
            [r"\addlinespace[3pt]", rf"\multicolumn{{14}}{{@{{}}l}}{{\textit{{{heading}}}}} \\", r"\addlinespace[2pt]"]
        )
        for dataset in sorted(key for key, values in eligible.items() if (len(values) > 1) == multiple):
            group = selected[selected.dataset == dataset]
            contexts = ", ".join(str(int(c / 1000)) for c in eligible[dataset])
            cells = []
            for model in MODELS:
                rows = group[group.model == model]
                if len(rows) != 1:
                    raise ValueError(f"Expected one selection: {dataset}, {model}")
                row = rows.iloc[0]
                exponent = round(math.log10(row.learning_rate))
                context = str(int(row.context_size / 1000))
                if multiple and row.context_size == max(eligible[dataset]):
                    context = rf"\textbf{{{context}}}"
                cells.extend(["", rf"$10^{{{exponent}}}$", f"{row.weight_decay:g}", context])
            lines.append(
                " & ".join([LABELS.get(dataset, dataset.replace("_", " ").title()), contexts, *cells]) + r" \\"
            )
        if multiple:
            lines.append(r"\addlinespace[3pt]")
    choice_datasets = [dataset for dataset, contexts in eligible.items() if len(contexts) > 1]
    largest = {
        model: sum(
            selected.loc[(selected.dataset == dataset) & (selected.model == model), "context_size"].iloc[0]
            == max(eligible[dataset])
            for dataset in choice_datasets
        )
        for model in MODELS
    }
    choice_summary = (
        (
            f" Among datasets with a context-size choice, the largest cap is selected in "
            f"{largest['tabpfnv25']}/{len(choice_datasets)} cases for TabPFN-2.5, "
            f"{largest['tabiclv2']}/{len(choice_datasets)} for TabICLv2 and "
            f"{largest['mitra']}/{len(choice_datasets)} for Mitra."
        )
        if choice_datasets
        else ""
    )
    lines += [
        r"\bottomrule",
        r"\end{tabularx}",
        r"\caption[Eligible contexts and validation-selected hyperparameters.]{\textbf{Eligible contexts and validation-selected hyperparameters.} LR denotes learning rate and WD weight decay. Configurations are selected by mean inner-validation performance across seeds and folds. Context caps are given in thousands of rows and may exceed the available fine-tuning rows. Bold context values mark selection of the largest eligible cap where multiple sizes were tested."
        + choice_summary
        + "}",
        r"\label{tab:appendix-hpo-selections}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def resource_table(studies: Mapping[str, Mapping[str, float]]) -> str:
    """Render the measured GPU-hours of completed reproduction runs."""
    numbers = [float(value) for values in studies.values() for value in values.values()]
    if any(not math.isfinite(value) or value < 0 for value in numbers):
        raise ValueError("GPU-hours must be finite and nonnegative.")
    lines = [
        r"\begin{table}[H]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabularx}{\textwidth}{@{}Xrrrrr@{}}",
        r"\toprule",
        r"Experiment & TabPFNv2 & TabPFN-2.5 & TabICLv2 & Mitra & Total \\",
        r"\midrule",
    ]
    for study, values in studies.items():
        cells = [f"{values[model]:.1f}" if model in values else "--" for model in ["tabpfnv2", *MODELS]]
        lines.append(" & ".join([study, *cells, f"{sum(values.values()):.1f}"]) + r" \\")
    # The caption sums the displayed study totals, rounded to one decimal above.
    displayed_total = sum(Decimal(f"{sum(values.values()):.1f}") for values in studies.values())
    rounded_total = displayed_total.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    lines += [
        r"\bottomrule",
        r"\end{tabularx}",
        r"\caption[Recorded GPU-hours by experiment and model.]{\textbf{Recorded GPU-hours by experiment and model.} "
        f"The total is approximately {rounded_total:,} GPU-hours, including supporting experiments. "
        "Hardware and workloads differ, so these times do not directly compare model efficiency.}",
        r"\label{tab:appendix-resources}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"
