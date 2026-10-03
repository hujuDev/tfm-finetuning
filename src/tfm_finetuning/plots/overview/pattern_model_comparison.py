"""Cross-model score, rank, and critical-difference analysis."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from scipy import stats

from tfm_finetuning.plots.common.model_display import (
    CRITICAL_DIFFERENCE_ALPHA,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_ORDER,
    TASK_LABELS,
)
from tfm_finetuning.plots.common.plot_style import (
    latex_fraction_width,
    save_and_close_figure,
)
from tfm_finetuning.plots.metrics import metric_is_lower_better

DEFAULT_DECISIVE_WINNER_MARGIN = 0.005
WINNER_UNCERTAINTY_Z = 1.96


def _model_method_score_matrix(task_rows: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, str, str]]]:
    """Return a complete dataset-by-model-method score matrix for one task family."""
    columns: list[tuple[str, str, str]] = []
    series: list[pd.Series] = []
    for model in MODEL_ORDER:
        model_rows = task_rows[task_rows["model"] == model].set_index("dataset")
        if model_rows.empty:
            continue
        for method, value_column in [("icl", "baseline"), ("sft", "selected")]:
            comparison = f"{model}__{method}"
            columns.append((comparison, model, method))
            series.append(model_rows[value_column].rename(comparison))
    if not series:
        return pd.DataFrame(), []
    return pd.concat(series, axis=1).dropna(), columns


def _critical_difference_table(
    frame: pd.DataFrame,
    *,
    alpha: float = CRITICAL_DIFFERENCE_ALPHA,
    overall: bool = False,
) -> pd.DataFrame:
    """Rank all model-method combinations and calculate Friedman--Nemenyi statistics."""
    # Rank within each task metric first, then pool dataset ranks, never raw scores.
    ranked_groups = []
    columns = [
        (f"{model}__{method}", model, method)
        for model in MODEL_ORDER
        if model in set(frame["model"])
        for method in ("icl", "sft")
    ]
    for task_family, task_rows in frame.groupby("task_family", sort=False):
        matrix, _ = _model_method_score_matrix(task_rows)
        matrix = matrix.reindex(columns=[column[0] for column in columns]).dropna()
        metric = str(task_rows["score_metric"].iloc[0])
        ranks = matrix.rank(axis=1, ascending=metric_is_lower_better(metric), method="average")
        ranked_groups.append((str(task_family), ranks))
    if overall and ranked_groups:
        ranked_groups = [("overall", pd.concat([ranks for _, ranks in ranked_groups]))]
    records: list[dict[str, object]] = []
    for task_family, ranks in ranked_groups:
        comparison_count = ranks.shape[1]
        dataset_count = ranks.shape[0]
        if comparison_count < 3 or dataset_count < 2:
            continue
        matrix = ranks
        friedman_p = float(stats.friedmanchisquare(*(matrix[column] for column in matrix.columns)).pvalue)
        critical_difference = float(
            stats.studentized_range.ppf(1.0 - alpha, comparison_count, np.inf)
            * math.sqrt(comparison_count * (comparison_count + 1) / (12.0 * dataset_count))
        )
        records.extend(
            {
                "task_family": str(task_family),
                "comparison": comparison,
                "model": model,
                "method": method,
                "mean_rank": float(ranks[comparison].mean()),
                "dataset_count": dataset_count,
                "comparison_count": comparison_count,
                "friedman_p_value": friedman_p,
                "critical_difference": critical_difference,
                "alpha": alpha,
            }
            for comparison, model, method in columns
        )
    return pd.DataFrame.from_records(records)


def _sft_dataset_win_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Count within-model route wins and strict global SFT wins."""
    complete = frame.dropna(subset=["effect", "normalized_baseline", "normalized_selected"]).copy()
    records: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        model_rows = complete[complete["model"] == model]
        own_improvements = int((model_rows["effect"] > 0).sum())
        own_icl_wins = int((model_rows["effect"] < 0).sum())
        own_ties = int((model_rows["effect"] == 0).sum())
        global_wins = 0
        for _, dataset_rows in complete.groupby("dataset"):
            candidate = dataset_rows[dataset_rows["model"] == model]
            if candidate.empty:
                continue
            selected = float(candidate.iloc[0]["normalized_selected"])
            competitor_values: list[float] = []
            for _, row in dataset_rows.iterrows():
                competitor_values.append(float(row["normalized_baseline"]))
                if row["model"] != model:
                    competitor_values.append(float(row["normalized_selected"]))
            if competitor_values and selected > max(competitor_values):
                global_wins += 1
        records.append(
            {
                "model": model,
                "own_icl_improvement_count": own_improvements,
                "own_icl_win_count": own_icl_wins,
                "own_tie_count": own_ties,
                "global_best_count": global_wins,
                "dataset_count": int(model_rows["dataset"].nunique()),
            }
        )
    return pd.DataFrame.from_records(records)


def _winner_uncertainty(
    winner: dict[str, object],
    runner_up: dict[str, object],
    *,
    metric: str,
) -> tuple[float, float, float]:
    """Approximate top-versus-runner-up separation from reported standard errors.

    Cross-model run covariance is unavailable in the composed result index, so
    the standard errors are combined as if independent.  The returned z-score
    is therefore a diagnostic, not a paired significance test.
    """
    winner_score = float(winner["raw_score"])
    runner_up_score = float(runner_up["raw_score"])
    winner_se = float(winner["raw_se"])
    runner_up_se = float(runner_up["raw_se"])
    if metric_is_lower_better(metric):
        if winner_score <= 0 or runner_up_score <= 0:
            return np.nan, np.nan, np.nan
        gap = math.log(runner_up_score / winner_score)
        gap_se = math.hypot(winner_se / winner_score, runner_up_se / runner_up_score)
    else:
        gap = winner_score - runner_up_score
        gap_se = math.hypot(winner_se, runner_up_se)
    z_score = gap / gap_se if gap_se > 0 else np.nan
    return float(gap), float(gap_se), float(z_score)


def _global_winner_table(
    frame: pd.DataFrame,
    *,
    decisive_margin: float = DEFAULT_DECISIVE_WINNER_MARGIN,
) -> pd.DataFrame:
    """Return exact observed winners plus practical and uncertainty diagnostics.

    Candidate identity is the model together with its inference method: ICL or
    validation-selected SFT.  The normalized margin is the best score minus
    the runner-up score after direction-aware best-relative normalization.  A
    unique winner is called decisive only when this margin is greater than the
    configured threshold; smaller margins remain observed wins but are treated
    as effective ties for characteristic profiling.
    """
    if decisive_margin < 0:
        raise ValueError("The decisive-winner margin must be non-negative.")
    records: list[dict[str, object]] = []
    for dataset, dataset_rows in frame.groupby("dataset", sort=True):
        candidates: list[dict[str, object]] = []
        for _, row in dataset_rows.drop_duplicates("model").iterrows():
            for method, normalized_column, raw_column, se_column in (
                ("icl", "normalized_baseline", "baseline", "baseline_se"),
                ("sft", "normalized_selected", "selected", "selected_se"),
            ):
                normalized_score = pd.to_numeric(pd.Series([row.get(normalized_column)]), errors="coerce").iloc[0]
                raw_score = pd.to_numeric(
                    pd.Series([row.get(raw_column, row.get(normalized_column))]),
                    errors="coerce",
                ).iloc[0]
                if not np.isfinite(normalized_score) or not np.isfinite(raw_score):
                    continue
                raw_se = pd.to_numeric(pd.Series([row.get(se_column)]), errors="coerce").iloc[0]
                candidates.append(
                    {
                        "dataset": dataset,
                        "task_family": row["task_family"],
                        "score_metric": row.get("score_metric", ""),
                        "model": row["model"],
                        "method": method,
                        "raw_score": float(raw_score),
                        "raw_se": float(raw_se) if np.isfinite(raw_se) else np.nan,
                        "normalized_score": float(normalized_score),
                    }
                )
        if not candidates:
            continue
        candidates.sort(
            key=lambda candidate: (
                -float(candidate["normalized_score"]),
                MODEL_ORDER.index(str(candidate["model"]))
                if str(candidate["model"]) in MODEL_ORDER
                else len(MODEL_ORDER),
                0 if candidate["method"] == "sft" else 1,
            )
        )
        best = float(candidates[0]["normalized_score"])
        winners = [
            candidate
            for candidate in candidates
            if math.isclose(float(candidate["normalized_score"]), best, rel_tol=1e-10, abs_tol=1e-12)
        ]
        runner_up = candidates[1] if len(candidates) > 1 else None
        normalized_margin = best - float(runner_up["normalized_score"]) if runner_up else np.nan
        for candidate in winners:
            candidate_runner_up = next(
                (
                    other
                    for other in candidates
                    if (other["model"], other["method"]) != (candidate["model"], candidate["method"])
                ),
                None,
            )
            exact_tie_size = len(winners)
            decisive = bool(
                exact_tie_size == 1 and np.isfinite(normalized_margin) and normalized_margin > decisive_margin
            )
            if (
                candidate_runner_up is not None
                and np.isfinite(candidate["raw_se"])
                and np.isfinite(candidate_runner_up["raw_se"])
            ):
                raw_gap, raw_gap_se, uncertainty_z = _winner_uncertainty(
                    candidate,
                    candidate_runner_up,
                    metric=str(candidate["score_metric"]),
                )
            else:
                raw_gap, raw_gap_se, uncertainty_z = np.nan, np.nan, np.nan
            records.append(
                {
                    **candidate,
                    "strict_winner": exact_tie_size == 1,
                    "tie_size": exact_tie_size,
                    "candidate_count": len(candidates),
                    "runner_up_model": candidate_runner_up["model"] if candidate_runner_up else "",
                    "runner_up_method": candidate_runner_up["method"] if candidate_runner_up else "",
                    "runner_up_raw_score": candidate_runner_up["raw_score"] if candidate_runner_up else np.nan,
                    "runner_up_raw_se": candidate_runner_up["raw_se"] if candidate_runner_up else np.nan,
                    "runner_up_normalized_score": (
                        candidate_runner_up["normalized_score"] if candidate_runner_up else np.nan
                    ),
                    "normalized_margin": normalized_margin,
                    "decisive_margin_threshold": decisive_margin,
                    "decisive_winner": decisive,
                    "winner_status": "decisive" if decisive else "effective_tie",
                    "raw_gap": raw_gap,
                    "raw_gap_se_independent": raw_gap_se,
                    "uncertainty_z_independent": uncertainty_z,
                    "uncertainty_95_separated": bool(
                        np.isfinite(uncertainty_z) and uncertainty_z > WINNER_UNCERTAINTY_Z
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def _plot_sft_dataset_wins(frame: pd.DataFrame, output_dir: Path) -> Path:
    """Show route wins within each model and across all eight variants."""
    summary = _sft_dataset_win_table(frame).set_index("model").reindex(MODEL_ORDER)
    dataset_count = int(frame["dataset"].nunique())
    global_winners = _global_winner_table(frame)
    strict_winners = global_winners[global_winners["strict_winner"]]
    global_counts = strict_winners.groupby(["method", "model"])["dataset"].nunique()
    global_tie_count = int(global_winners.loc[~global_winners["strict_winner"], "dataset"].nunique())

    width_fraction = 0.48
    figure, axis = plt.subplots(figsize=(latex_fraction_width(width_fraction), 2.75))
    model_y_positions = np.arange(len(MODEL_ORDER), dtype=float)
    overall_y = float(len(MODEL_ORDER)) + 0.45
    bar_height = 0.56

    def draw_method_separator(*, y: float, boundary: float) -> None:
        axis.plot(
            [boundary, boundary],
            [y - bar_height / 2, y + bar_height / 2],
            color="#111827",
            linewidth=1.15,
            solid_capstyle="butt",
            zorder=4,
        )

    def draw_method_segment(*, y: float, left: int, count: int, model: str, method: str) -> None:
        if count <= 0:
            return
        color = MODEL_COLORS[model]
        facecolor = (*to_rgb(color), 0.38) if method == "icl" else color
        axis.barh(
            y,
            count,
            left=left,
            height=bar_height,
            color=facecolor,
            edgecolor=color,
            linewidth=0.9,
            zorder=2,
        )

    for y, model in zip(model_y_positions, MODEL_ORDER, strict=True):
        icl_count = int(summary.loc[model, "own_icl_win_count"])
        tie_count = int(summary.loc[model, "own_tie_count"])
        sft_count = int(summary.loc[model, "own_icl_improvement_count"])
        draw_method_segment(y=y, left=0, count=icl_count, model=model, method="icl")
        if tie_count:
            axis.barh(
                y,
                tie_count,
                left=icl_count,
                height=bar_height,
                color="#d7dee8",
                edgecolor="#64748b",
                linewidth=0.9,
                zorder=2,
            )
        draw_method_segment(
            y=y,
            left=icl_count + tie_count,
            count=sft_count,
            model=model,
            method="sft",
        )
        if icl_count + tie_count > 0 and sft_count > 0:
            draw_method_separator(y=y, boundary=icl_count + tie_count)
        if icl_count:
            axis.text(icl_count / 2, y, str(icl_count), ha="center", va="center", fontsize=8.0, color="#253047")
        if tie_count:
            axis.text(
                icl_count + tie_count / 2,
                y,
                str(tie_count),
                ha="center",
                va="center",
                fontsize=8.0,
                color="#253047",
            )
        if sft_count:
            red, green, blue = to_rgb(MODEL_COLORS[model])
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            axis.text(
                icl_count + tie_count + sft_count / 2,
                y,
                str(sft_count),
                ha="center",
                va="center",
                fontsize=8.0,
                color="#253047" if luminance > 0.58 else "white",
            )

    method_bounds: dict[str, tuple[int, int]] = {}
    left = 0
    for method in ("icl", "sft"):
        method_start = left
        for model in MODEL_ORDER:
            count = int(global_counts.get((method, model), 0))
            draw_method_segment(y=overall_y, left=left, count=count, model=model, method=method)
            left += count
        method_bounds[method] = (method_start, left)
    if global_tie_count:
        axis.barh(
            overall_y,
            global_tie_count,
            left=left,
            height=bar_height,
            color="#d7dee8",
            edgecolor="#64748b",
            linewidth=0.9,
            zorder=2,
        )
        left += global_tie_count
    icl_start, icl_end = method_bounds["icl"]
    sft_start, sft_end = method_bounds["sft"]
    if icl_end > icl_start and sft_end > sft_start:
        draw_method_separator(y=overall_y, boundary=icl_end)

    for method, label in (("icl", "ICL"), ("sft", "SFT")):
        start, end = method_bounds[method]
        if end > start:
            axis.text(
                (start + end) / 2,
                overall_y - 0.43,
                f"{label} {end - start}",
                ha="center",
                va="center",
                fontsize=8.0,
                fontweight="semibold",
                color="#253047",
            )

    y_positions = [*model_y_positions, overall_y]
    axis.set_yticks(y_positions, [*[MODEL_LABELS[model] for model in MODEL_ORDER], "Best overall"])
    for tick, model in zip(axis.get_yticklabels()[:-1], MODEL_ORDER, strict=True):
        tick.set_color(MODEL_COLORS[model])
        tick.set_fontweight("semibold")
    axis.get_yticklabels()[-1].set_fontweight("bold")
    axis.axhline(len(MODEL_ORDER) - 0.30, color="#aeb8c6", linewidth=0.8, zorder=1)
    axis.set_ylim(overall_y + 0.62, -0.52)
    axis.set_xlim(0, dataset_count)
    axis.set_xticks([*range(0, dataset_count, 10), dataset_count])
    axis.set_xlabel(f"Dataset count (of {dataset_count})", fontsize=9.0)
    axis.grid(axis="x", color="#d7dee8", linewidth=0.65, alpha=0.85, zorder=0)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.legend(
        handles=[
            Patch(facecolor=(*to_rgb("#64748b"), 0.38), edgecolor="#64748b", label="ICL"),
            Patch(facecolor="#64748b", edgecolor="#64748b", label="SFT"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.17),
        ncol=2,
        frameon=False,
        fontsize=9.0,
    )
    figure.subplots_adjust(left=0.37, right=0.97, top=0.82, bottom=0.16)
    return save_and_close_figure(
        figure,
        output_dir / "sft_dataset_wins.pdf",
        tight=False,
        inclusion_fraction=width_fraction,
    )


def _critical_difference_groups(rows: pd.DataFrame) -> list[tuple[float, float]]:
    """Return maximal contiguous mean-rank groups within the Nemenyi distance."""
    ordered = rows.sort_values("mean_rank").reset_index(drop=True)
    critical_difference = float(ordered["critical_difference"].iloc[0])
    groups: list[tuple[float, float]] = []
    furthest_group_end = -1
    for left in range(len(ordered)):
        right = left
        while (
            right + 1 < len(ordered)
            and float(ordered.loc[right + 1, "mean_rank"] - ordered.loc[left, "mean_rank"]) <= critical_difference
        ):
            right += 1
        if right > left and right > furthest_group_end:
            groups.append(
                (
                    float(ordered.loc[left, "mean_rank"]),
                    float(ordered.loc[right, "mean_rank"]),
                )
            )
            furthest_group_end = right
    return groups


def _draw_critical_difference_panel(
    ax: plt.Axes,
    rows: pd.DataFrame,
    *,
    title: str | None = None,
    contain_side_labels: bool = False,
) -> None:
    """Draw one conventional left-to-right critical-difference rank panel."""
    ordered = rows.sort_values("mean_rank").reset_index(drop=True)
    comparison_count = int(ordered["comparison_count"].iloc[0])
    dataset_count = int(ordered["dataset_count"].iloc[0])
    critical_difference = float(ordered["critical_difference"].iloc[0])
    task_family = str(ordered["task_family"].iloc[0])
    axis_y = 0.52 if contain_side_labels else 0.58
    if contain_side_labels:
        ax.set_xlim(-3.2, comparison_count + 6.5)
    else:
        ax.set_xlim(0.0, comparison_count + 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()
    resolved_title = f"{TASK_LABELS[task_family]} · common datasets n={dataset_count}" if title is None else title
    if resolved_title:
        ax.text(
            (comparison_count + 1.0) / 2.0,
            0.98,
            resolved_title,
            ha="center",
            va="top",
            fontsize=10.0,
            fontweight="bold",
        )
    ax.plot([1.0, comparison_count], [axis_y, axis_y], color="#374151", linewidth=0.8)
    for rank in range(1, comparison_count + 1):
        ax.plot([rank, rank], [axis_y - 0.025, axis_y + 0.025], color="#374151", linewidth=0.8)
        ax.text(rank, axis_y + 0.045, str(rank), ha="center", va="bottom", fontsize=9.0)

    cd_y = 0.72 if contain_side_labels else 0.78
    ax.plot([1.0, 1.0 + critical_difference], [cd_y, cd_y], color="#111827", linewidth=1.0)
    ax.plot([1.0, 1.0], [cd_y - 0.025, cd_y + 0.025], color="#111827", linewidth=1.0)
    ax.plot(
        [1.0 + critical_difference, 1.0 + critical_difference],
        [cd_y - 0.025, cd_y + 0.025],
        color="#111827",
        linewidth=1.0,
    )
    ax.text(1.0 + critical_difference / 2.0, cd_y + 0.025, "CD", ha="center", va="bottom", fontsize=9.0)

    for group_index, (left_rank, right_rank) in enumerate(_critical_difference_groups(ordered)):
        group_y = axis_y - 0.055 - group_index * 0.035
        ax.plot(
            [left_rank, right_rank],
            [group_y, group_y],
            color="#111827",
            linewidth=2.0,
            solid_capstyle="round",
        )

    split = math.ceil(len(ordered) / 2)
    left_rows = ordered.iloc[:split]
    right_rows = ordered.iloc[split:].iloc[::-1]
    label_levels = np.linspace(0.39, 0.06, split)
    for side, label_rows in [("left", left_rows), ("right", right_rows)]:
        for label_y, (_, row) in zip(label_levels, label_rows.iterrows(), strict=False):
            rank = float(row["mean_rank"])
            model = str(row["model"])
            method = str(row["method"])
            color = MODEL_COLORS[model]
            is_sft = method == "sft"
            line_style = "-" if is_sft else "--"
            marker_face = color if is_sft else "white"
            if side == "left":
                elbow_x = 1.60 if contain_side_labels else 0.82
                text_x = 1.50 if contain_side_labels else 0.72
                alignment = "right"
            else:
                elbow_x = comparison_count - 0.50 if contain_side_labels else comparison_count + 0.18
                text_x = comparison_count - 0.40 if contain_side_labels else comparison_count + 0.28
                alignment = "left"
            ax.plot(
                [rank, rank, elbow_x],
                [axis_y, label_y, label_y],
                color=color,
                linewidth=0.75,
                linestyle=line_style,
            )
            ax.scatter(
                [rank],
                [axis_y],
                s=19,
                facecolor=marker_face,
                edgecolor=color,
                linewidth=0.9,
                zorder=3,
            )
            ax.text(
                text_x,
                label_y,
                f"[{rank:.2f}] {_compact_cd_model_label(model)} {method.upper()}",
                ha=alignment,
                va="center",
                fontsize=8.0,
                color=color,
                fontweight="semibold" if is_sft else "normal",
                clip_on=False,
            )


def _compact_cd_model_label(model: str) -> str:
    """Keep rank labels within the narrow main-text CD canvas."""
    return {
        "tabpfnv2": "PFNv2",
        "tabpfnv25": "PFN-2.5",
        "tabiclv2": "ICLv2",
        "mitra": "Mitra",
    }.get(model, MODEL_LABELS.get(model, model))


def _plot_overall_critical_differences(comparisons: pd.DataFrame, output_dir: Path) -> Path:
    """Render pooled dataset ranks, e.g. from `_critical_difference_table(frame, overall=True)`."""
    figure, axis = plt.subplots(figsize=(latex_fraction_width(1.0), 1.8))
    _draw_critical_difference_panel(axis, comparisons, title="", contain_side_labels=True)
    axis.set_xlim(-2.8, 12.3)
    figure.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.04)
    return save_and_close_figure(
        figure,
        output_dir / "critical_difference_overall.pdf",
        tight=False,
        inclusion_fraction=1.0,
    )
