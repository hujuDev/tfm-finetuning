"""Five-setting normalized-score comparison.

Example: render_grouped_natural_domains(matched_rows, fixed_rows, Path("figure.pdf")).
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator

from plots.common.model_display import MODEL_COLORS, MODEL_LABELS
from plots.common.plot_style import FULL_WIDTH_INCHES
from plots.metrics import best_relative_normalized_score, metric_is_lower_better
from plots.transfer.natural_domain_data import MODEL_ORDER, PANEL_LABELS, PANEL_ORDER, ROUTE_ORDER


def render_grouped_natural_domains(matched_rows, fixed_rows, output: Path) -> Path:
    """Compare both context protocols with one normalization per target rotation."""
    rows = {"matched": matched_rows, "fixed": fixed_rows}
    best = {}
    for group in rows.values():
        for r in group:
            k = (r.panel, r.dataset)
            vals = [r.baseline, r.final] + ([best[k]] if k in best else [])
            best[k] = min(vals) if metric_is_lower_better(r.metric) else max(vals)
    points = defaultdict(list)
    for protocol, group in rows.items():
        for r in group:
            for phase, v in [("icl", r.baseline), ("sft", r.final)]:
                points[protocol, r.panel, r.model, r.route, phase].append(
                    best_relative_normalized_score(v, best[r.panel, r.dataset], r.metric)
                )
    for values in points.values():
        assert len(values) == 3
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
    output.parent.mkdir(parents=True, exist_ok=True)
    handles = [Line2D([], [], color=MODEL_COLORS[m], lw=3, label=MODEL_LABELS[m]) for m in MODEL_ORDER] + [
        Line2D([], [], color="#555", marker="o", mfc="white", ls="", label="ICL"),
        Line2D([], [], color="#555", marker="D", ls="", label="SFT"),
    ]

    def point(ax, protocol, panel, model, route, phase, x):
        v = points[protocol, panel, model, route, phase]
        mu = np.mean(v)
        se = np.std(v, ddof=1) / np.sqrt(3)
        ax.errorbar(
            x,
            mu,
            yerr=se,
            fmt="o" if phase == "icl" else "D",
            color=MODEL_COLORS[model],
            mfc="white" if phase == "icl" else MODEL_COLORS[model],
            ms=5,
            capsize=2,
            lw=0.9,
            zorder=3,
        )
        return mu

    limits = {"heart": (0.8, 1.005), "wids": (0.9, 1.005), "covertype": (0, 1.05)}

    def style(ax, panel):
        ax.set_ylim(*limits[panel])
        ax.set_ylabel("Normalized score")
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
        ax.set_yticks(
            {
                "heart": [0.8, 0.85, 0.9, 0.95, 1.0],
                "wids": [0.9, 0.92, 0.94, 0.96, 0.98, 1.0],
                "covertype": [0, 0.2, 0.4, 0.6, 0.8, 1.0],
            }[panel]
        )
        if panel in ("heart", "wids", "covertype"):
            ax.yaxis.set_minor_locator(
                FixedLocator([v / 100 for v in range(0, 101, 2) if v / 100 >= limits[panel][0]])
            )
            ax.tick_params(axis="y", which="minor", length=2.5, width=0.7, labelleft=False)
            ax.tick_params(axis="y", which="major", length=5)

    fig, axes = plt.subplots(3, 1, figsize=(FULL_WIDTH_INCHES, 6.2), squeeze=False)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.82, bottom=0.10, hspace=0.42, wspace=0.26)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=5,
        frameon=False,
        fontsize=8,
        columnspacing=1.1,
        handlelength=1.4,
        handletextpad=0.4,
    )
    for ri, panel in enumerate(PANEL_ORDER):
        ax = axes[ri, 0]
        style(ax, panel)
        ax.set_title(PANEL_LABELS[panel], loc="left", fontweight="bold", fontsize=10, pad=6)
        settings = [
            ("fixed", ROUTE_ORDER[0]),
            ("fixed", ROUTE_ORDER[1]),
            ("fixed", ROUTE_ORDER[2]),
            ("matched", ROUTE_ORDER[1]),
            ("matched", ROUTE_ORDER[2]),
        ]
        for j, (protocol, route) in enumerate(settings):
            for mi, model in enumerate(MODEL_ORDER):
                x = j + (mi - 1) * 0.22
                a = point(ax, protocol, panel, model, route, "icl", x - 0.045)
                b = point(ax, protocol, panel, model, route, "sft", x + 0.045)
                ax.plot([x - 0.045, x + 0.045], [a, b], color=MODEL_COLORS[model], alpha=0.4, lw=0.8)
        ax.set_xlim(-0.5, 4.5)
        labels = ["ICL: T\nSFT: T", "ICL: T\nSFT: S", "ICL: T\nSFT: T + S", "ICL: S\nSFT: S", "ICL: T + S\nSFT: T + S"]
        ax.set_xticks(range(5), labels if ri == 2 else [""] * 5)
        ax.tick_params(axis="x", labelsize=9)
        for x in [0.5, 2.5]:
            ax.axvline(x, color="#aaa", lw=0.8)
        if ri == 0:
            for x, label in [
                (0, "Target-group-only\nreference"),
                (1.5, "Vary SFT\ntrain data only"),
                (3.5, "Vary SFT + ICL\ntrain data"),
            ]:
                ax.text(
                    x, 1.24, label, transform=ax.get_xaxis_transform(), ha="center", fontsize=8.5, fontweight="bold"
                )
    fig.savefig(output)
    plt.close(fig)
    return output
