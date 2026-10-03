"""Select validation-only contextual-scale and heavy-tail calibration doses.

Example:
    UV_CACHE_DIR=/tmp/uv-cache uv run python \
      src/tfm_finetuning/experiments/transformations/calibration.py contextual-scale \
      results/distribution_shift_contextual_scale_calibration \
      --output /tmp/contextual-scale-selections.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any


@dataclass(frozen=True)
class CalibrationRow:
    dataset: str
    cell: str
    metric: str
    value: float
    spread: float | None = None
    feature_count: int | None = None
    rate: float | None = None
    magnitude_iqr: float | None = None


def select_contextual_scale(rows: list[CalibrationRow]) -> dict[str, Any]:
    """Select the smallest feature-count tier and spread crossing the stress gate."""
    selections: dict[str, float] = {}
    feature_counts: dict[str, int] = {}
    status: dict[str, str] = {}
    all_diagnostics: dict[str, list[dict[str, Any]]] = {}
    for dataset, dataset_rows in _by_dataset(rows).items():
        reference = {row.cell: row for row in dataset_rows if row.spread == 1.0}
        count_tiers = sorted(
            {int(row.feature_count or 1) for row in dataset_rows if row.spread is not None and row.spread != 1.0}
        )
        selected = None
        diagnostics = []
        for feature_count in count_tiers:
            tier = [row for row in dataset_rows if int(row.feature_count or 1) == feature_count]
            by_spread = _group(tier, "spread")
            for spread in (3.0, 10.0, 30.0, 100.0):
                degradations = _paired_degradations(reference, by_spread.get(spread, []))
                metric = dataset_rows[0].metric if dataset_rows else ""
                gate = _contextual_gate(degradations, metric=metric)
                diagnostics.append({"feature_count": feature_count, "spread": spread, **gate})
                if gate["passes"] and selected is None:
                    selected = {"feature_count": feature_count, "spread": spread}
            if selected is not None:
                break
        if selected is not None:
            selections[dataset] = float(selected["spread"])
            feature_counts[dataset] = int(selected["feature_count"])
            status[dataset] = "selected"
        else:
            status[dataset] = "excluded_no_validation_stress"
        all_diagnostics[dataset] = diagnostics
    return {
        "selections": selections,
        "feature_counts": feature_counts,
        "status": status,
        "diagnostics": all_diagnostics,
    }


def select_heavy_tail(rows: list[CalibrationRow]) -> dict[str, Any]:
    """Select rate first within magnitude tiers 10, 25, then 50 IQR."""
    selections: dict[str, dict[str, float]] = {}
    status: dict[str, str] = {}
    all_diagnostics: dict[str, list[dict[str, Any]]] = {}
    for dataset, dataset_rows in _by_dataset(rows).items():
        selected = None
        diagnostics = []
        # A zero-rate run is independent of magnitude. Current plans report it
        # once; the mapping also collapses equivalent rows from older grids.
        reference = {row.cell: row for row in dataset_rows if row.rate == 0.0}
        for magnitude in (10.0, 25.0, 50.0):
            tier = [row for row in dataset_rows if row.magnitude_iqr == magnitude]
            by_rate = _group(tier, "rate")
            for rate in (0.01, 0.05, 0.15, 0.30, 0.60, 1.0):
                degradations = _paired_degradations(reference, by_rate.get(rate, []))
                gate = _heavy_tail_gate(degradations)
                diagnostics.append({"rate": rate, "magnitude_iqr": magnitude, **gate})
                if gate["passes"] and selected is None:
                    selected = {"rate": rate, "magnitude_iqr": magnitude}
            if selected is not None:
                break
        selections[dataset] = selected or {"rate": 1.0, "magnitude_iqr": 50.0}
        status[dataset] = "selected" if selected is not None else "robust_at_maximum_tested_dose"
        all_diagnostics[dataset] = diagnostics
    return {"selections": selections, "status": status, "diagnostics": all_diagnostics}


def load_calibration_rows(root: Path) -> list[CalibrationRow]:
    """Load validation-only measurements from completed fresh calibration runs."""
    rows: list[CalibrationRow] = []
    for path in _finetune_report_paths(root):
        payload = json.loads(path.read_text(encoding="utf-8"))
        report = payload
        data_task = report.get("data_task") if isinstance(report.get("data_task"), dict) else {}
        shift = data_task.get("distribution_shift") if isinstance(data_task.get("distribution_shift"), dict) else {}
        pre = shift.get("pre_finite") if isinstance(shift.get("pre_finite"), dict) else {}
        metrics = report.get("zero_shot_metrics") or {}
        validation = metrics.get("val") if isinstance(metrics, dict) else None
        if not isinstance(validation, dict):
            continue
        metric = str(report.get("score_metric") or data_task.get("metric_name") or "").lower()
        value = validation.get(metric, validation.get("score"))
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        dataset = _dataset_id(data_task)
        split = report.get("split") if isinstance(report.get("split"), dict) else {}
        cell = ":".join(
            str(value)
            for value in (
                split.get("resolved_seed", split.get("base_seed", 0)),
                data_task.get("fold", 0),
                data_task.get("repeat", 0),
                data_task.get("sample", 0),
            )
        )
        contextual = pre.get("contextual_scale") if isinstance(pre.get("contextual_scale"), dict) else {}
        heavy = pre.get("heavy_tail_contamination") if isinstance(pre.get("heavy_tail_contamination"), dict) else {}
        rows.append(
            CalibrationRow(
                dataset=dataset,
                cell=cell,
                metric=metric,
                value=float(value),
                spread=_optional_float(contextual.get("scale_spread")),
                feature_count=_optional_int(contextual.get("contextualized_feature_count")),
                rate=_optional_float(heavy.get("rate")),
                magnitude_iqr=_optional_float(heavy.get("magnitude_iqr")),
            )
        )
    return rows


def _finetune_report_paths(source: Path) -> list[Path]:
    if not source.is_dir():
        raise FileNotFoundError(f"Expected a completed study directory: {source}")
    paths = []
    for marker in sorted(source.glob("*/run.json")):
        status = json.loads(marker.read_text())["status"]
        if status != "complete":
            raise ValueError(f"Calibration requires completed runs: {marker} is {status}")
        paths.append(marker.with_name("finetune_report.json"))
    return paths


def _paired_degradations(
    reference: dict[str, CalibrationRow],
    candidates: list[CalibrationRow],
) -> list[float]:
    out = []
    for candidate in candidates:
        baseline = reference.get(candidate.cell)
        if baseline is None or baseline.metric != candidate.metric:
            continue
        if candidate.metric == "roc_auc":
            out.append(baseline.value - candidate.value)
        elif abs(baseline.value) > 1e-12:
            out.append(candidate.value / baseline.value - 1.0)
    return out


def _contextual_gate(degradations: list[float], *, metric: str) -> dict[str, Any]:
    if not degradations:
        return {"passes": False, "cell_count": 0}
    threshold = 0.02 if metric == "roc_auc" else 0.10
    return _gate(degradations, median_threshold=threshold, cell_threshold=threshold)


def _heavy_tail_gate(degradations: list[float]) -> dict[str, Any]:
    return _gate(degradations, median_threshold=0.10, cell_threshold=0.05)


def _gate(
    degradations: list[float],
    *,
    median_threshold: float,
    cell_threshold: float,
) -> dict[str, Any]:
    if not degradations:
        return {"passes": False, "cell_count": 0}
    median_value = float(median(degradations))
    fraction = sum(value >= cell_threshold for value in degradations) / len(degradations)
    return {
        "passes": median_value >= median_threshold and fraction >= 2 / 3,
        "cell_count": len(degradations),
        "median_degradation": median_value,
        "degraded_cell_fraction": fraction,
    }


def _by_dataset(rows: list[CalibrationRow]) -> dict[str, list[CalibrationRow]]:
    grouped: dict[str, list[CalibrationRow]] = defaultdict(list)
    for row in rows:
        grouped[row.dataset].append(row)
    return dict(grouped)


def _group(rows: list[CalibrationRow], field: str) -> dict[float, list[CalibrationRow]]:
    grouped: dict[float, list[CalibrationRow]] = defaultdict(list)
    for row in rows:
        value = getattr(row, field)
        if value is not None:
            grouped[float(value)].append(row)
    return dict(grouped)


def _dataset_id(data_task: dict[str, Any]) -> str:
    value = str(data_task.get("dataset_id") or "").strip()
    if value:
        return value.lower().replace("-", "_")
    path = str(data_task.get("dataset_path") or "").rstrip("/")
    return path.rsplit("/", 1)[-1].lower().replace("-", "_")


def _optional_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _optional_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 1 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=("contextual-scale", "heavy-tail"))
    parser.add_argument(
        "results_root",
        type=Path,
        help="Directory containing completed calibration runs.",
    )
    parser.add_argument(
        "--additional-input",
        action="append",
        default=[],
        type=Path,
        help="Additional completed calibration study directory.",
    )
    parser.add_argument("--dataset", action="append", default=[], help="Restrict selection to one dataset.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = []
    for source in (args.results_root, *args.additional_input):
        rows.extend(load_calibration_rows(source))
    selected_datasets = {value.strip().lower().replace("-", "_") for value in args.dataset if value.strip()}
    if selected_datasets:
        rows = [row for row in rows if row.dataset in selected_datasets]
    if args.family == "contextual-scale":
        rows = [row for row in rows if row.spread is not None]
    else:
        rows = [row for row in rows if row.rate is not None and row.magnitude_iqr is not None]
    if not rows:
        raise SystemExit(f"No usable {args.family} calibration rows found in {args.results_root}")
    selections = select_contextual_scale(rows) if args.family == "contextual-scale" else select_heavy_tail(rows)
    payload = json.dumps(selections, indent=2, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
