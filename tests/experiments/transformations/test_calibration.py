from __future__ import annotations

from tfm_finetuning.experiments.transformations.calibration import (
    CalibrationRow,
    select_contextual_scale,
    select_heavy_tail,
)


def test_contextual_selector_uses_metric_gate_and_smallest_spread() -> None:
    rows = []
    for cell, baseline, shifted in (("a", 0.80, 0.79), ("b", 0.82, 0.79), ("c", 0.78, 0.75)):
        rows.append(CalibrationRow("demo", cell, "roc_auc", baseline, spread=1))
        rows.append(CalibrationRow("demo", cell, "roc_auc", shifted, spread=3))

    payload = select_contextual_scale(rows)

    assert payload["selections"] == {"demo": 3.0}
    assert payload["feature_counts"] == {"demo": 1}
    assert payload["status"] == {"demo": "selected"}


def test_contextual_selector_escalates_feature_count_before_selecting_spread() -> None:
    rows = []
    for cell in ("a", "b", "c"):
        rows.extend(
            [
                CalibrationRow("demo", cell, "rmse", 10.0, spread=1, feature_count=1),
                CalibrationRow("demo", cell, "rmse", 10.2, spread=100, feature_count=1),
                CalibrationRow("demo", cell, "rmse", 11.2, spread=3, feature_count=3),
            ]
        )

    payload = select_contextual_scale(rows)

    assert payload["selections"] == {"demo": 3.0}
    assert payload["feature_counts"] == {"demo": 3}
    assert payload["status"] == {"demo": "selected"}


def test_contextual_selector_excludes_dataset_without_validation_stress() -> None:
    rows = []
    for cell in ("a", "b", "c"):
        rows.extend(
            [
                CalibrationRow("demo", cell, "rmse", 10.0, spread=1, feature_count=1),
                CalibrationRow("demo", cell, "rmse", 10.2, spread=100, feature_count=1),
                CalibrationRow("demo", cell, "rmse", 10.4, spread=100, feature_count=3),
            ]
        )

    payload = select_contextual_scale(rows)

    assert payload["selections"] == {}
    assert payload["feature_counts"] == {}
    assert payload["status"] == {"demo": "excluded_no_validation_stress"}


def test_heavy_tail_selector_escalates_magnitude_before_selecting_rate() -> None:
    rows = []
    for cell in ("a", "b", "c"):
        rows.extend(
            [
                CalibrationRow("demo", cell, "rmse", 10.0, rate=0, magnitude_iqr=10),
                CalibrationRow("demo", cell, "rmse", 10.2, rate=0.05, magnitude_iqr=10),
                CalibrationRow("demo", cell, "rmse", 11.2, rate=0.05, magnitude_iqr=25),
            ]
        )

    payload = select_heavy_tail(rows)

    assert payload["selections"] == {"demo": {"rate": 0.05, "magnitude_iqr": 25.0}}
    assert payload["status"] == {"demo": "selected"}



def test_fresh_calibration_uses_validation_only_and_rejects_failed_runs(tmp_path):
    import json

    import pytest

    from tfm_finetuning.experiments.transformations.calibration import load_calibration_rows

    run = tmp_path / "run-1"
    run.mkdir()
    marker = run / "run.json"
    marker.write_text(json.dumps({"status": "complete"}))
    (run / "finetune_report.json").write_text(json.dumps({
        "score_metric": "rmse",
        "data_task": {
            "dataset_name": "diamonds", "fold": 1,
            "distribution_shift": {"pre_finite": {"heavy_tail_contamination": {"rate": .05, "magnitude_iqr": 25}}},
        },
        "split": {"resolved_seed": 2},
        "zero_shot_metrics": {"val": {"rmse": 11.2}, "test": {"rmse": 999}},
    }))
    rows = load_calibration_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0].value == 11.2
    assert rows[0].cell == "2:1:0:0"
    assert (rows[0].rate, rows[0].magnitude_iqr) == (.05, 25)
    marker.write_text(json.dumps({"status": "failed"}))
    with pytest.raises(ValueError, match="requires completed runs"):
        load_calibration_rows(tmp_path)
