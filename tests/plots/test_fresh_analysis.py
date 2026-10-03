"""Exercise fresh reports through scientific aggregation and final artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from plots.overview.hpo_analysis import generate_hpo_analysis
from plots.overview.pattern_data import _canonical_frame
from plots.overview.result_table import dataset_result_table_latex
from plots.results import aggregate_metrics, metric_frame, read_runs
from plots.thesis_resource_analysis import generate as generate_resources


def write_run(
    root: Path,
    *,
    study="unified_foundation_baseline",
    dataset="houses",
    model="mitra",
    seed=0,
    metric="rmse",
    baseline=4.0,
    selected=2.0,
    final=3.0,
    val=None,
    learning_rate=0.0001,
    context=10000,
    run_id=None,
    status="complete",
) -> Path:
    """Write exactly the public runner's current output contract."""
    run_id = run_id or f"{dataset}-{model}-{seed}-{learning_rate}-{context}"
    directory = root / study / run_id
    directory.mkdir(parents=True, exist_ok=True)
    parameters = {"learningRate": learning_rate, "contextSize": context}
    config = {"foundation_model": {"learning_rate": learning_rate, "weight_decay": 0.01, "context_size": context}}
    metadata = dict(
        study=study,
        dataset=dataset,
        model=model,
        seed=seed,
        fold=0,
        repeat=0,
        sample=0,
        run_id=run_id,
        preprocessing="native",
        parameters=parameters,
        config=config,
        status=status,
    )
    report = {
        "config": config,
        "score_metric": metric,
        "zero_shot_metrics": {"val": {metric: baseline}, "test": {metric: baseline}},
        "selected_checkpoint_metrics": {"val": {metric: selected if val is None else val}, "test": {metric: selected}},
        "final_checkpoint_metrics": {"val": {metric: final}, "test": {metric: final}},
        "evaluation": {"selected_epoch": 2, "final_epoch": 100},
        "runtime": {"total_seconds": 3600},
    }
    (directory / "run.json").write_text(json.dumps(metadata))
    (directory / "finetune_report.json").write_text(json.dumps(report))
    return directory


def test_fresh_reports_preserve_paired_gain_uncertainty_and_feed_final_dataset_table(tmp_path):
    write_run(tmp_path, baseline=4, selected=2, final=8)
    write_run(tmp_path, seed=1, baseline=9, selected=3, final=6)
    metadata = {"houses": {"baseTrainRows": 100, "nFeaturesTotal": 5}}
    frame = _canonical_frame(tmp_path, metadata=metadata)
    row = frame.iloc[0]
    assert row.effect == pytest.approx((math.log(2) + math.log(3)) / 2)
    assert row.effect_se == pytest.approx(abs(math.log(3) - math.log(2)) / 2)
    assert row.selected_epoch == 2
    assert row.trainingTableCells == 500
    metrics = aggregate_metrics(
        metric_frame(tmp_path).query("split == 'test'"), ["dataset", "model", "metric", "score_metric", "task_family"]
    )
    table = dataset_result_table_latex(metrics, model_order=["mitra"], model_labels={"mitra": "Mitra"})
    assert "Houses" in table
    assert r"\bestresult{2.5}" in table


def test_completed_duplicate_corrupt_and_failed_runs_are_not_silently_pooled(tmp_path):
    write_run(tmp_path)
    write_run(tmp_path, run_id="duplicate")
    with pytest.raises(ValueError, match="Duplicate"):
        list(read_runs(tmp_path))
    (tmp_path / "unified_foundation_baseline" / "duplicate" / "run.json").unlink()
    directory = write_run(tmp_path, run_id="failed", seed=1, status="failed")
    with pytest.warns(UserWarning, match="Excluded failed"):
        assert len(list(read_runs(tmp_path))) == 1
    payload = json.loads((directory / "run.json").read_text())
    payload["status"] = "complete"
    (directory / "run.json").write_text(json.dumps(payload))
    (directory / "finetune_report.json").unlink()
    with pytest.raises(FileNotFoundError):
        list(read_runs(tmp_path))


def test_hpo_selects_validation_config_and_renders_final_artifacts_from_fresh_runs(tmp_path):
    results = tmp_path / "results"
    for model in ("tabpfnv25", "tabiclv2", "mitra"):
        for dataset, metric in (("churn", "roc_auc"), ("sdss17", "log_loss"), ("houses", "rmse")):
            baseline = 0.6 if metric == "roc_auc" else 4
            default = 0.7 if metric == "roc_auc" else 3
            write_run(results, model=model, dataset=dataset, metric=metric, baseline=baseline, selected=default)
            for lr, validation, test in (
                ((0.0001, 0.8, 0.65), (0.001, 0.7, 0.9)) if metric == "roc_auc" else ((0.0001, 1, 3), (0.001, 2, 0.5))
            ):
                for seed in (0, 1, 2):
                    write_run(
                        results,
                        study="foundation_regular_hpo_representative",
                        model=model,
                        dataset=dataset,
                        metric=metric,
                        baseline=baseline,
                        selected=test,
                        val=validation,
                        learning_rate=lr,
                        seed=seed,
                    )
    config = tmp_path / "hpo.toml"
    config.write_text(
        f'''[input]\nresults_root = "{results}"\n[output]\nroot = "{tmp_path / "analysis"}"\nthesis_assets = "{tmp_path / "assets"}"\n[analysis]\nmodels = ["tabpfnv25", "tabiclv2", "mitra"]\n'''
    )
    output = generate_hpo_analysis(config_path=config)
    assert output["selectedConfigurationCount"] == 9
    import pandas as pd

    selections = pd.read_csv(tmp_path / "analysis/selected_configurations.csv")
    assert selections.learning_rate.eq(0.0001).all()
    assert (tmp_path / "assets/hpo_test_gain_vs_default.pdf").read_bytes().startswith(b"%PDF")
    assert (tmp_path / "assets/hpo_selections.tex").is_file()


def test_resource_table_uses_fresh_measured_runtime_only(tmp_path):
    results = tmp_path / "results"
    write_run(results)
    write_run(results, study="dataset_transfer", dataset="covertype")
    write_run(results, study="notebook_example", dataset="churn")
    config = tmp_path / "resources.toml"
    config.write_text(
        f'''[input]\nresults_root = "{results}"\n[output]\nanalysis_dir = "{tmp_path / "analysis"}"\nthesis_asset_dir = "{tmp_path / "assets"}"\n'''
    )
    output = generate_resources(config)
    assert output["studies"] == {"Default in-distribution": {"mitra": 1.0}, "Cross-dataset transfer": {"mitra": 1.0}}
    assert "2 GPU-hours" in (tmp_path / "assets/resource_table.tex").read_text()


def test_overview_generates_all_final_outputs_from_fresh_reports(tmp_path):
    import tomllib

    from plots.overview.pattern_analysis import generate_section1_pattern_analysis

    results = tmp_path / "results"
    metadata = {}
    for task, metric in (("binary", "roc_auc"), ("multiclass", "log_loss"), ("regression", "rmse")):
        for index in range(6):
            dataset = f"{task}_{index}"
            metadata[dataset] = {"baseTrainRows": 100 * (index + 1), "nFeaturesTotal": index + 2}
            for model_index, model in enumerate(("tabpfnv2", "tabpfnv25", "tabiclv2", "mitra")):
                for preprocessing in ("shared", "native"):
                    for seed in (0, 1, 2):
                        baseline = 0.6 + 0.01 * model_index if task == "binary" else 4 + 0.1 * model_index
                        selected = (
                            baseline + 0.001 * (index + 1) if task == "binary" else baseline / (1.1 + 0.01 * index)
                        )
                        directory = write_run(
                            results,
                            dataset=dataset,
                            model=model,
                            seed=seed,
                            metric=metric,
                            baseline=baseline,
                            selected=selected,
                            final=selected,
                            run_id=f"{dataset}-{model}-{preprocessing}-{seed}",
                        )
                        marker = directory / "run.json"
                        payload = json.loads(marker.read_text())
                        payload["preprocessing"] = preprocessing
                        marker.write_text(json.dumps(payload))
    cache = tmp_path / "characteristics.json"
    cache.write_text(json.dumps({"datasets": metadata}))
    config = tmp_path / "overview.toml"
    config.write_text(f'''[input]
results_root = "{results}"
studies = ["unified_foundation_baseline"]
meta_feature_cache = "{cache}"
[output]
root = "{tmp_path / "analysis"}"
thesis_assets = "{tmp_path / "assets"}"
[analysis]
bootstrap_seed = 42
bootstrap_resamples = 10
default_preprocessing = "shared"
default_context_size = 10000
excluded_datasets = []
features = ["baseTrainRows", "nFeaturesTotal"]
''')
    summary = generate_section1_pattern_analysis(config_path=config)
    assert summary["datasetCount"] == 18
    manifest = tomllib.loads((Path(__file__).parents[2] / "configs/plots/final_figures.toml").read_text())
    for name in manifest["groups"]["overview"]["assets"]:
        assert (tmp_path / "assets" / Path(name).name).is_file()


def test_schedule_loader_pairs_fresh_update_horizons(tmp_path):
    from plots.transfer.transfer_data import (
        DATASET_ORDER,
        MODEL_ORDER,
        ROUTE_ORDER,
        TARGET_ROUTE,
        build_transfer_tables,
    )

    for dataset in DATASET_ORDER:
        for model in MODEL_ORDER:
            for seed in (0, 1, 2):
                for route in (TARGET_ROUTE, *ROUTE_ORDER[dataset]):
                    directory = write_run(
                        tmp_path,
                        study="dataset_transfer",
                        dataset=dataset,
                        model=model,
                        seed=seed,
                        metric="log_loss",
                        baseline=1,
                        selected=0.4,
                        run_id=f"{dataset}-{model}-{seed}-{route}",
                    )
                    marker = directory / "run.json"
                    metadata = json.loads(marker.read_text())
                    metadata["parameters"]["transferRoute"] = route
                    marker.write_text(json.dumps(metadata))
                    report_path = directory / "finetune_report.json"
                    report = json.loads(report_path.read_text())
                    report["dataset_transfer"] = {"route": route}
                    report["target_update_metrics"] = (
                        {
                            str(steps): {"selected_checkpoint_metrics": {"test": {"log_loss": value}}}
                            for steps, value in ((100, 0.8), (200, 0.6), (300, 0.5))
                        }
                        if route == TARGET_ROUTE
                        else {}
                    )
                    report_path.write_text(json.dumps(report))
    tables = build_transfer_tables(tmp_path)
    assert len(tables.runs) == 153
    contrasts = [r for r in tables.contrasts if r.comparison == "target_exposure"]
    assert contrasts
    assert all(row.gain == pytest.approx(math.log(2)) for row in contrasts)


def test_natural_domains_read_fresh_training_boundaries_and_frozen_context_views(tmp_path):
    from plots.transfer.natural_domain_data import (
        FIXED_TARGET_CONTEXT_POLICY,
        MODEL_ORDER,
        PANEL_DATASETS,
        PANEL_METRICS,
        ROUTE_MATCHED_CONTEXT_POLICY,
        ROUTE_ORDER,
        TARGET_PRODUCER_ROUTE,
        load_natural_domain_rows,
    )

    for panel, datasets in PANEL_DATASETS.items():
        for dataset in datasets:
            metric = PANEL_METRICS[panel]
            baseline, selected = (0.6, 0.7) if metric == "roc_auc" else (1, 0.5)
            for model in MODEL_ORDER:
                for seed in (0, 1, 2):
                    for context in (FIXED_TARGET_CONTEXT_POLICY, ROUTE_MATCHED_CONTEXT_POLICY):
                        study = (
                            "dataset_transfer_natural_domain_training"
                            if context == FIXED_TARGET_CONTEXT_POLICY
                            else "dataset_transfer_natural_domain"
                        )
                        for route in (TARGET_PRODUCER_ROUTE, *ROUTE_ORDER[1:]):
                            directory = write_run(
                                tmp_path,
                                study=study,
                                dataset=dataset,
                                model=model,
                                seed=seed,
                                metric=metric,
                                baseline=baseline,
                                selected=selected,
                                run_id=f"{dataset}-{model}-{seed}-{route}",
                            )
                            marker = directory / "run.json"
                            metadata = json.loads(marker.read_text())
                            metadata["parameters"]["transferRoute"] = route
                            marker.write_text(json.dumps(metadata))
                            path = directory / "finetune_report.json"
                            report = json.loads(path.read_text())
                            report["dataset_transfer"] = {
                                "route": route,
                                "context_policy": context,
                                "execution_mode": "train",
                            }
                            if context == ROUTE_MATCHED_CONTEXT_POLICY:
                                report["dataset_transfer"].update(
                                    route="route_matched_context_eval",
                                    execution_mode="checkpoint_eval",
                                    checkpoint_route=route,
                                )
                            report["target_update_metrics"] = {
                                "100": {
                                    "selected_checkpoint_metrics": report["selected_checkpoint_metrics"],
                                    "zero_shot_metrics": report["zero_shot_metrics"],
                                }
                            }
                            path.write_text(json.dumps(report))
    for fixed in (False, True):
        rows = load_natural_domain_rows(tmp_path, fixed_target_context=fixed)
        assert len(rows) == 81
        assert all(row.seed_count == 3 for row in rows)
        assert all(row.gain > 0 for row in rows)


def test_results_root_environment_only_overrides_default_input(tmp_path, monkeypatch):
    from plots.results import REPO_ROOT

    alternate = tmp_path / "environment_results"
    explicit = tmp_path / "configured_results"
    write_run(alternate, dataset="houses")
    write_run(explicit, dataset="churn")
    monkeypatch.setenv("RESULTS_ROOT", str(alternate))
    assert [run.dataset for run in read_runs(Path("results"))] == ["houses"]
    assert [run.dataset for run in read_runs(REPO_ROOT / "results")] == ["houses"]
    assert [run.dataset for run in read_runs(explicit)] == ["churn"]
