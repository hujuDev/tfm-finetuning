"""Synthetic integration checks for final artifacts; these are not research results.

The public planner and run writer emit current run envelopes. Transfer training
reports are built by the production report builder from deterministic fake stage
metrics. No dataset download, model loading, or gradient update occurs here.
"""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path

import pytest
import tomli_w

from experiments.configuration import dataset_key, expand_runs
from experiments.run_experiment import execute_run
from experiments.transfer.routes.resolve import resolve_transfer_route
from experiments.transfer.runner.executor import StageProduction, StageResult
from experiments.transfer.runner.reporting import training_report
from utils.config import read_yaml_mapping

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = tomllib.loads((ROOT / "configs/plots/final_figures.toml").read_text())
MODEL_IDS = {"mitra": "mitra", "tabiclv2": "tabicl_v2", "tabpfnv25": "tabpfn_v2_5", "tabpfnv2": "tabpfn_v2"}
REGRESSION = {
    "houses",
    "diamonds",
    "food_delivery_time",
    "physiochemical_protein",
    "miami_housing",
    "airfoil_self_noise",
    "healthcare_insurance_expenses",
    "superconductivity",
    "concrete_compressive_strength",
    "qsar_fish_toxicity",
}
MULTICLASS = {"covertype", "sdss17", "satimage", "splice", "anneal", "maternal_health_risk", "wine_quality"}


def metric_for(dataset):
    dataset = dataset_key(dataset)
    return (
        "rmse"
        if dataset in REGRESSION
        else "log_loss"
        if dataset in MULTICLASS or dataset.startswith("covertype_")
        else "roc_auc"
    )


def metrics(dataset, *, seed=0, adapted=False):
    metric = metric_for(dataset)
    value = (
        (0.72 if adapted else 0.65) + 0.001 * seed if metric == "roc_auc" else (1.1 if adapted else 1.5) + 0.001 * seed
    )
    split = {metric: value, "tabarena_metric": metric}
    if metric != "rmse":
        split["classification_report"] = {
            "0": {"recall": 0.8 + 0.001 * seed, "support": 80},
            "1": {"recall": 0.6 + 0.001 * seed, "support": 20},
        }
    return {"val": copy.deepcopy(split), "test": split}


def base_report(run):
    return {
        "config": run["config"],
        "score_metric": metric_for(run["dataset"]),
        "foundation_model": {"model_id": MODEL_IDS[run["model"]]},
        "data_task": {"dataset_id": run["dataset"], "fold": run["fold"]},
        "zero_shot_metrics": metrics(run["dataset"], seed=run["seed"]),
        "selected_checkpoint_metrics": metrics(run["dataset"], seed=run["seed"], adapted=True),
        "final_checkpoint_metrics": metrics(run["dataset"], seed=run["seed"], adapted=True),
        "evaluation": {"selected_epoch": 2, "final_epoch": run["config"]["foundation_model"].get("epochs", 100)},
        "runtime": {"total_seconds": 30 + run["seed"]},
    }


def transfer_report(run):
    route = resolve_transfer_route(run["config"])
    report = base_report(run)
    if route.execution_mode == "train":
        stages = []
        for stage in route.stages:
            baseline = metrics(route.target_dataset, seed=run["seed"])
            selected = metrics(route.target_dataset, seed=run["seed"], adapted=True)
            raw = {
                "metrics": selected,
                "zero-shot-metrics": baseline,
                "score_metric": metric_for(route.target_dataset),
                "checkpoint_selection": {"selected_step": stage.update_budget},
                "checkpoint_view_metrics": {
                    str(n): {"metrics": selected, "baseline_metrics": baseline} for n in stage.view_boundaries
                },
                "checkpoint_view_selection": {str(n): {"selected_step": n} for n in stage.view_boundaries},
            }
            checkpoint = Path("synthetic-checkpoint.pt")
            stages.append(
                StageResult(stage, StageProduction(checkpoint, raw, {n: checkpoint for n in stage.view_boundaries}))
            )
        report.update(training_report(route, tuple(stages)))
    else:
        # Same public schema as evaluate_frozen_checkpoint_bundle; no real weights required.
        evaluations = [
            {
                "target": target,
                "route": route.dependency.route_id,
                "score_metric": metric_for(target),
                "checkpoint_path": "synthetic-checkpoint.pt",
                "zero_shot_metrics": metrics(target, seed=run["seed"]),
                "selected_checkpoint_metrics": metrics(target, seed=run["seed"], adapted=True),
                "axes": {"contextPolicy": route.context_policy},
            }
            for target in route.evaluation_targets
        ]
        report.update(
            dataset_transfer={
                "route": route.route_id,
                "family": route.family,
                "execution_mode": "checkpoint_eval",
                "context_policy": route.context_policy,
                "checkpoint_route": route.dependency.route_id,
                "checkpoint_origin": route.dependency.checkpoint_origin,
            },
            evaluations=evaluations,
        )
        report.update(
            {key: evaluations[0][key] for key in ("score_metric", "zero_shot_metrics", "selected_checkpoint_metrics")}
        )
    return report


def transformation_report(run):
    report = base_report(run)
    preprocess = copy.deepcopy(run["config"].get("preprocess", {}))
    pre, post = {}, {}
    for step in preprocess.get("steps", []):
        payload = copy.deepcopy(preprocess.get(step, {}))
        if step == "subsample_train":
            payload["requested"] = payload["size"]
        elif step == "contextual_scale":
            payload["scale_spread"] = 3
            payload["transformation_setting"] = (
                "no_transformation" if payload["condition"] == "scale_1x_no_unit_column" else "matched_transformation"
            )
        elif step == "heavy_tail_contamination":
            payload["rate"] = 0 if run["parameters"]["heavyTailContamination"] == "original" else 0.05
        elif step == "rebalance_train":
            payload.update(requested_target_focal_rate=payload["target_focal_rate"], focal_label=0, other_label=1)
        if step in {"numeric_scale", "shuffle_feature_order"}:
            post[step] = payload
            if "fit_state" in payload:
                original, transformed = (
                    ("canonical", "permuted") if step == "shuffle_feature_order" else ("raw", "standardized")
                )
                report["distribution_shift_test_views"] = {
                    "enabled": True,
                    "same_fitted_state": True,
                    "fit_state": payload["fit_state"],
                    "primary_view": original,
                    "alternate_view": transformed,
                    **{
                        output: {view: report[input_] for view in (original, transformed)}
                        for output, input_ in (
                            ("baseline_metrics", "zero_shot_metrics"),
                            ("selected_checkpoint_metrics", "selected_checkpoint_metrics"),
                            ("final_checkpoint_metrics", "final_checkpoint_metrics"),
                        )
                    },
                }
        else:
            pre[step] = payload
    report["data_task"]["distribution_shift"] = {"pre_finite": pre, "post_finite": post}
    if run["study"].endswith("heavy_tail_diamonds"):
        report["epoch_evaluations"] = [
            {
                "split": "val",
                "epoch": epoch,
                "eval_source": "zero_shot_initial" if epoch == 0 else "in_memory_validation",
                "rmse": 1.5 if epoch == 0 else 1.1 + 1 / (epoch + 1),
            }
            for epoch in (0, 1, 10, 50)
        ]
    if run["study"].endswith("class_prior_context_only_downsampling"):
        full = "full_original_for_test_icl"
        balanced = "balanced_original_majority_downsampled_for_test_icl"
        report["test_context_comparison"] = {
            "policies": {balanced: {"majority_downsampling": {"focal_label": 0, "other_label": 1}}},
            "baseline_metrics": {policy: report["zero_shot_metrics"] for policy in (full, balanced)},
            "selected_checkpoint_metrics": {
                policy: report["selected_checkpoint_metrics"] for policy in (full, balanced)
            },
        }
    return report


def materialize(root, studies):
    for study in studies:
        recipe = read_yaml_mapping(ROOT / "configs/experiments" / f"{study}.yaml")
        for run in expand_runs(recipe, repo_root=ROOT):
            report = (
                transfer_report(run)
                if study.startswith("dataset_transfer")
                else transformation_report(run)
                if study.startswith("distribution_shift")
                else base_report(run)
            )
            execute_run(run, root, repo_root=ROOT, trainer=lambda config, output, *, force, report=report: report)


def group_config(tmp_path, group):
    config = tomllib.loads((ROOT / MANIFEST["groups"][group]["config"]).read_text())
    config["input"]["results_root"] = str(tmp_path / "results")
    for key in config["output"]:
        config["output"][key] = str(tmp_path / ("assets" if "thesis" in key else "analysis"))
    path = tmp_path / f"{group}.toml"
    path.write_text(tomli_w.dumps(config))
    return path


def assert_assets(tmp_path, group):
    for asset in MANIFEST["groups"][group]["assets"]:
        path = tmp_path / "assets" / Path(asset).name
        assert path.is_file(), path.name
        assert path.stat().st_size > 100
        if path.suffix == ".pdf":
            assert path.read_bytes().startswith(b"%PDF")


@pytest.mark.parametrize("group", ["transformations", "transfer", "natural_domains", "retention", "mitra_pipeline"])
def test_remaining_groups_render_from_current_public_run_contract(tmp_path, group):
    import importlib

    materialize(tmp_path / "results", MANIFEST["groups"][group]["studies"])
    config = group_config(tmp_path, group)
    module, entrypoint = MANIFEST["groups"][group]["entrypoint"].split(":")
    getattr(importlib.import_module(module), entrypoint)(config_path=config)
    assert_assets(tmp_path, group)
