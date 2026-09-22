"""Scientific grid expansion, execution status, and local/SLURM parity."""

import copy
import json
from pathlib import Path

import pytest

from experiments.configuration import expand_runs
from experiments.execution.bundle import write_bundle
from experiments.execution.catalog import load_catalog
from experiments.execution.planning import build_plan
from experiments.run_experiment import execute_run
from utils.config_io import read_yaml_mapping

ROOT = Path(__file__).resolve().parents[1]


def recipe(name="distribution_shift_label_noise"):
    return read_yaml_mapping(ROOT / "configs/experiments" / f"{name}.yaml")


def test_label_noise_exact_protocol():
    runs = expand_runs(recipe(), repo_root=ROOT)
    assert len(runs) == 180
    assert {r["seed"] for r in runs} == {0, 1, 2}
    assert {r["fold"] for r in runs} == {0}
    assert {r["config"]["preprocess"]["label_noise"]["rate"] for r in runs} == {0, 0.05, 0.15, 0.3}
    for run in runs:
        settings = run["config"]["foundation_model"]
        assert settings["final_eval_checkpoint"] == "best"
        assert settings["validation_split_seed"] == 0
        assert settings["test_context_includes_validation_data"] is True
        assert settings["preprocessing_protocol"] == "shared"


def test_small_datasets_have_three_folds_without_changing_training_seeds():
    runs = expand_runs(
        recipe("distribution_shift_feature_order"), repo_root=ROOT, datasets=["maternal-health-risk"], models=["mitra"]
    )
    assert len(runs) == 18
    assert {(r["fold"], r["seed"]) for r in runs} == {(f, s) for f in range(3) for s in range(3)}
    assert all(r["config"]["seed"] == r["seed"] for r in runs)


def test_hpo_removes_only_saturated_context_duplicates():
    runs = expand_runs(
        recipe("foundation_regular_hpo_representative"), repo_root=ROOT, datasets=["anneal"], models=["mitra"]
    )
    assert len(runs) == 81  # 3 folds * 3 seeds * 3 learning rates * 3 weight decays
    assert {r["config"]["foundation_model"]["context_size"] for r in runs} == {5000}
    large = expand_runs(
        recipe("foundation_regular_hpo_representative"), repo_root=ROOT, datasets=["covertype"], models=["mitra"]
    )
    assert {r["config"]["foundation_model"]["context_size"] for r in large} == {5000, 10000, 20000}


def test_alignment_and_update_only_are_separate_experiments():
    for name, transform in [("feature_order", "shuffle_feature_order"), ("numeric_scale", "numeric_scale")]:
        for suffix, field in [("", "update_state"), ("_alignment", "fit_state")]:
            runs = expand_runs(recipe(f"distribution_shift_{name}{suffix}"), repo_root=ROOT)
            assert len(runs) == 162
            assert all(field in r["config"]["preprocess"][transform] for r in runs)


def test_filter_order_does_not_change_run_identity():
    all_runs = expand_runs(recipe(), repo_root=ROOT)
    subset = expand_runs(recipe(), repo_root=ROOT, datasets=["jm1"], models=["mitra"])
    assert {r["run_id"] for r in subset} == {
        r["run_id"] for r in all_runs if r["dataset"] == "jm1" and r["model"] == "mitra"
    }
    with pytest.raises(ValueError, match="Unknown datasets"):
        expand_runs(recipe(), repo_root=ROOT, datasets=["typo"])


def test_resume_and_failure_status(tmp_path):
    run = expand_runs(recipe(), repo_root=ROOT, datasets=["jm1"], models=["mitra"])[0]
    calls = []

    def train(config, output, force=False):
        calls.append(config)
        return {"schema_version": 1, "config": config, "zero_shot_metrics": {"test": {"roc_auc": 0.7}}}

    output = execute_run(run, tmp_path, repo_root=ROOT, trainer=train)
    assert json.loads((output / "run.json").read_text())["status"] == "complete"
    execute_run(run, tmp_path, repo_root=ROOT, trainer=train)
    assert len(calls) == 1

    def fail(*args, **kwargs):
        raise RuntimeError("test training failure")

    with pytest.raises(RuntimeError, match="test training failure"):
        execute_run(run, tmp_path, repo_root=ROOT, trainer=fail, force=True)
    assert json.loads((output / "run.json").read_text())["status"] == "failed"
    with pytest.raises(RuntimeError, match="Incomplete run"):
        execute_run(run, tmp_path, repo_root=ROOT, trainer=train)


def test_all_recipes_local_slurm_same_grid(tmp_path):
    cluster = read_yaml_mapping(ROOT / "configs/slurm/hydra_cluster.yaml")
    for name in load_catalog(ROOT):
        local = build_plan(ROOT, name, None, tmp_path / name)
        remote = build_plan(ROOT, name, cluster, tmp_path / name)
        assert [(j["id"], j["runs"]) for j in local["jobs"]] == [
            (j["id"], j["runs"]) for j in remote["jobs"]
        ]
        actual = [run for group in local["jobs"] for run in group["runs"]]
        assert actual == expand_runs(recipe(name), repo_root=ROOT)


def test_filtered_bundle_runs_only_selection(tmp_path):
    plan = build_plan(
        ROOT,
        "unified_foundation_baseline",
        None,
        tmp_path / "launch",
        datasets=["churn"],
        models=["tabpfnv25"],
        preprocessing=["shared"],
    )
    assert len(plan["jobs"]) == 1
    job = plan["jobs"][0]
    assert len(job["runs"]) == 3
    assert job["datasets"] == job["result_datasets"] == ["churn"]
    write_bundle(plan)
    saved = json.loads((Path(plan["output"]) / "plan.json").read_text())
    runs = saved["jobs"][0]["runs"]
    assert len(runs) == 3 and {r["preprocessing"] for r in runs} == {"shared"}
    assert "script_path" not in job


def test_duplicate_groups_rejected():
    config = recipe()
    config["groups"].append(copy.deepcopy(config["groups"][0]))
    with pytest.raises(ValueError, match="Duplicate scientific run"):
        expand_runs(config, repo_root=ROOT)


def test_completed_transfer_producer_requires_its_weight_files(tmp_path):
    run = expand_runs(recipe("dataset_transfer"), repo_root=ROOT, datasets=["credit-g"], models=["mitra"])[0]

    def train(config, output, force=False):
        checkpoint = output / "selected.ckpt"
        checkpoint.write_text("weights")
        return {
            "schema_version": 1,
            "config": config,
            "dataset_transfer": {"execution_mode": "train"},
            "stage_checkpoints": {"target": str(checkpoint)},
        }

    output = execute_run(run, tmp_path, repo_root=ROOT, trainer=train)
    (output / "selected.ckpt").unlink()
    with pytest.raises(RuntimeError, match="weights are missing.*--force"):
        execute_run(run, tmp_path, repo_root=ROOT, trainer=train)
    execute_run(run, tmp_path, repo_root=ROOT, trainer=train, force=True)
    assert (output / "selected.ckpt").is_file()
