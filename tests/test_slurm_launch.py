"""Resource selection and scheduler-boundary tests using local fixtures only."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.execution.bundle import write_bundle
from experiments.execution.catalog import load_catalog
from experiments.execution.planning import build_plan
from experiments.execution.resources import resolve_resources
from experiments.execution.submission import submit_plan
from experiments.launch import main
from utils.config_io import read_yaml_mapping

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cluster():
    return read_yaml_mapping(ROOT / "configs/slurm/hydra_cluster.yaml")


def plan_for(tmp_path, cluster, name="distribution_shift_feature_order", **kwargs):
    return build_plan(ROOT, name, cluster, tmp_path / "bundle with spaces", **kwargs)


def test_every_pinned_experiment_has_a_resource_plan_and_wrapper(tmp_path, cluster):
    # Execute every wrapper with a fake Python command, including from another
    # working directory and a checkout path containing spaces.
    root = tmp_path / "repo with spaces"
    shutil.copytree(ROOT / "scripts", root / "scripts")
    capture = tmp_path / "capture-python"
    capture.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    capture.chmod(0o755)
    wrappers = []
    for script in sorted((root / "scripts/experiments").rglob("*.sh")):
        result = subprocess.run(
            ["bash", str(script), "--local", "--dataset", "dataset with spaces", "--dry-run"],
            cwd=tmp_path, env={**os.environ, "TFM_PYTHON": str(capture)},
            text=True, capture_output=True, check=True,
        )
        args = result.stdout.splitlines()
        assert args[:4] == ["-m", "experiments.launch", "--repo-root", str(root)]
        assert args[5:] == ["--local", "--dataset", "dataset with spaces", "--dry-run"]
        wrappers.append(args[4])
    assert len(wrappers) == len(set(wrappers))
    assert set(wrappers) == set(load_catalog(ROOT))
    for name in wrappers:
        plan = plan_for(tmp_path, cluster, name)
        assert plan["jobs"]
        assert all(len(job["runs"]) > 0 and job["resource_rule"] for job in plan["jobs"])
        assert all(job["resources"]["parameters"]["gpus_per_node"] == 1 for job in plan["jobs"])
    assert not (tmp_path / "bundle with spaces").exists()


def test_baseline_includes_main_panel_addendum_and_all_four_models(tmp_path, cluster):
    plan = plan_for(tmp_path, cluster, "unified_foundation_baseline")
    jobs = plan["jobs"]
    assert len({d for job in jobs for d in job["datasets"]}) == 57
    assert {m for job in jobs for m in job["models"]} == {"tabpfnv2", "tabpfnv25", "tabiclv2", "mitra"}
    assert sum(len(job["runs"]) for job in jobs) == 2184
    assert all(run["config"]["foundation_model"]["test_context_includes_validation_data"] for job in jobs for run in job["runs"])
    assert not any("v15" in job["resource_rule"] for job in jobs)
    v2 = [job for job in jobs if job["models"] == ["tabpfnv2"]]
    assert all(run["model"] == "tabpfnv2" for job in v2 for run in job["runs"])


@pytest.mark.parametrize(
    ("dataset", "profile"),
    [
        ("maternal_health_risk", "gpu_any_5h"),
        ("splice", "gpu_80gb_2d"),
    ],
)
def test_dataset_selects_recorded_hardware(tmp_path, cluster, dataset, profile):
    plan = plan_for(tmp_path, cluster, datasets=[dataset], models=["mitra"])
    assert len(plan["jobs"]) == 1
    assert plan["jobs"][0]["datasets"] == [dataset]
    assert plan["jobs"][0]["resources"]["profile"] == profile


def test_transfer_resource_choice_depends_on_route_not_just_target(tmp_path, cluster):
    plan = plan_for(tmp_path, cluster, "dataset_transfer", datasets=["satimage"], models=["mitra"])
    assert {j["resources"]["profile"] for j in plan["jobs"]} == {"gpu_any_5h", "gpu_80gb_5h_mem64g"}
    large = next(j for j in plan["jobs"] if j["resources"]["profile"].endswith("mem64g"))
    assert large["resources"]["parameters"]["mem_gb"] == 64
    assert "sequential_covertype_then_target" in [r["parameters"]["transferRoute"] for r in large["runs"]]


@pytest.mark.parametrize("kwargs", [{"datasets": ["splice", "typo"]}, {"models": ["mitra", "typo"]}])
def test_unknown_filters_fail_instead_of_dropping_requests(tmp_path, cluster, kwargs):
    with pytest.raises(ValueError):
        plan_for(tmp_path, cluster, **kwargs)


def test_missing_resource_mapping_fails_before_any_output(tmp_path, cluster):
    cluster["profiles"].pop("gpu_80gb_2d")
    with pytest.raises(ValueError, match="must exist"):
        plan_for(tmp_path, cluster, datasets=["splice"])
    assert not list(tmp_path.iterdir())


def test_cluster_mapping_keeps_requirements_but_changes_site_names(cluster):
    cluster["profiles"]["gpu_80gb_5h_mem64g"]["slurm_partition"] = "accelerated"
    cluster["profiles"]["gpu_80gb_5h_mem64g"]["slurm_constraint"] = "large-vram"
    resources = read_yaml_mapping(ROOT / "configs/resources.yaml")
    resolved = resolve_resources("gpu_80gb_5h_mem64g", resources, cluster)
    assert resolved["parameters"] == {
        "gpus_per_node": 1, "timeout_min": 300, "mem_gb": 64,
        "slurm_partition": "accelerated", "slurm_constraint": "large-vram",
        "slurm_array_parallelism": 8,
    }


@pytest.fixture
def scheduler(tmp_path, monkeypatch):
    # Only the scheduler boundary is fake: Submitit writes real scripts/pickles.
    bindir = tmp_path / "scheduler"
    bindir.mkdir()
    calls = tmp_path / "scheduler-calls.json"
    fail = tmp_path / "fail-second"
    script = bindir / "sbatch"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\nfrom pathlib import Path\n"
        f"p = Path({str(calls)!r})\n"
        "calls = json.loads(p.read_text()) if p.exists() else []\n"
        "calls.append(Path(sys.argv[1]).read_text())\n"
        "p.write_text(json.dumps(calls))\n"
        f"if Path({str(fail)!r}).exists() and len(calls) > 1: sys.exit(9)\n"
        "print(f'Submitted batch job {12344 + len(calls)}')\n"
    )
    script.chmod(0o755)
    srun = bindir / "srun"
    srun.write_text("#!/bin/sh\nexit 1\n")
    srun.chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep + os.environ["PATH"])
    return calls, fail


def test_submitit_serializes_each_selected_run_once(tmp_path, cluster, scheduler):
    import cloudpickle

    cluster['parameters']['slurm_array_parallelism'] = 2
    cluster['python'] = ['/custom path/python-wrapper', '--site-flag']
    plan = plan_for(tmp_path, cluster, datasets=['splice'], models=['mitra'])
    write_bundle(plan)
    jobs = submit_plan(plan)
    expected = plan['jobs'][0]['runs']
    assert len(jobs) == len(expected)
    calls = json.loads(scheduler[0].read_text())
    assert len(calls) == 1
    assert f'--array=0-{len(expected)-1}%2' in calls[0]
    assert '--constraint=80gb' in calls[0]
    assert '--time=2880' in calls[0]
    assert '--nodes=1' in calls[0]
    assert '--ntasks-per-node=1' in calls[0]
    assert "'/custom path/python-wrapper' --site-flag -u -m submitit.core._submit" in calls[0]
    for job, run in zip(jobs, expected, strict=True):
        with job.paths.submitted_pickle.open('rb') as handle:
            submitted = cloudpickle.load(handle)
        assert submitted.args == (run,)
        assert submitted.function.keywords['results_root'] == plan['results_root']
        assert submitted.function.keywords['cuda_devices'] is None
    records = json.loads((Path(plan['output']) / 'jobs.json').read_text())
    assert [r['job_id'] for r in records] == [job.job_id for job in jobs]
    assert [r['run_id'] for r in records] == [r['run_id'] for r in expected]
    with pytest.raises(FileExistsError):
        write_bundle(plan)


def test_default_preview_does_not_call_scheduler_or_write_files(tmp_path, capsys, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Preview called an external command")

    monkeypatch.setattr(subprocess, "run", unexpected)
    assert main(["distribution_shift_feature_order", "--dataset", "splice", "--output", str(tmp_path / "out")]) == 0
    assert "gpu_80gb_2d" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


def test_cli_requires_explicit_cluster_for_submission():
    with pytest.raises(SystemExit):
        main(["distribution_shift_feature_order", "--submit"])


def test_failed_submission_preserves_previous_job_ids(tmp_path, cluster, scheduler):
    scheduler[1].touch()
    plan = plan_for(tmp_path, cluster, datasets=['splice', 'maternal_health_risk'], models=['mitra'])
    assert len(plan['jobs']) == 2
    write_bundle(plan)
    with pytest.raises(RuntimeError, match='job IDs and logs are recorded'):
        submit_plan(plan)
    records = json.loads((Path(plan['output']) / 'jobs.json').read_text())
    assert len(json.loads(scheduler[0].read_text())) == 2
    assert [r['run_id'] for r in records] == [r['run_id'] for r in plan['jobs'][0]['runs']]


def test_invalid_submitit_parameters_fail_before_any_submission(tmp_path, cluster, scheduler):
    cluster['parameters']['slurm_parameter_typo'] = 1
    plan = plan_for(tmp_path, cluster, datasets=['splice'])
    write_bundle(plan)
    with pytest.raises(ValueError):
        submit_plan(plan)
    assert not scheduler[0].exists()


def test_same_dataset_keeps_experiment_specific_resource_rule(tmp_path, cluster):
    baseline = plan_for(tmp_path, cluster, "unified_foundation_baseline", datasets=["splice"], models=["mitra"])
    transformed = plan_for(tmp_path, cluster, datasets=["splice"], models=["mitra"])
    assert baseline["jobs"][0]["resources"]["profile"] == "gpu_40gb_5h"
    assert transformed["jobs"][0]["resources"]["profile"] == "gpu_80gb_2d"


def test_unknown_preprocessing_is_not_silently_ignored(tmp_path, cluster):
    with pytest.raises(ValueError, match="Preprocessing must be native or shared"):
        plan_for(tmp_path, cluster, "unified_foundation_baseline", preprocessing=["native", "typo"])


@pytest.mark.parametrize("key", ["nodes", "tasks_per_node", "slurm_nodes", "slurm_ntasks_per_node"])
def test_multi_process_runs_are_rejected_before_submission(tmp_path, cluster, scheduler, key):
    cluster["parameters"][key] = 2
    plan = plan_for(tmp_path, cluster, datasets=["splice"], models=["mitra"])
    write_bundle(plan)
    with pytest.raises(ValueError, match="one process on one node"):
        submit_plan(plan)
    assert not scheduler[0].exists()


def test_experiment_notebook_previews_and_submits_through_submitit(tmp_path, scheduler):
    """Execute the actual notebook cells against the fake scheduler, never training."""
    notebook = json.loads((ROOT / 'notebooks/hydra_experiments.ipynb').read_text())
    cells = {cell['id']: ''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code'}
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), str(ROOT / 'notebooks/hydra_experiments.ipynb'), 'exec')
    namespace = {}
    exec(cells['load-environment'], namespace)
    namespace['results'] = tmp_path / 'results'
    exec(cells['preview-experiment'], namespace)
    assert namespace['plan']['backend'] == 'local'
    assert namespace['EXPERIMENT_CONFIG'] == ROOT / 'configs/experiments/distribution_shift_label_noise.yaml'
    assert namespace['plan']['experiment'] == namespace['EXPERIMENT_CONFIG'].stem
    exec(cells['submit-experiment'], namespace)
    assert not Path(namespace['plan']['output']).exists()
    assert not scheduler[0].exists()
    preview = cells['preview-experiment'].replace('BACKEND = "local"', 'BACKEND = "slurm"')
    exec(preview, namespace)
    assert namespace['plan']['backend'] == 'slurm'
    launch = cells['submit-experiment'].replace('SUBMIT = False', 'SUBMIT = True')
    exec(launch, namespace)
    jobs = namespace['jobs']
    assert len(jobs) == sum(len(group['runs']) for group in namespace['plan']['jobs'])
    assert json.loads(scheduler[0].read_text())
    calls_before = scheduler[0].read_text()
    with pytest.raises(FileExistsError):
        exec(launch, namespace)
    assert scheduler[0].read_text() == calls_before


def test_slurm_defaults_to_current_python(tmp_path, cluster):
    plan = plan_for(tmp_path, cluster, datasets=['splice'], models=['mitra'])
    assert plan['python'] == [sys.executable]


def test_copied_configuration_is_loaded_and_saved_with_launch(tmp_path):
    import yaml

    original = ROOT / 'configs/experiments/distribution_shift_label_noise.yaml'
    original_bytes = original.read_bytes()
    config = read_yaml_mapping(original)
    config['settings']['foundation_model.epochs'] = 7
    config['groups'][0]['datasets'].append('churn')
    copied = tmp_path / 'custom-experiment.yaml'
    copied.write_text(yaml.safe_dump(config))
    plan = build_plan(ROOT, copied.stem, None, tmp_path / 'launch', config_path=copied,
                      datasets=['churn'], models=['tabpfnv25'])
    runs = [run for group in plan['jobs'] for run in group['runs']]
    assert runs and all(run['dataset'] == 'churn' for run in runs)
    assert all(run['config']['foundation_model']['epochs'] == 7 for run in runs)
    write_bundle(plan)
    saved = json.loads((tmp_path / 'launch/plan.json').read_text())
    assert saved['config_path'] == str(copied)
    assert saved['experiment_config'] == config
    assert original.read_bytes() == original_bytes
