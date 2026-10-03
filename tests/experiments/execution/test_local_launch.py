"""Exercise real Submitit local workers without model or dataset downloads."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tfm_finetuning.experiments.execution import launch, submission
from tfm_finetuning.experiments.execution.bundle import write_bundle
from tfm_finetuning.experiments.execution.planning import build_plan
from tfm_finetuning.experiments.execution.submission import run_one, submit_plan

ROOT = Path(__file__).resolve().parents[3]


def local_plan(tmp_path, **kwargs):
    return build_plan(ROOT, "unified_foundation_baseline", None, tmp_path / "bundle with spaces",
                      datasets=["splice"], models=["mitra"], preprocessing=["native"],
                      results_root=tmp_path / 'results', **kwargs)


def completed_results(plan):
    for group in plan['jobs']:
        for run in group['runs']:
            folder = Path(plan['results_root']) / run['study'] / run['run_id']
            folder.mkdir(parents=True)
            (folder / 'run.json').write_text(json.dumps({**run, 'status': 'complete'}))
            (folder / 'finetune_report.json').write_text('{}')


def test_local_preview_needs_no_cluster_or_executor(tmp_path, monkeypatch, capsys):
    original = launch.read_yaml_mapping
    def no_cluster(path):
        assert 'cluster' not in str(path)
        return original(path)
    def unexpected(*args, **kwargs):
        raise AssertionError('Preview created an executor')
    monkeypatch.setattr(launch, 'read_yaml_mapping', no_cluster)
    monkeypatch.setattr(submission, 'make_executor', unexpected)
    output = tmp_path / 'out'
    assert launch.main(['unified_foundation_baseline', '--local', '--dataset', 'splice', '--model', 'mitra',
                        '--output', str(output), '--json']) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan['python'] == [sys.executable]
    assert plan['backend'] == 'local'
    assert plan['jobs'][0]['runs']
    assert not output.exists()


@pytest.mark.parametrize('options', [['--run'], ['--local', '--submit'], ['--local', '--cluster', 'unused.yaml']])
def test_cli_rejects_mixed_execution_modes(options):
    with pytest.raises(SystemExit):
        launch.main(['unified_foundation_baseline', *options])


def test_local_submitit_jobs_reuse_completed_results_from_another_directory(tmp_path, monkeypatch):
    plan = local_plan(tmp_path)
    completed_results(plan)
    write_bundle(plan)
    monkeypatch.chdir(tmp_path)
    jobs = submit_plan(plan)
    assert len(jobs) == 3
    records = json.loads((Path(plan['output']) / 'jobs.json').read_text())
    for job, record, run in zip(jobs, records, plan['jobs'][0]['runs'], strict=True):
        expected = str(Path(plan['results_root']) / run['study'] / run['run_id'])
        assert job.result() == expected
        assert record['run_id'] == run['run_id']
        assert record['job_id'] == job.job_id
        assert f'Completed: {run["run_id"]}' in Path(record['stdout']).read_text()


def test_local_failure_stops_before_later_runs(tmp_path):
    plan = local_plan(tmp_path)
    completed_results(plan)
    failing = plan['jobs'][0]['runs'][1]
    marker = Path(plan['results_root']) / failing['study'] / failing['run_id'] / 'run.json'
    marker.write_text(json.dumps({'status': 'failed'}))
    write_bundle(plan)
    with pytest.raises(RuntimeError, match='Incomplete run'):
        submit_plan(plan)
    records = json.loads((Path(plan['output']) / 'jobs.json').read_text())
    assert len(records) == 2
    assert 'Incomplete run' in Path(records[-1]['stderr']).read_text()


def test_worker_preserves_explicit_local_gpu_visibility(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for key in ('REPO_ROOT', 'RESULTS_ROOT', 'CUDA_VISIBLE_DEVICES'):
        monkeypatch.setenv(key, 'before')
    def execute(run, results_root, *, repo_root, force):
        assert Path.cwd() == ROOT
        assert os.environ['REPO_ROOT'] == str(ROOT)
        assert os.environ['RESULTS_ROOT'] == str(tmp_path)
        assert os.environ['CUDA_VISIBLE_DEVICES'] == 'GPU-selected-by-user'
        assert run == {'run_id': 'test'} and force
        return results_root / 'test'
    monkeypatch.setattr(submission, 'execute_run', execute)
    assert run_one({'run_id': 'test'}, repo_root=str(ROOT), results_root=str(tmp_path), force=True,
                   cuda_devices='GPU-selected-by-user') == str(tmp_path / 'test')


@pytest.mark.parametrize("local", [False, True])
def test_wrapper_selects_local_python_even_when_apptainer_is_available(tmp_path, local):
    root = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts", root / "scripts")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    envbin = tmp_path / "env/bin"
    envbin.mkdir(parents=True)
    for path, label in [(bindir / "apptainer-python", "container"), (envbin / "python", "local")]:
        path.write_text(f'#!/bin/bash\nprintf "%s\\n" "{label}" "$@"\n')
        path.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k != "TFM_PYTHON"}
    env.update(VIRTUAL_ENV=str(envbin.parent), PATH=f"{bindir}:{env['PATH']}")
    args = ["--local"] if local else []
    result = subprocess.run(
        ["bash", str(root / "scripts/experiments/transformations/feature_order.sh"), *args, "--dry-run"],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.splitlines()[0] == "local"
    assert "distribution_shift_feature_order\n" in result.stdout
    assert ("--local\n" in result.stdout) == local
