"""Submit intact experiment runs through Submitit's local or SLURM executor.

Example: ``jobs = submit_plan(plan)`` after ``write_bundle(plan)``.
"""

from __future__ import annotations

import json
import os
import shlex
from functools import partial
from pathlib import Path

import submitit

from tfm_finetuning.experiments.run_experiment import execute_run
from tfm_finetuning.utils.logging_utils import raise_noisy_library_log_levels


def run_one(run: dict, *, repo_root: str, results_root: str, force=False, cuda_devices=None) -> str:
    """Run one resolved configuration in the worker, without index-based selection."""
    root = Path(repo_root)
    os.chdir(root)
    os.environ.update(REPO_ROOT=str(root), RESULTS_ROOT=results_root)
    if cuda_devices is not None:
        # LocalExecutor normally chooses GPU 0; preserve explicit caller visibility.
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    raise_noisy_library_log_levels()
    return str(execute_run(run, Path(results_root), repo_root=root, force=force))


def make_executor(plan: dict, group: dict) -> submitit.AutoExecutor:
    """Configure Submitit; selecting a backend never falls back to a different one."""
    parameters = {"name": f"tfm-{group['id']}", **group["resources"]["parameters"]}
    for key in ("nodes", "tasks_per_node", "slurm_nodes", "slurm_ntasks_per_node"):
        if parameters.get(key, 1) != 1:
            raise ValueError("Experiment runs require one process on one node; use job arrays for parallelism")
    backend = plan["backend"]
    executor = submitit.AutoExecutor(
        folder=Path(plan["output"]) / group["id"] / "%j",
        cluster=backend,
        **{f"{backend}_python": shlex.join(plan["python"])},
    )
    try:
        # Without an explicit task count, a site's CPU allocation can spawn duplicate workers.
        executor.update_parameters(**{"nodes": 1, "tasks_per_node": 1, **parameters})
    except (NameError, AssertionError) as exc:
        raise ValueError(str(exc)) from exc
    return executor


def submit_plan(plan: dict) -> list[submitit.Job]:
    """Submit SLURM arrays or wait for local jobs one at a time; return their handles.

    Submitit owns processes, arrays, logs and results. Job IDs are also saved so
    a later submission failure does not hide already submitted work.
    """
    output = Path(plan["output"])
    if not (output / "plan.json").is_file():
        raise ValueError("Write the launch plan before submitting it")
    local = plan["backend"] == "local"
    worker = partial(
        run_one,
        repo_root=plan["repo_root"],
        results_root=plan["results_root"],
        force=plan["force"],
        cuda_devices=os.environ.get("CUDA_VISIBLE_DEVICES") if local else None,
    )
    # Validate every group's executor parameters before submitting any jobs.
    executors = [(group, make_executor(plan, group)) for group in plan["jobs"]]
    jobs, records = [], []

    def record(job, run, group):
        jobs.append(job)
        records.append(
            {
                "group": group["id"],
                "run_id": run["run_id"],
                "job_id": job.job_id,
                "folder": str(job.paths.folder),
                "stdout": str(job.paths.stdout),
                "stderr": str(job.paths.stderr),
            }
        )

    def save_records():
        (output / "jobs.json").write_text(json.dumps(records, indent=2) + "\n")

    try:
        for group, executor in executors:
            if local:
                for run in group["runs"]:
                    job = executor.submit(worker, run)
                    record(job, run, group)
                    save_records()
                    print(f"Local Submitit job {job.job_id}: {job.paths.stdout}", flush=True)
                    try:
                        job.result()
                    except BaseException:
                        job.cancel()
                        raise
            else:
                submitted = executor.map_array(worker, group["runs"])
                for job, run in zip(submitted, group["runs"], strict=True):
                    record(job, run, group)
                save_records()
                print(f"Submitted {len(submitted)} jobs for {group['id']}; {output / group['id']}", flush=True)
    except Exception as exc:
        raise RuntimeError(f"Submitit execution failed; job IDs and logs are recorded in {output}: {exc}") from exc
    return jobs
