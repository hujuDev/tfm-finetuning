"""Notebook support for the same experiment CLI used by local and cluster runs."""

from __future__ import annotations

import json
import os
import signal
import subprocess
from collections.abc import Sequence
from pathlib import Path

from tfm_finetuning.utils.paths import get_repo_root


def run_command(argv: Sequence[str]) -> None:
    """Stream a CLI command from the checkout and stop it when interrupted.

    For example, ``run_command([sys.executable, '-m',
    'tfm_finetuning.experiments.run_experiment', '--config', str(config_path), '--dry-run'])``
    previews an experiment using the notebook's Python environment.
    """
    with subprocess.Popen(
        argv,
        cwd=get_repo_root(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=os.name == "posix",
    ) as process:
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, argv)
        except KeyboardInterrupt:
            if process.poll() is None:
                try:
                    # Also stop child training processes on Unix.
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGTERM)
                    else:
                        process.terminate()
                except ProcessLookupError:
                    pass
            raise


def load_comparison(runs: Sequence[dict], results: Path):
    """Tabulate original, validation-selected and last-checkpoint scores for completed runs.

    For example, ``load_comparison(expand_runs(config, repo_root=root), results)`` returns one
    row per run and split, and links each W&B run when tracking was enabled.
    """
    import pandas as pd
    from IPython.display import Markdown, display

    records = []
    for run in runs:
        folder = results / run["study"] / run["run_id"]
        state_path = folder / "run.json"
        report_path = folder / "finetune_report.json"
        if not state_path.is_file() or not report_path.is_file():
            raise RuntimeError(f"Run training first. Results are missing in {folder}.")
        if json.loads(state_path.read_text())["status"] != "complete":
            raise RuntimeError(f"This run has not completed. Inspect {state_path}.")
        report = json.loads(report_path.read_text())
        metric = report["score_metric"]
        for split in ("val", "test"):
            records.append(
                {
                    "preprocessing": run["preprocessing"],
                    "seed": run["seed"],
                    "split": split,
                    "metric": metric,
                    "ICL": report["zero_shot_metrics"][split][metric],
                    "validation-selected": report["selected_checkpoint_metrics"][split][metric],
                    "last checkpoint": report["final_checkpoint_metrics"][split][metric],
                    "selected epoch": report["evaluation"]["selected_epoch"],
                }
            )
        if report.get("wandb", {}).get("url"):
            display(Markdown(f"[W&B run]({report['wandb']['url']})"))
    return pd.DataFrame(records)
