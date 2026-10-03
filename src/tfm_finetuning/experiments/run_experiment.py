"""Run the fixed thesis experiment grids sequentially in the current process.

Example: ``python -m tfm_finetuning.experiments.run_experiment --config CONFIG --dry-run``.
Every completed run owns one directory with its config, metrics, and checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

from tfm_finetuning.experiments.configuration import expand_runs
from tfm_finetuning.utils.config import read_yaml_mapping
from tfm_finetuning.utils.logging_utils import raise_noisy_library_log_levels
from tfm_finetuning.utils.paths import get_paths, get_repo_root


def write_run(path: Path, payload: dict) -> None:
    """Atomically publish the current run status, e.g. ``status='complete'``."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def execute_run(run: dict, results_root: Path, *, repo_root: Path, force: bool = False, trainer=None) -> Path:
    """Execute one concrete run; completed reports are resumable without remote caches."""
    output = results_root / run["study"] / run["run_id"]
    output.mkdir(parents=True, exist_ok=True)
    # A filesystem lock prevents two local/SLURM workers from training the same run.
    import fcntl

    with (output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = output / "run.json"
        if marker.exists():
            previous = json.loads(marker.read_text())
            if previous.get("status") == "complete" and (output / "finetune_report.json").is_file() and not force:
                report = json.loads((output / "finetune_report.json").read_text())
                transfer = report.get("dataset_transfer", {})
                if transfer.get("execution_mode") == "train":
                    from tfm_finetuning.experiments.transfer.runner.executor import require_checkpoint

                    checkpoints = report.get("stage_checkpoints", {})
                    if not checkpoints:
                        raise RuntimeError(
                            f"Transfer producer has no checkpoint records at {output}; use --force to rerun"
                        )
                    for raw in checkpoints.values():
                        checkpoint = Path(raw)
                        try:
                            require_checkpoint(checkpoint if checkpoint.is_absolute() else output / checkpoint)
                        except (OSError, ValueError) as exc:
                            raise RuntimeError(
                                f"Transfer producer weights are missing at {output}; use --force to rerun"
                            ) from exc
                print(f"Completed: {run['run_id']}", flush=True)
                return output
            if not force:
                raise RuntimeError(f"Incomplete run at {output}; inspect the error and use --force to retry")
        config = json.loads(json.dumps(run["config"]))
        config["results_root"] = str(results_root)
        config.setdefault("paths", {})["results_root"] = str(results_root)
        state = {**run, "config": config, "status": "running"}
        write_run(marker, state)
        try:
            if trainer is None:
                from tfm_finetuning.experiments.datasets import ensure_dataset
                from tfm_finetuning.finetuning.model_finetune import main as trainer

                path = ensure_dataset(run["dataset"], config, repo_root=repo_root)
                config["data"]["path"] = str(path)
                state["config"] = config
                write_run(marker, state)
            result = trainer(config, output, force=force)
            if not (output / "finetune_report.json").is_file():
                if not isinstance(result, dict):
                    raise RuntimeError("Training returned without a metrics report")
                write_run(output / "finetune_report.json", result)
        except BaseException as exc:
            write_run(marker, {**state, "status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
            raise
        write_run(marker, {**state, "status": "complete"})
    return output


def main(argv=None) -> int:
    """Preview or execute the exact same grid expansion used by the launcher."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--model", action="append")
    parser.add_argument("--preprocessing", action="append", choices=["native", "shared"])
    parser.add_argument("--group")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    raise_noisy_library_log_levels()
    root = get_repo_root()
    os.chdir(root)
    recipe = read_yaml_mapping(args.config)
    runs = expand_runs(
        recipe,
        repo_root=root,
        datasets=args.dataset,
        models=args.model,
        preprocessing=args.preprocessing,
        group_name=args.group,
        smoke=args.smoke,
    )
    results_root = (args.results_root or get_paths(repo_root=root).results_root).resolve()
    if args.smoke:
        results_root /= "smoke"
    if args.dry_run:
        print(json.dumps({"results_root": str(results_root), "run_count": len(runs), "runs": runs}, indent=2))
        return 0
    for index, run in enumerate(runs, 1):
        print(f"[{index}/{len(runs)}] {run['run_id']}", flush=True)
        execute_run(run, results_root, repo_root=root, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
