"""Preview, materialize, run locally, or submit a pinned thesis experiment.

Example: ``python -m experiments.execution.launch distribution_shift_feature_order --dataset splice --local``.
"""

from __future__ import annotations

import argparse
import json
import secrets
import shlex
from datetime import UTC, datetime
from pathlib import Path

from experiments.execution.bundle import write_bundle
from experiments.execution.catalog import load_catalog
from experiments.execution.planning import build_plan
from experiments.execution.submission import submit_plan
from utils.config import read_yaml_mapping


def parse_args(argv=None) -> argparse.Namespace:
    """Parse explicit execution intent; the default action only prints a plan."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("experiment", nargs="?")
    parser.add_argument("--list", action="store_true", help="List pinned runnable experiment recipes")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--cluster", type=Path, help="Cluster mapping; required for submission")
    parser.add_argument("--local", action="store_true", help="Use this Python environment without SLURM")
    parser.add_argument("--dataset", action="append", default=[], help="Dataset name; repeat to select multiple")
    parser.add_argument("--model", action="append", default=[], help="Model key, e.g. mitra, tabpfnv25, tabpfnv2")
    parser.add_argument("--preprocessing", action="append", default=[], help="Preprocessing axis key")
    parser.add_argument("--scope", choices=("full", "smoke"), default="full")
    parser.add_argument("--force", action="store_true", help="Retry incomplete runs or rerun completed runs")
    parser.add_argument("--output", type=Path, help="New directory for the launch plan and Submitit logs")
    parser.add_argument("--results-root", type=Path, help="Training results directory")
    parser.add_argument("--json", action="store_true", help="Print the complete plan as JSON")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true", help="Preview only (the default)")
    action.add_argument("--write", action="store_true", help="Write the launch bundle without launching")
    action.add_argument("--run", action="store_true", help="Run sequential local jobs through Submitit")
    action.add_argument(
        "--submit", action="store_true", help="Submit SLURM arrays through Submitit"
    )
    args = parser.parse_args(argv)
    if not args.list and not args.experiment:
        parser.error("provide an experiment or --list")
    if args.local and (args.cluster or args.submit):
        parser.error("--local cannot be combined with --cluster or --submit")
    if args.run and not args.local:
        parser.error("--run requires --local; use --submit for SLURM")
    if args.submit and not args.cluster:
        parser.error("--submit requires an explicit --cluster configuration")
    return args


def show_plan(plan: dict) -> None:
    """Preview the intact runs and Submitit parameters without creating an executor."""
    total = sum(len(group["runs"]) for group in plan["jobs"])
    print(f"{plan['experiment']}: {len(plan['jobs'])} resource groups, {total} runs ({plan['scope']})")
    print(f"Submitit backend: {plan['backend']}; Python: {shlex.join(plan['python'])}")
    if plan["backend"] == "local":
        print("Local jobs run sequentially. Resource requirements describe the hardware needed.")
    if plan.get("note"):
        print(plan["note"])
    for group in plan["jobs"]:
        print(f"\n{group['id']}: {len(group['runs'])} runs")
        print("  datasets: " + ", ".join(group["datasets"]))
        if group["result_datasets"] != group["datasets"]:
            print("  evaluation targets: " + ", ".join(group["result_datasets"]))
        print("  models: " + ", ".join(group["models"]))
        print("  rule: " + group["resource_rule"])
        print("  reason: " + group["reason"])
        print("  resource profile: " + group["resources"]["profile"])
        print("  Submitit parameters: " + json.dumps(group["resources"]["parameters"]))


def main(argv=None) -> int:
    """Keep planning and execution separate so inspection never launches experiments."""
    args = parse_args(argv)
    root = args.repo_root.resolve()
    try:
        if args.list:
            for name, recipe in load_catalog(root).items():
                print(f"{name}\t{recipe['section']}\t{recipe['title']}")
            return 0
        cluster = None
        if not args.local:
            cluster_path = args.cluster or root / "configs/slurm/hydra_cluster.yaml"
            if not cluster_path.is_file():
                raise ValueError(f"Cluster configuration not found: {cluster_path}")
            cluster = read_yaml_mapping(cluster_path)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + secrets.token_hex(3)
        output = args.output or root / "results" / ("local" if args.local else "slurm") / stamp
        plan = build_plan(
            root,
            args.experiment,
            cluster,
            output,
            datasets=args.dataset,
            models=args.model,
            preprocessing=args.preprocessing,
            scope=args.scope,
            force=args.force,
            results_root=args.results_root,
        )
        if args.json:
            print(json.dumps(plan, indent=2))
        else:
            show_plan(plan)
        if args.write or args.submit or args.run:
            write_bundle(plan)
        if args.submit or args.run:
            submit_plan(plan)
        return 0
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        raise SystemExit(f"Experiment launch failed: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
