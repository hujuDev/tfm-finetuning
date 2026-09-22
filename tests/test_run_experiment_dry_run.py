from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_foundation_mainline_dry_run_emits_expanded_plan_without_results(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_root = tmp_path / "results"
    config_path = repo_root / "configs/experiments/unified_foundation_baseline.yaml"

    result = subprocess.run(
        [
            sys.executable,
            "src/experiments/run_experiment.py",
            "--config",
            config_path.as_posix(),
            "--dataset",
            "churn",
            "--dry-run",
        ],
        cwd=repo_root,
        env={**os.environ, "REPO_ROOT": str(repo_root), "PYTHONPATH": str(repo_root / "src"), "RESULTS_ROOT": str(results_root)},
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["runs"][0]["study"] == "unified_foundation_baseline"
    assert payload["run_count"] > 0
    assert payload["runs"][0]["dataset"] == "churn"
    assert not results_root.exists()
