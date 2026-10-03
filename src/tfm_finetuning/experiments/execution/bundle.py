"""Save a reviewable launch plan without starting Submitit jobs."""

from __future__ import annotations

import json
from pathlib import Path


def write_bundle(plan: dict) -> None:
    """Save fully resolved runs, e.g. ``write_bundle(build_plan(...))``."""
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
