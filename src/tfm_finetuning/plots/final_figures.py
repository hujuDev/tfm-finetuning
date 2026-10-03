"""Generate the final thesis figures and appendix tables from fresh runs.

Example: ``python -m tfm_finetuning.plots.final_figures --list`` lists all 35 outputs without
loading results. ``python -m tfm_finetuning.plots.final_figures --group transformations`` runs
one analysis group; omit ``--group`` to generate every group in dependency order.
"""

from __future__ import annotations

import argparse
import importlib
import json
import tomllib
from pathlib import Path

from tfm_finetuning.plots.results import validate_study

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs/plots/final_figures.toml"


def generate(config_path: Path = DEFAULT_CONFIG, *, groups: list[str] | None = None) -> dict[str, object]:
    """Run configured evidence-to-figure entrypoints and verify declared outputs."""
    config = tomllib.loads(config_path.read_text())
    selected = groups or list(config["groups"])
    unknown = set(selected) - set(config["groups"])
    if unknown:
        raise ValueError(f"Unknown final-figure groups: {sorted(unknown)}")
    output = REPO_ROOT / config["output"]["root"]
    result = {}
    validated = set()
    for name in selected:
        group = config["groups"][name]
        inputs = tomllib.loads((REPO_ROOT / group["config"]).read_text())["input"]
        for study in group.get("studies", []):
            if study not in validated:
                validate_study(REPO_ROOT / inputs["results_root"], study, repo_root=REPO_ROOT)
                validated.add(study)
        module, function = group["entrypoint"].split(":", 1)
        entrypoint = getattr(importlib.import_module(module), function)
        result[name] = entrypoint(config_path=REPO_ROOT / group["config"])
        missing = [filename for filename in group["assets"] if not (output / filename).is_file()]
        if missing:
            raise RuntimeError(f"Group {name} did not generate declared final assets: {missing}")
    return result


def main() -> None:
    """List or regenerate the submitted artifacts using their explicit input configs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--group", action="append", dest="groups", help="Run one named analysis group; repeatable.")
    parser.add_argument("--list", action="store_true", help="Print expected artifacts and input configurations.")
    args = parser.parse_args()
    if args.list:
        config = tomllib.loads(args.config.read_text())
        for name, group in config["groups"].items():
            print(f"{name}: {group['config']}")
            for asset in group["assets"]:
                print(f"  {asset}")
        return
    if Path.cwd().resolve() != REPO_ROOT:
        raise RuntimeError(f"Run figure generation from the repository root: {REPO_ROOT}")
    print(json.dumps(generate(args.config, groups=args.groups), indent=2))


if __name__ == "__main__":
    main()
