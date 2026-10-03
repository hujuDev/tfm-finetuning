"""Explicit run directories and reproducible configuration snapshots.

Example: ``prepare_output(config, Path("results/demo"), force=False)``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import tomli_w


def prepare_output(config: dict[str, Any], output: Path, *, force: bool) -> Path:
    """Reserve the caller's directory; require an explicit overwrite for existing runs."""
    output = output.expanduser().resolve()
    entries = list(output.iterdir()) if output.exists() else []
    marker = output / "run.json"
    try:
        state = json.loads(marker.read_text())
        managed = (
            state.get("schema_version") == 1
            and state.get("status") in {"running", "failed", "complete"}
            and isinstance(state.get("config"), dict)
            and bool(state.get("study"))
            and bool(state.get("run_id"))
        )
    except (OSError, ValueError, AttributeError):
        managed = False
    controls = {"run.json", ".lock"} if managed else set()
    artifacts = [path for path in entries if path.name not in controls]
    if artifacts:
        if not force:
            raise FileExistsError(f"Run directory is not empty: {output}. Choose another output or use --force.")
        if not managed and not (output / "resolved_config.toml").is_file():
            raise ValueError(f"Refusing to overwrite a directory without resolved_config.toml: {output}")
        # Preserve the directory and lock inode while removing this run's old artifacts.
        for path in artifacts:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
    output.mkdir(parents=True, exist_ok=True)
    snapshot = json.loads(json.dumps(config, default=str))
    (output / "resolved_config.toml").write_text(tomli_w.dumps(snapshot), encoding="utf-8")
    return output
