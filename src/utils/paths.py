from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_env_file(env_path: Path) -> None:
    """Load key=value pairs from an .env file into the environment."""
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_repo_root(from_path: Path | None = None) -> Path:
    """Resolve the repository root from environment variables or sentinel files."""
    if "REPO_ROOT" in os.environ:
        return Path(os.environ["REPO_ROOT"]).resolve()
    if from_path is None:
        from_path = Path(__file__)
    start = from_path.resolve()
    if start.is_file():
        start = start.parent
    repo_root = None
    for candidate in [start, *start.parents]:
        env_path = candidate / ".env"
        if env_path.exists():
            load_env_file(env_path)
            if "REPO_ROOT" in os.environ:
                return Path(os.environ["REPO_ROOT"]).resolve()
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
            repo_root = candidate
            break
    if repo_root is None:
        repo_root = start
    os.environ.setdefault("REPO_ROOT", str(repo_root))
    return Path(os.environ["REPO_ROOT"]).resolve()


@dataclass(frozen=True)
class RepoPaths:
    """Resolved repository paths for results, logs, and checkpoints."""

    repo_root: Path
    results_root: Path
    logs_root: Path
    saves_root: Path


def get_paths(config: dict | None = None, *, repo_root: Path | None = None) -> RepoPaths:
    """Build RepoPaths using config overrides and environment variables."""
    repo_root = repo_root or get_repo_root(Path(__file__))
    paths_cfg = {}
    if isinstance(config, dict):
        paths_cfg = config.get("paths", {}) or {}

    def pick(name: str, env_key: str, default_rel: str) -> Path:
        if isinstance(paths_cfg, dict) and name in paths_cfg:
            value = Path(str(paths_cfg[name])).expanduser()
            return value if value.is_absolute() else repo_root / value
        env_val = os.environ.get(env_key)
        if env_val:
            value = Path(env_val).expanduser()
            return value if value.is_absolute() else repo_root / value
        return repo_root / default_rel

    return RepoPaths(
        repo_root=repo_root,
        results_root=pick("results_root", "RESULTS_ROOT", "results"),
        logs_root=pick("logs_root", "LOGS_ROOT", "logs"),
        saves_root=pick("saves_root", "SAVES_ROOT", "saves"),
    )
