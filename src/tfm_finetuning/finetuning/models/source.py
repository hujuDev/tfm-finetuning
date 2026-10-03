from __future__ import annotations

import importlib.util
import tomllib
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

if TYPE_CHECKING:
    from pathlib import Path


def _split_uv_git_source(git_source: str) -> tuple[str | None, str | None]:
    source_without_commit, separator, commit = git_source.partition("#")
    split_result = urlsplit(source_without_commit)
    remote_url = urlunsplit((split_result.scheme, split_result.netloc, split_result.path, "", ""))
    return remote_url or None, commit if separator and commit else None


def _locked_git_source_payload(repo_root: Path, *, package_name: str) -> tuple[str | None, str | None]:
    lock_path = repo_root / "uv.lock"
    if not lock_path.exists():
        return None, None
    try:
        lock_payload = tomllib.loads(lock_path.read_text())
    except tomllib.TOMLDecodeError:
        return None, None
    for package in lock_payload.get("package", []):
        if package.get("name") != package_name:
            continue
        source = package.get("source")
        if not isinstance(source, dict):
            return None, None
        git_source = source.get("git")
        if not isinstance(git_source, str):
            return None, None
        return _split_uv_git_source(git_source)
    return None, None


def installed_package_version_payload(
    *,
    repo_root: Path,
    package_name: str,
    module_name: str,
    package_version: str | None,
) -> dict[str, Any]:
    """Build reproducibility metadata for an installed Git package.

    Example:
        >>> payload = installed_package_version_payload(
        ...     repo_root=Path("/missing"),
        ...     package_name="missing",
        ...     module_name="missing",
        ...     package_version=None,
        ... )
        >>> payload["source_available"]
        False
    """
    remote_url, git_commit = _locked_git_source_payload(repo_root, package_name=package_name)
    spec = importlib.util.find_spec(module_name)
    package_path = None
    if spec is not None and spec.origin is not None:
        package_path = str(spec.origin)
    return {
        "source_path": None,
        "source_available": remote_url is not None,
        "source_resolved_path": package_path,
        "remote_url": remote_url,
        "git_commit": git_commit,
        "git_describe": git_commit[:7] if git_commit else None,
        "package_version": package_version,
    }
