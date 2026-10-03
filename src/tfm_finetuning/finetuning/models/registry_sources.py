"""Installed-package discovery for registry entries."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
from pathlib import Path
from typing import Any


def ensure_tabicl_on_path(*, repo_root: Path | None = None) -> Path:
    """Return the installed TabICL package directory.

    Example:
        >>> ensure_tabicl_on_path(repo_root=Path.cwd()).name == "tabicl"
        True
    """
    del repo_root
    spec = importlib.util.find_spec("tabicl")
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError(
            "The TabICL package is not importable. Run `uv sync` so the "
            "pyproject.toml Git dependency for TabICL is installed."
        )
    return Path(spec.origin).resolve().parent


def ensure_mitra_on_path(*, repo_root: Path | None = None) -> Path:
    """Return the installed AutoGluon Mitra package directory.

    Example:
        >>> ensure_mitra_on_path(repo_root=Path.cwd()).name in {"mitra", "models"}
        True
    """
    del repo_root
    spec = importlib.util.find_spec("autogluon.tabular.models.mitra.sklearn_interface")
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError(
            "The AutoGluon Mitra sklearn interface is not importable. Run `uv sync` so "
            "the pinned AutoGluon package with Mitra support is installed."
        )
    return Path(spec.origin).resolve().parent


def installed_module_version(module_name: str) -> str | None:
    """Return a module's declared version when import metadata exposes one."""
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    version = getattr(module, "__version__", None)
    if version is not None:
        return str(version)
    try:
        about = importlib.import_module(f"{module_name}.__about__")
    except Exception:
        return None
    about_version = getattr(about, "__version__", None)
    return str(about_version) if about_version is not None else None


def installed_distribution_version(*names: str) -> str | None:
    """Return the first installed distribution version among aliases."""
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def mitra_version_payload(source_path: Path) -> dict[str, Any]:
    """Describe the installed Mitra source used by a registry entry."""
    return {
        "source_available": True,
        "source_type": "installed_package",
        "source_resolved_path": str(source_path),
        "package_name": "autogluon.tabular",
        "package_version": installed_module_version("autogluon")
        or installed_distribution_version("autogluon.tabular", "autogluon"),
        "remote_url": None,
        "git_commit": None,
    }


__all__ = (
    "ensure_mitra_on_path",
    "ensure_tabicl_on_path",
    "installed_distribution_version",
    "installed_module_version",
    "mitra_version_payload",
)
