"""Provenance of installed packages pinned in uv.lock, from Git or from a package registry."""

from __future__ import annotations

from pathlib import Path

from tfm_finetuning.finetuning.models.source import installed_package_version_payload

GIT_LOCK = """
version = 1

[[package]]
name = "tabicl"
version = "2.1.1"
source = { git = "https://github.com/soda-inria/tabicl.git#86b12f989fb6a1bab2b09e4a867a37a91366ae2d" }
"""

REGISTRY_LOCK = """
version = 1

[[package]]
name = "tabpfn"
version = "9.2.0"
source = { registry = "https://pypi.org/simple" }
"""


def _payload(tmp_path: Path, lock: str, package: str, version: str | None) -> dict:
    (tmp_path / "uv.lock").write_text(lock)
    return installed_package_version_payload(
        repo_root=tmp_path, package_name=package, module_name=package, package_version=version
    )


def test_git_install_resolves_to_the_locked_commit(tmp_path: Path) -> None:
    payload = _payload(tmp_path, GIT_LOCK, "tabicl", "2.1.1")

    assert payload["source_available"] is True
    assert payload["git_commit"] == "86b12f989fb6a1bab2b09e4a867a37a91366ae2d"
    assert payload["remote_url"] == "https://github.com/soda-inria/tabicl.git"


def test_registry_install_is_identified_by_its_version(tmp_path: Path) -> None:
    payload = _payload(tmp_path, REGISTRY_LOCK, "tabpfn", "9.2.0")

    assert payload["source_available"] is True
    assert payload["git_commit"] is None
    assert payload["remote_url"] is None
    assert payload["package_version"] == "9.2.0"


def test_installed_version_that_differs_from_the_lock_is_not_available(tmp_path: Path) -> None:
    assert _payload(tmp_path, REGISTRY_LOCK, "tabpfn", "9.0.0")["source_available"] is False
