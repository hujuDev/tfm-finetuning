from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tfm_finetuning.finetuning.models.base import FoundationModelAdapter
from tfm_finetuning.finetuning.models.registry_sources import (
    ensure_mitra_on_path,
    ensure_tabicl_on_path,
    installed_module_version,
    mitra_version_payload,
)
from tfm_finetuning.finetuning.models.source import installed_package_version_payload
from tfm_finetuning.finetuning.models.tabpfn.official import ensure_official_tabpfn_on_path
from tfm_finetuning.finetuning.models.tabpfn.versions import TABPFN_VERSIONS, TabPFNVersion, get_tabpfn_version
from tfm_finetuning.utils.paths import get_repo_root

TABICL_V2_CLASSIFIER_CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"
TABICL_V2_REGRESSOR_CHECKPOINT = "tabicl-regressor-v2-20260212.ckpt"
MITRA_CLASSIFIER_CHECKPOINT = "autogluon/mitra-classifier"
MITRA_REGRESSOR_CHECKPOINT = "autogluon/mitra-regressor"


class FoundationModelRegistryError(RuntimeError):
    """Raised when a foundation model cannot be resolved."""


@dataclass(frozen=True)
class FoundationModelSpec:
    """Registry entry for an adapter-backed foundation model.

    Example:
        >>> spec = get_foundation_model_spec("tabpfn_v2_5")
        >>> spec.display_name
        'TabPFN v2.5'
    """

    model_id: str
    display_name: str
    adapter_cls_path: str
    supports_classification: bool
    supports_regression: bool
    classifier_checkpoint: str | None
    regressor_checkpoint: str | None
    source_path: Path
    version_payload: dict[str, Any]

    @property
    def adapter_cls(self) -> type[FoundationModelAdapter]:
        """Import and return the adapter class lazily."""
        module_name, _, class_name = self.adapter_cls_path.rpartition(".")
        if not module_name or not class_name:
            raise FoundationModelRegistryError(f"Invalid adapter path for {self.model_id}: {self.adapter_cls_path}")
        module = importlib.import_module(module_name)
        adapter_cls = getattr(module, class_name)
        if not isinstance(adapter_cls, type):
            raise FoundationModelRegistryError(f"Adapter is not a class for {self.model_id}: {self.adapter_cls_path}")
        return adapter_cls

    def ensure_source_available(self) -> None:
        """Fail loudly when package source metadata is unavailable."""
        if self.version_payload.get("source_available"):
            return
        raise FoundationModelRegistryError(
            f"{self.model_id} requires installed Git package metadata for {self.source_path}. "
            "Run `uv sync` before launching foundation-model runs."
        )


def _tabpfn_spec(
    repo_root: Path,
    *,
    version: TabPFNVersion,
) -> FoundationModelSpec:
    source_path = ensure_official_tabpfn_on_path(repo_root=repo_root)
    payload = installed_package_version_payload(
        repo_root=repo_root,
        package_name="tabpfn",
        module_name="tabpfn",
        package_version=installed_module_version("tabpfn"),
    )
    return FoundationModelSpec(
        model_id=version.model_id,
        display_name=version.display_name,
        adapter_cls_path="tfm_finetuning.finetuning.models.tabpfn.model.TabPFNAdapter",
        supports_classification=True,
        supports_regression=True,
        classifier_checkpoint=version.classifier_checkpoint,
        regressor_checkpoint=version.regressor_checkpoint,
        source_path=source_path,
        version_payload={
            **payload,
            "checkpoint_identifier_classifier": version.classifier_checkpoint,
            "checkpoint_identifier_regressor": version.regressor_checkpoint,
        },
    )


def _mitra_spec(repo_root: Path) -> FoundationModelSpec:
    source_path = ensure_mitra_on_path(repo_root=repo_root)
    return FoundationModelSpec(
        model_id="mitra",
        display_name="Mitra",
        adapter_cls_path="tfm_finetuning.finetuning.models.mitra.model.MitraAdapter",
        supports_classification=True,
        supports_regression=True,
        classifier_checkpoint=MITRA_CLASSIFIER_CHECKPOINT,
        regressor_checkpoint=MITRA_REGRESSOR_CHECKPOINT,
        source_path=source_path,
        version_payload={
            **mitra_version_payload(source_path),
            "checkpoint_identifier_classifier": MITRA_CLASSIFIER_CHECKPOINT,
            "checkpoint_identifier_regressor": MITRA_REGRESSOR_CHECKPOINT,
        },
    )


def _tabicl_spec(repo_root: Path) -> FoundationModelSpec:
    source_path = ensure_tabicl_on_path(repo_root=repo_root)
    payload = installed_package_version_payload(
        repo_root=repo_root,
        package_name="tabicl",
        module_name="tabicl",
        package_version=installed_module_version("tabicl"),
    )
    return FoundationModelSpec(
        model_id="tabicl_v2",
        display_name="TabICLv2",
        adapter_cls_path="tfm_finetuning.finetuning.models.tabicl.model.TabICLAdapter",
        supports_classification=True,
        supports_regression=True,
        classifier_checkpoint=TABICL_V2_CLASSIFIER_CHECKPOINT,
        regressor_checkpoint=TABICL_V2_REGRESSOR_CHECKPOINT,
        source_path=source_path,
        version_payload={
            **payload,
            "checkpoint_identifier_classifier": TABICL_V2_CLASSIFIER_CHECKPOINT,
            "checkpoint_identifier_regressor": TABICL_V2_REGRESSOR_CHECKPOINT,
        },
    )


def get_foundation_model_spec(
    model_id: str,
    *,
    repo_root: Path | None = None,
) -> FoundationModelSpec:
    """Return one foundation-model spec by stable model id.

    Example:
        >>> get_foundation_model_spec("tabicl_v2").classifier_checkpoint
        'tabicl-classifier-v2-20260212.ckpt'
    """
    root = repo_root or get_repo_root(Path(__file__))
    normalized = str(model_id).strip().lower()
    tabpfn_version = get_tabpfn_version(normalized)
    if tabpfn_version is not None:
        spec = _tabpfn_spec(root, version=tabpfn_version)
        return spec
    factories = {
        "tabicl_v2": _tabicl_spec,
        "mitra": _mitra_spec,
    }
    try:
        spec = factories[normalized](root)
    except KeyError as exc:
        supported = ", ".join(sorted([*factories, *(version.model_id for version in TABPFN_VERSIONS)]))
        raise FoundationModelRegistryError(
            f"Unknown foundation model_id={model_id!r}; expected one of {supported}."
        ) from exc
    return spec


def iter_foundation_model_specs(*, repo_root: Path | None = None) -> list[FoundationModelSpec]:
    """Return all foundation model specs in canonical order."""
    return [
        *(get_foundation_model_spec(version.model_id, repo_root=repo_root) for version in TABPFN_VERSIONS),
        get_foundation_model_spec("tabicl_v2", repo_root=repo_root),
        get_foundation_model_spec("mitra", repo_root=repo_root),
    ]
